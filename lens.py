"""Tournament prompts read at every depth with the logit lens.

tournament.py reads log P(first outcome) and log P(second outcome) from the model's output. Here the
same prompts are read at every depth: the residual stream at the last prompt position, passed through
the final norm and the unembedding, as in ../coinflip/logit_lens.py. Every pair is scored in both
orders, so lens_analysis.py can fit per-task utilities at each depth.

Depth convention (as ../coinflip): depth d is the residual stream after d decoder blocks. d = 0 is
the embeddings; d >= 1 is the output of block d-1 (a forward hook on layers[d-1], HF
hidden_states[d] for d < L). Depth L through the lens is the model's own output, and every prompt
checks that it matches the model's logits.

Per prompt and depth it saves log P of the two outcome tokens (log-softmax over the full vocab), the
lens's top token and its probability, and the entropy. Per prompt it also saves the model's own
log-probs, which should reproduce tournament.py's (same prompts, padding and batch size).

Output: outputs/lens/<run_id>/<stage>__<condition>.npz, uploaded to the W&B run (project tournament,
job_type logit_lens) as each condition finishes. Arrays are [prompts] or [prompts, depths], prompts in
select_pairs order, both orders (first/second hold task ids).

Usage, on a GPU pod: uv run python lens.py configs/lens-olmo3.1-32b-instruct.yaml
"""

import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml
from transformers import AutoTokenizer

from analyze import read_jsonl
from prompts import build_prompt, outcome_token_ids
from tournament import check_batching, load_model, select_pairs

# At depth L the lens must match the model's own outcome log-probs this closely. Both are bf16 logits
# from the same hidden state, but matmuls of different shapes can round differently; a wrong layer or
# norm would be off by whole nats. (../coinflip: OLMo's lens matched its output exactly.)
FINAL_TOLERANCE_NATS = 0.5


def decoder_parts(model):
    """(embedding, blocks, final norm, unembedding) of a Llama-style HF causal LM (OLMo 3 included)."""
    return model.model.embed_tokens, model.model.layers, model.model.norm, model.lm_head


@torch.no_grad()
def lens_score(model, tokenizer, items, batch_size):
    """items: [(prompt_text, (first_id, second_id))]. Returns {name: array} in item order.

    Padding as tournament.score: right-padded, length-sorted batches, so every row is computed as it
    would be alone. Hooks keep each row's residual at its last real token, at every depth.
    """
    embed, blocks, norm, head = decoder_parts(model)
    n_depths = len(blocks) + 1
    encoded = [tokenizer(p, add_special_tokens=False).input_ids for p, _ in items]
    order = sorted(range(len(items)), key=lambda k: len(encoded[k]))
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    out = dict(
        lp_first=np.zeros((len(items), n_depths), np.float32),
        lp_second=np.zeros((len(items), n_depths), np.float32),
        top_id=np.zeros((len(items), n_depths), np.int32),
        top_p=np.zeros((len(items), n_depths), np.float32),
        entropy=np.zeros((len(items), n_depths), np.float32),
        model_lp_first=np.zeros(len(items), np.float32),
        model_lp_second=np.zeros(len(items), np.float32),
        prompt_tokens=np.array([len(e) for e in encoded], np.int32),
    )

    store = {}

    def keep(depth):
        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            store[depth] = h[store["rows"], store["last"].to(h.device)].to(head.weight.device)
        return hook

    handles = [embed.register_forward_hook(keep(0))]
    handles += [block.register_forward_hook(keep(i + 1)) for i, block in enumerate(blocks)]
    try:
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            width = max(len(encoded[k]) for k in batch)
            input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
            mask = torch.zeros((len(batch), width), dtype=torch.long)
            for row, k in enumerate(batch):
                input_ids[row, :len(encoded[k])] = torch.tensor(encoded[k])
                mask[row, :len(encoded[k])] = 1
            last = mask.sum(-1) - 1
            store.clear()
            store["rows"], store["last"] = torch.arange(len(batch)), last
            n_keep = int(width - last.min())  # as tournament.score
            logits = model(input_ids=input_ids.to(model.device), attention_mask=mask.to(model.device),
                           logits_to_keep=n_keep).logits
            cols = (last - (width - n_keep)).to(logits.device)
            model_lp = torch.log_softmax(logits[torch.arange(len(batch), device=logits.device), cols].float(), -1)

            hidden = torch.stack([store[d] for d in range(n_depths)], 1)  # [B, depths, D], model dtype
            lens_logits = head(norm(hidden)).float()  # [B, depths, V]
            if not (torch.isfinite(lens_logits).all() and torch.isfinite(model_lp).all()):
                raise RuntimeError("non-finite logits")
            lp = torch.log_softmax(lens_logits, -1)
            del lens_logits
            ids = torch.tensor([items[k][1] for k in batch], device=lp.device)  # [B, 2]
            outcome_lp = lp.gather(2, ids[:, None, :].expand(-1, n_depths, -1))  # [B, depths, 2]
            top = lp.max(-1)
            entropy = -(lp.exp() * lp).sum(-1)
            model_outcome = model_lp.to(lp.device).gather(1, ids)  # [B, 2]
            for name, value in (("lp_first", outcome_lp[..., 0]), ("lp_second", outcome_lp[..., 1]),
                                ("top_id", top.indices), ("top_p", top.values.exp()), ("entropy", entropy),
                                ("model_lp_first", model_outcome[:, 0]), ("model_lp_second", model_outcome[:, 1])):
                out[name][batch] = value.cpu().numpy()
            del lp
    finally:
        for h in handles:
            h.remove()
    return out


def main():
    cfg = yaml.safe_load(open(sys.argv[1]))
    tasks = read_jsonl(cfg["tasks_file"])
    system_prompt = cfg.get("system_prompt", "none")
    rng = random.Random(cfg.get("seed", 0))
    if not torch.cuda.is_available() and cfg.get("require_gpu"):
        raise SystemExit("require_gpu is set but torch sees no usable GPU")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    run = wandb.init(project=cfg["wandb_project"], name=cfg.get("run_name"), job_type="logit_lens",
                     group=cfg.get("group"), config={**cfg, "n_tasks": len(tasks)})
    out_dir = Path("outputs") / "lens" / run.id
    out_dir.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump({**cfg, "wandb_run": run.path}, open(out_dir / "run_config.yaml", "w"))
    run.save(str(out_dir / "run_config.yaml"), base_path=str(out_dir), policy="now")

    pairs = select_pairs(len(tasks), cfg.get("max_pairs"), cfg.get("seed", 0))
    ordered = pairs + [(b, a) for a, b in pairs]
    print(f"{len(tasks)} tasks, {len(pairs)} pairs, {len(ordered)} prompts per condition", flush=True)

    for m in cfg["models"]:
        stage = m["stage"]
        tokenizer = AutoTokenizer.from_pretrained(m["name"])
        template_tokenizer = AutoTokenizer.from_pretrained(m.get("chat_template_from", m["name"]))
        model = load_model(m["name"], device, dtype)
        print(f"{m['name']}: {len(decoder_parts(model)[1])} blocks", flush=True)
        for condition in m["conditions"]:
            outcome_ids = tuple(outcome_token_ids(condition, tokenizer))
            items = [(build_prompt(condition, tasks[a]["prompt"], tasks[b]["prompt"], template_tokenizer,
                                   system_prompt), outcome_ids) for a, b in ordered]
            print(f"\n[{stage}/{condition}] example prompt:\n{items[0][0]}\n", flush=True)
            bc = cfg["batching_check"]
            check_batching(model, tokenizer, items, cfg["batch_size"], bc["n_prompts"], bc["tolerance"], rng)

            start, chunk, parts = time.time(), cfg.get("progress_every", 1000), []
            for i in range(0, len(items), chunk):
                parts.append(lens_score(model, tokenizer, items[i:i + chunk], cfg["batch_size"]))
                done = min(i + chunk, len(items))
                rate = done / (time.time() - start)
                print(f"  {stage}/{condition}: {done}/{len(items)} prompts, {rate:.1f}/s, "
                      f"ETA {(len(items) - done) / rate / 60:.1f} min", flush=True)
            res = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}

            final_diff = np.maximum(np.abs(res["lp_first"][:, -1] - res["model_lp_first"]),
                                    np.abs(res["lp_second"][:, -1] - res["model_lp_second"]))
            print(f"  depth L lens vs model output: max |diff| {final_diff.max():.2e} nats, "
                  f"median {np.median(final_diff):.2e}", flush=True)
            run.summary[f"{stage}/{condition}/max_final_lens_diff_nats"] = float(final_diff.max())
            if final_diff.max() > FINAL_TOLERANCE_NATS:
                raise RuntimeError(f"depth-L lens differs from the model's output by {final_diff.max():.3f} nats")
            mass = np.exp(res["lp_first"]) + np.exp(res["lp_second"])
            readable = np.flatnonzero(np.median(mass, 0) >= 0.1)
            print(f"  median outcome mass >= 0.1 at depths {readable.tolist()}", flush=True)

            path = out_dir / f"{stage}__{condition}.npz"
            np.savez_compressed(path, first=np.array([tasks[a]["id"] for a, _ in ordered]),
                                second=np.array([tasks[b]["id"] for _, b in ordered]),
                                outcome_ids=np.array(outcome_ids), **res)
            run.save(str(path), base_path=str(out_dir), policy="now")
            run.summary[f"{stage}/{condition}/seconds"] = time.time() - start
        del model
        torch.cuda.empty_cache()
    run.finish()


if __name__ == "__main__":
    main()
