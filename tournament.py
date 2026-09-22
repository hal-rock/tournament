"""Preference tournament: score every task pair, in both orders, under each
condition (see prompts.py), and fit per-task utilities (see analyze.py).

Each prompt is one forward pass; we read the log-probs of the two outcome
tokens at the last position. Nothing is generated, except by the optional
phrasing probe, which samples a few tokens of the Assistant's unprefilled reply
to check how it naturally phrases a choice.

Usage: uv run python tournament.py configs/local.yaml
"""

import csv
import gc
import itertools
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import wandb
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from analyze import analyze, log_to_wandb, read_jsonl
from prompts import build_prompt, outcome_token_ids


def score(model, tokenizer, items, batch_size, top_k=5):
    """items: [(prompt_text, (first_id, second_id))]. Returns one dict per item, in order.

    Batches are left-padded (so the last position is the real last token for
    every row) and length-sorted to keep padding small. Position ids are
    counted from each row's first real token.
    """
    encoded = [tokenizer(p, add_special_tokens=False).input_ids for p, _ in items]
    order = sorted(range(len(items)), key=lambda k: len(encoded[k]))
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    results = [None] * len(items)
    for start in range(0, len(order), batch_size):
        batch = order[start:start + batch_size]
        width = max(len(encoded[k]) for k in batch)
        input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for row, k in enumerate(batch):
            input_ids[row, width - len(encoded[k]):] = torch.tensor(encoded[k])
            mask[row, width - len(encoded[k]):] = 1
        positions = (mask.cumsum(-1) - 1).clamp(min=0)
        with torch.no_grad():
            logits = model(
                input_ids=input_ids.to(model.device), attention_mask=mask.to(model.device),
                position_ids=positions.to(model.device), logits_to_keep=1,
            ).logits[:, -1].float()
        logp = torch.log_softmax(logits, dim=-1).cpu()
        top = logp.topk(top_k)
        for row, k in enumerate(batch):
            a, b = items[k][1]
            results[k] = dict(
                logp_first=logp[row, a].item(), logp_second=logp[row, b].item(),
                prompt_tokens=len(encoded[k]),
                top_tokens=json.dumps([[tokenizer.decode([i]), round(math.exp(v), 4)]
                                       for v, i in zip(top.values[row].tolist(), top.indices[row].tolist())]),
            )
    return results


def check_batching(model, tokenizer, items, batch_size, n_prompts, tolerance, rng):
    """Score a few prompts of mixed lengths alone and together in one padded
    batch; padding/masking bugs show up as a mismatch. Returns the max
    |difference| in log-odds, and raises if it exceeds tolerance or is NaN."""
    sample = rng.sample(items, min(n_prompts, len(items)))
    alone = score(model, tokenizer, sample, batch_size=1)
    together = score(model, tokenizer, sample, batch_size=max(batch_size, len(sample)))
    diffs = [abs((x["logp_first"] - x["logp_second"]) - (y["logp_first"] - y["logp_second"]))
             for x, y in zip(alone, together)]
    worst = max(diffs)
    lengths = [r["prompt_tokens"] for r in alone]
    print(f"  batching check: {len(sample)} prompts, {min(lengths)}-{max(lengths)} tokens, "
          f"max |log-odds diff| = {worst:.2e} (tolerance {tolerance})", flush=True)
    if not worst <= tolerance:  # also catches NaN
        raise RuntimeError(f"batched and unbatched scores differ by {worst} > {tolerance}")
    return worst


def phrasing_probe(model, tokenizer, template_tokenizer, tasks, pairs, system_prompt, probe_cfg, rng):
    """Sample the start of the Assistant's reply with no prefill, for a few random pairs."""
    chosen = rng.sample(pairs, min(probe_cfg["n_pairs"], len(pairs)))
    prompts = [build_prompt("assistant", tasks[a]["prompt"], tasks[b]["prompt"], template_tokenizer,
                            system_prompt, prefill=False) for a, b in chosen]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    enc = tokenizer(prompts, add_special_tokens=False, padding=True, padding_side="left",
                    return_tensors="pt").to(model.device)
    torch.manual_seed(rng.randrange(2**31))
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=probe_cfg["max_new_tokens"], do_sample=True, temperature=1.0,
                             num_return_sequences=probe_cfg["n_samples"], pad_token_id=pad_id)
    completions = tokenizer.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    rows = []
    for k, text in enumerate(completions):
        a, b = chosen[k // probe_cfg["n_samples"]]
        rows.append([tasks[a]["id"], tasks[b]["id"], k % probe_cfg["n_samples"], text])
        print(f"  phrasing probe: {text!r}")
    return rows


def main():
    cfg = yaml.safe_load(open(sys.argv[1]))
    tasks = read_jsonl(cfg["tasks_file"])
    system_prompt = cfg.get("system_prompt", "none")
    batch_size = cfg["batch_size"]
    rng = random.Random(cfg.get("seed", 0))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    if cfg.get("require_gpu") and device != "cuda":
        raise SystemExit("require_gpu is set but torch sees no usable GPU (check the CUDA warning above)")
    run = wandb.init(project=cfg["wandb_project"], name=cfg.get("run_name"), job_type="tournament",
                     config={**cfg, "n_tasks": len(tasks)})
    out_dir = Path("outputs") / run.id
    out_dir.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump({**cfg, "wandb_run": run.path}, open(out_dir / "config.yaml", "w"))

    # Every unordered pair of task indices; each is scored in both orders.
    pairs = list(itertools.combinations(range(len(tasks)), 2))
    ordered = pairs + [(b, a) for a, b in pairs]
    print(f"{len(tasks)} tasks, {len(pairs)} pairs, {len(ordered)} prompts per condition")

    rows, probe_rows, examples = [], [], {}
    for m in cfg["models"]:
        stage = m["stage"]
        print(f"Loading {m['name']} on {device} ({dtype})", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(m["name"])
        template_tokenizer = AutoTokenizer.from_pretrained(m.get("chat_template_from", m["name"]))
        model = AutoModelForCausalLM.from_pretrained(m["name"], dtype=dtype, device_map=device).eval()

        for condition in m["conditions"]:
            outcome_ids = tuple(outcome_token_ids(condition, tokenizer))
            items = [(build_prompt(condition, tasks[a]["prompt"], tasks[b]["prompt"], template_tokenizer, system_prompt),
                      outcome_ids) for a, b in ordered]
            examples[f"{stage}/{condition}"] = items[0][0]
            print(f"\n[{stage}/{condition}] example prompt:\n{items[0][0]}\n", flush=True)

            bc = cfg["batching_check"]
            run.summary[f"{stage}/{condition}/batching_max_diff"] = check_batching(
                model, tokenizer, items, batch_size, bc["n_prompts"], bc["tolerance"], rng)

            start = time.time()
            scored = []
            for chunk in range(0, len(items), cfg.get("progress_every", 1000)):
                scored += score(model, tokenizer, items[chunk:chunk + cfg.get("progress_every", 1000)],
                                batch_size, cfg.get("top_k_tokens", 5))
                rate = len(scored) / (time.time() - start)
                print(f"  {stage}/{condition}: {len(scored)}/{len(items)} prompts, {rate:.1f}/s, "
                      f"ETA {(len(items) - len(scored)) / rate / 60:.1f} min", flush=True)
            run.summary[f"{stage}/{condition}/seconds"] = time.time() - start

            for (a, b), s in zip(ordered, scored):
                rows.append(dict(
                    model=m["name"], stage=stage, condition=condition,
                    first=tasks[a]["id"], second=tasks[b]["id"],
                    logp_first=s["logp_first"], logp_second=s["logp_second"],
                    log_odds=s["logp_first"] - s["logp_second"],
                    # How much probability the two outcome tokens get at all.
                    outcome_mass=math.exp(s["logp_first"]) + math.exp(s["logp_second"]),
                    prompt_tokens=s["prompt_tokens"], top_tokens=s["top_tokens"],
                ))

        if m.get("phrasing_probe"):
            probe_rows += [[stage] + r for r in phrasing_probe(
                model, tokenizer, template_tokenizer, tasks, pairs, system_prompt, m["phrasing_probe"], rng)]

        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    with open(out_dir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    log = {"results": wandb.Table(columns=list(rows[0]), data=[list(r.values()) for r in rows])}
    if probe_rows:
        log["phrasing_probe"] = wandb.Table(columns=["stage", "first", "second", "sample", "completion"], data=probe_rows)
    run.log(log)
    prompts_art = wandb.Artifact("example-prompts", type="prompts")
    with prompts_art.new_file("examples.yaml") as f:
        yaml.safe_dump(examples, f, allow_unicode=True, width=10_000)
    run.log_artifact(prompts_art)

    log_to_wandb(run, analyze(rows, tasks, out_dir / "analysis", seed=cfg.get("seed", 0)))
    run.finish()


if __name__ == "__main__":
    main()
