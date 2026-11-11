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
import gzip
import itertools
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml
from torch import nn
from transformers import (ApertusConfig, ApertusForCausalLM, AutoModelForCausalLM, AutoTokenizer, DeepseekV3Config,
                          DeepseekV3ForCausalLM)
from transformers.utils import cached_file

from analyze import analyze, log_to_wandb, read_jsonl
from prompts import build_prompt, outcome_token_ids


def score(model, tokenizer, items, batch_size, top_k=5):
    """items: [(prompt_text, (first_id, second_id))]. Returns one dict per item, in order.

    Batches are right-padded and length-sorted to keep padding small. Under a causal
    mask the real tokens never see the padding, so every row is computed exactly as
    it would be alone, whatever the model does with attention_mask or positions
    (DeepSeek V4, for one, compresses KV over windows of absolute positions, which
    left padding would shift). Logits are kept only at each row's last real token.
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
            input_ids[row, :len(encoded[k])] = torch.tensor(encoded[k])
            mask[row, :len(encoded[k])] = 1
        last = mask.sum(-1) - 1
        # Keep logits for the last n_keep positions, enough to reach every row's last real
        # token (few, as batches are length-sorted). An int, not a tensor of positions:
        # accelerate moves tensor arguments to the first GPU, but the hidden states it
        # indexes sit on the last one.
        n_keep = int(width - last.min())
        with torch.no_grad():
            logits = model(
                input_ids=input_ids.to(model.device), attention_mask=mask.to(model.device), logits_to_keep=n_keep,
            ).logits
        cols = (last - (width - n_keep)).to(logits.device)
        logits = logits[torch.arange(len(batch), device=logits.device), cols].float()
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


def write_csv_gz(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def check_batching(model, tokenizer, items, batch_size, n_prompts, tolerance, rng):
    """Score a few prompts of mixed lengths alone and together in one padded
    batch. A padding/masking bug shifts most prompts, so this raises if the median
    |difference| in log-odds exceeds tolerance (or any is NaN). bf16 kernels also
    round differently by batch shape, which some models amplify into rare outliers
    of a few nats (see configs/models/gemma4-31b.yaml), hence the median. Returns
    the max |difference|."""
    sample = rng.sample(items, min(n_prompts, len(items)))
    alone = score(model, tokenizer, sample, batch_size=1)
    together = score(model, tokenizer, sample, batch_size=max(batch_size, len(sample)))
    diffs = [abs((x["logp_first"] - x["logp_second"]) - (y["logp_first"] - y["logp_second"]))
             for x, y in zip(alone, together)]
    worst, median = max(diffs), float(np.median(diffs))
    lengths = [r["prompt_tokens"] for r in alone]
    print(f"  batching check: {len(sample)} prompts, {min(lengths)}-{max(lengths)} tokens, "
          f"|log-odds diff| median {median:.2e}, max {worst:.2e} (tolerance on the median {tolerance})", flush=True)
    if not (median <= tolerance and worst == worst):  # worst == worst is False for NaN
        raise RuntimeError(f"batched and unbatched scores differ: median {median}, max {worst} (tolerance {tolerance})")
    return worst


def phrasing_probe(model, tokenizer, template_tokenizer, tasks, pairs, system_prompt, probe_cfg, rng,
                   turn_format=None, close_think=False, strip_think=False):
    """Sample the start of the Assistant's reply with no prefill, for a few random pairs."""
    chosen = rng.sample(pairs, min(probe_cfg["n_pairs"], len(pairs)))
    prompts = [build_prompt("assistant", tasks[a]["prompt"], tasks[b]["prompt"], template_tokenizer,
                            system_prompt, prefill=False, turn_format=turn_format, close_think=close_think,
                            strip_think=strip_think)
               for a, b in chosen]
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


def select_pairs(n_tasks, max_pairs=None, seed=0):
    """Unordered pairs of task indices: all of them, or a random max_pairs of them.
    The draw depends only on (n_tasks, max_pairs, seed), so runs on the same task file
    with the same settings score the same pairs whatever else their configs change."""
    pairs = list(itertools.combinations(range(n_tasks), 2))
    if max_pairs is None or max_pairs >= len(pairs):
        return pairs
    return sorted(random.Random(f"pairs/{seed}").sample(pairs, max_pairs))


class ApertusPrunedHeadForCausalLM(ApertusForCausalLM):
    """Apertus 1.5's text model, as in ../coinflip (coinflip.py). The checkpoints are multimodal (`apertus1p5`), which
    released transformers can't read, but the text part is Apertus 1.0's `ApertusModel` unchanged, with input
    embeddings for the 266,752-id text+image+audio vocabulary and an LM head over the 131,072 text ids only. That is
    all swiss-ai's own `Apertus1p5TextForCausalLM` adds (swiss-ai/transformers@0c5e12f, modular_apertus1p5.py; its copy
    of modeling_apertus.py and the xIELU activation match transformers 5.17). Text-only input never reaches the image
    or audio tokenizers: the multimodal forward is embed -> language model -> lm_head."""

    def __init__(self, config):
        super().__init__(config)
        self.lm_head = nn.Linear(config.hidden_size, config.output_vocab_size, bias=False)
        self.post_init()


def load_apertus1p5_text(name, revision, dtype, device_map, max_memory=None):
    """Load the text model of an Apertus 1.5 checkpoint (see ApertusPrunedHeadForCausalLM), failing on any text
    weight missing or misshapen and on any skipped weight outside the image/audio tokenizers."""
    text = json.load(open(cached_file(name, "config.json", revision=revision)))["text_config"]
    assert text["model_type"] == "apertus1p5_text" and text["qk_norm"] and not text["post_norm"], text
    assert not text["tie_word_embeddings"] and text["output_vocab_size"] < text["vocab_size"], text
    config = ApertusConfig(**{k: v for k, v in text.items() if k not in ("model_type", "qk_norm", "post_norm")})
    model, info = ApertusPrunedHeadForCausalLM.from_pretrained(
        name, revision=revision, config=config, dtype=dtype, device_map=device_map, max_memory=max_memory,
        key_mapping={r"^model\.language_model\.": "model."}, output_loading_info=True)
    assert not info["missing_keys"] and not info["mismatched_keys"], (info["missing_keys"], info["mismatched_keys"])
    skipped = {k.split(".")[1] for k in info["unexpected_keys"]}
    assert skipped <= {"vision_tokenizer", "audio_tokenizer"}, skipped
    print(f"  apertus1p5 text model: skipped {len(info['unexpected_keys'])} image/audio tokenizer tensors", flush=True)
    return model


# Kimi K2's config.json fields that transformers' DeepseekV3 doesn't read, with the values its code assumes (sigmoid
# scores, top-k by score plus the correction bias, every layer from first_k_dense_replace on an MoE, no attention bias,
# no multi-token-prediction layer). The remote modeling_deepseek.py reads them; asserting them makes the two agree.
KIMI_FIXED = {"scoring_func": "sigmoid", "topk_method": "noaux_tc", "moe_layer_freq": 1, "attention_bias": False,
              "num_nextn_predict_layers": 0}


def kimi_k2_config(name, revision):
    """Kimi K2's config.json as transformers' native DeepseekV3Config. Its `model_type: kimi_k2` maps to remote code
    (configuration_deepseek.py, modeling_deepseek.py) written for transformers 4.48, which 5.17 doesn't know; the
    architecture is DeepseekV3ForCausalLM's (the config names it), with Kimi's sizes: 384 routed experts, 64 heads, one
    expert group, the first layer dense, yarn RoPE (factor 32, theta 50,000). The FP8 block quantization config is kept, so
    the weights load as stored (transformers' finegrained FP8). `rope_scaling` becomes rope_parameters as for DeepSeek V3's
    own config; scripts/patch_load_check.py checks the keys and shapes against the checkpoint without weights, and the
    resulting rotary frequencies and attention scale against the remote code's formulas (replication/patch_load_check.md).
    The checkpoint's per-layer rotary_emb.inv_freq buffers are a stale default (plain RoPE over hidden / heads = 112
    dimensions) that neither implementation uses; transformers skips them."""
    raw = json.load(open(cached_file(name, "config.json", revision=revision)))
    assert raw["model_type"] == "kimi_k2" and raw["architectures"] == ["DeepseekV3ForCausalLM"], raw.get("model_type")
    fixed = {k: raw.get(k) for k in KIMI_FIXED}
    assert fixed == KIMI_FIXED, fixed
    drop = ("model_type", "architectures", "auto_map", "transformers_version", "torch_dtype")
    config = DeepseekV3Config(**{k: v for k, v in raw.items() if k not in drop})
    assert config.rope_parameters["rope_type"] == "yarn" and config.quantization_config["quant_method"] == "fp8"
    return config


def load_kimi_k2(name, revision, dtype, device_map, max_memory=None, attn_implementation=None):
    """Kimi K2 as transformers' native DeepseekV3ForCausalLM (kimi_k2_config), failing on any weight missing or
    misshapen and on any skipped weight but the per-layer rotary inv_freq buffers the checkpoint carries (the native model
    computes its own, from the config)."""
    attn = {"attn_implementation": attn_implementation} if attn_implementation else {}
    model, info = DeepseekV3ForCausalLM.from_pretrained(
        name, revision=revision, config=kimi_k2_config(name, revision), dtype=dtype, device_map=device_map,
        max_memory=max_memory, output_loading_info=True, **attn)
    assert not info["missing_keys"] and not info["mismatched_keys"], (info["missing_keys"], info["mismatched_keys"])
    skipped = {k for k in info["unexpected_keys"] if not k.endswith(".self_attn.rotary_emb.inv_freq")}
    assert not skipped, sorted(skipped)[:10]
    return model


# `loader` in a model's config, for checkpoints AutoModelForCausalLM can't read.
LOADERS = {"apertus1p5_text": load_apertus1p5_text, "kimi_k2": load_kimi_k2}


def load_model(name, device, dtype, revision=None, max_memory=None, loader=None, attn_implementation=None):
    """On GPU, spread the model over all visible GPUs (one GPU if it fits there), and
    refuse to run with layers offloaded to CPU or disk: that is orders of magnitude slower.
    max_memory: {gpu: limit}, for a model that only just fits. loader: a key of LOADERS.
    attn_implementation: e.g. "eager", where the attention weights are needed (rollouts.py)."""
    attn = {"attn_implementation": attn_implementation} if attn_implementation else {}
    if loader is not None:
        load = lambda device_map: LOADERS[loader](name, revision, dtype, device_map, max_memory, **attn).eval()
    else:
        load = lambda device_map: AutoModelForCausalLM.from_pretrained(
            name, revision=revision, dtype=dtype, device_map=device_map, max_memory=max_memory, **attn).eval()
    if device != "cuda":
        return load(device)
    model = load("auto")
    placement = {str(t.device) for t in itertools.chain(model.parameters(), model.buffers())}
    if placement & {"cpu", "meta"}:
        raise SystemExit(f"{name} does not fit on the visible GPUs (devices: {sorted(placement)})")
    print(f"  placed on {len(placement)} device(s): {sorted(placement)}", flush=True)
    return model


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
    # group/tags tie together runs analyzed jointly (e.g. one run per model size; see scale.py).
    run = wandb.init(project=cfg["wandb_project"], name=cfg.get("run_name"), job_type="tournament",
                     group=cfg.get("group"), tags=cfg.get("tags"), config={**cfg, "n_tasks": len(tasks)})
    out_dir = Path("outputs") / run.id
    out_dir.mkdir(parents=True, exist_ok=True)
    # Not "config.yaml": W&B keeps its own file of that name in the run and would overwrite this one.
    yaml.safe_dump({**cfg, "wandb_run": run.path}, open(out_dir / "run_config.yaml", "w"))
    # Each finished condition is written to results/<stage>__<condition>.csv.gz and
    # uploaded to the W&B run's files right away, so a crash (or a failed batching
    # check) keeps it. (Too many rows for a W&B table at 300 tasks.) Recover with
    # analyze.py on the run dir or the W&B run path.
    run.save(str(out_dir / "run_config.yaml"), base_path=str(out_dir), policy="now")

    # Unordered pairs of task indices (all, or a max_pairs subsample); each is scored in both orders.
    pairs = select_pairs(len(tasks), cfg.get("max_pairs"), cfg.get("seed", 0))
    ordered = pairs + [(b, a) for a, b in pairs]
    print(f"{len(tasks)} tasks, {len(pairs)} pairs, {len(ordered)} prompts per condition")

    rows, examples = [], {}
    for m in cfg["models"]:
        stage = m["stage"]
        # revision: an HF branch, e.g. an intermediate checkpoint of ../coinflip's DPO retraining.
        revision = m.get("revision")
        print(f"Loading {m['name']}{'@' + revision if revision else ''} on {device} ({dtype})", flush=True)
        # tokenizer_from: for a repo without usable tokenizer files (OLMo 2 32B SFT has no tokenizer.json).
        tokenizer = AutoTokenizer.from_pretrained(m.get("tokenizer_from", m["name"]),
                                                  revision=None if "tokenizer_from" in m else revision)
        template_tokenizer = AutoTokenizer.from_pretrained(m.get("chat_template_from", m["name"]),
                                                           revision=None if "chat_template_from" in m else revision)
        model = load_model(m["name"], device, dtype, revision, m.get("max_memory"), m.get("loader"))
        if "use_cache" in m:  # no KV cache: scoring doesn't need one, and it costs memory on a tight fit
            model.config.use_cache = m["use_cache"]
        if m.get("loader"):  # a loader of our own: check it reads, with an answer a base model can't miss
            sanity = tokenizer("The capital of France is Paris, and the capital of Germany is", return_tensors="pt")
            with torch.no_grad():
                top = model(sanity.input_ids.to(model.device)).logits[0, -1].argmax().item()
            print(f"  sanity: ...the capital of Germany is -> {tokenizer.decode([top])!r}", flush=True)
            if tokenizer.decode([top]).strip() != "Berlin":
                raise SystemExit(f"{m['name']} fails the sanity prompt")

        if "doc_prefix" in m:
            # Raw-text conditions open with it: it must tokenize to the model's special token(s), not characters.
            prefix_ids = tokenizer.encode(m["doc_prefix"], add_special_tokens=False)
            assert prefix_ids and all(i in tokenizer.all_special_ids or i in tokenizer.added_tokens_decoder
                                      for i in prefix_ids), f"doc_prefix {m['doc_prefix']!r} is not special tokens: {prefix_ids}"

        for condition in m["conditions"]:
            outcome_ids = tuple(outcome_token_ids(condition, tokenizer))
            items = [(build_prompt(condition, tasks[a]["prompt"], tasks[b]["prompt"], template_tokenizer, system_prompt,
                                   turn_format=m.get("turn_format"), close_think=m.get("close_think", False),
                                   strip_think=m.get("strip_think", False), doc_prefix=m.get("doc_prefix")),
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

            new_rows = []
            for (a, b), s in zip(ordered, scored):
                new_rows.append(dict(
                    model=m["name"] + (f"@{revision}" if revision else ""), stage=stage, condition=condition,
                    first=tasks[a]["id"], second=tasks[b]["id"],
                    logp_first=s["logp_first"], logp_second=s["logp_second"],
                    log_odds=s["logp_first"] - s["logp_second"],
                    # How much probability the two outcome tokens get at all.
                    outcome_mass=math.exp(s["logp_first"]) + math.exp(s["logp_second"]),
                    prompt_tokens=s["prompt_tokens"], top_tokens=s["top_tokens"],
                ))
            path = out_dir / "results" / f"{stage}__{condition}.csv.gz"
            write_csv_gz(path, new_rows)
            run.save(str(path), base_path=str(out_dir), policy="now")
            # Keep what the analysis needs in memory; top_tokens is only for spot checks.
            rows += [{k: v for k, v in r.items() if k != "top_tokens"} for r in new_rows]

        if m.get("phrasing_probe"):
            probe_rows = [[stage] + r for r in phrasing_probe(
                model, tokenizer, template_tokenizer, tasks, pairs, system_prompt, m["phrasing_probe"], rng,
                m.get("turn_format"), m.get("close_think", False), m.get("strip_think", False))]
            run.log({f"phrasing_probe/{stage}": wandb.Table(
                columns=["stage", "first", "second", "sample", "completion"], data=probe_rows)})

        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    prompts_art = wandb.Artifact("example-prompts", type="prompts")
    with prompts_art.new_file("examples.yaml") as f:
        yaml.safe_dump(examples, f, allow_unicode=True, width=10_000)
    run.log_artifact(prompts_art)

    log_to_wandb(run, analyze(rows, tasks, out_dir / "analysis", seed=cfg.get("seed", 0),
                              stage_order=[m["stage"] for m in cfg["models"]]))
    run.finish()


if __name__ == "__main__":
    main()
