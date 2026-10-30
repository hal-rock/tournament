"""Residual activations for the preference probe (PLAN_preference_probe.md, steps 1-2).

Runs every prompt of probe_prompts.extraction_items through one model, one prompt per forward pass (no
padding, so no batch-shape drift), and keeps the residual stream at each prompt's read positions at the
OUTPUT of every block (matskit at="post"; HF hidden_states[b+1]; block b = 0..L-1, as ../coinflip/circuits).

Saved per read site, averaged over the fixed partners (4, or 3 for a partner task itself):

    single_eot.npy, single_last.npy   [tasks, L, D]      the task alone (Gilg et al.'s format)
    assistant_task.npy                [tasks, 2, L, D]   the task's last token, slot A / slot B, Assistant prompt
    coinflip_task.npy                 [tasks, 2, L, D]   the same in the coinflip prompt
    coinflip_label_b.npy              [tasks, L, D]      " tails" with the task in slot A
    choice_task.npy, choice_label_b.npy [tasks, L, D]    --kinds choice: ../coinflip's direct-choice prompt, task
                                                         in slot A, its closing token and " B" (one partner)

Stored as bf16 bit patterns (int16; read with load_acts), so values are exactly the model's, with no
fp16 overflow from massive activations. tasks.json lists the task order. items.jsonl has one row per
prompt: kind, task, partner, slot, and L = log P(first outcome) - log P(second outcome) at the last
token (assistant: " A" vs " B"; coinflip: " heads" vs " tails"), for a check against the tournament's
utilities (choice: " A" vs " B"). summary.json has provenance, timings, and the max |activation|.

Usage (GPU pod): uv run python probe_acts.py <model> <stage> [--limit N] [--out outputs/probe]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from matskit import acts, io, models
from probe_prompts import DEFAULT_KINDS, all_tasks, extraction_items, load_design, outcome_ids

SITES = {  # name -> (kind, read, has slot axis)
    "single_eot": ("single", "eot", False),
    "single_last": ("single", "last", False),
    "assistant_task": ("assistant", "task", True),
    "coinflip_task": ("coinflip", "task", True),
    "coinflip_label_b": ("coinflip", "label_b", False),
    # --kinds choice (label-token follow-up): ../coinflip's direct-choice prompt, task in slot A, one partner
    "choice_task": ("choice", "task", False),
    "choice_label_b": ("choice", "label_b", False),
}


def to_bits(x: torch.Tensor) -> np.ndarray:
    return x.to(torch.bfloat16).view(torch.int16).numpy()


def load_acts(path, mmap=True) -> np.ndarray:
    """A saved site as a float32-convertible array view: returns the int16 bits; use as_float() per slice."""
    return np.load(path, mmap_mode="r" if mmap else None)


def as_float(bits: np.ndarray) -> np.ndarray:
    """bf16 bit patterns (int16) -> float32."""
    return torch.from_numpy(np.ascontiguousarray(bits)).view(torch.bfloat16).float().numpy()


@torch.no_grad()
def extract(model, tok, items, task_ids, progress_every=200):
    n_layers = models.n_layers(model)
    D = models.text_config(model).hidden_size
    row = {t: i for i, t in enumerate(task_ids)}
    kinds = {it.kind for it in items}
    sites = {k: v for k, v in SITES.items() if v[0] in kinds}
    sums, counts = {}, {}
    for name, (_, _, slotted) in sites.items():
        shape = (len(task_ids), 2) if slotted else (len(task_ids),)
        sums[name] = torch.zeros(*shape, n_layers, D)
        counts[name] = torch.zeros(*shape)
    first_second = {k: outcome_ids(tok, k) for k in ("assistant", "coinflip", "choice")}
    records, max_abs, t0 = [], 0.0, time.time()
    layers = list(range(n_layers))
    for k, item in enumerate(items):
        reads = list(item.read.items())
        pos = torch.tensor([p for _, p in reads])
        ids = torch.tensor([item.ids])
        with acts.capture(model, layers, at="post", reduce=lambda h, l: h[0, pos.to(h.device)].float().cpu()) as cap:
            logits = model(input_ids=ids.to(models.input_device(model)), use_cache=False, logits_to_keep=1).logits
        h = torch.stack([cap[l] for l in layers], 1)  # [reads, L, D]
        if not torch.isfinite(h).all() or not torch.isfinite(logits).all():
            raise RuntimeError(f"non-finite activations or logits at item {k} ({item.meta})")
        max_abs = max(max_abs, h.abs().max().item())
        rec = dict(kind=item.kind, **item.meta, n_tokens=len(item.ids))
        if item.kind in first_second:
            lp = torch.log_softmax(logits[0, -1].float(), -1)
            a, b = first_second[item.kind]
            rec["L"] = (lp[a] - lp[b]).item()
        records.append(rec)
        r = row[item.meta["task"]]
        for j, (read, _) in enumerate(reads):
            name = f"{item.kind}_{read}"
            if name not in sites:
                continue
            idx = (r, "ab".index(item.meta["slot"])) if SITES[name][2] else (r,)
            sums[name][idx] += h[j]
            counts[name][idx] += 1
        if progress_every and (k + 1) % progress_every == 0:
            el = time.time() - t0
            print(f"item {k + 1}/{len(items)}  {el / 60:.1f} min elapsed, ~{el / (k + 1) * (len(items) - k - 1) / 60:.1f} "
                  "min left", flush=True)
    means = {}
    for name in sites:
        c = counts[name]
        assert (c > 0).all(), f"{name}: some tasks have no reads"
        means[name] = sums[name] / c.reshape(*c.shape, 1, 1)
    return means, records, dict(max_abs=max_abs, seconds=time.time() - t0, n_items=len(items))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("stage")
    ap.add_argument("--limit", type=int, help="first N tasks only (smoke tests)")
    ap.add_argument("--out", default="outputs/probe")
    ap.add_argument("--cpu", action="store_true", help="tiny local models: fp32 on CPU, no sanity check")
    ap.add_argument("--kinds", nargs="+", default=list(DEFAULT_KINDS),
                    help="prompt kinds (single assistant coinflip choice); a non-default set writes items_<kinds>.jsonl "
                         "and summary_<kinds>.json beside the default run's files")
    a = ap.parse_args()
    suffix = "" if tuple(a.kinds) == DEFAULT_KINDS else "_" + "_".join(a.kinds)

    out = Path(a.out) / a.stage / "acts"
    out.mkdir(parents=True, exist_ok=True)
    kw = dict(device_map="cpu", dtype="float32", check=False) if a.cpu else {}
    model, tok = models.load(a.model, **kw)
    design = load_design()
    task_ids = list(all_tasks(design))[:a.limit] if a.limit else list(all_tasks(design))
    items = extraction_items(tok, design, limit=a.limit, kinds=tuple(a.kinds))
    print(f"{a.model}: {len(items)} prompts over {len(task_ids)} tasks, {models.n_layers(model)} blocks", flush=True)
    means, records, info = extract(model, tok, items, task_ids)
    for name, m in means.items():
        np.save(out / f"{name}.npy", to_bits(m))
    (out / "tasks.json").write_text(json.dumps(task_ids))
    with open(out / f"items{suffix}.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    summary = dict(model=a.model, stage=a.stage, layers="block outputs (at='post'), 0..L-1",
                   n_layers=models.n_layers(model), d_model=models.text_config(model).hidden_size,
                   design=load_design() | {"benign_pairs": len(design["benign_pairs"])},
                   provenance=io.provenance(), **info)
    (out / f"summary{suffix}.json").write_text(json.dumps(summary, indent=1))
    print(f"done: {info['n_items']} prompts in {info['seconds'] / 60:.1f} min; max |h| {info['max_abs']:.0f}; "
          f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
