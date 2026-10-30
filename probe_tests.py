"""Is the utility probe's direction causal for the Assistant's choice and for the coinflip? (PLAN_preference_probe.md,
step 3.) Runs on a GPU pod, one model at a time.

Prompts (probe_prompts.test_pairs): HP puts the preferred task first (heads / A), TP second.
    standard   ../coinflip's 25 pairs (5 benign x 5 extremely harmful), token-identical to circuits/pairs.py
    benign     design["benign_pairs"]: 300 non-harmful pairs whose DPO Assistant utilities differ by >= 2
Formats: coinflip (L = logit " heads" - " tails") and choice (../coinflip's direct choice, prefill "I choose Task",
L = logit " A" - " B"). effect = (L_HP - L_TP) / 2, as in ../coinflip.

Directions (outputs/probe/directions.npz, written by probe_readout.py from the fits; one unit vector per block):
    probe_dpo         the DPO model's probe (target dpo/assistant, all 300 tasks, pairwise Assistant site)
    probe_sft         the SFT model's probe, same fit
    probe_dpo_benign  the DPO model's probe fit on the 257 non-harmful tasks only
    rand0..2          random Gaussian directions (seeded), the controls
Layers are block outputs (matskit at="post"), as everywhere in this repo's probe code.

Arms, per prompt:
    steer   add +c * s_b * u_b to every token of task A and -c * s_b * u_b to every token of task B at block b, where
            u_b is the unit direction and s_b the mean residual norm at block b (over the standard prompts, every
            position but 0). Single blocks, and "together" (every steering block at once, c / n_blocks each, so the
            total addition matches a single block's). Reads dL = L(steered) - L(unsteered row of the same batch).
    swap    at block b, shift each task's tokens along u_b so that the slot's mean projection becomes what the
            other order's prompt has in that slot (HP's slot A gets TP's slot-A mean projection, i.e. the other
            task's, and so on). If the prompt depended on the tasks only through u_b, HP would turn into TP:
            share = ((L_TP' - L_TP) - (L_HP' - L_HP)) / 4 / effect, 1 = all of the effect, 0 = none.
            "together" sets every task token's projection at every swap block to its unpatched value plus that
            block's shift (acts.clamp), so the shifts don't stack.
Label-token arms (--label, 2026-10-29 follow-up), at the label B outcome word (" tails" / " B"), which comes after
task A's text and before task B's, so it can only carry task A's value. Its probes (label_<format>_<probe> in
directions.npz) are fit at that token with task A's DPO Assistant utility as the target, same names as above:
    steer_label  +c * s_b * u_b at the label token only, same c grid and blocks as steer
    swap_label   the label token's projection on its probe set to the other order's value, single blocks and together
    swap_both    swap_label plus the task-token swap along the task probe of the same name
Each prompt's arms run in batches of identical rows (no padding), each batch with one unsteered row; the
unsteered row is also compared with an unbatched forward (L_unbatched), to show the batch-shape drift.

Output: outputs/probe/<stage>/tests/rows.jsonl (one row per arm and prompt) and summary.json (norms, timings).
--only-swap-together reruns just the all-blocks swap into rows_swap_together.jsonl (the first runs, 2026-10-29,
added the shifts at every block, which stacked; probe_effects.py uses the rerun's rows for those arms).
--label writes rows_label.jsonl.

Usage: uv run python probe_tests.py <model> <stage> [--sets standard benign] [--limit N] [--cpu]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from matskit import acts, io, models
from probe_prompts import load_design, outcome_ids, test_pairs

STEER_BLOCKS = list(range(15, 24))
SWAP_BLOCKS = list(range(15, 24))
C_STANDARD = [0.025, 0.05, 0.1, 0.2, 0.4]  # x mean residual norm; the probe projection's SD across tasks is ~0.1
C_BENIGN = [0.05, 0.1]
BENIGN_BLOCKS = [17, 20, 23, "together"]
PROBE_DIRS = ["probe_dpo", "probe_sft", "probe_dpo_benign"]
N_RANDOM = 3


def load_directions(path, n_layers, d_model, seed=0, prefix=""):
    """{name: [L, D] unit vectors}: the probes from directions.npz (keys prefix + name) plus seeded random ones.
    prefix "label_coinflip_" / "label_choice_" gives the label B token's probes for that prompt (seed 1 randoms)."""
    out = {}
    if Path(path).exists():
        with np.load(path) as f:
            out = {k: torch.tensor(f[prefix + k], dtype=torch.float32) for k in PROBE_DIRS if prefix + k in f}
    g = torch.Generator().manual_seed(seed)
    for i in range(N_RANDOM):
        out[f"rand{i}"] = torch.randn(n_layers, d_model, generator=g)
    return {k: v / v.norm(dim=-1, keepdim=True) for k, v in out.items()}


def seg_coef(item, values):
    """[T] coefficients: values[segment] on that segment's tokens, 0 elsewhere."""
    return torch.tensor([values.get(s, 0.0) for s in item.seg], dtype=torch.float32)


@torch.no_grad()
def baseline(model, item, blocks):
    """Unbatched L and the residual [T, D] at each of `blocks` (block outputs)."""
    ids = torch.tensor([item.ids])
    with acts.capture(model, blocks, at="post", reduce=lambda h, l: h[0].float().cpu()) as cap:
        logits = model(input_ids=ids.to(models.input_device(model)), use_cache=False, logits_to_keep=1).logits
    return logits[0, -1].float(), {b: cap[b] for b in blocks}


def logit_diff(logits, ids):
    a, b = ids
    return (logits[..., a] - logits[..., b]).float()


def task_positions(item):
    return torch.tensor([i for i, x in enumerate(item.seg) if x in ("task_a", "task_b")])


def clamp_positions(item, where):
    """Token positions a clamp edits: "task" = every token of both tasks, "label" = the label B outcome word."""
    return task_positions(item) if where == "task" else torch.tensor([item.read["label_b"]])


@torch.no_grad()
def run_arms(model, item, arms, out_ids, batch_size):
    """arms: [(label dict, add, clamp)], add = {block: (coef [T], vec [D])} (acts.steer_tokens), clamp = {(block, where):
    (unit direction [D], projections [P] at clamp_positions(item, where))} (acts.clamp). Returns (L per arm, L of the
    unsteered row per arm's batch). Each batch is identical copies of the prompt plus one unsteered row."""
    T = len(item.ids)
    Ls, bases = [], []
    for s in range(0, len(arms), batch_size - 1):
        chunk = arms[s:s + batch_size - 1]
        B = len(chunk) + 1  # row 0 unsteered
        D = next(v for _, add, cl in chunk for v in (*[a[1] for a in add.values()], *[c[0] for c in cl.values()])).shape[0]
        additions, clamps = {}, {}
        for b in sorted({b for _, add, _ in chunk for b in add}):
            coef, vec = torch.zeros(B, T), torch.zeros(B, D)
            for r, (_, add, _) in enumerate(chunk, start=1):
                if b in add:
                    coef[r], vec[r] = add[b]
            additions[b] = (coef, vec)
        for key in sorted({k for _, _, cl in chunk for k in cl}):
            pos = clamp_positions(item, key[1])
            dirs, vals, rows = torch.zeros(B, D), torch.zeros(B, len(pos)), torch.zeros(B, dtype=torch.bool)
            dirs[:, 0] = 1.0  # rows left alone still need a unit vector (clamp normalizes every row)
            for r, (_, _, cl) in enumerate(chunk, start=1):
                if key in cl:
                    dirs[r], vals[r] = cl[key]
                    rows[r] = True
            clamps[key] = (pos, dirs, vals, rows)
        ids = torch.tensor([item.ids] * B).to(models.input_device(model))
        with contextlib.ExitStack() as stack:
            if additions:
                stack.enter_context(acts.steer_tokens(model, additions, at="post"))
            for (b, _), (pos, dirs, vals, rows) in clamps.items():  # one clamp per block and position set
                stack.enter_context(acts.clamp(model, {b: dirs}, {b: vals}, at="post", positions=pos, rows=rows))
            logits = model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[:, -1].float()
        if not torch.isfinite(logits).all():
            raise RuntimeError("non-finite logits")
        L = logit_diff(logits, out_ids).cpu()
        Ls += L[1:].tolist()
        bases += [L[0].item()] * len(chunk)
    return Ls, bases


def steer_arms(item, dirs, norms, cs, blocks):
    """+c on task A's tokens, -c on task B's, along each direction, per block config."""
    sign = seg_coef(item, {"task_a": 1.0, "task_b": -1.0})
    arms = []
    for name, U in dirs.items():
        for c in cs:
            for cfg in blocks:
                bl = STEER_BLOCKS if cfg == "together" else [cfg]
                scale = c / len(bl)
                add = {b: (sign * scale * norms[b], U[b]) for b in bl}
                arms.append((dict(arm="steer", direction=name, c=c, blocks=str(cfg)), add, {}))
    return arms


def task_swap_clamps(item, other, h_self, h_other, U, blocks):
    """{(b, "task"): (u_b, targets)}: every task token's projection set to its unpatched value plus its slot's shift
    (the other order's slot mean projection minus this prompt's)."""
    def slot_mean(it, h, u):
        p = h @ u
        return {s: p[torch.tensor([x == s for x in it.seg])].mean().item() for s in ("task_a", "task_b")}

    pos = task_positions(item)
    out = {}
    for b in blocks:
        mine, theirs = slot_mean(item, h_self[b], U[b]), slot_mean(other, h_other[b], U[b])
        shift = seg_coef(item, {s: theirs[s] - mine[s] for s in mine})
        out[b, "task"] = (U[b], (h_self[b] @ U[b])[pos] + shift[pos])
    return out


def label_swap_clamps(item, other, h_other, U, blocks):
    """{(b, "label"): (u_b, target)}: the label B token's projection set to the other order's label B projection."""
    return {(b, "label"): (U[b], (h_other[b][other.read["label_b"]] @ U[b]).reshape(1)) for b in blocks}


def swap_arms(item, other, h_self, h_other, dirs, blocks):
    """Move each slot's tokens along u_b by (other prompt's slot mean projection - this prompt's).

    A single block b adds the shift there (acts.steer_tokens). "together" instead sets every task token's
    projection at every block b to (its unpatched projection + block b's shift) (acts.clamp): adding the shifts at
    each block would stack them, since a shift at block 15 is still in the stream at blocks 16-23."""
    arms = []
    for name, U in dirs.items():
        clamps = task_swap_clamps(item, other, h_self, h_other, U, SWAP_BLOCKS)
        for cfg in blocks:
            label = dict(arm="swap", direction=name, blocks=str(cfg))
            if cfg == "together":
                arms.append((label, {}, clamps))
            else:
                (u, target), pos = clamps[cfg, "task"], task_positions(item)
                arms.append((label, {cfg: (torch.zeros(len(item.ids)).index_put((pos,), target - h_self[cfg][pos] @ u),
                                           u)}, {}))
    return arms


def label_swap_arms(item, other, h_self, h_other, ldirs, tdirs, blocks):
    """swap_label: the label B token's projection on its own probe set to the other order's value; swap_both: that
    plus the task-token swap along the task probe of the same name. Single blocks and "together" (every swap block),
    all as clamps (at a single block, setting a projection equals adding the shift)."""
    arms = []
    for name, U in ldirs.items():
        lab = label_swap_clamps(item, other, h_other, U, SWAP_BLOCKS)
        task = task_swap_clamps(item, other, h_self, h_other, tdirs[name], SWAP_BLOCKS)
        for cfg in blocks:
            bl = SWAP_BLOCKS if cfg == "together" else [cfg]
            only = {k: v for k, v in lab.items() if k[0] in bl}
            both = {**only, **{k: v for k, v in task.items() if k[0] in bl}}
            arms.append((dict(arm="swap_label", direction=name, blocks=str(cfg)), {}, only))
            arms.append((dict(arm="swap_both", direction=name, blocks=str(cfg)), {}, both))
    return arms


def label_steer_arms(item, ldirs, norms, cs, blocks):
    """steer_label: +c * s_b * u_b at the label B token only (u_b its own probe, which reads task A's value), same
    dose scale as the task-token steering; "together" splits c over the steering blocks."""
    arms = []
    at = item.read["label_b"]
    for name, U in ldirs.items():
        for c in cs:
            for cfg in blocks:
                bl = STEER_BLOCKS if cfg == "together" else [cfg]
                add = {}
                for b in bl:
                    coef = torch.zeros(len(item.ids))
                    coef[at] = c / len(bl) * norms[b]
                    add[b] = (coef, U[b])
                arms.append((dict(arm="steer_label", direction=name, c=c, blocks=str(cfg)), add, {}))
    return arms


def residual_norms(model, items, blocks):
    """Mean residual norm per block over all positions but 0."""
    tot, n = {b: 0.0 for b in blocks}, 0
    for it in items:
        _, h = baseline(model, it, blocks)
        for b in blocks:
            tot[b] += h[b][1:].norm(dim=-1).sum().item()
        n += len(it.ids) - 1
    return {b: tot[b] / n for b in blocks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("stage")
    ap.add_argument("--sets", nargs="+", default=["standard", "benign"])
    ap.add_argument("--formats", nargs="+", default=["coinflip", "choice"])
    ap.add_argument("--directions", default="outputs/probe/directions.npz")
    ap.add_argument("--limit", type=int, help="first N pairs per set (smoke tests)")
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--out", default="outputs/probe")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--only-swap-together", action="store_true",
                    help="rerun just the all-blocks swap arms, into rows_swap_together.jsonl")
    ap.add_argument("--label", action="store_true",
                    help="the label-token arms (steer_label, swap_label, swap_both), into rows_label.jsonl")
    a = ap.parse_args()

    out = Path(a.out) / a.stage / "tests"
    out.mkdir(parents=True, exist_ok=True)
    kw = dict(device_map="cpu", dtype="float32", check=False) if a.cpu else {}
    model, tok = models.load(a.model, **kw)
    L, D = models.n_layers(model), models.text_config(model).hidden_size
    global STEER_BLOCKS, SWAP_BLOCKS, BENIGN_BLOCKS
    if L < 24:  # tiny test models
        STEER_BLOCKS = SWAP_BLOCKS = list(range(1, L))
        BENIGN_BLOCKS = [1, "together"]
    dirs = load_directions(a.directions, L, D)
    label_dirs = {f: load_directions(a.directions, L, D, seed=1, prefix=f"label_{f}_") for f in a.formats}
    design = load_design()
    blocks = sorted(set(STEER_BLOCKS) | set(SWAP_BLOCKS))

    std_items = [it for f in a.formats for _, _, hp, tp in test_pairs(tok, f, design, "standard") for it in (hp, tp)]
    norms = residual_norms(model, std_items[: (a.limit or len(std_items))], blocks)
    print(f"mean residual norms: { {b: round(v, 1) for b, v in norms.items()} }", flush=True)

    tag = "_swap_together" if a.only_swap_together else "_label" if a.label else ""
    rows_path = out / f"rows{tag}.jsonl"
    rows_path.unlink(missing_ok=True)
    t0 = time.time()
    for which in a.sets:
        cs = C_STANDARD if which == "standard" else C_BENIGN
        sblocks = STEER_BLOCKS + ["together"] if which == "standard" else BENIGN_BLOCKS
        wblocks = SWAP_BLOCKS + ["together"] if which == "standard" else BENIGN_BLOCKS
        keep = (lambda k: True) if which == "standard" else (lambda k: k not in ("rand1", "rand2"))
        use_dirs = {k: v for k, v in dirs.items() if keep(k)}
        for fmt in a.formats:
            use_ldirs = {k: v for k, v in label_dirs[fmt].items() if keep(k)}
            out_ids = outcome_ids(tok, fmt)
            pairs = test_pairs(tok, fmt, design, which)[: a.limit]
            for k, (p, d, hp, tp) in enumerate(pairs):
                base = {}
                for it in (hp, tp):
                    logits, h = baseline(model, it, blocks)
                    base[it.meta["order"]] = (logit_diff(logits, out_ids).item(), h)
                rows = []
                for it, other in ((hp, tp), (tp, hp)):
                    order, oo = it.meta["order"], other.meta["order"]
                    if a.only_swap_together:
                        arms = swap_arms(it, other, base[order][1], base[oo][1], use_dirs, ["together"])
                    elif a.label:
                        arms = label_steer_arms(it, use_ldirs, norms, cs, sblocks)
                        arms += label_swap_arms(it, other, base[order][1], base[oo][1], use_ldirs, use_dirs, wblocks)
                    else:
                        arms = steer_arms(it, use_dirs, norms, cs, sblocks)
                        arms += swap_arms(it, other, base[order][1], base[oo][1], use_dirs, wblocks)
                    Ls, bases = run_arms(model, it, arms, out_ids, a.batch_size)
                    for (label, *_), Lv, Lb in zip(arms, Ls, bases):
                        rows.append(dict(stage=a.stage, set=which, format=fmt, pair=k, pref=p, dispref=d, order=order,
                                         **label, L=Lv, L_base=Lb, L_unbatched=base[order][0]))
                with open(rows_path, "a") as f:
                    for r in rows:
                        f.write(json.dumps(r) + "\n")
                if (k + 1) % 25 == 0 or k + 1 == len(pairs):
                    print(f"{which}/{fmt}: pair {k + 1}/{len(pairs)}  {(time.time() - t0) / 60:.1f} min", flush=True)
    summary = dict(model=a.model, stage=a.stage, norms=norms, steer_blocks=STEER_BLOCKS, swap_blocks=SWAP_BLOCKS,
                   c_standard=C_STANDARD, c_benign=C_BENIGN, benign_blocks=BENIGN_BLOCKS,
                   directions=sorted(dirs), label_directions={f: sorted(v) for f, v in label_dirs.items()},
                   batch_size=a.batch_size, seconds=time.time() - t0, provenance=io.provenance())
    (out / f"summary{tag}.json").write_text(json.dumps(summary, indent=1))
    print(f"done in {(time.time() - t0) / 60:.1f} min; wrote {out}", flush=True)


if __name__ == "__main__":
    main()
