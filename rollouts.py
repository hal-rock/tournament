"""Rollouts with the coin's " up" swapped (README, "Rollouts with the coin's ' up' swapped"). Exploratory; one model per run,
on a GPU pod. Hypothesis-generating: does the Assistant's actual reply to the coin's task change when " up" is swapped?

Prompt (prompts.CONDITIONS["coin_task_heads" / "coin_task_tails"]): the coinflip prompt word for word through "it came up",
then "heads, so please do Task A." or "tails, so please do Task B."; the user turn is closed and the Assistant replies.

Per model:
- Pairs: n task pairs far apart in the model's own coin utilities (its sweep), non-harmful tasks only (refusals would
  dominate), the Assistant's utilities agreeing on which task is better, no task used twice. The coin-preferred task is
  Task A in even-numbered pairs and Task B in odd ones. Both outcomes for every pair.
- Conditions: none (unpatched) and swap: the residual entering every block at " up" replaced with the other order's (the same
  tasks in the opposite slots, so " up" carries the coin's lean toward the other slot), as in report_patch.py.
- Coupled sampling: n_samples replies per condition at temperature 1, by Gumbel-max with noise seeded per (sample, step), the
  same in both conditions. A none/swap pair of replies is token-identical until the patch changes which token wins, so the
  first divergence shows where (if anywhere) " up" matters. Each batch is one prompt: the none and swap samples plus a
  duplicate of none's sample 0, which must come out identical to it (a check that rows are computed alike).
- Per reply token, teacher-forced in both conditions: its log-prob under none and under swap, and the attention it pays to
  " up", averaged over heads and over the blocks from the lens onset on.
- Lens onset: over lens_pairs pilot pairs (both orders, outside the sweep), the logit lens at " up" (matskit.acts.logit_lens on
  the residual entering each block, and the output) gives heads - tails per depth; onset is the shallowest depth from which its
  pair effect correlates >= 0.5 with the output's at every deeper depth. Blocks from there on read keys from a residual that
  already carries the coin's lean.

Output: outputs/rollouts/<family>/<run>/rows.jsonl (one row per reply) and lens.json; uploaded to W&B (group "rollouts").
The page (rollouts_page.py) is built locally from these.

Usage: uv run python rollouts.py configs/rollouts/<model>.yaml [--n-pairs 2 --max-new-tokens 32]  (smoke test)
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer, LogitsProcessor, LogitsProcessorList

from analyze import pearson, read_jsonl
from matskit import acts, io
from matskit.models import input_device, layers as get_layers
from prompts import build_prompt, outcome_token_ids
from report_patch import gather_reducer, pilot_pairs
from tournament import load_model, select_pairs as sweep_pairs

ONSET_R = 0.5


def far_pairs(family, tasks, n):
    """[(i, j)] task indices, i = Task A: the n most coin-separated non-harmful pairs, Assistant agreeing, no task reused.
    The coin-preferred task is Task A in even-numbered pairs and Task B in odd ones."""
    from coin_report import sweep_rows
    from report import task_arrays
    rows = sweep_rows(family)
    U, _, benign, _ = task_arrays(rows)
    C, A = U["instruct/coinflip"], U["instruct/assistant"]
    index = {t["id"]: k for k, t in enumerate(tasks)}
    ok = np.flatnonzero(benign)
    cand = sorted(((abs(C[a] - C[b]), a, b) for x, a in enumerate(ok) for b in ok[x + 1:]
                   if np.sign(C[a] - C[b]) == np.sign(A[a] - A[b])), reverse=True)
    used, out = set(), []
    for _, a, b in cand:
        if a in used or b in used:
            continue
        pref, other = (a, b) if C[a] > C[b] else (b, a)
        pair = (pref, other) if len(out) % 2 == 0 else (other, pref)
        out.append(tuple(index[rows[k]["id"]] for k in pair))
        used |= {a, b}
        if len(out) == n:
            break
    return out


class CoupledGumbel(LogitsProcessor):
    """Adds Gumbel noise seeded by (row seed, step), so greedy decoding samples at temperature 1 and rows with the same seed
    share their noise: replies under different conditions are coupled."""

    def __init__(self, seeds, prompt_len):
        self.seeds, self.prompt_len = seeds, prompt_len

    def __call__(self, input_ids, scores):
        step = input_ids.shape[1] - self.prompt_len
        out = scores.float().clone()
        for r, seed in enumerate(self.seeds):
            g = torch.Generator(device=scores.device).manual_seed(seed * 1_000_003 + step)
            u = torch.rand(scores.shape[1], generator=g, device=scores.device, dtype=torch.float64).clamp(1e-12, 1 - 1e-12)
            out[r] += (-torch.log(-torch.log(u))).float()
        return out


def prefix(tok, first, second):
    """Ids of the coinflip prompt through " up" (the last token), and the index of " up"."""
    ids = tok(build_prompt("coinflip", first, second, tok), add_special_tokens=False).input_ids
    assert tok.decode(ids[-1]).strip() == "up"
    return ids, len(ids) - 1


@torch.no_grad()
def up_residuals(model, ids, up, n_layers):
    """The residual entering every block at " up" of one prompt: {layer: [1, 1, D]} (CPU)."""
    pos = torch.tensor([[up]])
    with acts.capture(model, list(range(n_layers)), at="pre", reduce=gather_reducer(pos)) as store:
        model(input_ids=torch.tensor([ids]).to(input_device(model)), use_cache=False)
    return {l: store[l] for l in range(n_layers)}


@torch.no_grad()
def lens_onset(model, tok, tasks, pairs, coin_ids, n_layers):
    """Per depth (0 = embeddings ... n_layers = output), r between the lens's heads - tails pair effect at " up" and the
    output's, over the pairs; and the onset depth (see the module docstring)."""
    eff = []
    for i, j in pairs:
        per_order = []
        for a, b in ((i, j), (j, i)):
            ids, up = prefix(tok, tasks[a]["prompt"], tasks[b]["prompt"])
            pos = torch.tensor([up])
            with acts.capture(model, list(range(n_layers)), at="pre", reduce=acts.positions_reducer(pos)) as pre, \
                    acts.capture(model, [n_layers - 1], at="post", reduce=acts.positions_reducer(pos)) as post:
                model(input_ids=torch.tensor([ids]).to(input_device(model)), use_cache=False)
            hs = torch.cat([*(pre[l] for l in range(n_layers)), post[n_layers - 1]])  # [L + 1, D]
            lo = acts.logit_lens(model, hs)[:, coin_ids].cpu()
            per_order.append((lo[:, 0] - lo[:, 1]).numpy())
        eff.append((per_order[0] - per_order[1]) / 2)
    eff = np.array(eff)  # [pairs, L + 1]
    r = np.array([pearson(eff[:, d], eff[:, -1]) if eff[:, d].std() > 0 else 0.0 for d in range(n_layers + 1)])
    onset = next((d for d in range(n_layers + 1) if np.all(r[d:] >= ONSET_R)), n_layers)
    return r, int(onset)


@torch.no_grad()
def generate_group(model, tok, ids, up, source, seeds, max_new, n_layers):
    """Replies to one prompt: rows none x seeds, swap x seeds, and none with seeds[0] again. Returns [(condition, seed,
    reply ids, finished)]."""
    conds = ["none"] * len(seeds) + ["swap"] * len(seeds) + ["none_dup"]
    row_seeds = list(seeds) + list(seeds) + [seeds[0]]
    B, D = len(conds), source[0].shape[-1]
    swap_rows = torch.tensor([c == "swap" for c in conds])
    values = {l: source[l].expand(B, 1, D) for l in range(n_layers)}
    eos = tok.eos_token_id if isinstance(tok.eos_token_id, list) else [tok.eos_token_id]
    gen_eos = model.generation_config.eos_token_id
    eos = sorted(set(eos) | set(gen_eos if isinstance(gen_eos, list) else [gen_eos] if gen_eos is not None else []))
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos[0]
    x = torch.tensor([ids] * B).to(input_device(model))
    with acts.replace(model, values, at="pre", positions=torch.full((B, 1), up), rows={l: swap_rows for l in values}):
        out = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=max_new, do_sample=False,
                             logits_processor=LogitsProcessorList([CoupledGumbel(row_seeds, len(ids))]),
                             eos_token_id=eos, pad_token_id=pad, temperature=None, top_p=None, top_k=None,
                             use_cache=True)
    res = []
    for r, (c, s) in enumerate(zip(conds, row_seeds)):
        reply = out[r, len(ids):].tolist()
        stop = next((k for k, t in enumerate(reply) if t in eos), None)
        res.append((c, s, reply[:stop] if stop is not None else reply, stop is not None))
    return res


@torch.no_grad()
def score(model, ids, up, reply, source, rel_layers, n_layers):
    """Teacher-forced, both conditions in one batch (row 0 none, row 1 swap): log-prob of each reply token, and the attention
    each position after " up" pays to it (mean over heads and over rel_layers). Returns (logp [2, R], attn [2, T - up - 1])."""
    seq = ids + reply
    x = torch.tensor([seq, seq]).to(input_device(model))
    blocks, store = get_layers(model), {}
    hooks = []
    for l in rel_layers:
        def hook(m, args, out, l=l):
            w = out[1]
            assert w is not None and w.dim() == 4, "need eager attention weights"
            store[l] = w[:, :, up + 1:, up].float().mean(1).cpu()  # [2, T - up - 1]
        hooks.append(blocks[l].self_attn.register_forward_hook(hook))
    values = {l: source[l].expand(2, 1, source[l].shape[-1]) for l in range(n_layers)}
    rows = torch.tensor([False, True])
    try:
        with acts.replace(model, values, at="pre", positions=torch.full((2, 1), up), rows={l: rows for l in values}):
            logits = model(input_ids=x, use_cache=False, logits_to_keep=len(reply) + 1).logits
    finally:
        for h in hooks:
            h.remove()
    lp = torch.log_softmax(logits[:, :-1].float(), -1)  # predictions for reply tokens
    tgt = torch.tensor(reply, device=lp.device)
    logp = lp[:, torch.arange(len(reply)), tgt].cpu()
    attn = torch.stack([store[l] for l in rel_layers]).mean(0)
    return logp, attn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--n-pairs", type=int)
    ap.add_argument("--max-new-tokens", type=int)
    ap.add_argument("--lens-pairs", type=int)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    for k in ("n_pairs", "max_new_tokens", "lens_pairs"):
        if getattr(a, k) is not None:
            cfg[k] = getattr(a, k)
    import wandb
    run = wandb.init(project="tournament", group="rollouts", job_type="rollouts", name=cfg["run_name"],
                     tags=[cfg["family"]], config=cfg)
    out = Path("outputs/rollouts") / cfg["family"] / run.id
    out.mkdir(parents=True, exist_ok=True)
    tasks = read_jsonl(cfg["tasks_file"])
    m = cfg["model"]
    tok = AutoTokenizer.from_pretrained(m["name"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(m["name"], device, torch.bfloat16 if device == "cuda" else torch.float32,
                       max_memory=m.get("max_memory"), attn_implementation="eager")
    n_layers = len(get_layers(model))
    coin_ids = outcome_token_ids("coinflip", tok)

    lens = pilot_pairs(len(tasks), cfg["lens_pairs"], sweep_pairs(len(tasks), 15000, 0), 0)
    r, onset = lens_onset(model, tok, tasks, lens, coin_ids, n_layers)
    rel_layers = list(range(min(onset, n_layers - 1), n_layers))
    print(f"lens onset depth {onset} of {n_layers}; r by depth: " + " ".join(f"{d}:{v:.2f}" for d, v in enumerate(r)),
          flush=True)
    (out / "lens.json").write_text(json.dumps({"r": r.tolist(), "onset": onset, "rel_layers": rel_layers}))

    pairs = far_pairs(cfg["family"], tasks, cfg["n_pairs"])
    path = out / "rows.jsonl"
    dup_ok = True
    for p, (i, j) in enumerate(pairs):
        first, second = tasks[i]["prompt"], tasks[j]["prompt"]
        _, up = prefix(tok, first, second)
        other_ids, other_up = prefix(tok, second, first)
        assert other_up == up, "the two orders must put ' up' at the same index"
        source = up_residuals(model, other_ids, up, n_layers)
        for outcome in ("heads", "tails"):
            ids = tok(build_prompt(f"coin_task_{outcome}", first, second, tok), add_special_tokens=False).input_ids
            assert ids[:up + 1] == prefix(tok, first, second)[0], "the prompt must extend the coinflip through ' up'"
            seeds = [cfg["seed"] * 10_000 + p * 100 + (outcome == "tails") * 10 + k for k in range(cfg["n_samples"])]
            replies = generate_group(model, tok, ids, up, source, seeds, cfg["max_new_tokens"], n_layers)
            none0 = next(rep for c, s, rep, _ in replies if c == "none" and s == seeds[0])
            dup = next(rep for c, s, rep, _ in replies if c == "none_dup")
            dup_ok &= dup == none0
            for cond, seed, reply, finished in replies:
                if cond == "none_dup" or not reply:
                    continue
                logp, attn = score(model, ids, up, reply, source, rel_layers, n_layers)
                io.append_jsonl(path, dict(
                    pair=p, a=tasks[i]["id"], b=tasks[j]["id"], outcome=outcome, condition=cond,
                    sample=seeds.index(seed), seed=seed, up=up, prompt_ids=ids, reply_ids=reply, finished=finished,
                    logp_none=logp[0].tolist(), logp_swap=logp[1].tolist(),
                    attn_none=attn[0].tolist(), attn_swap=attn[1].tolist()))
            print(f"  pair {p} {outcome}: reply lengths " + ", ".join(f"{c}{s % 10}:{len(rep)}" for c, s, rep, _ in replies),
                  flush=True)
    print(f"duplicate rows identical: {dup_ok}", flush=True)
    run.summary.update({"lens_onset": onset, "duplicates_identical": dup_ok})
    run.save(str(path), base_path=str(out), policy="now")
    run.save(str(out / "lens.json"), base_path=str(out), policy="now")
    run.finish()
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
