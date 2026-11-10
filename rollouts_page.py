"""Pages and a summary for rollouts.py runs (README, "Rollouts with the coin's ' up' swapped"). Local; needs only tokenizers.

For each model: one matskit.view page with every reply, the none and swap replies of the same sample side by side (a group),
and two per-token scores:
  attention to " up"     how much the position that chose this token attended to " up" (that is, the attention of the
                           position before it), in the reply's own condition, mean over heads and the blocks from the lens
                           onset on
  Δ log-prob, swap − none  the reply token's log-prob with " up" swapped minus unpatched (reply tokens only): what the swap does
                           to each word, whichever condition wrote it
Meta per reply: the tasks, which one the coin assigned and whether it's the one the coin prefers, the sample, whether the
coupled partner is identical, and the first token where they differ.

Writes outputs/rollouts/<family>/rollouts.html (gitignored: WildChat tasks can appear in prompts) and prints a summary:
per model, how many coupled pairs are identical, where they first diverge, mean |Δ log-prob| per reply token, and r between
attention to " up" and |Δ log-prob|.

Usage: uv run python rollouts_page.py outputs/rollouts/<family>/<run>/rows.jsonl [...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from analyze import read_jsonl
from matskit import io, view
from prompts import CONDITIONS

MODELS = {"gemma4-31b": "google/gemma-4-31B-it", "llama-70b": "unsloth/Llama-3.3-70B-Instruct",
          "glm-4.5-air": "zai-org/GLM-4.5-Air", "olmo3.1-32b": "allenai/Olmo-3.1-32B-Instruct"}
LABEL = {"gemma4-31b": "Gemma 4 31B", "llama-70b": "Llama 3.3 70B", "glm-4.5-air": "GLM-4.5-Air", "olmo3.1-32b": "OLMo 3.1 32B"}


def labels(tok, ids, body):
    """role, turn, kind per prompt token: template before the user's text, the user's text, then the Assistant header."""
    text = tok.decode(ids)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    assert enc.input_ids == ids, "the prompt must re-tokenize to the ids the model read"
    a = text.index(body)
    b = a + len(body)
    role, turn, kind = [], [], []
    for s, e in enc.offset_mapping:
        if e <= a:
            role.append("user"), turn.append(0), kind.append("template")
        elif s < b:
            role.append("user"), turn.append(0), kind.append("content")
        else:
            role.append("assistant"), turn.append(1), kind.append("template")
    return role, turn, kind


def first_divergence(x, y):
    """Index of the first differing token (None if identical)."""
    for k, (s, t) in enumerate(zip(x, y)):
        if s != t:
            return k
    return None if len(x) == len(y) else min(len(x), len(y))


def build(path, out_dir=Path("outputs/rollouts")):
    family = Path(path).parts[-3]
    tok = AutoTokenizer.from_pretrained(MODELS[family])
    tasks = {t["id"]: t for t in read_jsonl("data/tasks_n300_max500_seed0.jsonl")}
    lens = json.loads((Path(path).parent / "lens.json").read_text())
    rows = io.read_jsonl(path)
    by = {(r["pair"], r["outcome"], r["sample"], r["condition"]): r for r in rows}
    transcripts, stats = [], dict(n=0, identical=0, first=[], dlogp=[], attn=[], len_none=[], len_swap=[])
    for (p, outcome, k, cond), r in sorted(by.items()):
        other = by.get((p, outcome, k, "swap" if cond == "none" else "none"))
        a, b = tasks[r["a"]], tasks[r["b"]]
        body = CONDITIONS[f"coin_task_{outcome}"]["body"].format(first=a["prompt"].strip(), second=b["prompt"].strip())
        role, turn, kind = labels(tok, r["prompt_ids"], body)
        n_p, n_r, up = len(r["prompt_ids"]), len(r["reply_ids"]), r["up"]
        ids = r["prompt_ids"] + r["reply_ids"]
        attn = [None] * (up + 2) + r[f"attn_{cond}"][:-1]  # token t shows position t-1's attention (it chose token t)
        dlogp = [None] * n_p + (np.array(r["logp_swap"]) - np.array(r["logp_none"])).tolist()
        div = first_divergence(r["reply_ids"], other["reply_ids"]) if other else None
        assigned = a if outcome == "heads" else b
        pref = p % 2 == 0  # rollouts.far_pairs: the coin-preferred task is Task A in even-numbered pairs
        assigned_pref = (outcome == "heads") == pref
        meta = {"model": LABEL[family], "pair": p, "Task A": a["id"], "Task B": b["id"], "outcome": outcome,
                "assigned": f"{assigned['id']} (the coin's {'preferred' if assigned_pref else 'other'} task)",
                "condition": cond, "sample": k, "same as partner": div is None,
                "first divergence (reply token)": div, "finished": r["finished"], "lens onset depth": lens["onset"]}
        title = f"pair {p} · {outcome} → {'preferred' if assigned_pref else 'other'} task · sample {k} · {cond}"
        t = view.from_ids(tok, ids, role=role + ["assistant"] * n_r, turn=turn + [1] * n_r, kind=kind + ["content"] * n_r,
                          sampled=[False] * n_p + [True] * n_r, title=title, group=f"pair {p} · {outcome} · sample {k}",
                          meta=meta, token_scores={"attention to \" up\" (choosing this token)": attn, "Δ log-prob, swap − none": dlogp},
                          score_labels={"Δ log-prob, swap − none": ("less likely with \" up\" swapped",
                                                                    "more likely with \" up\" swapped")},
                          turn_flags={} if r["finished"] else {1: ("warn", f"cut at {n_r} tokens")})
        transcripts.append(t)
        if cond == "none":
            stats["n"] += 1
            stats["identical"] += div is None
            stats["first"].append(div if div is not None else n_r)
            d = np.abs(np.array(r["logp_swap"]) - np.array(r["logp_none"]))
            stats["dlogp"] += d.tolist()
            stats["attn"] += r["attn_none"][n_p - up - 2:n_p - up - 2 + n_r]  # positions that chose the reply tokens
            stats["len_none"].append(n_r)
            stats["len_swap"].append(len(other["reply_ids"]) if other else np.nan)
    note = (f"{LABEL[family]}: replies to the coinflip's assigned task, unpatched (none) and with the coin's \" up\" swapped for "
            f"the other order's at every block (swap). Each none/swap pair shares its sampling noise, so the two are identical "
            f"until the swap changes which token wins. Attention to \" up\" is averaged over heads and blocks {lens['rel_layers'][0]}"
            f"-{lens['rel_layers'][-1]} (from the lens onset, depth {lens['onset']}). All replies shown; no selection.")
    out = Path(out_dir) / family / "rollouts.html"
    view.write(out, transcripts, title=f"{LABEL[family]}: rollouts with \" up\" swapped", note=note)
    d, at = np.array(stats["dlogp"]), np.array(stats["attn"])
    print(f"{LABEL[family]}: {stats['identical']}/{stats['n']} coupled pairs identical; first divergence at reply token "
          f"median {np.median(stats['first']):.0f} (of median length {np.median(stats['len_none']):.0f}); "
          f"mean |Δ log-prob| per reply token {d.mean():.3f}, 99th pct {np.percentile(d, 99):.3f}; "
          f"r(attention to up, |Δ log-prob|) {np.corrcoef(at, d)[0, 1]:+.2f}; lens onset {lens['onset']}")
    return out


def main():
    for p in sys.argv[1:]:
        build(p)


if __name__ == "__main__":
    main()
