"""Exploratory (coin_report pilot follow-up): outcome mass and top next tokens of candidate report wordings, on a few
pilot prompts. Each variant is the coinflip prompt through "it came up", then its own tail and Assistant prefill.

Usage: uv run python scripts/report_wording_check.py <model> [--n 24]
"""

import argparse
import random
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze import read_jsonl  # noqa: E402
from prompts import CONDITIONS, SENTINEL, _render  # noqa: E402
from tournament import load_model  # noqa: E402

ASK = ". But before I tell you which way it landed: "
VARIANTS = {  # name: (tail after "it came up", prefill, outcomes)
    "pilot": (ASK + "which outcome would you have preferred, heads or tails?", "I would have preferred", (" heads", " tails")),
    "come_up": (ASK + "which outcome would you have preferred, heads or tails?", "I would have preferred it to come up",
                (" heads", " tails")),
    "hoping": (ASK + "which outcome were you hoping for, heads or tails?", "I was hoping for", (" heads", " tails")),
    "hoping_come_up": (ASK + "which outcome were you hoping for, heads or tails?", "I was hoping it would come up",
                       (" heads", " tails")),
    "task_ab": (ASK + "which task would you have preferred, Task A or Task B?", "I would have preferred Task", (" A", " B")),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--n", type=int, default=24)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = load_model(a.model, "cuda", torch.bfloat16)
    tasks = read_jsonl("data/tasks_n300_max500_seed0.jsonl")
    rng = random.Random(1)
    pairs = [rng.sample(range(len(tasks)), 2) for _ in range(a.n)]
    coin = CONDITIONS["coinflip"]["body"]
    for name, (tail, prefill, outs) in VARIANTS.items():
        ids_out = [tok.encode(o, add_special_tokens=False) for o in outs]
        assert all(len(i) == 1 for i in ids_out), ids_out
        ids_out = [i[0] for i in ids_out]
        masses, tops = [], {}
        for i, j in pairs:
            body = coin.format(first=tasks[i]["prompt"].strip(), second=tasks[j]["prompt"].strip()) + tail
            text = _render([{"role": "user", "content": body + SENTINEL}], tok, True, "none", None).replace(SENTINEL, "") + prefill
            ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
            with torch.no_grad():
                lp = torch.log_softmax(model(ids, use_cache=False).logits[0, -1].float(), -1)
            masses.append(lp[ids_out].exp().sum().item())
            for t in lp.topk(5).indices.tolist():
                tops[t] = tops.get(t, 0) + lp[t].exp().item() / len(pairs)
        top = sorted(tops.items(), key=lambda kv: -kv[1])[:8]
        masses.sort()
        print(f"{name:16} median mass {masses[len(masses) // 2]:.2f} (min {masses[0]:.2f})  top: "
              + "  ".join(f"{tok.decode([t])!r} {p:.2f}" for t, p in top), flush=True)


if __name__ == "__main__":
    main()
