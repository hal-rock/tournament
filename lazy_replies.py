"""Lazy personas, uptake check (README, "Lazy personas"): free replies to the Assistant-choice prompt under each phrasing.

The scored conditions prefill "I'll choose Task", so they can't show whether the model takes the persona up at all (talks
like a pirate, plays a cat) or only its style. This samples the reply with no prefill, for a few fixed task pairs:
- conditions: assistant_youare_<persona> and assistant_answeras_<persona> from the config (configs/lazy/<model>-<persona>.yaml),
  plus the plain "assistant" with --default (once per model);
- pairs: n_pairs of the sweep's pairs with both tasks from WildChat, Alpaca or MATH (the sources the analysis is about),
  drawn with a fixed seed, so every model and persona gets the same pairs; one order (the drawn one);
- n_samples replies per pair at temperature 1 with the model's own generation defaults otherwise (top-k / top-p from its
  generation_config), max_new_tokens each, seeded per condition.

Not a measurement: the page (lazy_page.py) is for reading, and the printed tally of "Task A" / "Task B" in each reply's opening
is a rough look only.

Output: outputs/lazy_replies/<family>/<persona>.jsonl, one row per reply: condition, pair, task ids, sample, prompt_ids, reply_ids,
reply text, and whether it ended (EOS) before max_new_tokens.

Usage: uv run python lazy_replies.py configs/lazy/<model>-<persona>.yaml [--default] [--n-pairs 2 --max-new-tokens 16]
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import torch
import yaml
from transformers import AutoTokenizer

from analyze import read_jsonl
from matskit import io
from prompts import build_prompt
from tournament import load_model, select_pairs

ARBITRARY = ("WILDCHAT", "ALPACA", "MATH")


def reply_pairs(tasks, n_pairs, max_pairs=15000, seed=0):
    """n_pairs of the sweep's pairs (tournament.select_pairs) with both tasks in ARBITRARY, by a fixed draw."""
    pool = [(a, b) for a, b in select_pairs(len(tasks), max_pairs, seed)
            if tasks[a]["origin"] in ARBITRARY and tasks[b]["origin"] in ARBITRARY]
    return random.Random("lazy_replies").sample(pool, n_pairs)


def chosen_task(text):
    """'A' / 'B' if the reply's first 300 characters name exactly one of "Task A" / "Task B", else None."""
    found = set(re.findall(r"\bTask ([AB])\b", text[:300]))
    return found.pop() if len(found) == 1 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--default", action="store_true", help="also sample the plain assistant prompt")
    ap.add_argument("--n-pairs", type=int, default=6)
    ap.add_argument("--n-samples", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=150)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    (m,) = cfg["models"]
    family = cfg["tags"][0]
    persona = cfg["run_name"].removeprefix(f"lazy-{family}-")
    conditions = [c for c in m["conditions"] if c.startswith("assistant_")]
    assert conditions == [f"assistant_youare_{persona}", f"assistant_answeras_{persona}"], conditions
    if args.default:
        conditions = ["assistant"] + conditions
    tasks = read_jsonl(cfg["tasks_file"])
    pairs = reply_pairs(tasks, args.n_pairs, cfg["max_pairs"], cfg["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(m["name"])
    model = load_model(m["name"], device, torch.bfloat16 if device == "cuda" else torch.float32,
                       max_memory=m.get("max_memory"))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    out = Path("outputs/lazy_replies") / family / f"{persona}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    for ci, cond in enumerate(conditions):
        prompts = [build_prompt(cond, tasks[a]["prompt"], tasks[b]["prompt"], tok, cfg["system_prompt"], prefill=False)
                   for a, b in pairs]
        ids = [tok(p, add_special_tokens=False).input_ids for p in prompts]
        enc = tok(prompts, add_special_tokens=False, padding=True, padding_side="left", return_tensors="pt").to(
            model.device)
        torch.manual_seed(1000 + ci)  # the same seed for a condition's position in every config
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=True, temperature=1.0,
                                 num_return_sequences=args.n_samples, pad_token_id=pad,
                                 use_cache=m.get("use_cache", True))
        tally = {"A": 0, "B": 0, None: 0}
        for k, row in enumerate(gen[:, enc.input_ids.shape[1]:].tolist()):
            p = k // args.n_samples
            ended = any(t in eos for t in row)
            reply = row[:next((j + 1 for j, t in enumerate(row) if t in eos), len(row))]
            text = tok.decode(reply, skip_special_tokens=True)
            tally[chosen_task(text)] += 1
            io.append_jsonl(out, dict(family=family, model=m["name"], persona=persona, condition=cond, pair=p,
                                      first=tasks[pairs[p][0]]["id"], second=tasks[pairs[p][1]]["id"],
                                      sample=k % args.n_samples, prompt_ids=ids[p], reply_ids=reply, reply=text,
                                      ended=ended))
            print(f"  {cond} pair {p}: {text[:160]!r}", flush=True)
        print(f"{cond}: Task A {tally['A']}, Task B {tally['B']}, neither or both {tally[None]}", flush=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
