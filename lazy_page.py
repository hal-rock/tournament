"""Lazy personas, uptake check: one matskit.view page per model with the free replies lazy_replies.py sampled.

Each transcript is the exact prompt and reply ids the model read and wrote. Transcripts of the same task pair are one group,
so a pair's replies under the default, "You are a ___." and "In your next message, answer as a ___." sit side by side. Meta:
persona, phrasing, the tasks, which task the reply names in its opening (lazy_replies.chosen_task), whether it ended before
max_new_tokens.

Writes outputs/lazy_replies/<family>/replies.html (gitignored: WildChat tasks appear in the prompts).

Usage: uv run python lazy_page.py [--models llama-70b ...]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoTokenizer

from analyze import read_jsonl
from lazy import MODELS
from lazy_replies import chosen_task
from matskit import io, view
from models import MODEL_INFO
from prompts import CONDITIONS, split_condition
from rollouts_page import labels

ORDER = ("assistant", "youare", "answeras")


def phrasing(cond):
    return "default" if cond == "assistant" else cond.split("_")[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    args = ap.parse_args()
    tasks = {t["id"]: t["prompt"].strip() for t in read_jsonl("data/tasks_n300_max500_seed0.jsonl")}
    for family in args.models:
        files = sorted((Path("outputs/lazy_replies") / family).glob("*.jsonl"))
        if not files:
            print(f"{family}: no replies yet")
            continue
        rows = [r for f in files for r in io.read_jsonl(f)]
        seen, keep = set(), []
        for r in rows:  # the default replies come from one pod per model; keep one copy if there are more
            key = (r["condition"], r["pair"], r["sample"])
            if key not in seen:
                seen.add(key)
                keep.append(r)
        tok = AutoTokenizer.from_pretrained(MODELS[family])
        rank = lambda r: (r["pair"], r["condition"] != "assistant", r["persona"] if r["condition"] != "assistant" else "",
                          ORDER.index(phrasing(r["condition"]) if r["condition"] != "assistant" else "assistant"),
                          r["sample"])
        out = []
        for r in sorted(keep, key=rank):
            first, second = tasks[r["first"]], tasks[r["second"]]
            body = CONDITIONS[split_condition(r["condition"])[0]]["body"].format(first=first, second=second)
            role, turn, kind = labels(tok, r["prompt_ids"], body)
            n = len(r["reply_ids"])
            ph = phrasing(r["condition"])
            persona = "-" if ph == "default" else r["persona"]
            chose = chosen_task(r["reply"])
            out.append(view.from_ids(
                tok, r["prompt_ids"] + r["reply_ids"], role=role + ["assistant"] * n, turn=turn + [1] * n,
                kind=kind + ["content"] * n, sampled=[False] * len(r["prompt_ids"]) + [True] * n,
                title=f"pair {r['pair']} · {persona} · {ph} · sample {r['sample']}",
                group=f"pair {r['pair']}",
                meta={"persona": persona, "phrasing": ph, "Task A": r["first"], "Task B": r["second"],
                      "names": f"Task {chose}" if chose else "neither or both", "ended": r["ended"]},
                turn_flags=None if r["ended"] else {1: ("warn", f"cut at {n} tokens")}))
        page = Path("outputs/lazy_replies") / family / "replies.html"
        view.write(page, out, title=f"{MODEL_INFO[MODELS[family]][0]}: free replies under lazy personas",
                   note="Six task pairs from WildChat, Alpaca and MATH (a fixed random draw, the same for every model), two "
                        "replies each per condition, sampled at temperature 1 with the model's own generation defaults; "
                        "the Assistant's turn is not prefilled. Exact prompt and reply ids.")


if __name__ == "__main__":
    main()
