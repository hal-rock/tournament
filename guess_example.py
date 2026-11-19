"""A shareable example of the guess prompt (README, "Shorter questions"), on Gemma 4 31B-it (Hal, 2026-11-19).

The prompt is the one the guess run scored (W&B 7vblk8tv, condition coin_predict_think_bold; prompts.build_prompt with
Gemma's own template and no system prompt, as report_patch.py renders it): the coinflip user turn word for word through
"it came up", then the user withholds the outcome and asks for a guess, and the Assistant's turn is prefilled up to the
bold it opens for its answer. The next-token probabilities of "heads" / "tails" are the run's unpatched ones.

The pair is picked from the run's 1,000 pairs by rule: both tasks from MATH or Alpaca (no WildChat or harmful tasks),
both short (under 200 characters together), the Assistant clearly preferring the Task B / tails task (sweep utilities
jrl62npf, A_i - A_j < -2), and the guess's pair effect close to the median of all clear pairs (3.6 nats against a
median of 3.4). Gemma rather than Llama: Llama's guess leans +5.7 nats toward heads in every pair, as much as its pair
effect, so a single order of a Llama prompt mostly shows the heads bias; Gemma's lean is +1.2 nats.

Usage: uv run --with playwright python guess_example.py
  -> outputs/coin_report/gemma4-31b/guess_example.html (the matskit.view page; outputs/ is gitignored)
     figures/coin_report_guess_example.png (its screenshot)
"""

import csv
import json
import math
from pathlib import Path

from transformers import AutoTokenizer

from matskit import view
from prompts import CONDITIONS, build_prompt

REPO, CONDITION = "google/gemma-4-31B-it", "coin_predict_think_bold"
FIRST, SECOND = "alpaca_13796", "competition_math_11278"  # Task A (heads), Task B (tails); Gemma prefers Task B
TASKS = "data/tasks_n300_max500_seed0.jsonl"
RUNS = Path("outputs/coin_report/gemma4-31b")
GUESS, PREFERENCE, SWEEP = "7vblk8tv", "fb5mlfde", "jrl62npf"  # the guess, the "heads or tails?" preference, the sweep


def scored(run, first, second):
    """(P(heads), P(tails)) of the unpatched pass for this order, from the run's rows (log-odds and outcome mass)."""
    for line in open(RUNS / run / "rows.jsonl"):
        r = json.loads(line)
        if (r["a"], r["b"]) == (first, second):
            log_odds, mass = r["report"]["none"], r["mass"]["none"]
            p_heads = mass / (1 + math.exp(-log_odds))
            return p_heads, mass - p_heads
    raise KeyError((run, first, second))


def main():
    tasks = {t["id"]: t for t in map(json.loads, open(TASKS))}
    utility = {r["id"]: float(r["instruct/assistant"])
               for r in csv.DictReader(open(Path("outputs") / SWEEP / "analysis" / "utilities.csv"))}
    tok = AutoTokenizer.from_pretrained(REPO)
    first, second = tasks[FIRST]["prompt"], tasks[SECOND]["prompt"]
    spec = CONDITIONS[CONDITION]
    text = build_prompt(CONDITION, first, second, tok)
    assert text.endswith(spec["prefill"]), text[-60:]
    body = spec["body"].format(first=first.strip(), second=second.strip())

    heads, tails = scored(GUESS, FIRST, SECOND)
    swapped = scored(GUESS, SECOND, FIRST)
    pref, pref_swapped = scored(PREFERENCE, FIRST, SECOND), scored(PREFERENCE, SECOND, FIRST)
    gap = utility[SECOND] - utility[FIRST]
    assert gap > 2 and tails > heads and swapped[0] > swapped[1], "not the example the docstring describes"

    parts = [dict(text=body, role="user", turn=0), dict(text=spec["prefill"], role="assistant", turn=1)]
    t = view.from_exact(
        tok, text, parts, title=f"Gemma 4 31B-it: the Assistant's guess at the withheld coin ({FIRST} vs {SECOND})",
        meta={"guess, next token": f"heads {heads:.2f} / tails {tails:.2f}",
              "tasks swapped (the math task is heads)": f"heads {swapped[0]:.2f} / tails {swapped[1]:.2f}",
              "stated preference, \"heads or tails?\"": (f"tails {pref[1]:.2f} here, heads {pref_swapped[0]:.2f} swapped"),
              "the Assistant's preference": f"Task B (the math task) by {gap:.1f} in sweep utilities"})
    # The run reads the next token after the prefill: show it at the cut, as from_open_turn does for its prompts
    # (from_open_turn re-renders the Assistant turn, which for Gemma 4 isn't the generation prompt + prefill the run used).
    t.cut = {"label": "the run reads the next token here: heads or tails",
             "next": [("heads", heads), ("tails", tails)]}
    page = view.write(RUNS / "guess_example.html", [t], title="The Assistant's guess at the coin: an example prompt",
                      note=("The prompt as Gemma 4 31B-it read it in the guess run (its own chat template, no system "
                            "prompt). The user's message is the coinflip prompt through \"it came up\", then a question; "
                            "the Assistant's turn is prefilled through \"**\". Pair picked by rule (MATH/Alpaca, short, a "
                            "clear preference for the tails task, pair effect near the median): see guess_example.py."),
                      view="template")
    view.png(page, Path("figures") / "coin_report_guess_example.png", view="template")


if __name__ == "__main__":
    main()
