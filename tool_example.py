"""A shareable example of one tool-call coin prompt (README, "The coin as a tool call") on DeepSeek V4 Flash (Hal, 2026-11-12).

The prompt is the one the run scored (prompts.build_prompt, condition coinflip_tool, toolframe's deepseek4 family: V4's
reference encoder in chat mode, flip_coin declared in the system slot, no other system prompt): the user asks for a coin
flip, the Assistant's turn is a prefilled call to flip_coin, and the prompt is cut inside the tool's result, where the
next token is `heads` or `tails`. The pair is a short, benign one with a clear coin preference over both orders: a MATH
task as Task A against a STRESS-TEST request as Task B. The next-token probabilities are the scored run's.

Usage: uv run --with playwright python tool_example.py
  -> outputs/replication/tool-deepseek-v4-flash/example_prompt.html (the matskit.view page)
     figures/deepseek-v4-flash/tool_example_prompt.png (its screenshot)
"""

import csv
import gzip
import json
import math
from pathlib import Path

from transformers import AutoTokenizer

from matskit import view
from prompts import CONDITIONS, build_prompt

KEY, REPO = "deepseek-v4-flash", "deepseek-ai/DeepSeek-V4-Flash"
FIRST, SECOND = "competition_math_773", "stresstest_27_1202_neutral"
TASKS = "data/tasks_n300_max500_seed0.jsonl"
OUT = Path("outputs/replication") / f"tool-{KEY}"
RESULT_OPEN = '{"result": "'


def scored(first, second):
    """(P(heads), P(tails), prompt tokens) for this order, from the run's results."""
    with gzip.open(OUT / "results" / "instruct__coinflip_tool.csv.gz", "rt", newline="") as fh:
        for r in csv.DictReader(fh):
            if (r["first"], r["second"]) == (first, second):
                return math.exp(float(r["logp_first"])), math.exp(float(r["logp_second"])), int(r["prompt_tokens"])
    raise KeyError((first, second))


def main():
    tasks = {t["id"]: t for t in map(json.loads, open(TASKS))}
    tok = AutoTokenizer.from_pretrained(REPO)
    first, second = tasks[FIRST]["prompt"], tasks[SECOND]["prompt"]
    text = build_prompt("coinflip_tool", first, second, tok, "none", True, tool_family="deepseek4", tool_repo=REPO)
    assert text.endswith(RESULT_OPEN), text[-40:]
    p_heads, p_tails, n_tokens = scored(FIRST, SECOND)
    assert n_tokens == len(tok(text, add_special_tokens=False).input_ids), "not the prompt the run scored"
    swapped = scored(SECOND, FIRST)
    body = CONDITIONS["coinflip_tool"]["body"].format(first=first.strip(), second=second.strip())
    parts = [dict(text=body, role="user", turn=1), dict(text=RESULT_OPEN, role="tool", turn=3)]
    t = view.from_exact(
        tok, text, parts, title=f"DeepSeek V4 Flash: the coin as a tool call ({FIRST} vs {SECOND})",
        meta={"next token": f"heads {p_heads:.2f} / tails {p_tails:.2f}",
              "tasks swapped": f"heads {swapped[0]:.2f} / tails {swapped[1]:.2f}",
              "prefilled": "the Assistant's call to flip_coin and the start of the tool's result"})
    page = view.write(OUT / "example_prompt.html", [t], title="The coin as a tool call: an example prompt",
                      note=("The prompt as DeepSeek V4 Flash read it in the tool run (V4's reference encoder, chat mode; "
                            "flip_coin declared in the system slot, no other system prompt). Everything after the user's "
                            "message is prefilled; the run reads the next token, heads or tails. Pair picked by hand: "
                            "short, benign, with a clear coin preference in both orders."))
    view.png(page, Path("figures") / KEY / "tool_example_prompt.png", view="template")


if __name__ == "__main__":
    main()
