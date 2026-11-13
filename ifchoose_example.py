"""A shareable example of one reworded-instruction prompt and what Llama 3.3 70B does with it (Hal, 2026-11-13).

The shortest and longest MATH tasks of the 300 (16 and 452 characters), under "...choose the one with the longer description."
The user's message is shown verbatim, cut from the exact rendered prompt (Llama 3.3's chat template, no system prompt).
Below it are the coin's next-token probabilities after "it came up" from the scored run (W&B j07k5gix), as shown and with
the tasks swapped, next to the Assistant's own choice under the same instruction.

Usage: uv run python ifchoose_example.py   ->  figures/ifchoose_example_prompt.png
"""

import csv
import gzip
import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from analyze import read_jsonl
from matskit import plotstyle
from prompts import build_prompt
from report import INK, INK_2, RULE
from user_choice import TASKS_FILE

SHORT, LONG = "competition_math_6100", "competition_math_835"
RUN, SWEEP = "j07k5gix", "tvlqj8a0"  # the reworded length run and the sweep (Llama 3.3 70B)
HEADER = "<|start_header_id|>user<|end_header_id|>\n\n"


def probs(stem, run):
    """{(first, second): (P(first outcome), P(second outcome))} for the example pair, both orders."""
    out = {}
    with gzip.open(Path("outputs") / run / "results" / f"{stem}.csv.gz", "rt", newline="") as fh:
        for r in csv.DictReader(fh):
            if {r["first"], r["second"]} == {SHORT, LONG}:
                out[r["first"], r["second"]] = (math.exp(float(r["logp_first"])), math.exp(float(r["logp_second"])))
    return out


def main():
    from transformers import AutoTokenizer
    plotstyle.setup()
    tasks = {t["id"]: t for t in read_jsonl(TASKS_FILE)}
    tok = AutoTokenizer.from_pretrained("unsloth/Llama-3.3-70B-Instruct")
    rendered = build_prompt("coinflip_ifchoose_longer", tasks[SHORT]["prompt"], tasks[LONG]["prompt"], tok)
    user = rendered[rendered.index(HEADER) + len(HEADER):]  # the user's turn as the model read it, cut after "it came up"
    lines = []
    for para in user.split("\n"):
        lines += textwrap.wrap(para, 112, drop_whitespace=True) or [""]
    text = "\n".join(lines).replace("$", r"\$")  # literal dollar signs, not mathtext

    coin = probs("instruct__coinflip_ifchoose_longer", RUN)
    choice = probs("instruct__assistant_ifchoose_longer", RUN)
    plain = probs("instruct__coinflip", SWEEP)
    shown, swapped = coin[SHORT, LONG], coin[LONG, SHORT]  # (P(heads), P(tails))
    rows = [  # (label, P(longer task), P(shorter task))
        ("Coin, as shown\n(longer task is Task B, tails)", shown[1], shown[0]),
        ("Coin, tasks swapped\n(longer task is Task A, heads)", swapped[0], swapped[1]),
        ("The Assistant's own choice, same instruction\n(\" A\" / \" B\" after \"I'll choose Task\")",
         choice[SHORT, LONG][1], choice[SHORT, LONG][0]),
    ]
    assert abs(choice[LONG, SHORT][0] - rows[2][1]) < 1e-3, "the Assistant's choice differs between orders"

    n_lines = len(lines)
    prompt_h = 0.172 * n_lines + 0.42
    fig = plt.figure(figsize=(10.5, prompt_h + 3.55))
    top = fig.add_axes([0.03, 1 - (prompt_h + 0.5) / fig.get_figheight(), 0.94, prompt_h / fig.get_figheight()])
    top.axis("off")
    top.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.0,rounding_size=0.02", transform=top.transAxes,
                                 facecolor="#f4f2ee", edgecolor=RULE, lw=1))
    top.text(0.015, 1 - 0.2 / prompt_h, text, transform=top.transAxes, va="top", ha="left", family="DejaVu Sans Mono",
             fontsize=8.6, color=INK, linespacing=1.35)
    fig.text(0.03, 1 - 0.32 / fig.get_figheight(), "User's message (verbatim)", fontsize=10.5, color=INK, weight="bold")
    fig.text(0.97, 1 - 0.32 / fig.get_figheight(), "Llama 3.3 70B Instruct, the model's own chat template, no system prompt",
             fontsize=8.5, color=INK_2, ha="right")

    bars = fig.add_axes([0.31, 0.55 / fig.get_figheight(), 0.66, 2.15 / fig.get_figheight()])
    long_c, short_c = plotstyle.SERIES[1], plotstyle.SERIES[0]
    for i, (label, p_long, p_short) in enumerate(rows):
        y = len(rows) - 1 - i
        other = max(0.0, 1 - p_long - p_short)
        bars.barh(y, p_long, color=long_c, height=0.62, edgecolor="white", lw=2)
        bars.barh(y, p_short, left=p_long, color=short_c, height=0.62, edgecolor="white", lw=2)
        bars.barh(y, other, left=p_long + p_short, color=RULE, height=0.62, edgecolor="white", lw=2)
        bars.text(0.012, y, f"longer task {p_long:.2f}", va="center", ha="left", color="white", fontsize=9, weight="bold")
        if p_short > 0.12:
            bars.text(p_long + 0.012, y, f"shorter {p_short:.2f}", va="center", ha="left", color="white", fontsize=9,
                      weight="bold")
        bars.text(-0.015, y, label, va="center", ha="right", color=INK, fontsize=9)
    bars.set_xlim(0, 1)
    bars.set_ylim(-0.5, len(rows) - 0.5)
    bars.axis("off")
    fig.text(0.31, (0.55 + 2.15 + 0.1) / fig.get_figheight(),
             "Next-token probability after \"it came up\" (gray: other tokens, such as \" ...\")", fontsize=8.8, color=INK_2)
    avg_long, avg_short = (shown[1] + swapped[0]) / 2, (shown[0] + swapped[1]) / 2
    fig.text(0.03, 0.16 / fig.get_figheight(),
             f"Averaged over both orders, the coin gives the longer task {avg_long:.2f} and the shorter {avg_short:.2f}. "
             f"Without the instruction, Llama's coin says \" heads\" about {(plain[SHORT, LONG][0] + plain[LONG, SHORT][0]) / 2:.2f} "
             "whichever task is Task A.", fontsize=8.6, color=INK_2)
    plotstyle.save(fig, "ifchoose_example_prompt", Path("figures"))


if __name__ == "__main__":
    main()
