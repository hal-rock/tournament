"""Slide figure: the two tournament prompts side by side, on one random task pair.

  figures/prompt_examples.png

Prompts are rendered with prompts.build_prompt and the OLMo 3.1 Instruct chat
template (no system prompt), exactly as scored. The pair is drawn at random
from the 300-task set, restricted to tasks of at most MAX_CHARS characters so
the panels stay legible, and to two different task sources (origins).

Usage: uv run python figures.py [--seed N]
"""

import argparse
import json
import random
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from transformers import AutoTokenizer

from prompts import CONDITIONS, build_prompt

TASKS = "data/tasks_n300_max500_seed0.jsonl"
TEMPLATE = "allenai/Olmo-3.1-32B-Instruct"
MAX_CHARS = 200
WIDTH = 50  # characters per line in each panel
OUT = Path("figures")
PANELS = {
    "assistant": "Assistant choice",
    "coinflip": "Coinflip (Assistant does the task)",
}
INK, MUTED, RULE = "#0b0b0b", "#8a8984", "#d6d5d0"


def wrap(prompt):
    """Wrap each line of the rendered prompt, indenting continuations of 'Task X:' lines."""
    lines = []
    for line in prompt.split("\n"):
        indent = " " * (line.index(":") + 2) if line.startswith("Task ") else ""
        lines += textwrap.wrap(line, WIDTH, subsequent_indent=indent) or [""]
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tasks = [json.loads(line) for line in open(TASKS)]
    short = [t for t in tasks if t["n_chars"] <= MAX_CHARS]
    rng = random.Random(args.seed)
    first, second = rng.sample(short, 2)
    while first["origin"] == second["origin"]:  # show two different task sources
        first, second = rng.sample(short, 2)
    print(f"Task A: {first['id']} ({first['origin']})\nTask B: {second['id']} ({second['origin']})")

    tokenizer = AutoTokenizer.from_pretrained(TEMPLATE)
    panels = {c: wrap(build_prompt(c, first["prompt"], second["prompt"], tokenizer)) for c in PANELS}
    n_lines = max(len(v) for v in panels.values())

    line_h = 0.19  # inches
    fig_h = 1.6 + n_lines * line_h
    fig = plt.figure(figsize=(11, fig_h), dpi=200)
    fig.patch.set_facecolor("white")
    y_top = 1 - 0.35 / fig_h
    for col, (condition, lines) in enumerate(panels.items()):
        x = 0.03 + col * 0.5
        fig.text(x, y_top, PANELS[condition], fontsize=12, fontweight="bold", color=INK, va="top")
        for i, line in enumerate(lines):
            template_only = line.startswith("<|im_start|>") and line.endswith(("user", "assistant"))
            fig.text(x, y_top - (0.45 + i * line_h) / fig_h, line, family="monospace", fontsize=8.8,
                     color=MUTED if template_only else INK, va="top", parse_math=False)
        a, b = CONDITIONS[condition]["outcomes"]
        fig.text(x, 0.3 / fig_h, f'Next token:  P("{a}")  vs  P("{b}")', fontsize=10.5,
                 fontweight="bold", color=INK, va="bottom")
    fig.add_artist(plt.Line2D([0.5, 0.5], [0.08, 0.95], color=RULE, lw=1))

    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "prompt_examples.png", facecolor="white")


if __name__ == "__main__":
    main()
