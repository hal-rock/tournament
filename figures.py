"""Slide figures for the tournament.

  figures/prompt_examples.png          the two prompts side by side, on one random task pair
  figures/scatter_vs_instruct.png      each stage's coinflip utilities vs the final Instruct
                                       Assistant utilities, one fit line, r over all / benign tasks
  figures/by_origin_<stage>.png        one stage's coinflip vs that stage's Assistant utilities,
                                       with a fit line and r per task origin

Prompts are rendered with prompts.build_prompt and the OLMo 3.1 Instruct chat
template (no system prompt), exactly as scored. The pair is drawn at random
from the 300-task set, restricted to tasks of at most MAX_CHARS characters so
the panels stay legible, and to two different task sources (origins).

Utilities are read from analyze.py's utilities.csv for run d8ysbybo (no length
control). "Benign" is analyze.py's definition: not BailBench and not topic
harmful_request.

Usage: uv run python figures.py [--seed N] [--utilities PATH]
"""

import argparse
import csv
import json
import random
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

from analyze import HARM_ORIGINS, HARM_TOPICS, ORIGIN_STYLE, pearson
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
UTILITIES = "outputs/d8ysbybo/analysis/utilities.csv"
STAGES = {"base": "Base", "sft": "SFT", "dpo": "DPO", "instruct": "Instruct"}
ORIGIN_LABELS = {"MATH": "MATH", "ALPACA": "Alpaca", "WILDCHAT": "WildChat",
                 "STRESS_TEST": "STRESS-TEST", "BAILBENCH": "BailBench"}
N_BOOTSTRAP = 2000


def wrap(prompt):
    """Wrap each line of the rendered prompt, indenting continuations of 'Task X:' lines."""
    lines = []
    for line in prompt.split("\n"):
        indent = " " * (line.index(":") + 2) if line.startswith("Task ") else ""
        lines += textwrap.wrap(line, WIDTH, subsequent_indent=indent) or [""]
    return lines


def prompt_figure(seed):
    tasks = [json.loads(line) for line in open(TASKS)]
    short = [t for t in tasks if t["n_chars"] <= MAX_CHARS]
    rng = random.Random(seed)
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

    fig.savefig(OUT / "prompt_examples.png", facecolor="white")
    plt.close(fig)


def load_utilities(path):
    """utilities.csv -> ({column: array}, origins, benign mask)."""
    rows = list(csv.DictReader(open(path)))
    cols = {k: np.array([float(r[k]) for r in rows]) for k in rows[0] if "/" in k}
    origins = np.array([r["origin"] for r in rows])
    benign = np.array([r["topic"] not in HARM_TOPICS and r["origin"] not in HARM_ORIGINS for r in rows])
    return cols, origins, benign


def scatter_points(ax, x, y, origins):
    for origin, (color, marker) in ORIGIN_STYLE.items():
        m = origins == origin
        ax.scatter(x[m], y[m], s=20, c=color, marker=marker, edgecolors="white", linewidths=0.5,
                   label=ORIGIN_LABELS[origin], zorder=3)


def style_axes(ax):
    ax.axhline(0, color=RULE, lw=0.8, zorder=1)
    ax.axvline(0, color=RULE, lw=0.8, zorder=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8.5)


def fit_line(ax, x, y, color, lw=1.8):
    xs = np.array([x.min(), x.max()])
    ax.plot(xs, np.polyval(np.polyfit(x, y, 1), xs), color=color, lw=lw, zorder=4)


def bootstrap_r(x, y, rng):
    idx = rng.integers(0, len(x), (N_BOOTSTRAP, len(x)))
    rs = np.sort([pearson(x[i], y[i]) for i in idx])
    return rs[int(0.025 * N_BOOTSTRAP)], rs[int(0.975 * N_BOOTSTRAP)]


def scatter_vs_instruct(U, origins, benign):
    x = U["instruct/assistant"]
    fig, axes = plt.subplots(1, len(STAGES), figsize=(14, 3.9), dpi=200, sharex=True)
    for ax, (stage, label) in zip(axes, STAGES.items()):
        y = U[f"{stage}/coinflip"]
        scatter_points(ax, x, y, origins)
        fit_line(ax, x, y, INK)
        r, rb = pearson(x, y), pearson(x[benign], y[benign])
        ax.set_title(f"{label} coinflip", loc="left", fontsize=11, color=INK, pad=20)
        ax.text(0, 1.03, f"r = {r:.2f}   (benign only: {rb:.2f})", transform=ax.transAxes,
                fontsize=9.5, color=INK)
        ax.set_xlabel("Instruct Assistant utility", color=MUTED, fontsize=9)
        style_axes(ax)
        print(f"vs instruct/assistant  {stage:9} r = {r:.3f}  benign r = {rb:.3f}")
    axes[0].set_ylabel("Coinflip utility (log-odds)", color=MUTED, fontsize=9)
    axes[-1].legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(1.02, 1), handletextpad=0.2)
    fig.suptitle(f"Coinflip utilities at each stage vs the final Instruct model's Assistant utilities "
                 f"({len(x)} tasks; y-axes differ between panels; benign = {benign.sum()} tasks)",
                 x=0.01, ha="left", fontsize=10, color=MUTED)
    fig.tight_layout()
    fig.savefig(OUT / "scatter_vs_instruct.png", facecolor="white")
    plt.close(fig)


def by_origin(U, origins, stage, rng):
    x, y = U[f"{stage}/assistant"], U[f"{stage}/coinflip"]
    fig, ax = plt.subplots(figsize=(10.5, 5), dpi=200)
    scatter_points(ax, x, y, origins)
    handles, labels = ax.get_legend_handles_labels()
    for i, origin in enumerate(ORIGIN_STYLE):
        m = origins == origin
        fit_line(ax, x[m], y[m], ORIGIN_STYLE[origin][0], lw=1.6)
        r, (lo, hi) = pearson(x[m], y[m]), bootstrap_r(x[m], y[m], rng)
        labels[i] = f"{ORIGIN_LABELS[origin]:12} r = {r:5.2f}  [{lo:.2f}, {hi:.2f}]  n = {m.sum()}"
        print(f"{stage:9} {origin:12} r = {r:.3f} [{lo:.3f}, {hi:.3f}] n = {m.sum()}")
    ax.legend(handles, labels, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1),
              prop={"family": "monospace", "size": 8.5}, handletextpad=0.3)
    ax.set_title(f"{STAGES[stage]}: coinflip vs Assistant utilities, within each task source",
                 loc="left", fontsize=11, color=INK)
    ax.set_xlabel(f"{STAGES[stage]} Assistant utility (log-odds)", color=MUTED, fontsize=9.5)
    ax.set_ylabel(f"{STAGES[stage]} coinflip utility (log-odds)", color=MUTED, fontsize=9.5)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT / f"by_origin_{stage}.png", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--utilities", default=UTILITIES)
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    prompt_figure(args.seed)
    U, origins, benign = load_utilities(args.utilities)
    scatter_vs_instruct(U, origins, benign)
    rng = np.random.default_rng(0)
    for stage in STAGES:
        by_origin(U, origins, stage, rng)


if __name__ == "__main__":
    main()
