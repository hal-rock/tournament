"""The outcome lean of the Assistant's coin answers (exploratory, Hal, 2026-11-19; README, "Shorter questions", "The guess's
outcome lean").

Every report question is scored in both task orders, and the pair effect (half the difference of the two orders' log-odds)
cancels any lean toward one outcome word. This script measures that lean instead: per pair, the mean of the two orders'
log-odds (first outcome minus second: heads - tails, or " A" - " B" for the Task A/B control), averaged over the 1,000
pilot pairs, with a 95% bootstrap CI over pairs. Next to it, for scale, the mean size of the pair effect, and the share of
single prompts whose top answer names the outcome of the task the Assistant clearly prefers (|A_i - A_j| > 2 in the sweep's
utilities), split by whether that task is heads (Task A) or tails (Task B). The user-turn coin at " up", read in the same
forward passes (the rows' "coin"), is the baseline.

Models: the "Shorter questions" runs (Gemma 4 31B, Llama 3.3 70B, GLM-4.5-Air) and the replication's finding 8 runs (Llama
3.1 405B, Nemotron-4 340B, DeepSeek V3, Kimi K2, DeepSeek V4 Flash), all from outputs/coin_report/<model>/<run>/rows.jsonl.

    uv run python scripts/guess_lean.py   ->  outputs/coin_report/guess_lean.csv, figures/coin_report_guess_lean.png
"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from matskit import plotstyle, stats

ROWS = Path("outputs/coin_report")
SWEEP = {"gemma4-31b": "outputs/jrl62npf", "llama-70b": "outputs/tvlqj8a0", "glm-4.5-air": "outputs/i5vejs3b"}
MODELS = {  # key: (label, {question: run}); the coin at " up" comes from the guess run's rows
    "gemma4-31b": ("Gemma 4 31B", {"preference, heads or tails?": "fb5mlfde", "short preference": "jtlbqu7f",
                                    "guess": "7vblk8tv"}),
    "llama-70b": ("Llama 3.3 70B", {"preference, heads or tails?": "8eod0l7e", "short preference": "o3jwywnx",
                                     "guess": "ly2ng7l0"}),
    "glm-4.5-air": ("GLM-4.5-Air", {"preference, heads or tails?": "3y2ncefl", "short preference": "zu4wux9u",
                                     "guess": "or7e7o6a"}),
    "llama-405b": ("Llama 3.1 405B", {"short preference": "ycs5go29", "guess": "01la3hzo", "Task A/B (control)": "l8yiup2o"}),
    "nemotron4-340b": ("Nemotron-4 340B", {"short preference": "gxtyjauq", "guess": "a22g64uf",
                                           "Task A/B (control)": "gxkgx4cx"}),
    "deepseek-v3": ("DeepSeek V3", {"short preference": "gzualmt2", "guess": "r7zjpyo0", "Task A/B (control)": "0aw26562"}),
    "kimi-k2": ("Kimi K2", {"short preference": "gso9m014", "guess": "081zwk4v", "Task A/B (control)": "xlrye2ib"}),
    "deepseek-v4-flash": ("DeepSeek V4 Flash", {"short preference": "i75ggejl", "guess": "kkde34ba",
                                                "Task A/B (control)": "5uv38f7i"}),
}
CLEAR = 2.0  # |A_i - A_j| for "the Assistant clearly prefers one task", as in the README's guess table


def utilities(key):
    path = Path(SWEEP.get(key, f"outputs/replication/{key}")) / "analysis" / "utilities.csv"
    return {r["id"]: float(r["instruct/assistant"]) for r in csv.DictReader(open(path))}


def both_orders(run_dir, field):
    """(a, b, x_ab, x_ba): each pair's log-odds (first outcome - second) in both orders, task a first in x_ab."""
    rows = {(r["a"], r["b"]): r for r in map(json.loads, open(run_dir / "rows.jsonl"))}
    get = (lambda r: r["coin"]) if field == "coin" else (lambda r: r["report"]["none"])
    out = [(a, b, get(rows[a, b]), get(rows[b, a])) for (a, b) in rows if a < b and (b, a) in rows]
    assert len(out) == len(rows) // 2, f"{run_dir}: pairs missing an order"
    return out


def measure(pairs, U):
    a, b, x_ab, x_ba = zip(*pairs)
    x_ab, x_ba = np.array(x_ab), np.array(x_ba)
    lean = (x_ab + x_ba) / 2
    lo, hi = stats.bootstrap_ci(lean, seed=0)
    gap = np.array([U[i] - U[j] for i, j in zip(a, b)])  # > 0: the Assistant prefers task a
    # Single prompts of clear pairs: does the top answer name the preferred task's outcome? Task a is first (heads) in x_ab.
    first_pref, second_pref = [], []  # the preferred task is heads / tails in that prompt
    for g, xab, xba in zip(gap, x_ab, x_ba):
        if abs(g) <= CLEAR:
            continue
        if g > 0:  # a preferred: heads in x_ab, tails in x_ba
            first_pref.append(xab > 0), second_pref.append(xba < 0)
        else:  # b preferred: tails in x_ab, heads in x_ba
            second_pref.append(xab < 0), first_pref.append(xba > 0)
    return {"pairs": len(lean), "lean": lean.mean(), "lean lo": lo, "lean hi": hi,
            "pair effect (mean size)": np.abs(x_ab - x_ba).mean() / 2,
            "follows the task across orders": np.mean((x_ab > 0) != (x_ba > 0)),
            "clear prompts": len(first_pref) + len(second_pref),
            "names preferred, preferred is heads/A": np.mean(first_pref),
            "names preferred, preferred is tails/B": np.mean(second_pref)}


def plot(rows, path):
    guess = [r for r in rows if r["question"] == "guess"]
    labels = [r["label"] for r in guess]
    y = np.arange(len(guess))[::-1]
    heads_c, tails_c = plotstyle.SERIES[0], plotstyle.SERIES[1]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True, gridspec_kw={"width_ratios": [1, 1.1]})

    for yy, r in zip(y, guess):
        e = r["pair effect (mean size)"]
        ax1.barh(yy, 2 * e, left=-e, height=0.5, color=plotstyle.GRID,
                 zorder=1, label="± mean pair effect" if yy == y[0] else None)
        ax1.plot([r["lean lo"], r["lean hi"]], [yy, yy], color=plotstyle.INK, lw=2, zorder=3, solid_capstyle="round")
        ax1.plot(r["lean"], yy, "o", ms=8, color=plotstyle.INK, mec="white", mew=1.5, zorder=4,
                 label="lean, order-averaged (95% CI)" if yy == y[0] else None)
    ax1.axvline(0, color=plotstyle.MUTED, lw=1, zorder=2)
    ax1.set_xlabel("nats: + toward heads, - toward tails")
    ax1.set_yticks(y, labels)
    ax1.set_title("The guess's lean toward one outcome word", loc="left", fontsize=10.5, weight="bold")
    lim = 1.1 * max(r["pair effect (mean size)"] for r in guess)
    ax1.set_xlim(-lim, lim)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=8, frameon=False)
    ax1.grid(axis="y", visible=False)

    for yy, r in zip(y, guess):
        h, t = 100 * r["names preferred, preferred is heads/A"], 100 * r["names preferred, preferred is tails/B"]
        ax2.plot([h, t], [yy, yy], color="#cfcdc7", lw=2, zorder=1)
        ax2.plot(h, yy, "o", ms=9, color=heads_c, mec="white", mew=1.5, zorder=3,
                 label="when the preferred task is heads" if yy == y[0] else None)
        ax2.plot(t, yy, "o", ms=9, color=tails_c, mec="white", mew=1.5, zorder=3,
                 label="when it is tails" if yy == y[0] else None)
    ax2.axvline(50, color=plotstyle.MUTED, lw=1, ls=(0, (3, 3)), zorder=0)
    ax2.set_xlim(-2, 102)
    ax2.set_xlabel("% of single prompts (clear pairs, |A_i - A_j| > 2)")
    ax2.set_title("How often one prompt's guess names the preferred task's outcome", loc="left", fontsize=10.5,
                  weight="bold")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=8, frameon=False)
    ax2.grid(axis="y", visible=False)
    for ax in (ax1, ax2):  # the earlier three models above, the replication's five below
        ax.axhline(y[2] - 0.5, color="#cfcdc7", lw=1)
    ax2.text(1, y[2] - 0.38, "Shorter questions run", fontsize=8, color=plotstyle.MUTED, va="bottom")
    ax2.text(1, y[2] - 0.62, "replication, larger models", fontsize=8, color=plotstyle.MUTED, va="top")
    fig.suptitle("The Assistant's guess at the withheld coin: a lean toward one outcome word, on top of its preference",
                 x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"wrote {path.resolve()}")


def main():
    plotstyle.setup()
    rows = []
    for key, (label, runs) in MODELS.items():
        U = utilities(key)
        for question, run in [("coin at \" up\" (user turn)", runs["guess"])] + list(runs.items()):
            field = "coin" if question.startswith("coin") else "report"
            m = measure(both_orders(ROWS / key / run, field), U)
            rows.append({"model": key, "label": label, "question": question, "run": run, **m})
    out = ROWS / "guess_lean.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out.resolve()}\n")
    print(f"{'model':18s} {'question':30s} {'lean [95% CI]':>22s} {'|pair|':>7s} {'follows':>8s} "
          f"{'pref heads/A':>12s} {'pref tails/B':>12s}")
    for r in rows:
        print(f"{r['label']:18s} {r['question']:30s} {r['lean']:+6.2f} [{r['lean lo']:+.2f}, {r['lean hi']:+.2f}] "
              f"{r['pair effect (mean size)']:7.2f} {r['follows the task across orders']:8.0%} "
              f"{r['names preferred, preferred is heads/A']:12.0%} {r['names preferred, preferred is tails/B']:12.0%}")
    plot(rows, Path("figures") / "coin_report_guess_lean.png")


if __name__ == "__main__":
    main()
