"""The shorter questions (README, "Shorter questions: the preferred outcome without 'heads or tails', and a guess"): per model,
the earlier "heads or tails?" preference question and the two new ones side by side, from report_patch.py runs.

Each run goes through coin_report.analyze_run (full per-run report and scatter), then:
  outputs/coin_report/questions.csv     one row per run: wording, mass, primary flag, the unpatched answer's r with the coin and
                                        with the Assistant (sweep A_i - A_j), its slope on the coin, and the swap arm's measures
  figures/coin_report_questions.png     per model and question (primary wording): r(answer, Assistant), slope(answer on coin)
                                        and the share through " up", with 95% bootstrap CIs over pairs
  figures/coin_report_guess_bias.png    per model: the unpatched guess against the Assistant's preference, pair by pair

Usage: uv run python coin_questions.py outputs/coin_report/<model>/<run>/rows.jsonl [...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import pearson
from coin_report import LABEL, analyze_run, mark_primary, question, sweep_rows
from matskit import plotstyle
from openers import write_csv
from report import INK, INK_2, RULE, style

QUESTIONS = {"coin_report": 'preferred outcome, "heads or tails?"', "coin_report_short": "preferred outcome, short",
             "coin_predict": "guess", "coin_predict_think": "guess", "coin_predict_my": "guess"}
GROUP = {"coin_report": "pref_full", "coin_report_short": "pref_short", "coin_predict": "guess", "coin_predict_think": "guess",
         "coin_predict_my": "guess"}
SERIES_NAME = {"pref_full": 'preferred, "heads or tails?"', "pref_short": "preferred, short", "guess": "guess"}
STATS = ("r(report, Assistant (sweep A_i - A_j))", "r(report, coin)", "slope(report on coin)", "swap: r(d, coin)",
         "swap: share", "kappa (swap+ctrl)", "lambda (swap+ctrl)", "swap_early: r(d, coin)", "swap_late: r(d, coin)",
         "ctrl: r(d, coin)", "swap_line: share")


def assistant_diff(family, D):
    """A_i - A_j per pair from the model's sweep (the default Assistant's utilities)."""
    srows = sweep_rows(family)
    U = np.array([float(r["instruct/assistant"]) for r in srows])
    pos = {r["id"]: k for k, r in enumerate(srows)}
    return U[[pos[t] for t in D["a"]]] - U[[pos[t] for t in D["b"]]]


def plot_summary(rows, path):
    panels = (("r(report, Assistant (sweep A_i - A_j))", "unpatched answer: r with the Assistant's preference", 1),
              ("slope(report on coin)", "unpatched answer: slope on the coin (nats per nat)", 1),
              ("swap: share", "share reversed by swapping \" up\" (%)", 100))
    families = [f for f in LABEL if any(r["family"] == f for r in rows)]
    groups = [g for g in SERIES_NAME if any(GROUP[question(r["report"])] == g for r in rows)]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 4.2))
    for ax, (stat, title, scale) in zip(axes, panels):
        for k, g in enumerate(groups):
            for i, f in enumerate(families):
                sel = [r for r in rows if r["family"] == f and r["primary"] and GROUP[question(r["report"])] == g]
                if not sel:
                    continue
                (r,) = sel
                e, lo, hi = scale * r[stat], scale * r[stat + "_lo"], scale * r[stat + "_hi"]
                ax.errorbar(i + (k - (len(groups) - 1) / 2) * 0.22, e, yerr=[[e - lo], [hi - e]], color=plotstyle.SERIES[k],
                            marker="os^"[k], ms=7, lw=0, elinewidth=1.5, mec="white", mew=0.8,
                            label=SERIES_NAME[g] if i == 0 else None)
        ax.axhline(0, color=RULE, lw=0.8)
        ax.set_xticks(range(len(families)), [LABEL[f] for f in families], fontsize=9)
        ax.set_title(title, color=INK, fontsize=10)
        style(ax)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, frameon=False, fontsize=9, loc="lower center", ncol=len(groups))
    fig.suptitle("Which Assistant answers read the coin's \" up\": three questions (primary wording per model)", color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    plotstyle.save(fig, path.stem, path.parent)


def plot_guess_bias(results, path):
    guesses = [(f, rep, D) for f, rep, D, *_ in results if GROUP[question(rep)] == "guess" and D["primary"]]
    if not guesses:
        return
    fig, axes = plt.subplots(1, len(guesses), figsize=(4.0 * len(guesses), 3.8), squeeze=False)
    for ax, (f, rep, D) in zip(axes[0], guesses):
        x, y = D["A"], D["r"]["none"]
        ax.scatter(x, y, s=8, alpha=0.5, color=plotstyle.SERIES[0], edgecolors="none")
        xs = np.array([x.min(), x.max()])
        ax.plot(xs, np.polyval(np.polyfit(x, y, 1), xs), color=INK_2, lw=1.2)
        ax.axhline(0, color=RULE, lw=0.8)
        ax.axvline(0, color=RULE, lw=0.8)
        ax.set_title(f"{LABEL[f]} ({'bold' if rep.endswith('_bold') else 'plain'})\nr = {pearson(x, y):+.2f}",
                     fontsize=10, color=INK)
        ax.set_xlabel("Assistant's preference, sweep A_i - A_j", fontsize=9, color=INK_2)
        ax.set_ylabel("guess, unpatched (nats, pair effect)", fontsize=9, color=INK_2)
        style(ax)
    fig.suptitle("Does the Assistant's guess at the withheld outcome lean toward the task it prefers?", color=INK)
    fig.tight_layout()
    plotstyle.save(fig, path.stem, path.parent)


def main():
    plotstyle.setup()
    results = [analyze_run(Path(p)) for p in sys.argv[1:]]
    rows = []
    for family, report, D, est, lo, hi in results:
        row = dict(family=family, report=report, question=QUESTIONS[question(report)],
                   prefill="bold" if report.endswith("_bold") else "plain", mass=float(np.median(D["mass"]["none"])),
                   coin_sd=float(D["c"].std()), answer_sd=float(D["r"]["none"].std()))
        for k in STATS:
            row |= {k: est[k], k + "_lo": lo[k], k + "_hi": hi[k]}
        rows.append(row)
        D["A"] = assistant_diff(family, D)
    mark_primary(rows)
    for (family, report, D, *_), row in zip(results, rows):
        D["primary"] = row["primary"]
    write_csv(Path("outputs/coin_report/questions.csv"), rows)
    print("\nPer model and question (* = primary wording):")
    for r in sorted(rows, key=lambda r: (list(LABEL).index(r["family"]), list(QUESTIONS).index(question(r["report"])),
                                         r["prefill"])):
        print(f"  {'*' if r['primary'] else ' '} {LABEL[r['family']]:14} {r['report']:24} mass {r['mass']:.2f}  SD answer "
              f"{r['answer_sd']:.2f} coin {r['coin_sd']:.2f}  " + "  ".join(
                  f"{k} {r[k]:+.3f} [{r[k + '_lo']:+.3f}, {r[k + '_hi']:+.3f}]" for k in STATS))
    plot_summary(rows, Path("figures/coin_report_questions.png"))
    plot_guess_bias(results, Path("figures/coin_report_guess_bias.png"))


if __name__ == "__main__":
    main()
