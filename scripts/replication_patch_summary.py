"""Finding 8 across the replication models (README, "Replication across larger models", "Finding 8: scope"): the swap share
and kappa of each report question per model, with their bootstrap CIs, from coin_report.analyze_run on each run's rows.

    uv run python scripts/replication_patch_summary.py [key ...]

Writes outputs/replication/patching_summary.csv and figures/replication_patching.png, and moves each run's diagnostic figure
(coin_report.plot) to figures/<key>/patch_<report>.png. Finding 8's call per question: the swap share above 0 with kappa's
CI above 0; the Task A/B question is the control, expected null.
"""

import csv
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import coin_report  # noqa: E402
from matskit import plotstyle  # noqa: E402

MODELS = {"llama-405b": "Llama 3.1 405B", "nemotron4-340b": "Nemotron-4 340B", "deepseek-v3": "DeepSeek V3",
          "kimi-k2": "Kimi K2", "deepseek-v4-flash": "DeepSeek V4 Flash"}
QUESTIONS = {"coin_report_short_coin_bold": "short preference", "coin_predict_think_bold": "guess",
             "coin_report_ab": "Task A/B (control)"}
STATS = ("swap: share", "kappa (swap+ctrl)", "swap: r(d, coin)", "ctrl: share", "r(report, coin)",
         "r(report, Assistant (sweep A_i - A_j))")


def main(keys):
    plotstyle.setup()
    rows = []
    for key in keys:
        for path in sorted(Path("outputs/coin_report", key).glob("*/rows.jsonl")):
            family, report, D, est, lo, hi = coin_report.analyze_run(path)
            if report not in QUESTIONS:
                continue
            src = Path(f"figures/coin_report_pilot_{family}_{report}.png")
            if src.exists():
                Path("figures", key).mkdir(parents=True, exist_ok=True)
                shutil.move(src, Path("figures", key, f"patch_{report}.png"))
            row = dict(model=key, report=report, run=path.parent.name, pairs=len(D["c"]),
                       report_mass=float(np.median(D["mass"]["none"])), coin_mass=float(np.median(D["coin_mass"])))
            for k in STATS:
                row |= {k: est[k], k + " lo": lo[k], k + " hi": hi[k]}
            q = report != "coin_report_ab"
            row["call"] = (("replicates" if est["swap: share"] > 0 and lo["kappa (swap+ctrl)"] > 0 else "doesn't") if q
                           else "control")
            rows.append(row)
    out = Path("outputs/replication/patching_summary.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")
    for r in rows:
        print(f"  {MODELS[r['model']]:18} {QUESTIONS[r['report']]:20} share {100 * r['swap: share']:5.1f}% "
              f"[{100 * r['swap: share lo']:.1f}, {100 * r['swap: share hi']:.1f}]  kappa {r['kappa (swap+ctrl)']:+.3f} "
              f"[{r['kappa (swap+ctrl) lo']:+.3f}, {r['kappa (swap+ctrl) hi']:+.3f}]  r(d, coin) {r['swap: r(d, coin)']:+.2f}"
              f"  {r['call']}")
    plot(rows, Path("figures/replication_patching.png"))


def plot(rows, path):
    keys = [k for k in MODELS if any(r["model"] == k for r in rows)]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 0.75 * len(keys) + 1.6), sharey=True)
    for ax, (stat, title, scale) in zip(axes, (("swap: share", 'Share of the answer\'s coin-aligned part via " up" (%)', 100),
                                               ("kappa (swap+ctrl)", "κ: answer shift per nat of coin swapped in", 1))):
        for qi, (q, label) in enumerate(QUESTIONS.items()):
            color = plotstyle.SERIES[qi]
            for yi, key in enumerate(keys):
                r = next((r for r in rows if r["model"] == key and r["report"] == q), None)
                if r is None:
                    continue
                y = yi + (qi - 1) * 0.22
                ax.plot([scale * r[stat + " lo"], scale * r[stat + " hi"]], [y, y], color=color, lw=2,
                        solid_capstyle="round")
                ax.plot(scale * r[stat], y, "o", ms=8, color=color, mec="white", mew=1.5,
                        label=label if yi == 0 else None)
        ax.axvline(0, color="0.6", lw=0.8)
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_yticks(range(len(keys)), [MODELS[k] for k in keys])
    axes[0].invert_yaxis()
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle('Finding 8: swapping the coin\'s " up" from the other task order (95% bootstrap CIs, 1,000 pairs)',
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    plotstyle.save(fig, path.stem, outdir=path.parent)


if __name__ == "__main__":
    main(sys.argv[1:] or list(MODELS))
