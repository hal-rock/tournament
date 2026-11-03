"""Slide figures: a ladder's coinflip at each post-training stage vs its final model's Assistant utilities.

The style of figures/scatter_vs_instruct.png (figures.py), one figure per ladder, both fit on the sweep's
15,000 pairs. That figure is OLMo 3.1 32B on all 44,850 pairs; figures/scatter_vs_instruct_olmo3.png is the
same ladder on the 15,000, to compare with OLMo 2's like for like. The x-axis is the final model's Assistant
in every panel, so these r differ from the README's tables, which pair each stage with its own Assistant.

  figures/scatter_vs_instruct_olmo2.png
  figures/scatter_vs_instruct_olmo3.png

Needs outputs/olmo2/analysis (olmo2.py) and outputs/d8ysbybo/analysis-15000pairs-stages (olmo2.py --olmo3-only).

Usage: uv run python figures_stages.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze import pearson
from figures import INK, MUTED, OUT, STAGES, fit_line, load_utilities, scatter_points, style_axes

LADDERS = {"olmo2": ("OLMo 2 32B", "outputs/olmo2/analysis/utilities.csv"),
           "olmo3": ("OLMo 3.1 32B", "outputs/d8ysbybo/analysis-15000pairs-stages/utilities.csv")}


def scatter_vs_final(U, origins, benign, model, path):
    x = U["instruct/assistant"]
    fig, axes = plt.subplots(1, len(STAGES), figsize=(14, 3.9), dpi=200, sharex=True)
    for ax, (stage, label) in zip(axes, STAGES.items()):
        y = U[f"{stage}/coinflip"]
        scatter_points(ax, x, y, origins)
        fit_line(ax, x, y, INK)
        r, rb = pearson(x, y), pearson(x[benign], y[benign])
        ax.set_title(f"{label} coinflip", loc="left", fontsize=11, color=INK, pad=20)
        ax.text(0, 1.03, f"r = {r:.2f}   (benign only: {rb:.2f})", transform=ax.transAxes, fontsize=9.5, color=INK)
        ax.set_xlabel("Instruct Assistant utility", color=MUTED, fontsize=9)
        style_axes(ax)
        print(f"{model:14} vs instruct/assistant  {stage:9} r = {r:.3f}  benign r = {rb:.3f}  y range {y.min():+.2f} to {y.max():+.2f}")
    axes[0].set_ylabel("Coinflip utility (log-odds)", color=MUTED, fontsize=9)
    axes[-1].legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(1.02, 1), handletextpad=0.2)
    fig.suptitle(f"{model}: coinflip utilities at each stage vs the final Instruct model's Assistant utilities "
                 f"({len(x)} tasks; y-axes differ between panels; benign = {benign.sum()} tasks)",
                 x=0.01, ha="left", fontsize=10, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def main():
    for key, (model, utilities) in LADDERS.items():
        scatter_vs_final(*load_utilities(utilities), model, OUT / f"scatter_vs_instruct_{key}.png")


if __name__ == "__main__":
    main()
