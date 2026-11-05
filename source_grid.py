"""Per-source scatter grids: one figure per model, task sources across the columns, one comparison per row.

Each panel is one row's comparison (x utilities against y utilities) within one task source, with its least-squares line and
r [95% bootstrap CI over the source's tasks]. Rows share a y axis, so a row reads as one measure across sources. This is the
figure form of the per-source correlation tables (Hal's preference, 2026-11-05): the scatter makes the r readable without
the regression-coefficient abstractions.

Usage from an analysis script:
    plot_source_grid([Row("System prompt coin\\nvs default Assistant", a0, c_sys, "Default Assistant (nats)",
                          "coin (nats)", color), ...], origins, "Llama 3.3 70B: ...", Path("figures/x.png"))
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import ORIGIN_STYLE, pearson
from matskit import plotstyle
from report import INK, INK_2, RULE, style

SOURCES = tuple(ORIGIN_STYLE)
SOURCE_LABEL = {"MATH": "MATH", "ALPACA": "Alpaca", "WILDCHAT": "WildChat", "STRESS_TEST": "STRESS-TEST",
                "BAILBENCH": "BailBench"}


@dataclass
class Row:
    label: str  # row label at the left, e.g. "System prompt coin\nvs default Assistant"
    x: np.ndarray  # utilities over all tasks, in the task order of `origins`
    y: np.ndarray
    xlabel: str
    ylabel: str
    color: str


def r_ci(x, y, n_boot=2000, seed=0):
    """r and its 95% bootstrap CI over the tasks given."""
    rng = np.random.default_rng(seed)
    boot = [pearson(x[i], y[i]) for i in (rng.choice(len(x), len(x)) for _ in range(n_boot))]
    return pearson(x, y), *np.percentile(boot, [2.5, 97.5])


def plot_source_grid(rows, origins, title, path, n_boot=2000):
    """One panel per (row, source). Returns {(row label, source): (r, lo, hi)}."""
    plotstyle.setup()
    fig, axes = plt.subplots(len(rows), len(SOURCES), figsize=(3.0 * len(SOURCES), 2.55 * len(rows)),
                             sharey="row", squeeze=False)
    stats = {}
    for i, row in enumerate(rows):
        for j, src in enumerate(SOURCES):
            ax = axes[i, j]
            m = origins == src
            x, y = row.x[m], row.y[m]
            ax.scatter(x, y, s=14, color=row.color, edgecolors="white", linewidths=0.5, zorder=3)
            xs = np.array([x.min(), x.max()])
            ax.plot(xs, np.polyval(np.polyfit(x, y, 1), xs), color=INK_2, lw=1.2, zorder=4)
            ax.axhline(0, color=RULE, lw=0.6, zorder=1)
            r, lo, hi = stats[row.label, src] = r_ci(x, y, n_boot)
            ax.set_title(f"r = {r:+.2f} [{lo:+.2f}, {hi:+.2f}]", fontsize=8.5, color=INK, pad=3)
            ax.tick_params(labelsize=7)
            ax.set_xlabel(row.xlabel, fontsize=7.5, color=INK_2, labelpad=1)
            if j == 0:
                ax.set_ylabel(f"{row.label}\n\n{row.ylabel}", fontsize=8.5, color=INK)
            if i == 0:
                ax.annotate(f"{SOURCE_LABEL[src]} (n = {int(m.sum())})", (0.5, 1.28), xycoords="axes fraction",
                            ha="center", fontsize=10, color=INK, fontweight="bold")
            style(ax)
    fig.suptitle(title, color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    path = Path(path)
    plotstyle.save(fig, path.stem, path.parent)
    return stats
