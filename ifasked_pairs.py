"""Shallow instructions, exploratory (after the results; Hal, 2026-11-12): how often each pairwise choice goes to the shorter task.

For every scored pair (both orders), the order-averaged log-odds d = (lo(a, b) - lo(b, a)) / 2 says which task the Assistant
(or the coin) picks. This reads the raw pairs, not the fitted utilities, and depends on length only through which task is
shorter, so log or raw length give the same numbers. Per model and condition:
- the share of pairs (lengths not tied) whose pick is the shorter task, overall and by the length ratio (longer / shorter);
- r of the fitted utilities with raw length, next to log length (and Spearman, which is the same for both).

Conditions: the default Assistant and coin (the sweep's raw results) and the shorter / longer instructions, first wording (W&B
group "ifasked") or reworded (--variant ifchoose, group "ifchoose").
No CIs: pairs share tasks, so a binomial CI would be too narrow; read the differences against the main analysis's CIs.

  outputs/<variant>/pairs.csv          one row per model, condition and length-ratio bin
  figures/<variant>_pairs_length.png

Usage: uv run python ifasked_pairs.py [--variant ifchoose]
"""

import csv
import gzip
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import pearson, read_jsonl
from matskit import plotstyle
from openers import write_csv
from report import INK, INK_2, RULE, style
from user_choice import TASKS_FILE

# The sweep's run (default Assistant and coin) per model; the run holding the length conditions is found in the W&B group.
RUNS = {"llama-70b": ("Llama 3.3 70B", "tvlqj8a0"), "gemma4-31b": ("Gemma 4 31B", "jrl62npf"),
        "glm-4.5-air": ("GLM-4.5-Air", "i5vejs3b")}
CONDITIONS = {  # name: (which run, file stem; {v} is the variant)
    "A_0": ("sweep", "instruct__assistant"), "A_shorter": ("new", "instruct__assistant_{v}_shorter"),
    "A_longer": ("new", "instruct__assistant_{v}_longer"),
    "C_0": ("sweep", "instruct__coinflip"), "C_shorter": ("new", "instruct__coinflip_{v}_shorter"),
    "C_longer": ("new", "instruct__coinflip_{v}_longer")}


def length_run(variant, family):
    """The id of the variant's W&B run for this model that scored the length conditions."""
    import personas
    (run,) = [r for r in personas.collect_runs(variant)[family]
              if f"assistant_{variant}_shorter" in r.config["models"][0]["conditions"]]
    return run.id
BINS = [(1.0, 1.25), (1.25, 2.0), (2.0, 4.0), (4.0, np.inf)]
BIN_LABEL = ["< 1.25x", "1.25-2x", "2-4x", "> 4x"]


def raw_path(run_id, stem):
    path = Path("outputs") / run_id / "results" / f"{stem}.csv.gz"
    if not path.exists():
        import wandb
        from analyze import fetch_wandb
        fetch_wandb(wandb.Api().run(f"halrockwell-mats/tournament/{run_id}"), Path("outputs") / run_id)
    return path


def pair_picks(path):
    """{(a, b): d} with a < b and d > 0 meaning a is picked, from the two orders' log-odds (pairs with both orders)."""
    lo = {}
    with gzip.open(path, "rt", newline="") as fh:
        for r in csv.DictReader(fh):
            lo[r["first"], r["second"]] = float(r["log_odds"])
    return {(a, b): (lo[a, b] - lo[b, a]) / 2 for (a, b) in lo if a < b and (b, a) in lo}


def shorter_shares(picks, length):
    """(share of untied pairs picking the shorter task, per-bin shares, per-bin counts)."""
    d = np.array(list(picks.values()))
    la = np.array([length[a] for a, _ in picks])
    lb = np.array([length[b] for _, b in picks])
    keep = la != lb
    d, la, lb = d[keep], la[keep], lb[keep]
    picked_shorter = (d > 0) == (la < lb)
    ratio = np.maximum(la, lb) / np.minimum(la, lb)
    in_bin = [(ratio >= lo) & (ratio < hi) for lo, hi in BINS]
    return picked_shorter.mean(), [picked_shorter[m].mean() for m in in_bin], [int(m.sum()) for m in in_bin]


def spearman(x, y):
    rank = lambda v: np.argsort(np.argsort(v, kind="stable"), kind="stable").astype(float)
    return pearson(rank(x), rank(y))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="ifasked", choices=["ifasked", "ifchoose"])
    variant = ap.parse_args().variant
    plotstyle.setup()
    length = {t["id"]: len(t["prompt"].strip()) for t in read_jsonl(TASKS_FILE)}
    table, shares = [], {}
    for family, (label, sweep) in RUNS.items():
        new = length_run(variant, family)
        print(f"\n{label}: share of pairs whose pick is the shorter task (lengths not tied)")
        print(f"  {'':10} {'all':>6}  " + "  ".join(f"{b:>8}" for b in BIN_LABEL))
        for name, (which, stem) in CONDITIONS.items():
            picks = pair_picks(raw_path(sweep if which == "sweep" else new, stem.format(v=variant)))
            overall, by_bin, counts = shorter_shares(picks, length)
            shares[family, name] = (overall, by_bin)
            print(f"  {name:10} {overall:6.3f}  " + "  ".join(f"{s:8.3f}" for s in by_bin))
            table += [dict(model=label, condition=name, ratio_bin=b, n_pairs=n, share_shorter=s)
                      for b, s, n in zip(["all"] + BIN_LABEL, [overall] + by_bin, [sum(counts)] + counts)]
        print(f"  pairs per bin: " + "  ".join(f"{b} {n}" for b, n in zip(BIN_LABEL, counts)))
        # Utilities (fitted) against raw and log length.
        util = {}
        for path in (Path("outputs") / sweep / "analysis" / "utilities.csv",
                     Path("outputs") / variant / family / "analysis" / "utilities.csv"):
            with open(path, newline="") as fh:
                for r in csv.DictReader(fh):
                    util.setdefault(r["id"], {}).update(r)
        ids = sorted(util)
        n = np.array([length[i] for i in ids], float)
        print("  r of utilities with length: raw / log / Spearman")
        for name, col in (("A_0", "instruct/assistant"), ("A_shorter", f"instruct/assistant_{variant}_shorter"),
                          ("A_longer", f"instruct/assistant_{variant}_longer"), ("C_0", "instruct/coinflip"),
                          ("C_shorter", f"instruct/coinflip_{variant}_shorter"),
                          ("C_longer", f"instruct/coinflip_{variant}_longer")):
            u = np.array([float(util[i][col]) for i in ids])
            print(f"    {name:10} {pearson(u, n):+.2f} / {pearson(u, np.log(n)):+.2f} / {spearman(u, n):+.2f}")
    write_csv(Path("outputs") / variant / "pairs.csv", table)

    # Figure: per model, share picking the shorter task by length-ratio bin; Assistant (circles) and coin (diamonds).
    fig, axes = plt.subplots(1, len(RUNS), figsize=(4.8 * len(RUNS), 4.3), sharey=True, squeeze=False)
    styles = {"0": (INK_2, "default"), "shorter": (plotstyle.SERIES[0], '"shorter"'),
              "longer": (plotstyle.SERIES[1], '"longer"')}
    x = np.arange(len(BINS))
    for ax, (family, (label, _)) in zip(axes[0], RUNS.items()):
        for who, marker, ls in (("A", "o", "-"), ("C", "D", "--")):
            for cond, (color, name) in styles.items():
                ax.plot(x, shares[family, f"{who}_{cond}"][1], color=color, marker=marker, ls=ls, lw=1.6, ms=7,
                        mec="white", mew=0.8, label=f"{'Assistant' if who == 'A' else 'coin'}, {name}")
        ax.axhline(0.5, color=RULE, ls=":", lw=1)
        ax.set_xticks(x, BIN_LABEL)
        ax.set_ylim(0, 1)
        ax.set_xlabel("longer description / shorter description", color=INK_2, fontsize=9)
        ax.set_title(label, color=INK, fontsize=10)
        style(ax)
    axes[0, 0].set_ylabel("share of pairs picking the shorter task", color=INK_2, fontsize=9)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, frameon=False, fontsize=9, loc="lower center", ncol=3)
    fig.suptitle("How often each pairwise pick goes to the shorter task, by how different the lengths are\n"
                 "(solid circles: Assistant's choice; dashed diamonds: coin)", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    plotstyle.save(fig, f"{variant}_pairs_length", Path("figures"))
    print(f"\nWrote outputs/{variant}/pairs.csv and figures/{variant}_pairs_length.png")


if __name__ == "__main__":
    main()
