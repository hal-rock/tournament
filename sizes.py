"""OLMo 3 7B vs. 32B along the Instruct post-training ladder (base, SFT, DPO, Instruct).

Combines two stages runs scored on the same 300 tasks and all pairs: the 32B's d8ysbybo
(configs/olmo3-32b-stages-n300.yaml) and the 7B's (configs/olmo3-7b-stages-n300.yaml).
Measures are prefixed by size ("7b:sft/coinflip"). Per size and stage, think.py's
comparison table: r of coinflip with Assistant utilities (all tasks, non-harmful only,
log length partialled out), within-origin slope, utility SDs, benign-minus-harmful gaps,
and r_shift (stage - base, in both conditions). Also the 7B's coinflip against the 32B's
Assistant utilities at the same stage (does the 7B coinflip track the preferences the 32B
Assistant has, if not its own?), the correlation of each measure across sizes, and mean
utilities by origin.

Usage: uv run python sizes.py [run_7b] [run_32b]
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import N_BOOTSTRAP, ORIGIN_STYLE, pearson
from report import INK, RULE, SERIES, errorbar, fetch_wandb, stratified_resamples, style, task_arrays
from think import comparison_rows

RUNS = {"7b": "halrockwell-mats/tournament/st9cz0uw", "32b": "halrockwell-mats/tournament/d8ysbybo"}
STAGES = ("base", "sft", "dpo", "instruct")
CONDITIONS = ("assistant", "coinflip")


def load(path):
    """(utilities.csv rows, summary) of a W&B run, fetched to outputs/<id>/analysis once."""
    import wandb
    run_id = path.split("/")[-1]
    out = Path("outputs") / run_id / "analysis"
    if not (out / "utilities.csv").exists():
        fetch_wandb(wandb.Api().run(path), out)
    with open(out / "utilities.csv", newline="") as fh:
        return list(csv.DictReader(fh)), json.loads((out / "summary.json").read_text())


def comparisons():
    out = [(f"{size} {s}", f"{size}:{s}/assistant", f"{size}:{s}/coinflip") for size in RUNS for s in STAGES]
    return out + [(f"7b {s} coinflip vs 32b Assistant", f"32b:{s}/assistant", f"7b:{s}/coinflip")
                  for s in STAGES if s != "base"]


def plot_by_stage(table):
    by = {r["comparison"]: r for r in table}
    panels = (("r", "r(Assistant, coinflip)"), ("r_shift", "r of shifts from base"),
              ("coinflip_harm_gap", "coinflip benign - harmful (nats)"),
              ("assistant_harm_gap", "Assistant benign - harmful (nats)"))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4))
    x = np.arange(len(STAGES))
    for ax, (key, title) in zip(axes, panels):
        for k, (size, color, marker) in enumerate(zip(RUNS, SERIES, "os")):
            rows = [by[f"{size} {s}"] for s in STAGES]
            est, lo, hi = (np.array([r[f"{key}{suf}"] for r in rows]) for suf in ("", "_ci_low", "_ci_high"))
            errorbar(ax, x + (k - 0.5) * 0.12, est, lo, hi, color, marker, f"OLMo 3 {size.upper()}")
        ax.axhline(0, color=RULE, lw=0.8, zorder=1)
        ax.set_xticks(x, STAGES)
        ax.set_title(title, fontsize=11, color=INK)
        style(ax)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("OLMo 3 7B vs. 32B by post-training stage (95% CI over tasks)", fontsize=11)
    fig.tight_layout()
    return fig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_7b", nargs="?", default=RUNS["7b"])
    p.add_argument("run_32b", nargs="?", default=RUNS["32b"])
    args = p.parse_args()

    U, summary, rows0 = {}, {}, None
    for size, path in (("7b", args.run_7b), ("32b", args.run_32b)):
        rows, s = load(path)
        if rows0 is None:
            rows0 = rows
        assert [r["id"] for r in rows] == [r["id"] for r in rows0], "runs differ in their task sets"
        U.update({f"{size}:{m}": u for m, u in task_arrays(rows)[0].items() if "-" not in m.split("/")[0]})
        summary.update({f"{size}:{k}": v for k, v in s.items()})
    _, origins, benign, loglen = task_arrays(rows0)
    resamples = stratified_resamples(origins, np.random.default_rng(0), N_BOOTSTRAP)
    table = comparison_rows(U, origins, benign, loglen, summary, resamples, comparisons())
    cross = [dict(stage=s, condition=c, r_7b_32b=pearson(U[f"7b:{s}/{c}"], U[f"32b:{s}/{c}"]))
             for s in STAGES for c in CONDITIONS]
    by_origin = [dict(origin=g, n=int((origins == g).sum()),
                      **{f"{size}:{s}/{c}": float(U[f"{size}:{s}/{c}"][origins == g].mean())
                         for c in CONDITIONS for size in RUNS for s in STAGES})
                 for g in ORIGIN_STYLE if g in set(origins)]

    out_dir = Path("outputs") / "sizes"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("sizes", table), ("cross_size", cross), ("by_origin", by_origin)):
        with open(out_dir / f"{name}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    plot_by_stage(table).savefig(out_dir / "by_stage.png", dpi=150, bbox_inches="tight")

    print(f"{'comparison':32} {'r':>6} {'95% CI':>14} {'r benign':>9} {'r|len':>6} {'r shift':>18} {'slope':>7} "
          f"{'asst sd':>8} {'coin sd':>8} {'asst gap':>9} {'coin gap':>21} {'coin mass':>10} {'coin rel':>9}")
    for r in table:
        print(f"{r['comparison']:32} {r['r']:6.2f} [{r['r_ci_low']:5.2f}, {r['r_ci_high']:5.2f}] {r['r_benign']:9.2f} "
              f"{r['r_partial_length']:6.2f} {r['r_shift']:5.2f} [{r['r_shift_ci_low']:5.2f}, {r['r_shift_ci_high']:5.2f}] "
              f"{r['slope_within']:7.3f} {r['assistant_utility_sd']:8.2f} {r['coinflip_utility_sd']:8.3f} "
              f"{r['assistant_harm_gap']:9.2f} {r['coinflip_harm_gap']:6.2f} [{r['coinflip_harm_gap_ci_low']:5.2f}, "
              f"{r['coinflip_harm_gap_ci_high']:5.2f}] {r['coinflip_outcome_mass_mean']:10.2f} "
              f"{r['coinflip_reliability']:9.3f}")
    print("\nr(7B, 32B) by stage and condition:")
    for r in cross:
        print(f"  {r['stage']:9} {r['condition']:10} {r['r_7b_32b']:6.2f}")
    for c in CONDITIONS:
        cols = [f"{size}:{s}/{c}" for size in RUNS for s in STAGES]
        print(f"\nMean {c} utility by origin:\n{'origin':12} {'n':>4}" + "".join(f"{m:>16}" for m in cols))
        for r in by_origin:
            print(f"{r['origin']:12} {r['n']:4}" + "".join(f"{r[m]:16.2f}" for m in cols))
    print(f"\nWrote {out_dir}/sizes.csv, cross_size.csv, by_origin.csv and by_stage.png")


if __name__ == "__main__":
    main()
