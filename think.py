"""Bleedthrough in OLMo 3.1 32B Think next to the 3.1 Instruct, on the same tasks and pairs.

Combines the Think run (configs/olmo3-32b-think.yaml) with the OLMo 3 32B stages run
d8ysbybo's base and Instruct stages, refit on the pairs the Think run scored
(models.olmo_on_shared_pairs). The Think model's Assistant choice is read two ways
(see the config): after an empty think block ("think") and as the first words of its
reasoning ("think-inreasoning"); its coinflip is read once.

One row per (Assistant utilities, coinflip utilities) comparison, with the models.py
statistics: r (all tasks, non-harmful only, log length partialled out), within-origin
slope, utility SDs, benign-minus-harmful gaps, and r_shift, the correlation of the
post-training shifts (model - base) in the two conditions. Also: the correlation matrix
of all the utility scales, and per-origin mean utilities.

Usage: uv run python think.py [entity/project/run_id] [--update-wandb]
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
from models import CI_STATS, olmo_on_shared_pairs
from report import (DIVERGING, INK, fetch_wandb, origin_scatter, stage_analysis, stratified_resamples, task_arrays,
                    with_ci)

THINK_RUN = "halrockwell-mats/tournament/z6ldqzq2"
# (label, Assistant measure, coinflip measure). Measures of the Think run carry its stage
# names; base/ and instruct/ are d8ysbybo's.
COMPARISONS = (
    ("base", "base/assistant", "base/coinflip"),
    ("instruct", "instruct/assistant", "instruct/coinflip"),
    ("think", "think/assistant", "think/coinflip"),
    ("think, Assistant in reasoning", "think-inreasoning/assistant", "think/coinflip"),
    ("think coinflip vs instruct Assistant", "instruct/assistant", "think/coinflip"),
)
MEASURES = ("base/assistant", "instruct/assistant", "think/assistant", "think-inreasoning/assistant",
            "base/coinflip", "instruct/coinflip", "think/coinflip")


def shift(U, measure):
    """measure minus the base model's utilities in the same condition (None for base)."""
    stage, cond = measure.split("/")
    return None if stage == "base" else U[measure] - U[f"base/{cond}"]


def comparison_rows(U, origins, benign, loglen, summary, resamples):
    groups = [g for g in ORIGIN_STYLE if g in set(origins)]
    out = []
    for label, am, cm in COMPARISONS:
        a, c = U[am], U[cm]
        stats = stage_analysis(a, c, loglen, origins, benign, groups, resamples)
        row = dict(comparison=label, assistant=am, coinflip=cm,
                   assistant_utility_sd=float(a.std()), coinflip_utility_sd=float(c.std()),
                   assistant_reliability=summary.get(f"{am}/reliability", float("nan")),
                   coinflip_reliability=summary.get(f"{cm}/reliability", float("nan")),
                   assistant_outcome_mass_mean=summary.get(f"{am}/outcome_mass_mean", float("nan")),
                   coinflip_outcome_mass_mean=summary.get(f"{cm}/outcome_mass_mean", float("nan")))
        for k in CI_STATS:
            row.update({k: float(stats[k][0]), f"{k}_ci_low": float(stats[k][1]), f"{k}_ci_high": float(stats[k][2])})
        gap = with_ci(lambda i: np.array([u[i][benign[i]].mean() - u[i][~benign[i]].mean() for u in (a, c)]),
                      len(a), resamples)
        for j, cond in enumerate(("assistant", "coinflip")):
            row.update({f"{cond}_harm_gap": float(gap[0][j]), f"{cond}_harm_gap_ci_low": float(gap[1][j]),
                        f"{cond}_harm_gap_ci_high": float(gap[2][j])})
        sa, sc = shift(U, am), shift(U, cm)
        r_shift = (with_ci(lambda i: np.array([pearson(sa[i], sc[i])]), len(a), resamples)
                   if sa is not None and sc is not None else [np.array([np.nan])] * 3)
        row.update(r_shift=float(r_shift[0][0]), r_shift_ci_low=float(r_shift[1][0]),
                   r_shift_ci_high=float(r_shift[2][0]))
        out.append(row)
    return out


def plot_scatter(U, origins, table):
    shown = [r for r in table if r["comparison"] != "base"]
    fig, axes = plt.subplots(1, len(shown), figsize=(4.2 * len(shown), 4.4), squeeze=False)
    for ax, r in zip(axes[0], shown):
        origin_scatter(ax, U[r["assistant"]], U[r["coinflip"]], origins, fit_lines=True)
        ax.set_title(f"{r['comparison']}\nr = {r['r']:.2f} [{r['r_ci_low']:.2f}, {r['r_ci_high']:.2f}]",
                     fontsize=10, color=INK)
        ax.set_xlabel(f"{r['assistant']} utility")
        ax.set_ylabel(f"{r['coinflip']} utility")
    axes[0, -1].legend(frameon=False, fontsize=8)
    fig.suptitle("OLMo 3.1 32B: per-task utilities (log-odds); lines are fits within origin "
                 "(axes differ between panels)", fontsize=11)
    fig.tight_layout()
    return fig


def plot_matrix(M):
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    im = ax.imshow(M, cmap=DIVERGING, vmin=-1, vmax=1)
    for i, j in np.ndindex(M.shape):
        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9,
                color="white" if abs(M[i, j]) > 0.6 else INK)
    ax.set_xticks(range(len(MEASURES)), MEASURES, rotation=40, ha="right")
    ax.set_yticks(range(len(MEASURES)), MEASURES)
    ax.set_title("Pearson r across tasks", fontsize=11)
    for side in ax.spines.values():
        side.set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", nargs="?", default=THINK_RUN, help="W&B path of the Think run")
    p.add_argument("--update-wandb", action="store_true", help="log the tables and figures to the Think run")
    args = p.parse_args()

    import wandb
    run = wandb.Api().run(args.run)
    analysis = Path("outputs") / run.id / "analysis"
    fetch_wandb(run, analysis)
    with open(analysis / "utilities.csv", newline="") as fh:
        think_rows = list(csv.DictReader(fh))
    summary = json.loads((analysis / "summary.json").read_text())
    olmo_rows, olmo_summary = olmo_on_shared_pairs(run.config)
    assert [r["id"] for r in think_rows] == [r["id"] for r in olmo_rows], "runs differ in their task sets"
    summary.update({k: v for k, v in olmo_summary.items() if k.split("/")[0] in ("base", "instruct")})

    U, origins, benign, loglen = task_arrays(olmo_rows)
    U.update(task_arrays(think_rows)[0])
    resamples = stratified_resamples(origins, np.random.default_rng(0), N_BOOTSTRAP)
    table = comparison_rows(U, origins, benign, loglen, summary, resamples)
    M = np.corrcoef([U[m] for m in MEASURES])
    by_origin = [dict(origin=g, n=int((origins == g).sum()),
                      **{m: float(U[m][origins == g].mean()) for m in MEASURES})
                 for g in ORIGIN_STYLE if g in set(origins)]

    out_dir = Path("outputs") / "think"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("think", table), ("by_origin", by_origin)):
        with open(out_dir / f"{name}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    with open(out_dir / "correlations.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["measure", *MEASURES])
        w.writerows([[m, *(round(v, 4) for v in row)] for m, row in zip(MEASURES, M)])
    figures = {"scatter": plot_scatter(U, origins, table), "correlations": plot_matrix(M)}
    for name, fig in figures.items():
        fig.savefig(out_dir / f"{name}.png", dpi=150, bbox_inches="tight")

    print(f"{'comparison':38} {'r':>6} {'95% CI':>14} {'r benign':>9} {'r|len':>6} {'r shift':>16} {'slope':>7} "
          f"{'asst sd':>8} {'coin sd':>8} {'asst gap':>9} {'coin gap':>20} {'coin mass':>10}")
    for r in table:
        print(f"{r['comparison']:38} {r['r']:6.2f} [{r['r_ci_low']:5.2f}, {r['r_ci_high']:5.2f}] {r['r_benign']:9.2f} "
              f"{r['r_partial_length']:6.2f} {r['r_shift']:5.2f} [{r['r_shift_ci_low']:4.2f}, {r['r_shift_ci_high']:4.2f}] "
              f"{r['slope_within']:7.3f} {r['assistant_utility_sd']:8.2f} {r['coinflip_utility_sd']:8.3f} "
              f"{r['assistant_harm_gap']:9.2f} {r['coinflip_harm_gap']:6.2f} [{r['coinflip_harm_gap_ci_low']:5.2f}, "
              f"{r['coinflip_harm_gap_ci_high']:5.2f}] {r['coinflip_outcome_mass_mean']:10.2f}")
    print("\nReliability (split-half, Spearman-Brown) and mean outcome mass:")
    for m in MEASURES:
        print(f"  {m:30} {summary.get(f'{m}/reliability', float('nan')):6.3f} "
              f"{summary.get(f'{m}/outcome_mass_mean', float('nan')):6.2f}")
    print("\nPearson r across tasks:\n" + " " * 30 + " ".join(f"{m[:12]:>12}" for m in MEASURES))
    for m, row in zip(MEASURES, M):
        print(f"{m:30}" + " ".join(f"{v:12.2f}" for v in row))
    print("\nMean utility by origin:\n" + f"{'origin':12} {'n':>4}" + " ".join(f"{m[:12]:>12}" for m in MEASURES))
    for r in by_origin:
        print(f"{r['origin']:12} {r['n']:4}" + " ".join(f"{r[m]:12.2f}" for m in MEASURES))
    print(f"\nWrote {out_dir}/think.csv, by_origin.csv, correlations.csv and figures")

    if args.update_wandb:
        entity, project, run_id = args.run.split("/")
        wrun = wandb.init(project=project, entity=entity, id=run_id, resume="allow")
        wrun.log({"think/table": wandb.Table(columns=list(table[0]), data=[list(r.values()) for r in table]),
                  **{f"think/{name}": wandb.Image(str(out_dir / f"{name}.png")) for name in figures}})
        wrun.finish()


if __name__ == "__main__":
    main()
