"""Bleedthrough along ../coinflip's DPO retraining runs 5 and 1, checkpoint by checkpoint.

The measures and outcomes were pre-registered in the README ("DPO retraining checkpoints")
before any data. Collects the W&B runs of group "retrain" (one per checkpoint, see
configs/retrain/), and for each run and step t, with A_t and C_t the Assistant and coinflip
utilities and "benign" the non-harmful tasks (analyze.HARM_TOPICS / HARM_ORIGINS left out):

- r, r_benign: r(A_t, C_t) over all tasks and over benign ones;
- r_change, r_change_benign: r(A_t, C_t - C_0), the coinflip change since the starting SFT
  (step 0), which takes out the SFT's own coinflip structure (undefined at step 0);
- b: slope of C_t - C_0 on standardized A_t over benign tasks, in nats of coinflip change
  per Assistant SD: the amount of Assistant-aligned change, which does not shrink as the
  Assistant scale grows during DPO;
- harm_gap: mean coinflip utility, benign minus harmful (nats), the contrast the
  ../coinflip task pairs measure;
- r_benign_fixed: r_benign with step 600's Assistant utilities in place of A_t;
- assistant_sd, coinflip_sd, and for run 5 r_readings: r between the strip_think (primary)
  and close_think Assistant readings.

CIs are 95% task bootstraps within origin. Differences between steps (step 400 -> 600 for
the outcome, also 0 -> 300 and 300 -> 400) use the same resamples for both steps, so they
are paired. ../coinflip's own Assistant-framing change (its RESULTS.md) is shown alongside.
Not pre-registered, printed after the outcome: what the step 400 -> 600 coinflip change lines
up with (late_change), and b within each origin (origin_slopes).

Usage: uv run python retrain.py [--group retrain]
"""

import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import N_BOOTSTRAP, pearson
from report import INK, INK_2, RULE, SERIES, errorbar, fetch_wandb, origin_scatter, stratified_resamples, style, task_arrays

PROJECT = "halrockwell-mats/tournament"
RUNS = {"run1": "Run 1: Instruct SFT", "run5": "Run 5: Think SFT"}
COLOR = {"run1": SERIES[0], "run5": SERIES[1]}
MARKER = {"run1": "o", "run5": "s"}
STATS = ("r", "r_benign", "r_change", "r_change_benign", "b", "harm_gap", "r_benign_fixed",
         "assistant_sd", "coinflip_sd")
FIXED_STEP = 600
OUTCOME_STEPS = (400, 600)
DIFF_STEPS = ((400, 600), (0, 300), (300, 400))
# ../coinflip RESULTS.md, "Run 5" and "32B Instruct DPO replication (run 1)": Assistant
# framing, no system prompt, change from step 0 over its 25 task pairs (est, CI low, CI high;
# NaN where no CI is given).
NAN = float("nan")
COINFLIP_CHANGE = {
    "run5": {0: (0, 0, 0), 50: (-0.00, -0.03, 0.02), 100: (0.00, -0.02, 0.03), 160: (0.06, 0.03, 0.09),
             200: (0.13, 0.09, 0.16), 300: (0.34, 0.27, 0.41), 400: (0.44, 0.35, 0.54), 500: (0.43, 0.34, 0.52),
             600: (0.38, 0.29, 0.48)},
    "run1": {0: (0, 0, 0), 100: (0.01, -0.02, 0.03), 160: (0.08, 0.05, 0.11), 200: (0.18, 0.13, 0.22),
             300: (0.39, NAN, NAN), 400: (0.78, 0.58, 0.99), 600: (1.27, 0.95, 1.59)},
}


def checkpoint_stats(a, c, c0, a_fixed, benign):
    """The STATS of one checkpoint on one set of tasks (arrays over the same tasks)."""
    bm = benign
    dc = c - c0
    moved = bool(np.ptp(dc) > 0)  # at step 0, C_t - C_0 is all zeros
    ab = a[bm]
    b = float(np.mean((dc[bm] - dc[bm].mean()) * (ab - ab.mean())) / ab.std())
    return np.array([
        pearson(a, c), pearson(a[bm], c[bm]),
        pearson(a, dc) if moved else np.nan, pearson(ab, dc[bm]) if moved else np.nan,
        b, c[bm].mean() - c[~bm].mean(), pearson(a_fixed[bm], c[bm]), a.std(), c.std(),
    ])


def run_stats(U, run, steps, benign, resamples):
    """{step: (estimate, resampled)} of checkpoint_stats per step of one run; resampled is
    (n_resamples, len(STATS)), resample k using the same task indices at every step."""
    c0, a_fixed = U[run][0]["coinflip"], U[run][FIXED_STEP]["assistant"]
    out = {}
    for t in steps:
        a, c = U[run][t]["assistant"], U[run][t]["coinflip"]
        est = checkpoint_stats(a, c, c0, a_fixed, benign)
        boot = np.array([checkpoint_stats(a[i], c[i], c0[i], a_fixed[i], benign[i]) for i in resamples])
        out[t] = (est, boot)
    return out


def ci(boot):
    return np.nanpercentile(boot, 2.5, axis=0), np.nanpercentile(boot, 97.5, axis=0)


def diff(per_step, s, t):
    """(estimate, ci_low, ci_high) arrays over STATS of step t minus step s, paired."""
    (es, bs), (et, bt) = per_step[s], per_step[t]
    lo, hi = ci(bt - bs)
    return et - es, lo, hi


def classify(per_step):
    """The pre-registered outcome for one run (README): A, B, C or "mixed"."""
    k = {name: j for j, name in enumerate(STATS)}
    est, lo, hi = diff(per_step, *OUTCOME_STEPS)
    rises = [lo[k[m]] > 0 for m in ("r_benign", "b")]
    falls = [hi[k[m]] < 0 for m in ("r_benign", "b")]
    never = all(ci(per_step[t][1])[0][k["r_change_benign"]] <= 0 for t in (400, 500, 600) if t in per_step)
    if never:
        return "C"
    if any(rises):
        return "mixed" if any(falls) else "A"
    return "B"


def late_change(U, run, benign, resamples, s=400, t=600):
    """What the coinflip change from step s to t lines up with, over benign tasks: step t's
    Assistant utilities, the earlier change C_s - C_0, and the starting coinflip C_0.
    {name: (estimate, ci_low, ci_high)}."""
    a, cs, ct, c0 = U[run][t]["assistant"], U[run][s]["coinflip"], U[run][t]["coinflip"], U[run][0]["coinflip"]

    def stats(i):
        b = benign[i]
        late = (ct - cs)[i][b]
        return np.array([pearson(a[i][b], late), pearson((cs - c0)[i][b], late), pearson(c0[i][b], late),
                         late.std()])

    est = stats(np.arange(len(a)))
    lo, hi = ci(np.array([stats(i) for i in resamples]))
    return {name: (est[j], lo[j], hi[j]) for j, name in enumerate(("r_assistant", "r_earlier_change", "r_start",
                                                                   "sd"))}


def origin_slopes(U, run, origins, benign, steps=(300, 400, 600)):
    """{origin: {step: b within that origin}} over benign origins."""
    c0 = U[run][0]["coinflip"]
    out = {}
    for g in dict.fromkeys(origins[benign]):
        m = (origins == g) & benign
        out[g] = {t: checkpoint_stats(U[run][t]["assistant"][m], U[run][t]["coinflip"][m], c0[m],
                                      U[run][FIXED_STEP]["assistant"][m], np.ones(m.sum(), bool))[STATS.index("b")]
                  for t in steps}
    return out


def load(group):
    """Utilities of every finished run in the W&B group: ({run: {step: {measure: array}}},
    summary {(run, step): {key: value}}, task rows of the first run)."""
    import wandb
    U, summaries, rows0 = {}, {}, None
    for run in wandb.Api().runs(PROJECT, filters={"group": group}):
        if run.state != "finished":
            print(f"skipping {run.name} ({run.id}): {run.state}")
            continue
        name, step = run.name.split("-step")
        step = int(step)
        analysis = Path("outputs") / run.id / "analysis"
        fetch_wandb(run, analysis)
        with open(analysis / "utilities.csv", newline="") as fh:
            rows = list(csv.DictReader(fh))
        if rows0 is None:
            rows0 = rows
        assert [r["id"] for r in rows] == [r["id"] for r in rows0], f"{run.name}: different task set"
        u = task_arrays(rows)[0]
        stage = f"step{step:05d}"
        U.setdefault(name, {})[step] = {"assistant": u[f"{stage}/assistant"], "coinflip": u[f"{stage}/coinflip"],
                                        **({"assistant_close": u[f"{stage}-close/assistant"]}
                                           if f"{stage}-close/assistant" in u else {})}
        summary = json.loads((analysis / "summary.json").read_text())
        summaries[(name, step)] = {**{k.replace(f"{stage}/", ""): v for k, v in summary.items()
                                      if k.startswith(f"{stage}/")},
                                   "run_id": run.id}
    return U, summaries, rows0


def plot_trajectories(table, per_run):
    panels = (("r", "r(Assistant, coinflip), all tasks"),
              ("r_benign", "r(Assistant, coinflip), non-harmful"),
              ("r_change_benign", "r(Assistant, coinflip change), non-harmful"),
              ("b", "Assistant-aligned coinflip change,\nnon-harmful (nats per Assistant SD)"),
              ("harm_gap", "Coinflip harm gap,\nnon-harmful - harmful (nats)"),
              ("coinflip_change", "../coinflip Assistant change\n(25 task pairs, nats)"))
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6), sharex=True)
    for ax, (key, title) in zip(axes.flat, panels):
        ax.axvspan(*OUTCOME_STEPS, color="#f0efec", zorder=0)
        ends = []
        for k, run in enumerate(r for r in RUNS if r in per_run):
            if key == "coinflip_change":
                pts = sorted(COINFLIP_CHANGE[run].items())
                x = np.array([s for s, _ in pts], float)
                est, lo, hi = (np.array([v[j] for _, v in pts]) for j in range(3))
                lo, hi = np.where(np.isnan(lo), est, lo), np.where(np.isnan(hi), est, hi)
            else:
                rows = [r for r in table if r["run"] == run and not np.isnan(r[key])]
                x = np.array([r["step"] for r in rows], float)
                est, lo, hi = (np.array([r[f"{key}{s}"] for r in rows]) for s in ("", "_ci_low", "_ci_high"))
            errorbar(ax, x + (k - 0.5) * 6, est, lo, hi, COLOR[run], MARKER[run], RUNS[run])
            ends.append([x[-1] + 18, est[-1], run.replace("run", "run ")])
        # Direct labels at the line ends, pulled apart where the two lines end close together.
        y0, y1 = ax.get_ylim()
        if len(ends) == 2 and abs(ends[0][1] - ends[1][1]) < 0.06 * (y1 - y0):
            mid, up = (ends[0][1] + ends[1][1]) / 2, int(ends[1][1] > ends[0][1])
            ends[up][1], ends[1 - up][1] = mid + 0.03 * (y1 - y0), mid - 0.03 * (y1 - y0)
        for x_end, y_end, label in ends:
            ax.text(x_end, y_end, label, color=INK_2, fontsize=8, va="center")
        ax.axhline(0, color=RULE, lw=0.8, zorder=1)
        ax.set_title(title, fontsize=10, color=INK)
        ax.set_xlim(-25, 680)
        style(ax)
    for ax in axes[1]:
        ax.set_xlabel("DPO step (same step = same dose in both runs)")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Bleedthrough along the DPO retraining runs (95% CI over tasks; shaded: the pre-registered "
                 "step 400-600 comparison)", fontsize=11)
    fig.tight_layout()
    return fig


def plot_scatter(U, origins, benign, steps=(300, 600)):
    runs = [r for r in RUNS if r in U]
    fig, axes = plt.subplots(len(runs), len(steps), figsize=(4.4 * len(steps), 4.0 * len(runs)), squeeze=False)
    for row, run in zip(axes, runs):
        for ax, t in zip(row, steps):
            a = U[run][t]["assistant"]
            dc = U[run][t]["coinflip"] - U[run][0]["coinflip"]
            origin_scatter(ax, a[benign], dc[benign], origins[benign], fit_lines=True)
            ax.set_title(f"{RUNS[run]}, step {t}\nr = {pearson(a[benign], dc[benign]):.2f} (non-harmful tasks)",
                         fontsize=10, color=INK)
            ax.set_xlabel(f"Assistant utility at step {t} (log-odds)")
            ax.set_ylabel("coinflip change since step 0 (log-odds)")
    axes[0, -1].legend(frameon=False, fontsize=8)
    fig.suptitle("Coinflip change since the starting SFT vs. Assistant utility, non-harmful tasks "
                 "(axes differ between panels)", fontsize=11)
    fig.tight_layout()
    return fig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--group", default="retrain")
    args = p.parse_args()

    U, summaries, rows = load(args.group)
    _, origins, benign, _ = task_arrays(rows)
    resamples = stratified_resamples(origins, np.random.default_rng(0), N_BOOTSTRAP)
    per_run = {run: run_stats(U, run, sorted(U[run]), benign, resamples) for run in RUNS if run in U}

    table = []
    for run, per_step in per_run.items():
        for t, (est, boot) in per_step.items():
            lo, hi = ci(boot)
            s = summaries[(run, t)]
            row = dict(run=run, step=t, run_id=s["run_id"])
            for j, name in enumerate(STATS):
                row.update({name: float(est[j]), f"{name}_ci_low": float(lo[j]), f"{name}_ci_high": float(hi[j])})
            row.update({f"{cond}_{k}": s.get(f"{cond}/{k}", NAN) for cond in ("assistant", "coinflip")
                        for k in ("reliability", "outcome_mass_mean")})
            close = U[run][t].get("assistant_close")
            row["r_readings"] = pearson(U[run][t]["assistant"], close) if close is not None else NAN
            row["r_assistant_vs_fixed"] = pearson(U[run][t]["assistant"], U[run][FIXED_STEP]["assistant"])
            table.append(row)
    diffs = []
    for run, per_step in per_run.items():
        for s, t in DIFF_STEPS:
            if s in per_step and t in per_step:
                est, lo, hi = diff(per_step, s, t)
                diffs.append(dict(run=run, steps=f"{s}->{t}", **{
                    f"{name}{suffix}": float(v[j]) for j, name in enumerate(STATS)
                    for suffix, v in (("", est), ("_ci_low", lo), ("_ci_high", hi))}))

    out_dir = Path("outputs") / "retrain"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows_ in (("retrain", table), ("diffs", diffs)):
        with open(out_dir / f"{name}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_[0]))
            w.writeheader()
            w.writerows(rows_)
    figures = {"trajectory": plot_trajectories(table, per_run), "scatter": plot_scatter(U, origins, benign)}
    for name, fig in figures.items():
        fig.savefig(out_dir / f"{name}.png", dpi=150, bbox_inches="tight")
        shutil.copy(out_dir / f"{name}.png", Path("figures") / f"retrain_{name}.png")

    def fmt(r, k, prec=2):
        return f"{r[k]:+.{prec}f} [{r[f'{k}_ci_low']:+.{prec}f}, {r[f'{k}_ci_high']:+.{prec}f}]"

    print(f"{'run':5} {'step':>5} {'r':>22} {'r benign':>22} {'r change benign':>22} {'b':>22} {'harm gap':>22} "
          f"{'r benign fixed A':>17} {'asst sd':>8} {'coin sd':>8} {'readings r':>10} {'A vs A600':>9} "
          f"{'rel a/c':>11} {'mass a/c':>9}")
    for r in table:
        print(f"{r['run']:5} {r['step']:5} {fmt(r, 'r'):>22} {fmt(r, 'r_benign'):>22} {fmt(r, 'r_change_benign'):>22} "
              f"{fmt(r, 'b', 3):>22} {fmt(r, 'harm_gap'):>22} {r['r_benign_fixed']:17.2f} {r['assistant_sd']:8.2f} "
              f"{r['coinflip_sd']:8.3f} {r['r_readings']:10.3f} {r['r_assistant_vs_fixed']:9.3f} "
              f"{r['assistant_reliability']:5.3f}/{r['coinflip_reliability']:5.3f} "
              f"{r['assistant_outcome_mass_mean']:4.2f}/{r['coinflip_outcome_mass_mean']:4.2f}")
    print("\nPaired differences between steps:")
    for d in diffs:
        print(f"{d['run']:5} {d['steps']:>8}  r_benign {fmt(d, 'r_benign')}  b {fmt(d, 'b', 3)}  "
              f"r_change_benign {fmt(d, 'r_change_benign')}  harm gap {fmt(d, 'harm_gap')}  r {fmt(d, 'r')}")
    print("\nPre-registered outcome (step 400 -> 600):")
    for run, per_step in per_run.items():
        if all(t in per_step for t in OUTCOME_STEPS):
            print(f"  {run}: {classify(per_step)}")
    print("\nNot pre-registered. The coinflip change from step 400 to 600 (non-harmful tasks): its r with step 600's "
          "Assistant, with the earlier change (C_400 - C_0) and with the starting coinflip C_0, and its SD (nats):")
    for run in per_run:
        lc = late_change(U, run, benign, resamples)
        print(f"  {run}: " + "  ".join(f"{k} {v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]" for k, v in lc.items()))
    print("\nNot pre-registered. b within each origin (non-harmful tasks), steps 300 / 400 / 600:")
    for run in per_run:
        print(f"  {run}: " + " | ".join(f"{g.lower()} " + " / ".join(f"{v:+.3f}" for v in by_step.values())
                                       for g, by_step in origin_slopes(U, run, origins, benign).items()))
    print(f"\nWrote {out_dir}/retrain.csv, diffs.csv and figures (copied to figures/retrain_*.png)")


if __name__ == "__main__":
    main()
