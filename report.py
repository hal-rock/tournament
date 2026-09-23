"""Focused report on how the Assistant's preferences bleed through into the coinflip.

Works from analyze.py's outputs (fitted utilities and per-condition fit diagnostics),
so nothing is re-scored or re-fit from raw prompts.

Per stage, bleedthrough over tasks is measured by the correlation of coinflip with
Assistant utilities (all tasks, benign only, log length partialled out) and by slopes
in coinflip log-odds per Assistant log-odd. The task-type view fits, per stage,

    coinflip_i = beta * assistant_i + gamma * loglen_i + alpha[origin_i] + noise

beta is the within-origin slope: how much coinflip utility moves per unit of Assistant
utility, comparing tasks of the same origin. alpha[g] is origin g's offset, centered to
a task-weighted mean of zero: how far the whole group sits above (+) or below (-) where
beta would put it given its mean Assistant utility and length. A pooled slope (no
origins, no length) below beta means differences between origins bleed through less
than differences within them. Residuals from the model rank the tasks that over- or
under-bleed relative to others of their origin.

The decomposition table regresses the focus stage's coinflip utilities on the previous
stage's coinflip and the focus stage's Assistant utilities: how much of the focus
coinflip is the earlier coinflip structure carried over (and rescaled) versus a new
Assistant-aligned part. It is given over all tasks and over benign ones.

The non-Assistant coinflip utility of a stage is its coinflip utility minus the pooled
least-squares fit on the same stage's Assistant utility (no origin or length terms, so
those effects stay in it, to be explained). Its variance is partitioned by adjusted R^2
over blocks of predictors (the previous stage's coinflip, origin, topic, log length),
each alone and added in that order; group means over origins and topics hold the fit
fixed and bootstrap within each group.

CIs are 95% percentile bootstraps over tasks, resampled within origin to match how the
task set was drawn (fixed source quotas). They hold the fitted utilities fixed, so they
leave out the utilities' own noise (small: see split-half reliability).

Usage: uv run python report.py outputs/<run_id>          (after analyze.py on that dir)
       uv run python report.py entity/project/<run_id> [--update-wandb]
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from analyze import BASELINE_STAGE, HARM_ORIGINS, HARM_TOPICS, N_BOOTSTRAP, ORIGIN_STYLE, partial_r, pearson

# The stage where bleedthrough appears (offsets figure, residuals) and the stage before it.
FOCUS_STAGE = "dpo"
PRE_STAGE = "sft"
N_EXTREME = 10  # tasks printed at each end of the residual ranking
BLEED_STATS = ("r", "r_benign", "r_partial_length", "slope_pooled", "slope_within", "length_coef")
DIAGNOSTICS = ("reliability", "additive_r2", "slot_bias_mean", "outcome_mass_mean", "outcome_mass_min")
# Dataviz reference palette: categorical slots 1-3, text inks, and the blue <-> red
# diverging pair around a gray midpoint.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
INK, INK_2, RULE = "#0b0b0b", "#52514e", "#cccccc"
DIVERGING = LinearSegmentedColormap.from_list("diverging", ["#e34948", "#f0efec", "#2a78d6"])


def task_arrays(rows):
    """utilities.csv rows -> ({measure: utilities}, origins, benign mask, log task length)."""
    U = {m: np.array([float(r[m]) for r in rows]) for m in rows[0] if "/" in m}
    origins = np.array([r["origin"] for r in rows])
    benign = np.array([r["topic"] not in HARM_TOPICS and r["origin"] not in HARM_ORIGINS for r in rows])
    loglen = np.log([float(r["n_chars"]) for r in rows])
    return U, origins, benign, loglen


def stages_of(measures):
    """Stages run in both conditions, in column (config) order, skipping post-training shifts."""
    return [m.split("/")[0] for m in measures
            if m.endswith("/assistant") and not m.split("/")[0].endswith(f"-{BASELINE_STAGE}")
            and m.replace("/assistant", "/coinflip") in measures]


def stratified_resamples(groups, rng, n):
    """n index arrays, each resampling tasks with replacement within every group."""
    members = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    return [np.concatenate([rng.choice(m, len(m)) for m in members]) for _ in range(n)]


def offset_model(a, c, loglen, origins, groups):
    """Least-squares c = beta*a + gamma*loglen + alpha[origin]. Returns beta, gamma, the
    offsets alpha (in `groups` order, centered to task-weighted mean zero) and residuals."""
    D = (origins[:, None] == np.asarray(groups)[None, :]).astype(float)
    X = np.column_stack([a, loglen, D])
    coef = np.linalg.lstsq(X, c, rcond=None)[0]
    alpha = coef[2:]
    return coef[0], coef[1], alpha - D.mean(0) @ alpha, c - X @ coef


def within_slope(a, c, loglen, mask):
    """Slope of c on a within the masked tasks, adjusting for log length."""
    X = np.column_stack([a[mask], loglen[mask], np.ones(mask.sum())])
    return np.linalg.lstsq(X, c[mask], rcond=None)[0][0]


def ols(y, *covariates):
    """Least squares with an intercept: (coefficients of the covariates, R^2)."""
    X = np.column_stack([*covariates, np.ones(len(y))])
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ coef
    return coef[:-1], float(1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum())


DECOMPOSITION_STATS = ("r_pre", "r_assistant", "r_predictors", "r2_pre", "r2_assistant", "r2_both",
                       "coef_pre", "coef_assistant")


def decomposition(y, pre, assistant):
    """y ~ pre + assistant: each predictor's r with y and with each other, R^2 of each
    alone and both together, and the raw coefficients (y units per predictor unit)."""
    r_pre, r_asst = pearson(pre, y), pearson(assistant, y)
    coef, r2_both = ols(y, pre, assistant)
    return np.array([r_pre, r_asst, pearson(pre, assistant), r_pre ** 2, r_asst ** 2, r2_both, *coef])


def nonassistant(c, a):
    """Coinflip utility minus its pooled least-squares fit on Assistant utility, centered."""
    resid = c - ols(c, a)[0][0] * a
    return resid - resid.mean()


def dummies(labels):
    return [(labels == v).astype(float) for v in np.unique(labels)]


def adjusted_r2(y, blocks):
    """Adjusted R^2 of y on the columns of all blocks plus an intercept; parameters are
    counted by rank, so redundant dummies (e.g. topic within origin) cost nothing extra."""
    cols = [col for b in blocks for col in b]
    if not cols:
        return 0.0
    X = np.column_stack([*cols, np.ones(len(y))])
    r2 = ols(y, *cols)[1]
    p = np.linalg.matrix_rank(X) - 1
    return float(1 - (1 - r2) * (len(y) - 1) / (len(y) - p - 1))


def group_mean_ci(x, rng, n_boot):
    boots = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)]
    return (float(x.mean()), *(float(v) for v in np.percentile(boots, [2.5, 97.5])))


def with_ci(stat, n, resamples):
    """stat(index array) -> array; returns (estimate on all n tasks, ci_low, ci_high)."""
    lo, hi = np.percentile([stat(i) for i in resamples], [2.5, 97.5], axis=0)
    return stat(np.arange(n)), lo, hi


def stage_analysis(a, c, loglen, origins, benign, groups, resamples):
    """Bleedthrough stats and the task-type model for one stage. Bootstrapped quantities
    are (estimate, ci_low, ci_high); per-origin ones hold arrays in `groups` order."""
    G = len(groups)

    def stats(i):
        beta, gamma, offsets, _ = offset_model(a[i], c[i], loglen[i], origins[i], groups)
        b = benign[i]
        return np.array([pearson(a[i], c[i]), pearson(a[i][b], c[i][b]), partial_r(a[i], c[i], loglen[i]),
                         np.polyfit(a[i], c[i], 1)[0], beta, gamma, *offsets,
                         *(within_slope(a[i], c[i], loglen[i], origins[i] == g) for g in groups)])

    est, lo, hi = with_ci(stats, len(a), resamples)
    k = len(BLEED_STATS)
    out = {name: (est[j], lo[j], hi[j]) for j, name in enumerate(BLEED_STATS)}
    out["offset"] = (est[k:k + G], lo[k:k + G], hi[k:k + G])
    out["origin_slope"] = (est[k + G:], lo[k + G:], hi[k + G:])

    # Geometry for the offsets figure: group means of length-adjusted coinflip utility lie
    # offset[g] above the line of slope beta through the all-task mean.
    beta, gamma, _, resid = offset_model(a, c, loglen, origins, groups)
    y = c - gamma * (loglen - loglen.mean())
    out.update(
        residual=resid,
        x_group=np.array([a[origins == g].mean() for g in groups]),
        y_group=np.array([y[origins == g].mean() for g in groups]),
        line_intercept=y.mean() - beta * a.mean(),
        r2_slope_length=ols(c, a, loglen)[1],
        r2_with_origin=float(1 - (resid ** 2).sum() / ((c - c.mean()) ** 2).sum()),
    )
    return out


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK_2)


def errorbar(ax, x, est, lo, hi, color, marker, label, **kw):
    ax.errorbar(x, est, yerr=[est - lo, hi - est], color=color, marker=marker, ms=7, lw=2, elinewidth=1.5,
                capsize=0, mec="white", mew=0.8, label=label, zorder=3, **kw)


def plot_by_stage(stages, per_stage):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(stages))
    panels = (
        ("Correlation across tasks", "Pearson r (Assistant, coinflip)",
         (("r", "all tasks"), ("r_benign", "benign only"), ("r_partial_length", "log length partialled out"))),
        ("Slope", "coinflip log-odds per Assistant log-odd",
         (("slope_pooled", "pooled"), ("slope_within", "within origin, length-adjusted"))),
    )
    for ax, (title, ylabel, series) in zip(axes, panels):
        for k, ((key, label), color, marker) in enumerate(zip(series, SERIES, "osD")):
            est, lo, hi = (np.array([per_stage[s][key][j] for s in stages]) for j in range(3))
            errorbar(ax, x + (k - (len(series) - 1) / 2) * 0.08, est, lo, hi, color, marker, label)
        ax.axhline(0, color=RULE, lw=0.8, zorder=1)
        ax.set_xticks(x, stages)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11, color=INK)
        ax.legend(frameon=False, fontsize=9)
        style(ax)
    fig.suptitle("Assistant preferences in the coinflip, by training stage (95% CI over tasks)", fontsize=11)
    fig.tight_layout()
    return fig


def plot_cross_stage(stages, M):
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    lim = max(np.abs(M).max(), 0.1)
    im = ax.imshow(M, cmap=DIVERGING, vmin=-lim, vmax=lim)
    for i, j in np.ndindex(M.shape):
        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=10,
                color="white" if abs(M[i, j]) > 0.6 * lim else INK)
    ax.set_xticks(range(len(stages)), stages)
    ax.set_yticks(range(len(stages)), stages)
    ax.set_xlabel("coinflip utilities from stage")
    ax.set_ylabel("Assistant utilities from stage")
    ax.set_title("Pearson r across tasks", fontsize=11)
    for side in ax.spines.values():
        side.set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig


def origin_scatter(ax, x, y, origins, fit_lines=False):
    for origin, (color, marker) in ORIGIN_STYLE.items():
        m = origins == origin
        if not m.any():
            continue
        ax.scatter(x[m], y[m], s=22, c=color, marker=marker, label=origin.lower(), edgecolors="white",
                   linewidths=0.6, zorder=3)
        if fit_lines and m.sum() >= 3:
            xs = np.array([x[m].min(), x[m].max()])
            ax.plot(xs, np.polyval(np.polyfit(x[m], y[m], 1), xs), color=color, lw=1.5, zorder=4)
    ax.axhline(0, color=RULE, lw=0.8, zorder=1)
    ax.axvline(0, color=RULE, lw=0.8, zorder=1)
    style(ax)


def plot_scatter_by_stage(stages, U, origins, per_stage):
    fig, axes = plt.subplots(1, len(stages), figsize=(4.2 * len(stages), 4.4), squeeze=False)
    for ax, s in zip(axes[0], stages):
        origin_scatter(ax, U[f"{s}/assistant"], U[f"{s}/coinflip"], origins, fit_lines=True)
        est, lo, hi = per_stage[s]["r"]
        ax.set_title(f"{s}: r = {est:.2f} [{lo:.2f}, {hi:.2f}]", fontsize=10, color=INK)
        ax.set_xlabel("Assistant utility (log-odds)")
    axes[0, 0].set_ylabel("coinflip utility (log-odds)")
    axes[0, -1].legend(frameon=False, fontsize=8)
    fig.suptitle("Per-task utilities by stage; lines are least-squares fits within each origin "
                 "(axes differ between panels)", fontsize=11)
    fig.tight_layout()
    return fig


def plot_task_types(stages, groups, per_stage, focus):
    fig, (ax_s, ax_o) = plt.subplots(1, 2, figsize=(12.5, 5))
    x = np.arange(len(stages))
    for k, g in enumerate(groups):
        color, marker = ORIGIN_STYLE[g]
        est, lo, hi = (np.array([per_stage[s]["origin_slope"][j][k] for s in stages]) for j in range(3))
        errorbar(ax_s, x + (k - (len(groups) - 1) / 2) * 0.07, est, lo, hi, color, marker, g.lower())
    ax_s.plot(x, [per_stage[s]["slope_within"][0] for s in stages], color=INK_2, ls="--", lw=1,
              label="all origins (common slope)", zorder=2)
    ax_s.axhline(0, color=RULE, lw=0.8, zorder=1)
    ax_s.set_xticks(x, stages)
    ax_s.set_ylabel("coinflip log-odds per Assistant log-odd")
    ax_s.set_title("Slope within each origin, length-adjusted", fontsize=11)
    ax_s.legend(frameon=False, fontsize=8)
    style(ax_s)

    f = per_stage[focus]
    beta = f["slope_within"][0]
    off_est, off_lo, off_hi = f["offset"]
    xs = np.array([f["x_group"].min(), f["x_group"].max()])
    xs = xs + np.array([-0.08, 0.08]) * np.ptp(xs)
    ax_o.plot(xs, beta * xs + f["line_intercept"], color=INK_2, ls="--", lw=1, zorder=1,
              label=f"common slope {beta:.3f}")
    for k, g in enumerate(groups):
        color, marker = ORIGIN_STYLE[g]
        xg, yg = f["x_group"][k], f["y_group"][k]
        lo, hi = yg - (off_est[k] - off_lo[k]), yg + (off_hi[k] - off_est[k])
        errorbar(ax_o, np.array([xg]), np.array([yg]), np.array([lo]), np.array([hi]), color, marker, None)
        # Label on the side of the CI away from the line, where it can't collide with it.
        above = off_est[k] >= 0
        ax_o.annotate(f"{g.lower()}\n{off_est[k]:+.2f}", (xg, hi if above else lo), xytext=(0, 5 if above else -5),
                      textcoords="offset points", ha="center", va="bottom" if above else "top", fontsize=9,
                      color=INK)
    ax_o.axhline(0, color=RULE, lw=0.8, zorder=0)
    ax_o.set_xlabel("origin mean Assistant utility")
    ax_o.set_ylabel("origin mean coinflip utility, length-adjusted")
    ax_o.set_title(f"{focus}: origin offsets from the common slope (95% CI)", fontsize=11)
    ax_o.legend(frameon=False, fontsize=9)
    style(ax_o)
    fig.tight_layout()
    return fig


def plot_pre_stage(pre, focus, U, origins, correlates):
    fig, (ax_sc, ax_b) = plt.subplots(1, 2, figsize=(11.5, 4.5), gridspec_kw=dict(width_ratios=[1.1, 1]))
    origin_scatter(ax_sc, U[f"{pre}/coinflip"], U[f"{focus}/coinflip"], origins)
    ax_sc.set_xlabel(f"{pre} coinflip utility")
    ax_sc.set_ylabel(f"{focus} coinflip utility")
    ax_sc.legend(frameon=False, fontsize=8)
    names = list(correlates)
    est, lo, hi = (np.array([correlates[n][j] for n in names]) for j in range(3))
    y = np.arange(len(names))[::-1]
    ax_b.barh(y, est, height=0.5, color=SERIES[0], zorder=2)
    ax_b.errorbar(est, y, xerr=[est - lo, hi - est], fmt="none", ecolor=INK_2, elinewidth=1.2, zorder=3)
    ax_b.set_yticks(y, names)
    ax_b.axvline(0, color=RULE, lw=0.8)
    ax_b.set_xlabel(f"Pearson r with {pre} coinflip utility (95% CI)")
    style(ax_b)
    fig.suptitle(f"What the {pre} coinflip utilities track", fontsize=11)
    fig.tight_layout()
    return fig


def plot_nonassistant_groups(group_rows, show_stages, focus):
    """Group means (95% CI): the focus stage's Assistant utility on the left, the
    non-Assistant coinflip utility of each shown stage on the right."""
    n_origin = sum(r["kind"] == "origin" for r in group_rows)
    y = np.arange(len(group_rows))[::-1].astype(float)
    y[n_origin:] -= 0.6  # gap between the origin and topic blocks
    fig, (ax_a, ax_r) = plt.subplots(1, 2, figsize=(12, 0.3 * len(group_rows) + 1.8), sharey=True,
                                     gridspec_kw=dict(width_ratios=[1, 1.3]))

    def draw(ax, key, color, marker, label, dy):
        est, lo, hi = (np.array([r[f"{key}{suffix}"] for r in group_rows]) for suffix in ("", "_ci_low", "_ci_high"))
        ax.errorbar(est, y + dy, xerr=[est - lo, hi - est], fmt=marker, color=color, ms=6, elinewidth=1.5,
                    mec="white", mew=0.8, label=label, zorder=3)

    draw(ax_a, f"{focus}/assistant", INK_2, "o", None, 0)
    for k, s in enumerate(show_stages):
        draw(ax_r, f"{s}/nonassistant", SERIES[k], "os"[k], s, (k - (len(show_stages) - 1) / 2) * 0.25)
    ax_a.set_yticks(y, [f"{r['group'].lower()} ({r['n']})" for r in group_rows])
    for ax in (ax_a, ax_r):
        ax.axvline(0, color=RULE, lw=0.8, zorder=1)
        ax.axhline(y[n_origin - 1] - 0.8, color=RULE, lw=0.8, zorder=1)
        style(ax)
    ax_a.set_xlabel(f"{focus} Assistant utility (log-odds)")
    ax_r.set_xlabel("non-Assistant coinflip utility (log-odds)")
    ax_a.set_title("Assistant", fontsize=11)
    ax_r.set_title("Coinflip minus its fit on the same stage's Assistant utility", fontsize=11)
    ax_r.legend(frameon=False, fontsize=9)
    fig.suptitle("Group means by origin (top) and topic (bottom), task counts in parentheses", fontsize=11)
    fig.tight_layout()
    return fig


def fmt_ci(t, prec=2):
    return f"{t[0]:.{prec}f} [{t[1]:.{prec}f}, {t[2]:.{prec}f}]"


def report(rows, summary, out_dir, seed=0, n_boot=N_BOOTSTRAP):
    """Compute the report, write CSVs and PNGs to out_dir, and return what to log to W&B.
    summary: analyze.py's per-measure diagnostics (may be empty; those columns are then NaN)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    U, origins, benign, loglen = task_arrays(rows)
    stages = stages_of(list(U))
    groups = [g for g in ORIGIN_STYLE if g in set(origins)]
    if unknown := set(origins) - set(groups):
        raise ValueError(f"origins without a plot style: {sorted(unknown)}")
    resamples = stratified_resamples(origins, np.random.default_rng(seed), n_boot)
    per_stage = {s: stage_analysis(U[f"{s}/assistant"], U[f"{s}/coinflip"], loglen, origins, benign, groups,
                                   resamples) for s in stages}

    stage_rows = []
    for s in stages:
        row = dict(stage=s)
        for cond in ("assistant", "coinflip"):
            row[f"{cond}_utility_sd"] = float(U[f"{s}/{cond}"].std())
            for k in DIAGNOSTICS:
                row[f"{cond}_{k}"] = float(summary.get(f"{s}/{cond}/{k}", "nan"))
        for k in BLEED_STATS:
            row.update({k: float(per_stage[s][k][0]), f"{k}_ci_low": float(per_stage[s][k][1]),
                        f"{k}_ci_high": float(per_stage[s][k][2])})
        row.update(r2_slope_length=per_stage[s]["r2_slope_length"], r2_with_origin=per_stage[s]["r2_with_origin"])
        stage_rows.append(row)

    offset_rows = []
    for s in stages:
        f = per_stage[s]
        for k, g in enumerate(groups):
            offset_rows.append(dict(
                stage=s, origin=g, n=int((origins == g).sum()), mean_assistant=float(f["x_group"][k]),
                mean_coinflip_length_adjusted=float(f["y_group"][k]),
                **{f"{name}{suffix}": float(f[name][j][k]) for name in ("offset", "origin_slope")
                   for j, suffix in enumerate(("", "_ci_low", "_ci_high"))}))

    focus = FOCUS_STAGE if FOCUS_STAGE in stages else stages[-1]
    topics = np.array([r["topic"] or "none" for r in rows])
    resid = {s: nonassistant(U[f"{s}/coinflip"], U[f"{s}/assistant"]) for s in stages}
    partition_rows = []
    for k, s in enumerate(stages):
        prev = [U[f"{stages[k - 1]}/coinflip"]] if k else None
        blocks = dict(previous_coinflip=prev, origin=dummies(origins), topic=dummies(topics), log_length=[loglen])
        row = dict(stage=s, previous_stage=stages[k - 1] if k else "",
                   share_of_coinflip_variance=1 - pearson(U[f"{s}/coinflip"], U[f"{s}/assistant"]) ** 2,
                   sd=float(resid[s].std()))
        included = []
        for name, b in blocks.items():
            row[f"alone_{name}"] = adjusted_r2(resid[s], [b]) if b else float("nan")
            included += [b] if b else []
            row[f"cumulative_{name}"] = adjusted_r2(resid[s], included) if b else float("nan")
        row["unexplained"] = 1 - adjusted_r2(resid[s], included)
        partition_rows.append(row)

    group_rng = np.random.default_rng(seed + 1)
    group_rows = []
    for kind, labels in (("origin", origins), ("topic", topics)):
        for g in sorted(set(labels), key=lambda g: -resid[focus][labels == g].mean()):
            m = labels == g
            row = dict(kind=kind, group=g, n=int(m.sum()))
            for s in stages:
                for name, v in ((f"{s}/assistant", U[f"{s}/assistant"]), (f"{s}/nonassistant", resid[s])):
                    est, lo, hi = group_mean_ci(v[m], group_rng, n_boot)
                    row.update({name: est, f"{name}_ci_low": lo, f"{name}_ci_high": hi})
            group_rows.append(row)
    nonassistant_rows = [dict(id=rows[i]["id"], origin=rows[i]["origin"], topic=rows[i]["topic"],
                              **{f"{s}/nonassistant": float(resid[s][i]) for s in stages}, prompt=rows[i]["prompt"])
                         for i in np.argsort(-resid[focus])]

    M = np.array([[pearson(U[f"{sa}/assistant"], U[f"{sc}/coinflip"]) for sc in stages] for sa in stages])
    figures = {
        "bleedthrough_by_stage": plot_by_stage(stages, per_stage),
        "cross_stage_r": plot_cross_stage(stages, M),
        "scatter_by_stage": plot_scatter_by_stage(stages, U, origins, per_stage),
        "task_types": plot_task_types(stages, groups, per_stage, focus),
        "nonassistant_groups": plot_nonassistant_groups(
            group_rows, [s for s in (PRE_STAGE, FOCUS_STAGE) if s in stages] or [focus], focus),
    }

    residual_rows = []
    if FOCUS_STAGE in stages:
        f = per_stage[FOCUS_STAGE]
        order = np.argsort(-f["residual"])
        residual_rows = [dict(id=rows[i]["id"], origin=rows[i]["origin"], topic=rows[i]["topic"],
                              assistant=float(U[f"{FOCUS_STAGE}/assistant"][i]),
                              coinflip=float(U[f"{FOCUS_STAGE}/coinflip"][i]), residual=float(f["residual"][i]),
                              prompt=rows[i]["prompt"]) for i in order]

    decomposition_rows = []
    if PRE_STAGE in stages and FOCUS_STAGE in stages:
        target = U[f"{PRE_STAGE}/coinflip"]
        candidates = {"log task length": loglen, **{m: U[m] for m in (
            f"{BASELINE_STAGE}/coinflip", f"{PRE_STAGE}/assistant", f"{FOCUS_STAGE}/assistant",
            f"{FOCUS_STAGE}/coinflip") if m in U}}
        correlates = {name: with_ci(lambda i, x=x: pearson(x[i], target[i]), len(target), resamples)
                      for name, x in candidates.items()}
        figures["pre_stage_coinflip"] = plot_pre_stage(PRE_STAGE, FOCUS_STAGE, U, origins, correlates)

        names = dict(target=f"{FOCUS_STAGE}/coinflip", pre=f"{PRE_STAGE}/coinflip", assistant=f"{FOCUS_STAGE}/assistant")
        y, pre, asst = (U[names[k]] for k in ("target", "pre", "assistant"))
        for subset, mask in (("all", np.ones(len(y), bool)), ("benign", benign)):
            est, lo, hi = with_ci(lambda i: decomposition(y[i][mask[i]], pre[i][mask[i]], asst[i][mask[i]]),
                                  len(y), resamples)
            decomposition_rows.append(dict(subset=subset, n=int(mask.sum()), **names, **{
                f"{name}{suffix}": float(v[j]) for j, name in enumerate(DECOMPOSITION_STATS)
                for suffix, v in (("", est), ("_ci_low", lo), ("_ci_high", hi))}))

    paths = {}
    for name, fig in figures.items():
        paths[f"report/{name}"] = out_dir / f"{name}.png"
        fig.savefig(paths[f"report/{name}"], dpi=150)
        plt.close(fig)
    tables = dict(stage_summary=stage_rows, origin_offsets=offset_rows, decomposition=decomposition_rows,
                  residuals=residual_rows, nonassistant_partition=partition_rows, nonassistant_groups=group_rows,
                  nonassistant_tasks=nonassistant_rows)
    for name, table in tables.items():
        if table:
            with open(out_dir / f"{name}.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(table[0]))
                w.writeheader()
                w.writerows(table)

    print(f"\n{'stage':10} {'sd A':>6} {'sd C':>6} {'addR2 A':>8} {'addR2 C':>8} {'mass min C':>11}  "
          f"{'r':20} {'benign r':20} {'r | loglen':20} {'pooled slope':22} {'within slope':22} {'R2 +origin':>10}")
    for r, s in zip(stage_rows, stages):
        p = per_stage[s]
        print(f"{s:10} {r['assistant_utility_sd']:6.2f} {r['coinflip_utility_sd']:6.3f} "
              f"{r['assistant_additive_r2']:8.2f} {r['coinflip_additive_r2']:8.2f} {r['coinflip_outcome_mass_min']:11.3f}  "
              f"{fmt_ci(p['r']):20} {fmt_ci(p['r_benign']):20} {fmt_ci(p['r_partial_length']):20} "
              f"{fmt_ci(p['slope_pooled'], 3):22} {fmt_ci(p['slope_within'], 3):22} {p['r2_with_origin']:10.2f}")
    print("\nOrigin offsets (coinflip log-odds), within-origin slope in parentheses:")
    print(f"{'stage':10} " + " ".join(f"{g.lower():>24}" for g in groups))
    for s in stages:
        cells = []
        for k in range(len(groups)):
            o, sl = per_stage[s]["offset"], per_stage[s]["origin_slope"]
            cells.append(f"{o[0][k]:+.2f} [{o[1][k]:+.2f},{o[2][k]:+.2f}] ({sl[0][k]:.3f})")
        print(f"{s:10} " + " ".join(f"{c:>24}" for c in cells))
    if decomposition_rows:
        d = decomposition_rows[0]
        print(f"\n{d['target']} ~ {d['pre']} (pre) + {d['assistant']} (asst):")
        print(f"{'subset':8} {'n':>4} {'r pre':>6} {'r asst':>7} {'r pre,asst':>11} {'R2 pre':>7} {'R2 asst':>8} "
              f"{'R2 both':>8}  {'coef pre':22} {'coef asst':22}")
        for d in decomposition_rows:
            ci = {k: (d[k], d[f"{k}_ci_low"], d[f"{k}_ci_high"]) for k in ("coef_pre", "coef_assistant")}
            print(f"{d['subset']:8} {d['n']:4d} {d['r_pre']:6.2f} {d['r_assistant']:7.2f} {d['r_predictors']:11.2f} "
                  f"{d['r2_pre']:7.2f} {d['r2_assistant']:8.2f} {d['r2_both']:8.2f}  "
                  f"{fmt_ci(ci['coef_pre']):22} {fmt_ci(ci['coef_assistant'], 4):22}")
    if residual_rows:
        print(f"\n{FOCUS_STAGE}: tasks furthest above / below the offset model (coinflip residual):")
        for r in residual_rows[:N_EXTREME] + [None] + residual_rows[-N_EXTREME:]:
            if r is None:
                print("   ...")
                continue
            print(f"  {r['residual']:+.2f} {r['origin']:11} {r['topic']:18} A={r['assistant']:+6.1f} "
                  f"C={r['coinflip']:+.2f}  {r['prompt'][:80]!r}")
    blocks = ("previous_coinflip", "origin", "topic", "log_length")
    print("\nNon-Assistant coinflip utility: share of coinflip variance, then adjusted R^2 of each block "
          "alone / cumulative in this order:")
    print(f"{'stage':10} {'share':>6} {'sd':>6}  " + " ".join(f"{b:>18}" for b in blocks) + f" {'unexplained':>12}")
    for r in partition_rows:
        cells = [f"{r[f'alone_{b}']:.2f} / {r[f'cumulative_{b}']:.2f}" for b in blocks]
        print(f"{r['stage']:10} {r['share_of_coinflip_variance']:6.2f} {r['sd']:6.3f}  "
              + " ".join(f"{c:>18}" for c in cells) + f" {r['unexplained']:12.2f}")
    shown = [s for s in (PRE_STAGE, FOCUS_STAGE) if s in stages] or [focus]
    print(f"\nGroup means of non-Assistant coinflip utility (95% CI); {focus} Assistant utility for reference:")
    print(f"{'group':28} {'n':>4} {focus + ' assistant':>16} " + " ".join(f"{s + ' non-asst':>22}" for s in shown))
    for r in group_rows:
        cells = [fmt_ci(tuple(r[f"{s}/nonassistant{x}"] for x in ("", "_ci_low", "_ci_high"))) for s in shown]
        print(f"{r['kind'][0]}:{r['group'].lower():26} {r['n']:4d} {r[f'{focus}/assistant']:16.2f} "
              + " ".join(f"{c:>22}" for c in cells))
    print(f"\nTasks with the highest / lowest {focus} non-Assistant coinflip utility:")
    for r in nonassistant_rows[:N_EXTREME] + [None] + nonassistant_rows[-N_EXTREME:]:
        if r is None:
            print("   ...")
            continue
        print(f"  {r[f'{focus}/nonassistant']:+.2f} ({' '.join(f'{s} {r[f'{s}/nonassistant']:+.2f}' for s in shown if s != focus)})"
              f" {r['origin']:11} {r['topic']:18} {r['prompt'][:70]!r}")
    print(f"\nWrote report to {out_dir}")
    return dict(tables=tables, figures=paths)


def log_to_wandb(run, result):
    import wandb
    log = {f"report/{name}": wandb.Table(columns=list(t[0]), data=[list(r.values()) for r in t])
           for name, t in result["tables"].items() if t}
    log.update({name: wandb.Image(str(path)) for name, path in result["figures"].items()})
    run.log(log)


def fetch_wandb(run, out_dir):
    """Save a W&B run's utilities table and numeric summary to out_dir as utilities.csv and
    summary.json, the files analyze.py writes locally."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = run.summary["utilities"]["path"]
    run.file(path).download(root=str(out_dir), replace=True)
    table = json.loads((out_dir / path).read_text())
    with open(out_dir / "utilities.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(table["columns"])
        w.writerows(table["data"])
    summary = {k: v for k, v in run.summary.items() if isinstance(v, (int, float)) and not k.startswith("_")}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source", help="local run dir (outputs/<run_id>, analyzed) or W&B run path entity/project/run_id")
    p.add_argument("--update-wandb", action="store_true", help="log the report to the W&B run")
    args = p.parse_args()

    run = None
    if Path(args.source).is_dir():
        analysis = Path(args.source) / "analysis"
    else:
        import wandb
        run = wandb.Api().run(args.source)
        analysis = Path("outputs") / run.id / "analysis"
        fetch_wandb(run, analysis)
    with open(analysis / "utilities.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    summary_path = analysis / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    result = report(rows, summary, analysis.parent / "report")
    if args.update_wandb:
        if run is None:
            raise SystemExit("--update-wandb needs a W&B run path")
        import wandb
        live = wandb.init(entity=run.entity, project=run.project, id=run.id, resume="must")
        log_to_wandb(live, result)
        live.finish()


if __name__ == "__main__":
    main()
