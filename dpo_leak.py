"""Does DPO raise the coin's leakage, or the Assistant's preferences? (README section of that name.)

Pre-registered in the README (2026-11-22) before any Tulu 3 70B or OLMo 2 13B data; OLMo 2 32B, OLMo 3.1 32B and Apertus
1.5 70B were already on disk and are context. With A and C a stage's Assistant and coinflip utilities (nats) over the
300 tasks:

- sd_a = SD(A), how strongly the Assistant's preferences separate the tasks; sd_c = SD(C);
- b = cov(A, C) / sd_a, the leaked size: the coin's part along the Assistant, in coin nats per Assistant SD;
- lam = cov(A, C) / var(A) = b / sd_a, the leakage rate: coin nats per nat of Assistant utility. lam_benign and
  b_benign over non-harmful tasks; lam_within and b_within after removing each origin's mean from both;
- gap_a, gap_c: mean non-harmful minus harmful; gap_ratio = gap_c / gap_a;
- resid_sd: SD of C around its line on A; r = r(A, C).

A step s -> t (consecutive stages):
- gains t / s of sd_a, b, lam (b's gain is sd_a's times lam's), their benign and within-origin versions, gap_ratio and
  resid_sd;
- the leakage share S = log(lam gain) / log(b gain), the part of the leaked coin's growth that is a higher rate (1 - S is
  stronger Assistant preferences); share_benign and share_within the same over non-harmful tasks and within origin;
- each condition's gain as the slope of t's utilities on s's (k_a, k_c) and the r between them (r_a, r_c): is t mostly s
  turned up?
- the marginal leakage: the slope of C_t - C_s on A_t - A_s over tasks (what the step adds to the Assistant, and how much
  of it reaches the coin).

CIs are 95% task bootstraps within origin (2,000), the same resamples for every ladder (they all score the same 300
tasks), so steps and ladders are paired.

  outputs/dpo_leak/stages.csv   per ladder and stage
  outputs/dpo_leak/steps.csv    per ladder and step
  outputs/dpo_leak/report.txt   gate, calls, predictions (also printed)
  figures/dpo_leak_*.png

Usage: uv run python dpo_leak.py [--skip-missing]   (--skip-missing leaves out a new ladder with no finished runs)
"""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import N_BOOTSTRAP
from matskit import plotstyle
from models import olmo_on_shared_pairs
from olmo2 import SHARED_PAIRS, merged_fit
from openers import within_groups, write_csv
from report import stratified_resamples, task_arrays

OUT = Path("outputs/dpo_leak")
# key: label, stages in order, the DPO step (pre, post), where the utilities come from, and whether the ladder is one of
# the pre-registered tests (new data) or context seen before the pre-registration.
LADDERS = {
    "olmo2_32b": dict(label="OLMo 2 32B", stages=("base", "sft", "dpo", "instruct"), dpo=("sft", "dpo"),
                      source=("group", "olmo2", "outputs/olmo2/analysis"), test=False),
    "tulu3_70b": dict(label="Tulu 3 70B", stages=("base", "sft", "dpo", "instruct"), dpo=("sft", "dpo"),
                      source=("group", "tulu3", "outputs/tulu3/analysis"), test=True),
    "olmo2_13b": dict(label="OLMo 2 13B", stages=("base", "sft", "dpo", "instruct"), dpo=("sft", "dpo"),
                      source=("group", "olmo2-13b", "outputs/olmo2-13b/analysis"), test=True),
    "olmo3_32b": dict(label="OLMo 3.1 32B", stages=("base", "sft", "dpo", "instruct"), dpo=("sft", "dpo"),
                      source=("olmo3",), test=False),
    "apertus_70b": dict(label="Apertus 1.5 70B", stages=("base", "sft", "rl", "dpo"), dpo=("rl", "dpo"),
                        source=("file", "outputs/apertus/analysis"), test=False),
}
STAGE_LABELS = {"base": "base", "sft": "SFT", "dpo": "DPO", "instruct": "final", "rl": "RL"}
COLORS = {k: plotstyle.SERIES[j] for j, k in enumerate(LADDERS)}
MARKERS = {"olmo2_32b": "o", "tulu3_70b": "D", "olmo2_13b": "^", "olmo3_32b": "s", "apertus_70b": "v"}

# Task sets for the scale / leaked size / rate triple: (task mask from origins and benign, demean within origin?). The
# "arb" sets (Hal's request, 2026-11-22, exploratory, added before any new ladder was analysed) are the 225 MATH, Alpaca
# and WildChat tasks, where preferences are "more arbitrary" than BailBench's and STRESS-TEST's, pooled, within origin, and
# one origin at a time.
ARB = ("MATH", "ALPACA", "WILDCHAT")
SUBSETS = {
    "": (lambda o, bn: np.ones(len(o), bool), False),
    "benign": (lambda o, bn: bn, False),
    "within": (lambda o, bn: np.ones(len(o), bool), True),
    "arb": (lambda o, bn: np.isin(o, ARB), False),
    "arbw": (lambda o, bn: np.isin(o, ARB), True),
    "math": (lambda o, bn: o == "MATH", False),
    "alpaca": (lambda o, bn: o == "ALPACA", False),
    "wildchat": (lambda o, bn: o == "WILDCHAT", False),
    "rest": (lambda o, bn: ~np.isin(o, ARB), False),  # the complement, BailBench + STRESS-TEST (added with "arb")
}
SUBSET_LABELS = {"": "all 300 tasks", "benign": "non-harmful", "within": "within origin",
                 "arb": "MATH + Alpaca + WildChat", "arbw": "MATH + Alpaca + WildChat, within origin",
                 "math": "MATH", "alpaca": "Alpaca", "wildchat": "WildChat",
                 "rest": "BailBench + STRESS-TEST"}


def sfx(sub):
    return f"_{sub}" if sub else ""


STAGE_STATS = tuple(f"{q}{sfx(sub)}" for sub in SUBSETS for q in ("sd_a", "b", "lam")) + (
    "sd_c", "gap_a", "gap_c", "gap_ratio", "resid_sd", "r", "resid_sd_arb", "r_arb")
SK = {s: j for j, s in enumerate(STAGE_STATS)}
GAINS = tuple(f"{q}{sfx(sub)}" for sub in SUBSETS for q in ("sd_a", "b", "lam")) + (
    "sd_c", "gap_ratio", "resid_sd", "resid_sd_arb")
STEP_STATS = tuple(f"g_{g}" for g in GAINS) + tuple(f"share{sfx(sub)}" for sub in SUBSETS) + (
    "k_a", "k_c", "r_a", "r_c", "marginal", "r_marginal", "marginal_arb", "rel_coin", "align", "share_align")
TK = {s: j for j, s in enumerate(STEP_STATS)}

MIN_RELIABILITY = 0.9
LOW_MASS = 0.5
B_GAIN_BAR = 1.2  # the DPO step's b gain must have a CI above this for the call to be read


def slope(x, y):
    x, y = x - x.mean(), y - y.mean()
    return float((x * y).mean() / (x * x).mean())


def triple(a, c):
    """(sd_a, b, lam, residual SD, r) of C on A over these tasks."""
    a0, c0 = a - a.mean(), c - c.mean()
    cov, var = (a0 * c0).mean(), (a0 * a0).mean()
    sd_a = math.sqrt(var)
    return sd_a, cov / sd_a, cov / var, (c0 - cov / var * a0).std(), cov / (sd_a * c0.std())


def subset_arrays(a, c, sub, benign, origins):
    mask_fn, within = SUBSETS[sub]
    m = mask_fn(origins, benign)
    a, c = a[m], c[m]
    if within:
        a, c = within_groups(a, origins[m]), within_groups(c, origins[m])
    return a, c


def stage_vec(a, c, benign, origins):
    """STAGE_STATS of one stage over arrays of the same (possibly resampled) tasks."""
    vals = []
    for sub in SUBSETS:
        vals += triple(*subset_arrays(a, c, sub, benign, origins))[:3]
    _, _, _, resid_sd, r = triple(a, c)
    _, _, _, resid_arb, r_arb = triple(*subset_arrays(a, c, "arb", benign, origins))
    gap_a, gap_c = a[benign].mean() - a[~benign].mean(), c[benign].mean() - c[~benign].mean()
    return np.array(vals + [c.std(), gap_a, gap_c, gap_c / gap_a, resid_sd, r, resid_arb, r_arb])


def step_vec(st_s, st_t, a_s, c_s, a_t, c_t, benign, origins):
    """STEP_STATS of the step s -> t from both stages' STAGE_STATS and utilities (same tasks)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        g = {name: st_t[SK[name]] / st_s[SK[name]] for name in GAINS}
        share = lambda lam, b: math.log(lam) / math.log(b) if lam > 0 and b > 0 and b != 1 else math.nan
        da, dc = a_t - a_s, c_t - c_s
        arb = np.isin(origins, ARB)
        return np.array([g[n] for n in GAINS] + [share(g[f"lam{sfx(sub)}"], g[f"b{sfx(sub)}"]) for sub in SUBSETS] + [
            slope(a_s, a_t), slope(c_s, c_t), np.corrcoef(a_s, a_t)[0, 1], np.corrcoef(c_s, c_t)[0, 1],
            slope(da, dc), np.corrcoef(da, dc)[0, 1], slope(da[arb], dc[arb]),
            # Exploratory (after the data): lam's gain = rel_coin * align exactly. rel_coin is the coin's part off the
            # Assistant line growing relative to the Assistant (the whole coin getting more extreme); align is b over
            # that residual SD growing (the coin following the Assistant more closely); share_align its part of lam's gain.
            g["resid_sd"] / g["sd_a"], g["b"] / g["resid_sd"], share(g["b"] / g["resid_sd"], g["lam"]),
        ])


def ci(boot):
    return np.nanpercentile(boot, 2.5, axis=0), np.nanpercentile(boot, 97.5, axis=0)


def load_ladder(key):
    """(utilities.csv rows, summary dict) of one ladder, all stages fit together."""
    src = LADDERS[key]["source"]
    if src[0] == "group":
        return merged_fit(src[1], Path(src[2]))
    if src[0] == "olmo3":
        return olmo_on_shared_pairs(SHARED_PAIRS, stages=LADDERS[key]["stages"], cache_suffix="-stages")
    out = Path(src[1])
    with open(out / "utilities.csv", newline="") as fh:
        return list(csv.DictReader(fh)), json.loads((out / "summary.json").read_text())


def ladder_cells(U, stages, benign, origins, resamples):
    """{stage: (est, boot)} of STAGE_STATS and {(s, t): (est, boot)} of STEP_STATS for consecutive stages."""
    stage_cells = {}
    for s in stages:
        a, c = U[f"{s}/assistant"], U[f"{s}/coinflip"]
        stage_cells[s] = (stage_vec(a, c, benign, origins),
                          np.array([stage_vec(a[i], c[i], benign[i], origins[i]) for i in resamples]))
    step_cells = {}
    for s, t in zip(stages, stages[1:]):
        args = lambda idx: (U[f"{s}/assistant"][idx], U[f"{s}/coinflip"][idx], U[f"{t}/assistant"][idx],
                            U[f"{t}/coinflip"][idx], benign[idx], origins[idx])
        everything = np.arange(len(benign))
        est = step_vec(stage_cells[s][0], stage_cells[t][0], *args(everything))
        boot = np.array([step_vec(stage_cells[s][1][j], stage_cells[t][1][j], *args(i)) for j, i in enumerate(resamples)])
        step_cells[(s, t)] = (est, boot)
    return stage_cells, step_cells


def call(step_est, step_boot):
    """The pre-registered call on a ladder's DPO step."""
    g_b_lo = ci(step_boot)[0][TK["g_b"]]
    if not g_b_lo > B_GAIN_BAR:
        return "DPO doesn't scale the broad coin here"
    S = step_est[TK["share"]]
    s_lo, s_hi = (x[TK["share"]] for x in ci(step_boot))
    lam_lo, lam_hi = (x[TK["g_lam"]] for x in ci(step_boot))
    if S < 0.25 and lam_lo <= 1 <= lam_hi:
        return "explained away"
    if S > 0.75:
        return "leakage step"
    if S < 0.25:  # lam's gain CI excludes 1
        return "mostly preferences, rate up a little"
    if s_hi < 0.5:
        return "both, mostly preferences"
    if s_lo > 0.5:
        return "both, mostly leakage"
    return "both (not split)"


def fmt(est, lo, hi, d=2):
    return f"{est:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def gate(key, summary, stages):
    ok, lines = True, []
    for s in stages:
        for cond in ("assistant", "coinflip"):
            rel, mass = summary[f"{s}/{cond}/reliability"], summary[f"{s}/{cond}/outcome_mass_mean"]
            passed = rel >= MIN_RELIABILITY
            ok &= passed
            flag = "  (coin mass under 0.5: flagged)" if cond == "coinflip" and mass < LOW_MASS else ""
            lines.append(f"  {LADDERS[key]['label']:15} {s:9} {cond:10} reliability {rel:.3f}  mass {mass:.2f}: "
                         f"{'pass' if passed else 'FAIL'}{flag}")
    return ok, lines


def predictions(steps, stages_cells):
    """Score the README's seven predictions on the test ladders that are present."""
    out = []
    tests = [k for k in steps if LADDERS[k]["test"]]

    def dpo(k):
        return steps[k][LADDERS[k]["dpo"]]

    def stat(k, name):
        est, boot = dpo(k)
        lo, hi = ci(boot)
        return est[TK[name]], lo[TK[name]], hi[TK[name]]

    lines = []
    for k in tests:
        lab = LADDERS[k]["label"]
        g, lo, _ = stat(k, "g_b")
        lines.append(f"{lab}: b gain {g:.2f}, CI low {lo:.2f} -> {'holds' if g >= 1.5 and lo > 1.2 else 'fails'}")
    out.append(("1. Premise: b grows >= 1.5x at DPO (CI above 1.2)", lines))
    lines = []
    for k in tests:
        g = stat(k, "g_sd_a")[0]
        lines.append(f"{LADDERS[k]['label']}: sd_a gain {g:.2f} -> {'holds' if 1.3 <= g <= 2.5 else 'fails'}")
    out.append(("2. Stronger preferences: sd_a grows 1.3-2.5x at DPO", lines))
    lines = []
    for k in tests:
        c = call(*dpo(k))
        lines.append(f"{LADDERS[k]['label']}: {c} (most likely was 'both')"
                     + (" -> holds" if c.startswith("both") else " -> fails"))
    out.append(("3. The call (most likely 'both' in each)", lines))
    lines = []
    for k in tests:
        g = stat(k, "g_lam")[0]
        lines.append(f"{LADDERS[k]['label']}: lam gain {g:.2f} -> {'holds' if g > 1 else 'fails'}")
    out.append(("4. The rate rises at all (lam gain > 1)", lines))
    lines = []
    for k in tests:
        st = steps[k].get(("dpo", "instruct"))
        if st is None:
            continue
        g = st[0][TK["g_b"]]
        if k == "tulu3_70b":
            lines.append(f"Tulu 3 70B: b final/DPO {g:.3f} -> {'holds' if abs(g - 1) < 0.10 else 'fails'} (within 10%)")
        else:
            lines.append(f"{LADDERS[k]['label']}: b final/DPO {g:.3f} -> {'holds' if 1.10 <= g <= 1.30 else 'fails'} "
                         "(up 10-30%)")
    out.append(("5. The last stage", lines))
    lines = []
    for k in tests:
        m = stat(k, "marginal")[0]
        lam_sft = stages_cells[k][LADDERS[k]["dpo"][0]][0][SK["lam"]]
        lines.append(f"{LADDERS[k]['label']}: marginal {m:.3f} vs lam at SFT {lam_sft:.3f} -> "
                     f"{'holds' if m > lam_sft else 'fails'}")
    out.append(("6. Marginal leakage of the DPO step above lam at SFT", lines))
    lines = []
    for k in tests:
        g = stat(k, "g_lam")[0]
        parts = []
        ok = True
        for name in ("g_lam_benign", "g_lam_within"):
            v = stat(k, name)[0]
            same = (v > 1) == (g > 1)
            ok &= same and abs(v - g) <= 0.2
            parts.append(f"{name[2:]} {v:.2f}")
        lines.append(f"{LADDERS[k]['label']}: lam {g:.2f}, " + ", ".join(parts) + f" -> {'holds' if ok else 'fails'}")
    out.append(("7. Robust to task type (same direction, within 0.2 of lam's gain)", lines))
    return out


# ------------------------------------------------------------------------------------------------------------- figures

def fig_paths(stage_cells, keys):
    """sd_a against b, log-log, each ladder's path through its stages; dashed lines of constant lam."""
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    xs = np.geomspace(0.15, 12, 50)
    for lam in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5):
        ax.plot(xs, lam * xs, color="#c9c7c1", lw=0.8, ls=(0, (4, 3)), zorder=1)
        x_end, y_end = (11.5, lam * 11.5) if lam * 11.5 < 1.5 else (1.3 / lam, 1.3)
        ax.text(x_end, y_end, f"λ = {lam:g} ", color="#8a8880", fontsize=8, va="bottom", ha="right")
    for k in keys:
        st = LADDERS[k]["stages"]
        est = np.array([stage_cells[k][s][0] for s in st])
        x, y = est[:, SK["sd_a"]], est[:, SK["b"]]
        keep = y > 0
        alpha = 1.0 if k in ("olmo2_32b", "tulu3_70b", "olmo2_13b") else 0.55
        ax.plot(x[keep], y[keep], color=COLORS[k], lw=2, alpha=alpha, zorder=2)
        ax.scatter(x[keep], y[keep], color=COLORS[k], marker=MARKERS[k], s=48, edgecolor="white", linewidth=0.8,
                   alpha=alpha, zorder=3, label=LADDERS[k]["label"])
        for xi, yi, s, kp in zip(x, y, st, keep):
            if kp:  # final sits on top of DPO in every ladder that has both: put it above-left, the rest below-right
                above = s == "instruct"
                ax.annotate(STAGE_LABELS[s], (xi, yi), xytext=(-6, 6) if above else (6, -11),
                            textcoords="offset points", fontsize=7.5, color="#55534d", ha="right" if above else "left")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.15, 12)
    ax.set_ylim(0.003, 2)
    ax.set_xlabel("Assistant scale: SD of the Assistant's utilities (nats)")
    ax.set_ylabel("Leaked size b: the coin along the Assistant\n(coin nats per Assistant SD)")
    ax.set_title("Each stage's Assistant scale and leaked coin. Moving parallel to the dashed lines\n"
                 "is stronger preferences at a constant leakage rate λ; moving across them changes λ",
                 fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    return fig


def fig_gains(steps, keys):
    """Per step (columns) and ladder (rows): the gains in sd_a, lam and b, log scale, with CIs. S is printed where the
    leaked coin grows enough for it to be read (b gain's CI above B_GAIN_BAR)."""
    from matplotlib.ticker import FixedLocator, NullLocator
    step_names = ("base → SFT", "SFT → DPO (Apertus: SFT → RL)", "DPO → final (Apertus: RL → DPO)")
    measures = (("g_sd_a", "Assistant scale σ_A", plotstyle.SERIES[0]), ("g_lam", "leakage rate λ",
                plotstyle.SERIES[1]), ("g_b", "leaked coin b = σ_A·λ", "#3d3b36"))
    xmin, xmax = 0.5, 32
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 1.2 + 0.75 * len(keys)), sharey=True, sharex=True)
    for j, (title, ax) in enumerate(zip(step_names, axes)):
        for row, k in enumerate(keys):
            st = LADDERS[k]["stages"]
            s, t = list(zip(st, st[1:]))[j]
            est, boot = steps[k][(s, t)]
            lo, hi = ci(boot)
            for m, (name, _, color) in enumerate(measures):
                yv = row + (m - 1) * 0.24
                e = est[TK[name]]
                if not e > 0:
                    continue
                l, h = max(lo[TK[name]], xmin * 0.9), min(hi[TK[name]], xmax * 1.1)
                ax.plot([l, h], [yv, yv], color=color, lw=1.4, zorder=2, solid_capstyle="butt")
                ax.plot(e, yv, "o", color=color, ms=6, mec=plotstyle.SURFACE, mew=0.8, zorder=3)
            if lo[TK["g_b"]] > B_GAIN_BAR and np.isfinite(est[TK["share"]]):
                ax.text(0.98, row, f"S {est[TK['share']]:.2f}", transform=ax.get_yaxis_transform(), fontsize=8,
                        va="center", ha="right", color="#55534d")
        ax.axvline(1, color="#8a8880", lw=0.8, zorder=1)
        ax.set_xscale("log")
        ax.set_xlim(xmin, xmax)
        ax.xaxis.set_major_locator(FixedLocator([0.5, 1, 2, 4, 8, 16, 32]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels(["0.5", "1", "2", "4", "8", "16", "32"])
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlabel("gain, stage after / before (log scale)")
    axes[0].set_yticks(range(len(keys)), [LADDERS[k]["label"] for k in keys])
    axes[0].set_ylim(len(keys) - 0.5, -0.5)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, mec=plotstyle.SURFACE, ms=6, label=lab)
               for _, lab, c in measures]
    fig.legend(handles=handles, frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0.1, 1.0), fontsize=9)
    fig.text(0.1, -0.06, "On a log scale the leaked coin's gain is the sum of the other two. S = log(λ gain) / log(b gain), "
             "the share of the leaked coin's growth that is a higher rate; shown where b's gain has a CI above 1.2.",
             fontsize=8, color="#55534d")
    fig.tight_layout()
    return fig


def fig_subsets(steps, keys):
    """Each ladder's DPO step (rows), by task set (markers): the leakage rate's gain and the leakage share S, with CIs."""
    from matplotlib.ticker import FixedLocator, NullLocator
    subs = ("", "arb", "arbw", "math", "alpaca", "wildchat", "rest")
    colors = dict(zip(subs, ["#3d3b36", plotstyle.SERIES[0], "#86b6ef", plotstyle.SERIES[2], plotstyle.SERIES[3],
                             plotstyle.SERIES[6], plotstyle.SERIES[7]]))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 1.2 + 0.85 * len(keys)), sharey=True)
    for row, k in enumerate(keys):
        est, boot = steps[k][LADDERS[k]["dpo"]]
        lo, hi = ci(boot)
        for m, sub in enumerate(subs):
            yv = row + (m - (len(subs) - 1) / 2) * 0.12
            for ax, name in zip(axes, (f"g_lam{sfx(sub)}", f"share{sfx(sub)}")):
                j = TK[name]
                if not np.isfinite(est[j]):
                    continue
                l, h = lo[j], hi[j]
                lim = (0.5, 8) if name.startswith("g_") else (-0.6, 1.6)
                clipped = not (lim[0] <= l and h <= lim[1])  # unbounded where lam before the step is near 0
                l, h = max(l, lim[0]), min(h, lim[1])
                ax.plot([l, h], [yv, yv], color=colors[sub], lw=1.0 if clipped else 1.3, ls=":" if clipped else "-",
                        zorder=2, solid_capstyle="butt")
                ax.plot(est[j], yv, "o", color=colors[sub], ms=5, mec=plotstyle.SURFACE, mew=0.7, zorder=3,
                        label=SUBSET_LABELS[sub] if row == 0 and name.startswith("g_") else None)
    ax0, ax1 = axes
    ax0.axvline(1, color="#8a8880", lw=0.8, zorder=1)
    ax0.set_xscale("log")
    ax0.set_xlim(0.5, 8)
    ax0.xaxis.set_major_locator(FixedLocator([0.5, 1, 2, 4, 8]))
    ax0.xaxis.set_minor_locator(NullLocator())
    ax0.set_xticklabels(["0.5", "1", "2", "4", "8"])
    ax0.set_xlabel("leakage rate λ, gain at DPO (log scale)")
    ax0.set_title("Does DPO raise the leakage rate?", fontsize=10, loc="left")
    for x in (0, 0.25, 0.5, 0.75, 1):
        ax1.axvline(x, color="#8a8880" if x in (0, 1) else "#e2e0da", lw=0.8, zorder=1)
    ax1.set_xlim(-0.6, 1.6)
    ax1.set_xlabel("leakage share S of the coin's growth at DPO")
    ax1.set_title("How much of the coin's growth is the rate (S), not the Assistant's scale", fontsize=10, loc="left")
    ax0.set_yticks(range(len(keys)), [f"{LADDERS[k]['label']}\n({' → '.join(STAGE_LABELS[s] for s in LADDERS[k]['dpo'])})"
                                      for k in keys])
    ax0.set_ylim(len(keys) - 0.5, -0.5)
    fig.legend(*ax0.get_legend_handles_labels(), frameon=False, ncol=4, loc="lower left", bbox_to_anchor=(0.12, 1.0),
               fontsize=8.5)
    fig.text(0.12, -0.04, "Dotted: a CI cut at the axis (one origin's rate is near 0 before the step, so its gain is "
             "unbounded).", fontsize=8, color="#55534d")
    fig.tight_layout()
    return fig


def fig_scatter(U_by, keys):
    """Per ladder: A against C at the DPO step's two stages, with each stage's line (slope = lam)."""
    fig, axes = plt.subplots(1, len(keys), figsize=(4.0 * len(keys), 3.9), squeeze=False)
    for ax, k in zip(axes[0], keys):
        U = U_by[k]
        pre, post = LADDERS[k]["dpo"]
        for s, shade, z in ((pre, 0.45, 2), (post, 1.0, 3)):
            a, c = U[f"{s}/assistant"], U[f"{s}/coinflip"]
            ax.scatter(a - a.mean(), c - c.mean(), s=9, color=COLORS[k] if s == post else "#a3a19a",
                       alpha=0.55 if s == post else 0.45, edgecolor="none", zorder=z)
            lam = slope(a, c)
            xs = np.linspace((a - a.mean()).min(), (a - a.mean()).max(), 2)
            ax.plot(xs, lam * xs, color=COLORS[k] if s == post else "#8a8880", lw=2, zorder=4,
                    label=f"{STAGE_LABELS[s]}: σ_A {a.std():.2f}, λ {lam:.3f}")
        ax.axhline(0, color="#e2e0da", lw=0.8, zorder=1)
        ax.axvline(0, color="#e2e0da", lw=0.8, zorder=1)
        ax.set_title(LADDERS[k]["label"], fontsize=10, loc="left")
        ax.set_xlabel("Assistant utility (nats, centred)")
        ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    axes[0][0].set_ylabel("coinflip utility (nats, centred)")
    return fig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-missing", action="store_true", help="leave out a ladder whose runs aren't all finished")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    plotstyle.setup()

    loaded = {}
    for k in LADDERS:
        try:
            loaded[k] = load_ladder(k)
        except SystemExit as e:
            if not args.skip_missing:
                raise
            print(f"skipping {LADDERS[k]['label']}: {e}")
    keys = list(loaded)
    ids = [r["id"] for r in loaded[keys[0]][0]]
    for k in keys:
        assert [r["id"] for r in loaded[k][0]] == ids, f"{k}: task set differs"
    _, origins, benign, _ = task_arrays(loaded[keys[0]][0])
    resamples = stratified_resamples(origins, np.random.default_rng(0), N_BOOTSTRAP)

    report = []
    say = lambda s="": (print(s), report.append(s))
    say("Validation gate (test ladders; context ladders listed for reference):")
    for k in keys:
        ok, lines = gate(k, loaded[k][1], LADDERS[k]["stages"])
        for line in lines:
            say(line)
        if LADDERS[k]["test"]:
            say(f"  -> {LADDERS[k]['label']}: gate {'PASSES' if ok else 'FAILS'}")

    U_by, stage_cells, steps = {}, {}, {}
    for k in keys:
        U_by[k] = task_arrays(loaded[k][0])[0]
        stage_cells[k], steps[k] = ladder_cells(U_by[k], LADDERS[k]["stages"], benign, origins, resamples)

    stage_rows, step_rows = [], []
    for k in keys:
        for s in LADDERS[k]["stages"]:
            est, boot = stage_cells[k][s]
            lo, hi = ci(boot)
            stage_rows.append(dict(ladder=LADDERS[k]["label"], stage=s, **{
                f"{n}{suf}": v[SK[n]] for n in STAGE_STATS for suf, v in (("", est), ("_lo", lo), ("_hi", hi))}))
        for (s, t), (est, boot) in steps[k].items():
            lo, hi = ci(boot)
            step_rows.append(dict(ladder=LADDERS[k]["label"], step=f"{s}->{t}", **{
                f"{n}{suf}": v[TK[n]] for n in STEP_STATS for suf, v in (("", est), ("_lo", lo), ("_hi", hi))}))
    write_csv(OUT / "stages.csv", stage_rows)
    write_csv(OUT / "steps.csv", step_rows)

    say("\nPer stage: sd_a (Assistant scale), b (leaked coin, per Assistant SD), lam (coin nats per Assistant nat), r")
    for k in keys:
        for s in LADDERS[k]["stages"]:
            est, boot = stage_cells[k][s]
            lo, hi = ci(boot)
            g = lambda n, d=3: fmt(est[SK[n]], lo[SK[n]], hi[SK[n]], d)
            say(f"  {LADDERS[k]['label']:15} {STAGE_LABELS[s]:5}  sd_a {g('sd_a', 2)}  b {g('b')}  lam {g('lam')}  "
                f"lam_benign {est[SK['lam_benign']]:.3f}  lam_within {est[SK['lam_within']]:.3f}  "
                f"gap_ratio {est[SK['gap_ratio']]:.3f}  resid_sd {est[SK['resid_sd']]:.3f}  r {est[SK['r']]:.2f}")

    say("\nPer step: gains (after / before) and the leakage share S = log(lam gain) / log(b gain)")
    for k in keys:
        for (s, t), (est, boot) in steps[k].items():
            lo, hi = ci(boot)
            g = lambda n, d=2: fmt(est[TK[n]], lo[TK[n]], hi[TK[n]], d)
            say(f"  {LADDERS[k]['label']:15} {STAGE_LABELS[s]:>5} -> {STAGE_LABELS[t]:5}  sd_a x{g('g_sd_a')}  "
                f"b x{g('g_b')}  lam x{g('g_lam')}  S {g('share')}")
            say(f"  {'':15} {'':14}  benign: lam x{est[TK['g_lam_benign']]:.2f}, S {est[TK['share_benign']]:.2f}; "
                f"within: lam x{est[TK['g_lam_within']]:.2f}, S {est[TK['share_within']]:.2f}; gap ratio "
                f"x{est[TK['g_gap_ratio']]:.2f}; resid_sd x{est[TK['g_resid_sd']]:.2f}")
            say(f"  {'':15} {'':14}  k_a {est[TK['k_a']]:.2f} (r {est[TK['r_a']]:.3f}), k_c {est[TK['k_c']]:.2f} "
                f"(r {est[TK['r_c']]:.3f}); marginal leakage {g('marginal', 3)} (r {est[TK['r_marginal']]:.2f})")

    say("\nExploratory (after the data): lam's gain = rel_coin x align. rel_coin = resid_sd gain / sd_a gain (the coin's")
    say("part off the Assistant line growing relative to the Assistant); align = b gain / resid_sd gain (the coin following")
    say("the Assistant more closely); share_align = log(align) / log(lam gain).")
    for k in keys:
        for (s, t), (est, boot) in steps[k].items():
            lo, hi = ci(boot)
            g = lambda n, d=2: fmt(est[TK[n]], lo[TK[n]], hi[TK[n]], d)
            say(f"  {LADDERS[k]['label']:15} {STAGE_LABELS[s]:>5} -> {STAGE_LABELS[t]:5}  lam x{est[TK['g_lam']]:.2f} = "
                f"rel_coin x{g('rel_coin')} * align x{g('align')}; share_align {g('share_align')}")

    say("\nThe call on each ladder's DPO step (pre-registered for Tulu 3 70B and OLMo 2 13B; context otherwise):")
    for k in keys:
        pre, post = LADDERS[k]["dpo"]
        est, boot = steps[k][(pre, post)]
        say(f"  {LADDERS[k]['label']:15} ({STAGE_LABELS[pre]} -> {STAGE_LABELS[post]}): {call(est, boot)}"
            f"{'' if LADDERS[k]['test'] else '  [context, seen before the pre-registration]'}")

    if any(LADDERS[k]["test"] for k in keys):
        say("\nPredictions:")
        for head, lines in predictions({k: steps[k] for k in keys}, stage_cells):
            say(f"  {head}")
            for line in lines:
                say(f"    {line}")
    say("\nExploratory (Hal's request, added before any new ladder was analysed): the 225 MATH / Alpaca / WildChat tasks,")
    say("where preferences are more arbitrary than BailBench's and STRESS-TEST's. Per stage sd_a | b | lam; per step S:")
    arb_subs = ("arb", "arbw", "math", "alpaca", "wildchat", "rest")
    for k in keys:
        for s in LADDERS[k]["stages"]:
            est, boot = stage_cells[k][s]
            lo, hi = ci(boot)
            cells = []
            for sub in arb_subs:
                j = SK[f"lam_{sub}"]
                cells.append(f"{sub} {est[SK[f'sd_a_{sub}']]:.2f} | {est[SK[f'b_{sub}']]:.3f} | "
                             f"{fmt(est[j], lo[j], hi[j], 3)}")
            say(f"  {LADDERS[k]['label']:15} {STAGE_LABELS[s]:5}  " + "   ".join(cells)
                + f"   r_arb {est[SK['r_arb']]:.2f}")
        for (s, t), (est, boot) in steps[k].items():
            lo, hi = ci(boot)
            cells = []
            for sub in ("",) + arb_subs:
                g_l, sh = TK[f"g_lam{sfx(sub)}"], TK[f"share{sfx(sub)}"]
                cells.append(f"{sub or 'all'}: sd_a x{est[TK[f'g_sd_a{sfx(sub)}']]:.2f} lam x"
                             f"{fmt(est[g_l], lo[g_l], hi[g_l])} S {fmt(est[sh], lo[sh], hi[sh])}")
            say(f"  {LADDERS[k]['label']:15} {STAGE_LABELS[s]:>5} -> {STAGE_LABELS[t]:5}")
            for c_ in cells:
                say(f"      {c_}")
            say(f"      resid_sd_arb x{est[TK['g_resid_sd_arb']]:.2f}; marginal leakage (arb) {est[TK['marginal_arb']]:.3f}")
    (OUT / "report.txt").write_text("\n".join(report) + "\n")

    order = [k for k in ("olmo2_32b", "tulu3_70b", "olmo2_13b", "olmo3_32b", "apertus_70b") if k in keys]
    for name, fig in (("paths", fig_paths(stage_cells, order)), ("gains", fig_gains(steps, order)),
                      ("scatter", fig_scatter(U_by, [k for k in order if k in ("olmo2_32b", "tulu3_70b",
                                                                                  "olmo2_13b")])),
                      ("subsets", fig_subsets(steps, order))):
        path = plotstyle.save(fig, f"dpo_leak_{name}")
        plt.close(fig)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
