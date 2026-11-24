"""Replication round 2's new experiments (README, "Replication round 2", pre-registered 2026-11-23): B the tool's coin under
roleplay, C the length rules with the user-reported and the tool's coin, D the raw-text stories; and E, the new models'
reasoning check (A and P read inside the reasoning against after the empty think block; "agrees" if r >= 0.95). Findings 1-8 on the two new
models and the tool run's calls are replication_analysis.py's and tool_analysis.py's.

    uv run python round2_analysis.py [keys ...] [--n-boot 2000]

Keys: the first round's models (their round-2 results in outputs/replication/r2-<key>/) and the new ones (glm-5.3,
kimi-k3.1-pro: everything in outputs/replication/<key>/). Default: every key with round-2 results on disk.

Where each readout comes from (README, Gate):
- new models: their own run;
- first-round models: the round-2 conditions (T_rp, A_k, C_k, T_k, raw text) from r2-<key>. A_0, C_0 and T come from the
  round-1 runs (<key>, tool-<key>) if the model passes the gate (scripts/r2_gate.py: utilities r >= 0.999 and SD within 2%
  for all three), else from r2-<key>. The persona readouts round 2 doesn't re-score (A_sys, A_rp, C_sys, C_rp, T_sys) come
  from round 1 either way.

Utilities are replication_analysis.Model's (each condition's own fit, cached). CIs: 95% task bootstraps within origin with
round 1's resamples (replication_analysis.resamples_for, seed 0), so they're paired with every round-1 statistic. A statistic
whose cells fail the gate (split-half < 0.9 or median outcome mass < 0.2) is reported with its reason and gets no call.

Outputs: outputs/replication/round2/<key>/{stats.csv, calls.csv, report.txt}; outputs/replication/round2/calls.csv (all keys).
"""

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import replication_analysis as ra  # noqa: E402
from analyze import N_BOOTSTRAP, partial_r, pearson  # noqa: E402
from ifasked import agreement, fe_coef  # noqa: E402
from openers import slope_per_sd  # noqa: E402
from personas import pull_call  # noqa: E402
from personas_user import pulls  # noqa: E402
from replication import PERSONAS  # noqa: E402

OUT = Path("outputs/replication")
FIRST_ROUND = ["deepseek-v4-flash", "llama-405b", "deepseek-v3", "kimi-k2", "trinity-large", "nemotron4-340b"]
NEW = ["glm-5.3", "kimi-k3.1-pro"]
RULES = ("shorter", "longer")
STORIES = ("raw_okay", "raw_no_okay", "raw_story_ai", "raw_story_person")
GATE_R, GATE_SD = 0.999, 0.02
SEPARABLE = ra.SEPARABLE
REL_SLOPE = 0.25  # D's call: the raw-text section's retention bar, on the base-free relative slope


# ---------------------------------------------------------------------------------------------------------- readouts

def gate_passes(key):
    """Round 1's A_0, C_0 and T stand in for the re-scored ones only if all three pass (scripts/r2_gate.py)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
    from compare_engines import compare
    for cond, old in (("assistant", key), ("coinflip", key), ("coinflip_tool", f"tool-{key}")):
        new_path = OUT / f"r2-{key}" / "results" / f"instruct__{cond}.csv.gz"
        old_path = OUT / old / "results" / f"instruct__{cond}.csv.gz"
        if not new_path.exists() or not old_path.exists():
            if cond == "coinflip_tool" and not (OUT / f"tool-{key}").exists():
                continue  # Nemotron-4: no tool conditions
            return None  # not decidable yet
        c = compare(new_path, old_path)
        if not (c["r_utilities"] >= GATE_R and abs(c["sd_ratio"] - 1) <= GATE_SD):
            return False
    return True


def readouts(key):
    """({name: utilities}, {name: cell info}, origins, nonharm, loglen, note). Names: A0, C0, T; A_sys@p, A_rp@p, C_sys@p,
    C_rp@p, T_sys@p, T_rp@p; A_<rule>, C_<rule>, T_<rule>; raw_*."""
    models = {}

    def model(k):
        if k not in models:
            models[k] = ra.Model(k) if (OUT / k / "results").exists() else None
        return models[k]

    def take(src, condition):
        m = model(src)
        cell = f"instruct/{condition}"
        if m is None or cell not in m.U:
            return None
        return m.U[cell], m.cells[cell]

    new = key in NEW or not (OUT / f"r2-{key}").exists()
    r2 = key if new else f"r2-{key}"
    tool1 = key if new else f"tool-{key}"
    passes = True if new else gate_passes(key)
    base_src = {"A0": "assistant", "C0": "coinflip", "T": "coinflip_tool"}
    src = {}
    for name, cond in base_src.items():
        round1 = key if name != "T" else tool1
        src[name] = ((round1 if passes else r2), cond)
    for p in PERSONAS:
        src[f"A_sys@{p}"] = (key, f"assistant@{p}")
        src[f"A_rp@{p}"] = (key, f"assistant@{p}+roleplay")
        src[f"C_sys@{p}"] = (key, f"coinflip@{p}")
        src[f"C_rp@{p}"] = (key, f"coinflip@{p}+roleplay")
        src[f"T_sys@{p}"] = (tool1, f"coinflip_tool@{p}")
        src[f"T_rp@{p}"] = (r2, f"coinflip_tool@{p}+roleplay")
    for k in RULES:
        src[f"A_{k}"] = (r2, f"assistant_ifchoose_{k}")
        src[f"C_{k}"] = (r2, f"coinflip_ifchoose_{k}")
        src[f"T_{k}"] = (r2, f"coinflip_tool_ifchoose_{k}")
    for x in STORIES:
        src[x] = (r2, x)
    # The reasoning check (new models): A and P read inside the reasoning, against after the empty think block.
    src["P0"] = (key, "predict_user")
    src["A_inr"] = (r2, "assistant~inreasoning")
    src["P_inr"] = (r2, "predict_user~inreasoning")
    U, info = {}, {}
    for name, (s, cond) in src.items():
        got = take(s, cond)
        if got is not None:
            U[name], info[name] = got
            info[name] = {**info[name], "source": f"{s}/{cond}"}
    ref = next(m for m in models.values() if m is not None)
    loglen = np.log([len(t["prompt"]) for t in ref.tasks])  # n_chars, as ifasked.py's length rules
    note = "new model" if new else {True: "gate passes: A_0, C_0, T from round 1", False: "gate fails: A_0, C_0, T re-scored",
                                    None: "gate undecided: A_0, C_0, T from round 1"}[passes]
    return U, info, ref.origins, ref.nonharm, loglen, note


# -------------------------------------------------------------------------------------------------------- statistics

def statistics(U, origins, nonharm, loglen):
    """Every statistic the cells in U allow, as {name: value}; called on the full task set and on each resample."""
    out = {}
    gap = lambda x: float(x[nonharm].mean() - x[~nonharm].mean())
    has = lambda *names: all(n in U for n in names)
    sources = [(origins == o).astype(float) for o in np.unique(origins)[1:]]
    masks = [origins == o for o in np.unique(origins)]

    def within(x):
        y = x.astype(float).copy()
        for m in masks:
            y[m] -= y[m].mean()
        return y

    a0 = U.get("A0")
    for name, u in U.items():
        out[f"sd:{name}"], out[f"gap:{name}"] = float(u.std()), gap(u)
    # B. The tool's coin under roleplay.
    for p in PERSONAS:
        rp, sys_ = f"A_rp@{p}", f"A_sys@{p}"
        if has("A0", rp):
            out[f"B:{p}:sep_rp"] = pearson(U[rp], a0)
        if has("A0", sys_):
            out[f"B:{p}:sep_sys"] = pearson(U[sys_], a0)
        for coin in ("T", "C"):
            y = f"{coin}_rp@{p}"
            if has("A0", rp, y):
                b_def, b_per, w = pulls(U[y], a0, U[rp])
                out.update({f"B:{p}:{coin}_rp:b_def": b_def, f"B:{p}:{coin}_rp:b_per": b_per, f"B:{p}:{coin}_rp:w": w,
                            f"B:{p}:{coin}_rp:r_A_rp": pearson(U[y], U[rp])})
            ys = f"{coin}_sys@{p}"
            if has("A0", sys_, ys):
                out[f"B:{p}:{coin}_sys:w"] = pulls(U[ys], a0, U[sys_])[2]
                if has(y):  # the fixed target: the roleplay coin's pull toward the system-prompt persona, as a share
                    out[f"B:{p}:{coin}:rho_rp"] = pulls(U[y], a0, U[sys_])[1] / pulls(U[ys], a0, U[sys_])[1]
        if f"B:{p}:T_rp:w" in out and f"B:{p}:C_rp:w" in out:
            out[f"B:{p}:dw_T_minus_C"] = out[f"B:{p}:T_rp:w"] - out[f"B:{p}:C_rp:w"]
        if f"B:{p}:T_rp:w" in out and f"B:{p}:T_sys:w" in out:
            out[f"B:{p}:dw_rp_minus_sys"] = out[f"B:{p}:T_rp:w"] - out[f"B:{p}:T_sys:w"]
    # C. Length rules.
    for k in RULES:
        s = loglen if k == "longer" else -loglen
        if has("A0", f"A_{k}"):
            out[f"C:{k}:A:agree"] = agreement(U[f"A_{k}"], s)
            out[f"C:{k}:A:follow"] = out[f"C:{k}:A:agree"] - agreement(a0, s)
        for coin, default in (("C", "C0"), ("T", "T")):
            y = f"{coin}_{k}"
            if has("A0", y, default):
                out[f"C:{k}:{coin}:agree"] = agreement(U[y], s)
                out[f"C:{k}:{coin}:d_pr"] = partial_r(U[y], s, a0) - partial_r(U[default], s, a0)
        if f"C:{k}:T:d_pr" in out and f"C:{k}:C:d_pr" in out:
            out[f"C:{k}:d_pr_T_minus_C"] = out[f"C:{k}:T:d_pr"] - out[f"C:{k}:C:d_pr"]
    for coin in ("C", "T", "A"):
        if has(f"{coin}_shorter", f"{coin}_longer"):
            d = (U[f"{coin}_shorter"] - U[f"{coin}_longer"]) / 2
            out[f"C:contrast:{coin}"] = fe_coef(d, -loglen, sources)
    if "C:contrast:T" in out and "C:contrast:C" in out:
        out["C:contrast:T_minus_C"] = out["C:contrast:T"] - out["C:contrast:C"]
    # D. Raw-text stories, against the chat Assistant (A_0) and chat coin (C_0).
    if has("A0", "C0"):
        coin_slope = slope_per_sd(a0, U["C0"])
        out["D:r:C0"] = pearson(a0, U["C0"])
        for x in STORIES:
            if x in U:
                out[f"D:{x}:r"] = pearson(a0, U[x])
                out[f"D:{x}:r_within"] = pearson(within(a0), within(U[x]))
                out[f"D:{x}:rel_slope"] = slope_per_sd(a0, U[x]) / coin_slope
        for x in ("raw_story_ai", "raw_story_person"):
            for m in ("r", "rel_slope"):
                if f"D:{x}:{m}" in out and f"D:raw_no_okay:{m}" in out:
                    out[f"D:{x}:{m}_minus_no_okay"] = out[f"D:{x}:{m}"] - out[f"D:raw_no_okay:{m}"]
        for m in ("r", "rel_slope"):
            if f"D:raw_story_ai:{m}" in out and f"D:raw_story_person:{m}" in out:
                out[f"D:ai_minus_person:{m}"] = out[f"D:raw_story_ai:{m}"] - out[f"D:raw_story_person:{m}"]
    # E. The reasoning check: utilities read inside the reasoning against after the empty think block.
    for x, ref in (("A", "A0"), ("P", "P0")):
        if has(ref, f"{x}_inr"):
            out[f"E:{x}:r"] = pearson(U[ref], U[f"{x}_inr"])
            out[f"E:{x}:sd_ratio"] = float(U[f"{x}_inr"].std() / U[ref].std())
    return out


def bootstrap(U, origins, nonharm, loglen, n_boot=N_BOOTSTRAP):
    """{name: (estimate, lo, hi)}: 95% task bootstraps within origin, round 1's resamples."""
    est = statistics(U, origins, nonharm, loglen)
    names = sorted(est)
    boot = np.array([[s.get(n, np.nan) for n in names] for s in (
        statistics({k: u[idx] for k, u in U.items()}, origins[idx], nonharm[idx], loglen[idx])
        for idx in ra.resamples_for(origins, n_boot))])
    lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    return {n: (float(est[n]), float(lo[j]), float(hi[j])) for j, n in enumerate(names)}


# ------------------------------------------------------------------------------------------------------------- calls

def cell_problem(info, names):
    """Why any of the cells can't be read (missing, split-half < 0.9, median mass < 0.2), or ""."""
    bad = []
    for n in names:
        if n not in info:
            bad.append(f"{n} missing")
        elif info[n]["reliability"] < ra.MIN_RELIABILITY:
            bad.append(f"{n} split-half {info[n]['reliability']:.3f}")
        elif info[n]["median_mass"] < ra.MIN_MASS:
            bad.append(f"{n} median mass {info[n]['median_mass']:.2f}")
    return "; ".join(bad)


def three_way(lo, hi, above, below, neither):
    return above if lo > 0 else below if hi < 0 else neither


def fmt(S, name, prec=2):
    est, lo, hi = S[name]
    return f"{est:+.{prec}f} [{lo:+.{prec}f}, {hi:+.{prec}f}]"


def calls(S, info):
    """One row per call: experiment, persona or rule, measure, call, estimate with CI."""
    rows = []

    def row(exp, unit, measure, call, detail=""):
        rows.append(dict(experiment=exp, unit=unit, measure=measure, call=call, detail=detail))

    for p in PERSONAS:  # B
        problem = cell_problem(info, ["A0", f"A_rp@{p}", f"T_rp@{p}"])
        if problem:
            row("B", p, "T_rp fit on z(A_0), z(A_rp)", "not interpretable" if "missing" not in problem else "not run",
                problem)
            continue
        if abs(S[f"B:{p}:sep_rp"][0]) > SEPARABLE:
            row("B", p, "T_rp fit on z(A_0), z(A_rp)", "not separable", f"r(A_rp, A_0) {S[f'B:{p}:sep_rp'][0]:.2f}")
            continue
        g = lambda s: S[f"B:{p}:T_rp:{s}"]
        c = pull_call(g("b_def")[1], g("b_per")[1], g("b_per")[2])
        row("B", p, "T_rp fit on z(A_0), z(A_rp)", c,
            f"b_def {fmt(S, f'B:{p}:T_rp:b_def')}, b_per {fmt(S, f'B:{p}:T_rp:b_per')}, w {S[f'B:{p}:T_rp:w'][0]:.2f}")
        ok = c not in ("neither",)
        for name, label in (("dw_T_minus_C", "w_rp(T) - w_rp(C)"), ("dw_rp_minus_sys", "w_rp(T) - w_sys(T)")):
            key = f"B:{p}:{name}"
            if key in S and ok:
                row("B", p, label, three_way(S[key][1], S[key][2], "more", "less", "as much"), fmt(S, key))
    for k in RULES:  # C
        if not cell_problem(info, ["A0", f"A_{k}"]):
            row("C", k, "following: agreement(A_k) - agreement(A_0)",
                three_way(*S[f"C:{k}:A:follow"][1:], "follows", "against", "no change"), fmt(S, f"C:{k}:A:follow"))
        for coin, default in (("C", "C0"), ("T", "T")):
            problem = cell_problem(info, ["A0", f"{coin}_{k}", default])
            if problem:
                if "missing" not in problem or coin == "C":
                    row("C", k, f"d_pr ({coin})", "not interpretable" if "missing" not in problem else "not run", problem)
                continue
            key = f"C:{k}:{coin}:d_pr"
            row("C", k, f"d_pr ({coin})", three_way(S[key][1], S[key][2], "toward the rule", "away from the rule",
                                                     "no change"), fmt(S, key))
        key = f"C:{k}:d_pr_T_minus_C"
        if key in S and not cell_problem(info, [f"C_{k}", f"T_{k}", "C0", "T"]):
            row("C", k, "d_pr, tool minus user-reported", three_way(S[key][1], S[key][2], "the tool's coin follows more",
                                                                      "the tool's coin follows less", "as much"), fmt(S, key))
    for coin in ("C", "T"):
        key = f"C:contrast:{coin}"
        if key in S and not cell_problem(info, [f"{coin}_shorter", f"{coin}_longer"]):
            row("C", "both", f"mirrored length contrast ({coin})",
                three_way(S[key][1], S[key][2], "follows the length instruction", "against the length instruction",
                          "no length effect"), fmt(S, key))
    if "C:contrast:T_minus_C" in S and not cell_problem(info, [f"{c}_{r}" for c in "CT" for r in RULES]):
        key = "C:contrast:T_minus_C"
        row("C", "both", "contrast, tool minus user-reported", three_way(S[key][1], S[key][2], "the tool's coin follows more",
                                                                         "the tool's coin follows less", "as much"), fmt(S, key))
    for x in STORIES:  # D
        problem = cell_problem(info, ["A0", "C0", x])
        if problem:
            row("D", x, "relative slope", "not interpretable" if "missing" not in problem else "not run", problem)
            continue
        est, lo, hi = S[f"D:{x}:rel_slope"]
        c = ("tracks" if est >= REL_SLOPE and lo > 0 else "reversed" if est <= -REL_SLOPE and hi < 0
             else "absent" if abs(est) < REL_SLOPE else "unclear")
        row("D", x, "relative slope", c, f"{fmt(S, f'D:{x}:rel_slope')}; r(A_f, X_f) {fmt(S, f'D:{x}:r')}, within origin "
                                         f"{fmt(S, f'D:{x}:r_within')}")
    for x, ref in (("A", "A0"), ("P", "P0")):  # E
        if f"E:{x}:r" in S and not cell_problem(info, [ref, f"{x}_inr"]):
            row("E", x, "inside the reasoning vs after the empty block", "agrees" if S[f"E:{x}:r"][0] >= 0.95
                else "differs", f"r {fmt(S, f'E:{x}:r', 3)}, SD ratio {S[f'E:{x}:sd_ratio'][0]:.2f}")
    for x in ("raw_story_ai", "raw_story_person"):
        key = f"D:{x}:rel_slope_minus_no_okay"
        if key in S:
            row("D", x, "relative slope minus raw_no_okay", three_way(S[key][1], S[key][2], "above", "below", "no difference"),
                f"{fmt(S, key)}; r {fmt(S, f'D:{x}:r_minus_no_okay')}")
    return rows


# ----------------------------------------------------------------------------------------------------------- figures

LABEL = {"deepseek-v4-flash": "DeepSeek V4 Flash", "llama-405b": "Llama 3.1 405B", "deepseek-v3": "DeepSeek V3",
         "kimi-k2": "Kimi K2", "trinity-large": "Trinity Large Thinking", "nemotron4-340b": "Nemotron-4 340B",
         "glm-5.3": "GLM-5.3", "kimi-k3.1-pro": "Kimi K3.1 Pro"}


def read_stats(key):
    path = OUT / "round2" / key / "stats.csv"
    if not path.exists():
        return {}
    with open(path) as f:
        return {r["stat"]: (float(r["estimate"]), float(r["lo"]), float(r["hi"])) for r in csv.DictReader(f)}


def _dots(ax, rows, series, x_label, ref=None, ref_label=None):
    """A dot-and-CI panel: rows [(label, {series: stat tuple or None})] top to bottom; one color per series, offset."""
    import matplotlib.pyplot as plt  # noqa: F401
    from matskit import plotstyle
    from report import INK_2, RULE, style
    n = len(series)
    for i, (label, vals) in enumerate(rows):
        for j, (name, color) in enumerate(series):
            v = vals.get(name)
            if v is None:
                continue
            y = -i - (j - (n - 1) / 2) * 0.22  # the first series on top, as in the legend
            ax.plot([v[1], v[2]], [y, y], color=color, lw=1.8, solid_capstyle="round", zorder=2)
            ax.plot(v[0], y, "o", ms=6, color=color, mec=plotstyle.SURFACE, mew=1.2, zorder=3)
    ax.set_yticks([-i for i in range(len(rows))], [r[0] for r in rows], fontsize=8)
    ax.set_ylim(-len(rows) + 0.4, 0.6)
    ax.axvline(0, color=RULE, lw=0.8, zorder=0)
    if ref is not None:
        for r in (ref if isinstance(ref, (list, tuple)) else [ref]):
            ax.axvline(r, color=INK_2, lw=1, ls=(0, (3, 3)), zorder=1)
    ax.set_xlabel(x_label, fontsize=8.5)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", visible=False)
    style(ax)


def plot_cross_model(keys, outdir=Path("figures")):
    """One figure per experiment across models (estimates with 95% CIs): replication2_tool_roleplay.png (B),
    replication2_length.png (C), replication2_stories.png (D)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matskit import plotstyle
    from report import INK
    plotstyle.setup()
    stats = {k: read_stats(k) for k in keys}
    keys = [k for k in keys if stats[k]]
    if not keys:
        return
    blue, orange, green, yellow = plotstyle.SERIES[:4]

    def legend(fig, items, title):
        handles = [Line2D([], [], ls="none", marker="o", ms=7, color=c, label=lab) for lab, c in items]
        fig.suptitle(title, fontsize=10.5, color=INK, x=0.01, ha="left", y=0.995)
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.965), ncol=len(items), frameon=False,
                   fontsize=8.5)

    # B. The tool's coin under roleplay: w_rp for the tool's and the user-reported coin, and the tool's b_per.
    rows_w, rows_b = [], []
    for k in keys:
        S = stats[k]
        for p in PERSONAS:
            if f"B:{p}:T_rp:w" not in S:
                continue
            lab = f"{LABEL.get(k, k)}: {p}"
            rows_w.append((lab, {"T": S.get(f"B:{p}:T_rp:w"), "C": S.get(f"B:{p}:C_rp:w")}))
            rows_b.append((lab, {"T": S.get(f"B:{p}:T_rp:b_per"), "C": S.get(f"B:{p}:C_rp:b_per")}))
    if rows_w:
        fig, axes = plt.subplots(1, 2, figsize=(11, 0.32 * len(rows_w) + 1.8), sharey=True)
        series = [("T", blue), ("C", orange)]
        _dots(axes[0], rows_b, series, "b_per: pull toward the roleplay Assistant (coin nats per SD)")
        _dots(axes[1], [(l, {s: (v if v is None or abs(v[0]) < 5 else None) for s, v in d.items()}) for l, d in rows_w],
              series, "w_rp = b_per / (b_def + b_per)", ref=[0, 1])
        legend(fig, [("Tool reports the flip (T_rp)", blue), ("User reports the flip (C_rp)", orange)],
               "The coin under the roleplay persona: fit on the default and the roleplay Assistant")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        plotstyle.save(fig, "replication2_tool_roleplay", outdir)

    # C. Length rules: following (A), and each coin's move toward the rule (d_pr), per rule.
    rows_f, rows_d, rows_c = [], [], []
    for k in keys:
        S = stats[k]
        for rule in RULES:
            if f"C:{rule}:A:follow" not in S:
                continue
            lab = f"{LABEL.get(k, k)}: {rule}"
            rows_f.append((lab, {"A": S.get(f"C:{rule}:A:follow")}))
            rows_d.append((lab, {"C": S.get(f"C:{rule}:C:d_pr"), "T": S.get(f"C:{rule}:T:d_pr")}))
        if "C:contrast:C" in S:
            rows_c.append((LABEL.get(k, k), {"C": S.get("C:contrast:C"), "T": S.get("C:contrast:T"),
                                              "A": S.get("C:contrast:A")}))
    if rows_f:
        fig, axes = plt.subplots(1, 2, figsize=(11, 0.34 * len(rows_f) + 1.8), sharey=True)
        _dots(axes[0], rows_f, [("A", green)], "Assistant following: agreement with the rule minus the default's")
        _dots(axes[1], rows_d, [("C", orange), ("T", blue)], "coin toward the rule: d_pr (partial r given A_0, minus the default coin's)")
        legend(fig, [("Assistant's own choice", green), ("User reports the flip", orange), ("Tool reports the flip", blue)],
               "\"If I ask you to choose between tasks in this message, choose the one with the shorter / longer description.\"")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        plotstyle.save(fig, "replication2_length", outdir)

    # D. Raw-text stories: relative slope against the chat coin, and r with the chat Assistant.
    names = [("raw_okay", "first person, \"Okay, \"", blue), ("raw_no_okay", "first person, no \"Okay, \"", green),
             ("raw_story_ai", "story: AI assistant", orange), ("raw_story_person", "story: freelancer", yellow)]
    rows_s, rows_r = [], []
    for k in keys:
        S = stats[k]
        if not any(f"D:{x}:rel_slope" in S for x, _, _ in names):
            continue
        rows_s.append((LABEL.get(k, k), {x: S.get(f"D:{x}:rel_slope") for x, _, _ in names}))
        rows_r.append((LABEL.get(k, k), {x: S.get(f"D:{x}:r") for x, _, _ in names}))
    if rows_s:
        fig, axes = plt.subplots(1, 2, figsize=(11, 0.62 * len(rows_s) + 1.9), sharey=True)
        series = [(x, c) for x, _, c in names]
        _dots(axes[0], rows_s, series, "relative slope: slope on the chat Assistant / the chat coin's", ref=[0.25, 1])
        _dots(axes[1], rows_r, series, "r with the chat Assistant's utilities")
        legend(fig, [(lab, c) for _, lab, c in names], "The coinflip as raw text, against the chat Assistant (no base)")
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        plotstyle.save(fig, "replication2_stories", outdir)


# -------------------------------------------------------------------------------------------------------------- main

def analyze(key, n_boot=N_BOOTSTRAP):
    U, info, origins, nonharm, loglen, note = readouts(key)
    S = bootstrap(U, origins, nonharm, loglen, n_boot)
    rows = [{"model": key, **r} for r in calls(S, info)]
    out = OUT / "round2" / key
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "stats.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stat", "estimate", "lo", "hi"])
        w.writerows([n, *v] for n, v in sorted(S.items()))
    with open(out / "calls.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "experiment", "unit", "measure", "call", "detail"])
        w.writeheader()
        w.writerows(rows)
    lines = [f"{key}: {note}; {len(U)} readouts"]
    lines += [f"  {n}: {info[n]['source']}, split-half {info[n]['reliability']:.3f}, mass {info[n]['median_mass']:.2f}, "
              f"SD {U[n].std():.2f}" for n in U]
    lines += [f"  {r['experiment']} {r['unit']:10} {r['measure']:45} {r['call']:32} {r['detail']}" for r in rows]
    text = "\n".join(lines)
    (out / "report.txt").write_text(text + "\n")
    print(text)
    return S, rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("keys", nargs="*")
    ap.add_argument("--n-boot", type=int, default=N_BOOTSTRAP)
    a = ap.parse_args(argv)
    keys = a.keys or [k for k in FIRST_ROUND if (OUT / f"r2-{k}" / "results").exists()] + \
        [k for k in NEW if (OUT / k / "results").exists()]
    every = []
    for key in keys:
        every += analyze(key, a.n_boot)[1]
    plot_cross_model([k for k in FIRST_ROUND + NEW if (OUT / "round2" / k / "stats.csv").exists()])
    if every and not a.keys:
        with open(OUT / "round2" / "calls.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(every[0]))
            w.writeheader()
            w.writerows(every)


if __name__ == "__main__":
    main()
