"""The coin as a tool call (README, "The coin as a tool call"): the pre-registered measures for each model, with CIs, calls,
the on-policy check and figures.

Reads the tool run, outputs/replication/tool-<key>/ (score_replication.py on configs/replication/tool/<key>.yaml), and the
model's replication run, outputs/replication/<key>/, for the Assistant (A_0, A_sys, A_self), the user-reported coins (C_0,
C_sys, C_user, C_self) and the prediction of the user (P_self). Cells, gates and resamples are replication_analysis.py's:
the same 2,000 task resamples within origin (seed 0), so every difference with a replication cell is paired.

Names (README): T the tool's coin; Cd / Ad the user-reported coin / the Assistant's own choice with the tool declared but
unused; per persona p, T_sys@p, T_user@p, T_self@p (and C_..., A_sys, A_self, P_self from the replication).

Statistics (with "|within" and "|benign_sources" versions):
- r:X~A0 and slope:X~A0 (nats per A_0 SD) for X in T, Cd, C0; d_decl = r:T~A0 - r:Cd~A0, d_0 = r:T~A0 - r:C0~A0,
  d_block = r:Cd~A0 - r:C0~A0; ratio:T/Cd, ratio:T/C0, ratio:Cd/C0 (slope ratios);
- r:Ad~A0, sdratio:Ad/A0, r:T~Ad, r:Cd~Ad;
- per persona: fit:<cell>:{b_def,b_per,w,total} (X = c + b_def z(A_0) + b_per z(A_sys)) for T_sys, T_user, T_self, C_sys,
  C_user, C_self; rho_self:X@p and rho_user:X@p (b_per of the placement over b_per of the system prompt) for X in T, C;
  drho_self@p = rho_self:T - rho_self:C, drho_user@p likewise; dw_sys@p = w(T_sys) - w(C_sys); sep_sys@p = r(A_sys, A_0);
  Delta_X@p = r(X_self, P_self) - r(X_self, A_self), dDelta@p = Delta_T - Delta_C, sep_self@p = r(A_self, P_self).

Calls (pre-registered): 1 per model, on d_decl: "stronger" (CI above 0), "weaker" (below), "same"; 2: "unchanged" if
r(Ad, A0) >= 0.95, else "changed"; 3 per persona, on drho_self: "less" (CI below 0), "more" (above), "no difference",
"not separable" (|r(A_sys, A_0)| >= 0.9) or "unstable" (the CI of b_per(T_sys) or b_per(C_sys) includes 0). A cell failing
split-half 0.9 or median mass 0.2 makes its call "not interpretable"; a missing one, "not run".

Outputs:
  outputs/replication/tool-<key>/analysis/stats.csv, calls.csv, onpolicy.csv, report.txt
  outputs/replication/tool_summary.csv     every model's calls and key numbers
  figures/<key>/tool_coin.png              C_0, C_decl and T against A_0, and A_decl against A_0
  figures/<key>/tool_personas.png          rho_user and rho_self of the tool's coin against the user-reported coin
  figures/replication_tool.png             across models: r(coin, A_0) for the three coins, and drho_self per persona

Usage: uv run python tool_analysis.py [key ...] [--n-boot 2000]
"""

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):  # as replication_analysis.py
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import replication_analysis as ra  # noqa: E402
from analyze import N_BOOTSTRAP, pearson  # noqa: E402
from matskit import plotstyle  # noqa: E402
from openers import slope_per_sd, write_csv  # noqa: E402
from replication import PERSONAS  # noqa: E402
from report import INK, INK_2, RULE, origin_scatter, style  # noqa: E402

KEYS = ("llama-405b", "deepseek-v3", "kimi-k2", "deepseek-v4-flash", "trinity-large")
LABEL = {"llama-405b": "Llama 3.1 405B", "deepseek-v3": "DeepSeek V3", "kimi-k2": "Kimi K2",
         "deepseek-v4-flash": "DeepSeek V4 Flash", "trinity-large": "Trinity Large Thinking"}
R_UNCHANGED = 0.95
SUFFIXES = ra.FIT_SUFFIXES
PLAIN = {"T": "coinflip_tool", "Cd": "coinflip_tools_declared", "Ad": "assistant_tools_declared", "A0": "assistant",
         "C0": "coinflip"}


def persona_cells(p):
    return {f"T_sys@{p}": f"coinflip_tool@{p}", f"T_user@{p}": f"coinflip_tool@{p}+user",
            f"T_self@{p}": f"coinflip_tool@{p}+self", f"C_sys@{p}": f"coinflip@{p}", f"C_user@{p}": f"coinflip@{p}+user",
            f"C_self@{p}": f"coinflip@{p}+self", f"A_sys@{p}": f"assistant@{p}", f"A_self@{p}": f"assistant@{p}+self",
            f"P_self@{p}": f"predict_user@{p}+self"}


class Data:
    """One model's cells under short names, from its tool run and its replication run."""

    def __init__(self, key, root=None):
        rep, tool = ra.Model(key, root), ra.Model(f"tool-{key}", root)
        assert rep.ids == tool.ids, "the two runs' task sets differ"
        self.key, self.label = key, LABEL.get(key, rep.label)
        self.ids, self.origins, self.nonharm = rep.ids, rep.origins, rep.nonharm
        self.dir = tool.dir
        self.U, self.cells, self.pilots = {}, {}, set()
        names = dict(PLAIN)
        for p in PERSONAS:
            names.update(persona_cells(p))
        for short, cond in names.items():
            src = tool if "tool" in cond else rep
            k = f"instruct/{cond}"
            if k in src.U:
                self.U[short], self.cells[short] = src.U[k], src.cells[k]
            elif f"instruct/{cond}" in src.pilots:
                self.pilots.add(short)

    def personas(self):
        return [p for p in PERSONAS if f"T_sys@{p}" in self.U]


def gate(d, cells):
    """replication_analysis.gate on short names."""
    missing = [c for c in cells if c not in d.U]
    if missing:
        return "not run", "missing " + ", ".join(c + (" (pilot only)" if c in d.pilots else "") for c in missing)
    bad = []
    for c in cells:
        info = d.cells[c]
        if info["reliability"] < ra.MIN_RELIABILITY:
            bad.append(f"{c} split-half {info['reliability']:.3f}")
        if info["median_mass"] < ra.MIN_MASS:
            bad.append(f"{c} median mass {info['median_mass']:.2f}")
    return ("not interpretable", "; ".join(bad)) if bad else (None, "")


# ----------------------------------------------------------------------------------------------------------- statistics

def statistics(U, origins, nonharm, personas=PERSONAS):
    ctx = ra._Ctx(U, origins, nonharm)
    b = ctx.bsrc
    out = {}
    has = lambda *ks: all(k in U for k in ks)

    def r(name, ky, kx):
        if has(ky, kx):
            out[name] = pearson(U[ky], U[kx])
            out[name + "|within"] = pearson(ctx.W[ky], ctx.W[kx])
            out[name + "|benign_sources"] = pearson(U[ky][b], U[kx][b])

    def slope(name, ky, kx):
        if has(ky, kx):
            out[name] = slope_per_sd(U[kx], U[ky])
            out[name + "|within"] = slope_per_sd(ctx.W[kx], ctx.W[ky])
            out[name + "|benign_sources"] = slope_per_sd(U[kx][b], U[ky][b])

    for k, u in U.items():
        out[f"sd:{k}"] = float(u.std())
        out[f"gap:{k}"] = float(u[nonharm].mean() - u[~nonharm].mean())
    for c in ("T", "Cd", "C0"):
        r(f"r:{c}~A0", c, "A0")
        slope(f"slope:{c}~A0", c, "A0")
    ra._diff(out, "d_decl", "r:T~A0", "r:Cd~A0", SUFFIXES)
    ra._diff(out, "d_0", "r:T~A0", "r:C0~A0", SUFFIXES)
    ra._diff(out, "d_block", "r:Cd~A0", "r:C0~A0", SUFFIXES)
    for num, den in (("T", "Cd"), ("T", "C0"), ("Cd", "C0")):
        ra._ratio(out, f"ratio:{num}/{den}", f"slope:{num}~A0", f"slope:{den}~A0")
    r("r:Ad~A0", "Ad", "A0")
    if has("Ad", "A0"):
        out["sdratio:Ad/A0"] = float(U["Ad"].std() / U["A0"].std())
    r("r:T~Ad", "T", "Ad")
    r("r:Cd~Ad", "Cd", "Ad")

    for p in personas:
        if not has("A0", f"A_sys@{p}"):
            continue
        a0, ap = U["A0"], U[f"A_sys@{p}"]
        out[f"sep_sys@{p}"] = pearson(ap, a0)
        for c in ("T_sys", "T_user", "T_self", "C_sys", "C_user", "C_self"):
            if f"{c}@{p}" in U:
                ra._fit(out, f"fit:{c}@{p}", U[f"{c}@{p}"], a0, ap, ctx)
        for x in "TC":
            for place in ("self", "user"):
                ra._ratio(out, f"rho_{place}:{x}@{p}", f"fit:{x}_{place}@{p}:b_per", f"fit:{x}_sys@{p}:b_per")
        for place in ("self", "user"):
            ra._diff(out, f"drho_{place}@{p}", f"rho_{place}:T@{p}", f"rho_{place}:C@{p}", SUFFIXES)
        ra._diff(out, f"dw_sys@{p}", f"fit:T_sys@{p}:w", f"fit:C_sys@{p}:w", SUFFIXES)
        if has(f"A_self@{p}", f"P_self@{p}"):
            out[f"sep_self@{p}"] = pearson(U[f"A_self@{p}"], U[f"P_self@{p}"])
            for x in "TC":
                if f"{x}_self@{p}" in U:
                    r(f"r:{x}_self~P_self@{p}", f"{x}_self@{p}", f"P_self@{p}")
                    r(f"r:{x}_self~A_self@{p}", f"{x}_self@{p}", f"A_self@{p}")
                    ra._diff(out, f"Delta_{x}@{p}", f"r:{x}_self~P_self@{p}", f"r:{x}_self~A_self@{p}", SUFFIXES)
            ra._diff(out, f"dDelta@{p}", f"Delta_T@{p}", f"Delta_C@{p}", SUFFIXES)
    return out


def bootstrap(U, origins, nonharm, resamples, personas=PERSONAS):
    """{name: (estimate, ci_low, ci_high)}, as replication_analysis.bootstrap with this section's statistics."""
    est = statistics(U, origins, nonharm, personas)
    names = list(est)
    boot = np.full((len(resamples), len(names)), np.nan)
    for k, i in enumerate(resamples):
        s = statistics({c: u[i] for c, u in U.items()}, origins[i], nonharm[i], personas)
        boot[k] = [s.get(n, np.nan) for n in names]
    with np.errstate(invalid="ignore"):
        lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    return {n: (float(est[n]), float(lo[j]), float(hi[j])) for j, n in enumerate(names)}


# ---------------------------------------------------------------------------------------------------------------- calls

def _ci(S, name, prec=2):
    return ra._ci(S, name, prec) if name in S else "n/a"


def three_way(est_lo_hi, below, above, neither):
    _, lo, hi = est_lo_hi
    return below if hi < 0 else above if lo > 0 else neither


def calls(S, d):
    rows = []
    status, reason = gate(d, ["T", "Cd", "A0"])
    if status:
        rows.append(dict(call_no=1, persona="", call=status, key_number="", detail=reason))
    else:
        rows.append(dict(
            call_no=1, persona="", call=three_way(S["d_decl"], "weaker", "stronger", "same"),
            key_number=f"Δr_decl {S['d_decl'][0]:+.2f}",
            detail=f"r(T, A_0) {_ci(S, 'r:T~A0')}; r(C_decl, A_0) {_ci(S, 'r:Cd~A0')}; r(C_0, A_0) {_ci(S, 'r:C0~A0')}; "
                   f"Δr_decl {_ci(S, 'd_decl')}; Δr_0 {_ci(S, 'd_0')}; slope T / C_decl {_ci(S, 'ratio:T/Cd')}, "
                   f"T / C_0 {_ci(S, 'ratio:T/C0')}"))
    status, reason = gate(d, ["Ad", "A0"])
    if status:
        rows.append(dict(call_no=2, persona="", call=status, key_number="", detail=reason))
    else:
        rows.append(dict(
            call_no=2, persona="", call="unchanged" if S["r:Ad~A0"][0] >= R_UNCHANGED else "changed",
            key_number=f"r(A_decl, A_0) {S['r:Ad~A0'][0]:.3f}",
            detail=f"r(A_decl, A_0) {_ci(S, 'r:Ad~A0', 3)}; SD ratio {_ci(S, 'sdratio:Ad/A0')}; r(T, A_decl) "
                   f"{_ci(S, 'r:T~Ad')}; r(C_decl, A_decl) {_ci(S, 'r:Cd~Ad')}; tool block on the user-reported coin: "
                   f"r(C_decl, A_0) - r(C_0, A_0) {_ci(S, 'd_block')}, slope C_decl / C_0 {_ci(S, 'ratio:Cd/C0')}"))
    for p in d.personas():
        status, reason = gate(d, ["A0", f"A_sys@{p}", f"T_sys@{p}", f"T_self@{p}", f"C_sys@{p}", f"C_self@{p}"])
        if status:
            rows.append(dict(call_no=3, persona=p, call=status, key_number="", detail=reason))
            continue
        sep = S[f"sep_sys@{p}"][0]
        unstable = [c for c in ("T_sys", "C_sys") if S[f"fit:{c}@{p}:b_per"][1] <= 0 <= S[f"fit:{c}@{p}:b_per"][2]]
        call = ("not separable" if abs(sep) >= ra.SEPARABLE else "unstable" if unstable
                else three_way(S[f"drho_self@{p}"], "less", "more", "no difference"))
        secondary = ""
        if f"dDelta@{p}" in S:
            secondary = (f"; secondary: Δ_T {_ci(S, f'Delta_T@{p}')}, Δ_C {_ci(S, f'Delta_C@{p}')}, Δ_T - Δ_C "
                         f"{_ci(S, f'dDelta@{p}')}, r(A_self, P_self) {S[f'sep_self@{p}'][0]:.2f}"
                         + ("" if abs(S[f"sep_self@{p}"][0]) < ra.SEPARABLE else " (not separable)"))
        rows.append(dict(
            call_no=3, persona=p, call=call, key_number=f"Δρ_self {S[f'drho_self@{p}'][0]:+.2f}",
            detail=f"ρ_self T {_ci(S, f'rho_self:T@{p}')} vs C {_ci(S, f'rho_self:C@{p}')}, Δρ_self "
                   f"{_ci(S, f'drho_self@{p}')}; ρ_user T {_ci(S, f'rho_user:T@{p}')} vs C {_ci(S, f'rho_user:C@{p}')}, "
                   f"Δρ_user {_ci(S, f'drho_user@{p}')}; b_per(sys) T {_ci(S, f'fit:T_sys@{p}:b_per')}, C "
                   f"{_ci(S, f'fit:C_sys@{p}:b_per')}; w(sys) T {_ci(S, f'fit:T_sys@{p}:w')} vs C "
                   f"{_ci(S, f'fit:C_sys@{p}:w')}, Δw {_ci(S, f'dw_sys@{p}')}; r(A_sys, A_0) {sep:+.2f}"
                   + (f"; unstable: {', '.join(unstable)} b_per CI includes 0" if unstable else "") + secondary))
    return rows


def onpolicy(d):
    """Per tool-call condition: replies, calls, calls before any text (score_replication.py --onpolicy)."""
    rows = []
    for path in sorted((d.dir / "onpolicy").glob("instruct__*.jsonl")):
        lines = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        rows.append(dict(condition=lines[0]["condition"], replies=len(lines), calls=sum(x["calls"] for x in lines),
                         call_first=sum(x["call_first"] for x in lines)))
    return rows


# -------------------------------------------------------------------------------------------------------------- figures

COIN_STYLE = {"C0": (plotstyle.SERIES[1], "User reports the flip"), "Cd": (plotstyle.SERIES[2], "User reports, tool declared"),
              "T": (plotstyle.SERIES[0], "Tool reports the flip")}


def _r_title(S, name):
    return f"r = {S[name][0]:.2f} [{S[name][1]:.2f}, {S[name][2]:.2f}]" if name in S else ""


def plot_coin(d, S, path):
    panels = [(c, "A0", label, f"r:{c}~A0") for c, (_, label) in COIN_STYLE.items()] + [
        ("Ad", "A0", "Assistant's choice, tool declared", "r:Ad~A0")]
    panels = [p for p in panels if p[0] in d.U and p[1] in d.U]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(3.9 * len(panels), 4.0), squeeze=False)
    for ax, (y, x, label, name) in zip(axes[0], panels):
        origin_scatter(ax, d.U[x], d.U[y], d.origins)
        ax.set_title(f"{label}\n{_r_title(S, name)}", fontsize=9.5, color=INK, loc="left")
        ax.set_xlabel("Assistant utility (nats)", fontsize=9, color=INK_2)
        ax.set_ylabel(f"{'Assistant' if y == 'Ad' else 'coin'} utility (nats)", fontsize=9, color=INK_2)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, frameon=False, ncol=len(names), loc="lower center", fontsize=9)
    fig.suptitle(f"{d.label}: the coin when the user reports the flip and when a tool does, against the Assistant's own "
                 "choice (each panel has its own scale)", color=INK, fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    plotstyle.save(fig, path.stem, path.parent)


RHO_LIM = (-1.0, 3.0)  # ratios with a near-zero denominator run off this range; arrows mark CIs past it
PERSONA_COLOR = {p: plotstyle.SERIES[3 + i] for i, p in enumerate(PERSONAS)}  # apart from the coins' colors


def plot_personas(d, S, path):
    personas = [p for p in d.personas() if f"rho_self:T@{p}" in S]
    if not personas:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.55 * len(personas) + 1.8), sharey=True)
    for ax, place, title in zip(axes, ("user", "self"), ("User's instruction: ρ_user", "User's self-description: ρ_self")):
        for k, (x, (color, label)) in enumerate((("C", COIN_STYLE["C0"]), ("T", COIN_STYLE["T"]))):
            for yi, p in enumerate(personas):
                name = f"rho_{place}:{x}@{p}"
                if name not in S:
                    continue
                est, lo, hi = S[name]
                y = yi + (k - 0.5) * 0.28
                ax.plot([max(lo, RHO_LIM[0]), min(hi, RHO_LIM[1])], [y, y], color=color, lw=2, solid_capstyle="round")
                for edge, marker, off in ((RHO_LIM[0], "<", lo < RHO_LIM[0]), (RHO_LIM[1], ">", hi > RHO_LIM[1])):
                    if off:  # the CI runs past the axis
                        ax.plot(edge, y, marker, ms=6, color=color)
                if RHO_LIM[0] <= est <= RHO_LIM[1]:
                    ax.plot(est, y, "o", ms=7, color=color, mec="white", mew=1.2, label=label if yi == 0 else None)
        ax.set_xlim(RHO_LIM[0] - 0.1, RHO_LIM[1] + 0.1)
        ax.axvline(0, color=RULE, lw=0.8)
        ax.axvline(1, color=RULE, lw=0.8, ls=":")
        ax.set_title(f"{title}\n(b_per of the placement / b_per of the system prompt)", fontsize=9.5, color=INK, loc="left")
        style(ax)
    axes[0].set_yticks(range(len(personas)), personas)
    axes[0].invert_yaxis()
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle(f"{d.label}: how much each placement pulls the coin, relative to the persona as system prompt "
                 f"(95% CIs; arrows: past {RHO_LIM[0]:g} to {RHO_LIM[1]:g})",
                 color=INK, fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    plotstyle.save(fig, path.stem, path.parent)


def plot_models(results, path):
    keys = [k for k in KEYS if k in results]
    if not keys:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.75 * len(keys) + 1.8), sharey=True)
    ax = axes[0]
    for ci, (c, (color, label)) in enumerate(COIN_STYLE.items()):
        for yi, k in enumerate(keys):
            S = results[k]
            if f"r:{c}~A0" not in S:
                continue
            est, lo, hi = S[f"r:{c}~A0"]
            y = yi + (ci - 1) * 0.22
            ax.plot([lo, hi], [y, y], color=color, lw=2, solid_capstyle="round")
            ax.plot(est, y, "o", ms=7, color=color, mec="white", mew=1.2, label=label if yi == 0 else None)
    ax.axvline(0, color=RULE, lw=0.8)
    ax.set_title("r(coin, Assistant's own choice)", fontsize=10, color=INK, loc="left")
    style(ax)
    ax = axes[1]
    for pi, p in enumerate(PERSONAS):
        color = PERSONA_COLOR[p]
        for yi, k in enumerate(keys):
            S = results[k]
            if f"drho_self@{p}" not in S:
                continue
            est, lo, hi = S[f"drho_self@{p}"]
            y = yi + (pi - 1.5) * 0.17
            ax.plot([lo, hi], [y, y], color=color, lw=2, solid_capstyle="round")
            ax.plot(est, y, "o", ms=6, color=color, mec="white", mew=1.2, label=p if yi == 0 else None)
    ax.axvline(0, color=RULE, lw=0.8)
    ax.set_title("Δρ_self: the self-description's pull on the tool's coin\nminus on the user-reported coin",
                 fontsize=10, color=INK, loc="left")
    style(ax)
    axes[0].set_yticks(range(len(keys)), [LABEL[k] for k in keys])
    axes[0].invert_yaxis()
    for ax, ncol, x in ((axes[0], 3, 0.27), (axes[1], 4, 0.75)):
        fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=ncol, frameon=False, fontsize=8.5,
                   bbox_to_anchor=(x, 0.0))
    fig.suptitle("The coin as a tool call (95% task-bootstrap CIs)", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    plotstyle.save(fig, path.stem, path.parent)


# ----------------------------------------------------------------------------------------------------------------- main

def analyze(key, n_boot=N_BOOTSTRAP, figures=True, root=None, fig_root=None):
    d = Data(key, root)
    resamples = ra.resamples_for(d.origins, n_boot)
    S = bootstrap(d.U, d.origins, d.nonharm, resamples, d.personas())
    rows = calls(S, d)
    pol = onpolicy(d)
    out = d.dir / "analysis"
    write_csv(out / "stats.csv", [dict(stat=n, estimate=v[0], ci_low=v[1], ci_high=v[2]) for n, v in S.items()])
    write_csv(out / "calls.csv", rows)
    if pol:
        write_csv(out / "onpolicy.csv", pol)
    cells = [dict(cell=c, reliability=i["reliability"], median_mass=i["median_mass"], sd=i["sd"], slot_bias=i["slot_bias"])
             for c, i in d.cells.items()]
    write_csv(out / "cells.csv", cells)
    lines = [f"# {d.label}: the coin as a tool call", ""]
    lines += [f"call {r['call_no']}{' ' + r['persona'] if r['persona'] else ''}: {r['call']} ({r['key_number']}). "
              f"{r['detail']}" for r in rows]
    lines += ["", "on-policy (greedy replies that call flip_coin / that call it before any text):"]
    lines += [f"  {r['condition']}: {r['calls']}/{r['replies']}, first {r['call_first']}/{r['replies']}" for r in pol]
    lines += ["", "cells (split-half reliability, median outcome mass, SD):"]
    lines += [f"  {c['cell']}: {c['reliability']:.3f}, {c['median_mass']:.3f}, {c['sd']:.2f}" for c in cells]
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if figures:
        fig_root = Path(fig_root or ra.FIGURES)
        plot_coin(d, S, fig_root / key / "tool_coin.png")
        plot_personas(d, S, fig_root / key / "tool_personas.png")
    return S, rows, pol


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("keys", nargs="*", default=list(KEYS))
    ap.add_argument("--n-boot", type=int, default=N_BOOTSTRAP)
    a = ap.parse_args(argv)
    results, summary = {}, []
    for key in a.keys:
        if not (ra.OUT / f"tool-{key}" / "results").exists():
            print(f"{key}: no tool results yet")
            continue
        S, rows, pol = analyze(key, a.n_boot)
        results[key] = S
        plain = next((r for r in pol if r["condition"] == "coinflip_tool"), None)
        for r in rows:
            summary.append(dict(model=key, **r, onpolicy_calls=f"{plain['calls']}/{plain['replies']}" if plain else ""))
    if summary:
        write_csv(ra.OUT / "tool_summary.csv", summary)
        plot_models(results, ra.FIGURES / "replication_tool.png")


if __name__ == "__main__":
    main()
