"""The cross-model replication's analysis (README "Replication across larger models"): every finding's pre-registered call,
per model, with its CIs, gates, breakdowns and figures.

Reads outputs/replication/<key>/results/<stage>__<condition>.csv.gz, written by score_replication.py (and, for the earlier
Llama 70B and Gemma 4 31B runs, assembled by scripts/replication_legacy.py). A missing condition gives "not run", not an
error, so this runs on a model at any point of its run.

Readouts (README notation): A the Assistant's own choice, C the coin, U the user's choice, P the Assistant's prediction of the
user's choice; _b base, _f final, and A_0 = A_f. Per persona p, at the final stage: A_sys, C_sys with the persona as the system
prompt (`@p`; A_sys is the README's A_p); A_user, C_user as the user's instruction (`@p+user`); A, C, U, P_self as the user's
self-description (`@p+self`); A_roleplay, C_roleplay framed as roleplay (`@p+roleplay`; the README's A_r, C_r).

Utilities are each condition's fit on its own pairs (analyze.fit_utilities), with analyze.py's split-half reliability (the
same 50 splits), median outcome mass and slot bias; cached per results file in analysis/cells/.

Statistics, all with 95% task bootstraps within origin, one set of 2,000 resamples per model (report.stratified_resamples,
seed 0, as in every earlier section, so their CIs reproduce and differences here are paired):
- correlations, named "r:<y>~<x>", each over all 300 tasks, within origin ("|within": each source's mean removed), over the
  benign sources together ("|benign_sources": MATH, Alpaca, WildChat), over the non-harmful tasks ("|non_harmful", the usual
  257) and within each source ("|src:<SOURCE>"); "shift:<Y>~A" is r(A_f - A_b, Y_f - Y_b);
- the persona sections' fit X = c + b_def z(A_0) + b_per z(A_sys), w = b_per / (b_def + b_per) (personas_user.pulls), for
  every readout under every placement, named "fit:<cell>:<b_def|b_per|w|total>", with source fixed effects ("|within") and
  over the benign sources; the roleplay coin's own fit on z(A_0) and z(A_roleplay) is "f6:<p>:...";
- magnitudes: "sd:<cell>", "gap:<cell>" (mean of non-harmful minus harmful tasks), "slope:<Y>~<A>" (nats per Assistant SD);
- each finding's contrasts, "f<n>:..." (see the call functions below).

Calls (README pre-registration; categories replicates / doesn't / not separable / unstable / not interpretable / not run):
- a call needs every cell it uses: missing -> "not run"; split-half below 0.9 or median outcome mass below 0.2 -> "not
  interpretable" (reported, no call);
- 4: "not separable" unless |r(A_sys, A_0)| and |r(A_user, A_0)| are both below 0.9;
- 5 and 7: "not separable" unless |r(A_sys, A_0)| < 0.9 (not written in their calls: our reading, since both rest on the fit
  against A_sys, as the persona sections' calls do); 6: unless |r(A_roleplay, A_0)| < 0.9;
- 7: "unstable" when the CI of b_def + b_per includes 0 for A_self or P_self.

Outputs:
  outputs/replication/<key>/analysis/findings.csv   one row per finding (and persona for 4-7): call, key number, detail
  outputs/replication/<key>/analysis/stats.csv      every statistic with its CI
  outputs/replication/<key>/analysis/cells.csv      per condition: reliability, outcome mass, slot bias, SD
  outputs/replication/<key>/analysis/utilities.csv  per task, every condition's utilities
  outputs/replication/<key>/analysis/report.txt
  figures/<key>/summary.png         each call with the estimate, CI and threshold it rests on
  figures/<key>/coin.png            finding 1: the coin against the Assistant, base, final and post-training's shift
  figures/<key>/user_predict.png    findings 2-3: U and P against the Assistant, base, fixed target, final
  figures/<key>/sources_plain.png   the plain readouts within each task source (source_grid.py)
  figures/<key>/persona_pulls.png   findings 4-6: each placement's b_def against b_per, per persona
  figures/<key>/persona_w.png       findings 5-7: w of every readout by placement, per persona
  figures/<key>/sources_<p>.png     persona p's coins by placement within each task source
  (persona figures only once the model has persona cells)
  figures/replication_matrix.png    findings x models: the configured models, then any other keys given

Usage: uv run python replication_analysis.py <key> [<key> ...] [--n-boot 2000]
"""

import os

# Thousands of tiny least-squares fits: multithreaded BLAS only adds overhead (5x slower here). Set before numpy loads.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import gzip  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.lines import Line2D

from analyze import (HARM_ORIGINS, HARM_TOPICS, N_BOOTSTRAP, ORIGIN_STYLE, additive_r2, fit_utilities, pair_effects,
                     partial_r, pearson, read_jsonl, split_half)
from matskit import plotstyle
from openers import slope_per_sd, write_csv
from persona_placement import placement_call
from persona_readouts import readout_call
from personas import pull_call
from personas_user import pulls
from predict_user import delta_call
from replication import PERSONAS
from report import INK, INK_2, RULE, origin_scatter, stratified_resamples, style
from roleplay import compare_from
from source_grid import SOURCE_LABEL, Row, plot_source_grid
from user_choice import R_SHIFT, call_from

OUT = Path("outputs/replication")
CONFIGS = Path("configs/replication")
FIGURES = Path("figures")
DEFAULT_TASKS = "data/tasks_n300_max500_seed0.jsonl"
SEED = 0
SOURCES = tuple(ORIGIN_STYLE)
BENIGN_SOURCES = ("MATH", "ALPACA", "WILDCHAT")  # PLAN_replication.md, decision 7
MIN_RELIABILITY, MIN_MASS = 0.9, 0.2
SEPARABLE = 0.9  # "separable" is |r(persona Assistant, A_0)| below this
R_AP = 0.5  # finding 3: r(A_f, P_f) at least this
READOUT = {"A": "assistant", "C": "coinflip", "U": "user_choice", "P": "predict_user"}
READOUT_LABEL = {"A": "Assistant's own choice", "C": "Coin", "U": "User's choice", "P": "Assistant's prediction\nof the user"}
PLACE = {"none": "", "sys": "@{p}", "user": "@{p}+user", "self": "@{p}+self", "roleplay": "@{p}+roleplay"}
PLACE_LABEL = {"none": "No persona", "sys": "System prompt", "user": "User instruction", "self": "User self-description",
               "roleplay": "Roleplay"}
# Which readouts each placement scores at the final stage (replication.final_conditions).
PLACE_READOUTS = {"none": "ACUP", "sys": "AC", "user": "AC", "self": "ACUP", "roleplay": "AC"}
FIT_SUFFIXES = ("", "|within", "|benign_sources")
R_SUFFIXES = FIT_SUFFIXES + ("|non_harmful",) + tuple(f"|src:{s}" for s in SOURCES)
CATEGORIES = ("replicates", "doesn't", "not separable", "unstable", "not interpretable", "not run")
FINDINGS = {1: "Coin tracks the Assistant after post-training", 2: "User's choice becomes more Assistant-like",
            3: "Assistant's prediction of the user is Assistant-like", 4: "Coin follows a persona the user instructs",
            5: "Coin follows a self-description less", 6: "Coin follows a persona framed as roleplay",
            7: "Prediction follows a self-description more than A does"}


# ---------------------------------------------------------------------------------------------------------------- cells

def cell_name(x, stage="instruct", place=None, persona=None):
    """Short name of a cell: A_f, C_b, ..., or C_user@evil. With a persona and place "none", the plain final readout as seen
    from that persona's fits (X_none@p)."""
    if persona is None:
        return f"{x}_{'b' if stage == 'base' else 'f'}"
    return f"{x}_{place}@{persona}"


def condition_key(x, stage="instruct", place="none", persona=None):
    """The "stage/condition" key of a cell, as in the results file names."""
    return f"{stage}/{READOUT[x]}{PLACE[place].format(p=persona) if persona else ''}"


def short_of(key):
    """"instruct/coinflip@evil+user" -> "C_user@evil"."""
    stage, cond = key.split("/", 1)
    base, _, rest = cond.partition("@")
    x = {v: k for k, v in READOUT.items()}.get(base, base)
    if not rest:
        return cell_name(x, stage)
    persona, _, place = rest.partition("+")
    return f"{x}_{place or 'sys'}@{persona}"


def read_rows(path):
    with gzip.open(path, "rt", newline="") as fh:
        return [dict(stage=r["stage"], condition=r["condition"], first=r["first"], second=r["second"],
                     log_odds=float(r["log_odds"]), outcome_mass=float(r["outcome_mass"])) for r in csv.DictReader(fh)]


def cell_info(path, ids, cache_dir):
    """{u, reliability, split_half_r, median_mass, mean_mass, n_pairs, slot_bias, additive_r2} of one results file, cached
    in cache_dir and recomputed when the file changes. The split-half splits are analyze.py's (seed 0)."""
    path = Path(path)
    st = path.stat()
    sig = [st.st_size, st.st_mtime_ns, len(ids)]
    cache = Path(cache_dir) / (path.name.removesuffix(".csv.gz") + ".json")
    if cache.exists():
        info = json.loads(cache.read_text())
        if info.get("sig") == sig:
            info["u"] = np.array(info["u"])
            return info
    rows = read_rows(path)
    effects = pair_effects(rows)
    assert len(effects) == 1, f"{path}: more than one (stage, condition)"
    (key, eff), = effects.items()
    unknown = {t for p in eff["pairs"] for t in p} - set(ids)
    assert not unknown, f"{path}: {len(unknown)} tasks not in the tasks file"
    u = fit_utilities(ids, eff)
    name = "/".join(key)
    r_half, rel = split_half(ids, {key: eff}, {name: {key: 1.0}}, np.random.default_rng(SEED))[name]
    masses = np.array([r["outcome_mass"] for r in rows])
    info = dict(sig=sig, key=name, path=str(path), u=[float(v) for v in u], reliability=float(rel),
                split_half_r=float(r_half), median_mass=float(np.median(masses)), mean_mass=float(masses.mean()),
                n_pairs=len(eff["pairs"]), slot_bias=float(eff["bias"].mean()), additive_r2=additive_r2(ids, eff, u),
                sd=float(u.std()))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(info))
    info["u"] = np.array(info["u"])
    return info


def read_config(key, root=None):
    """The model's config: the run's own copy if scored, else configs/replication/<key>.yaml, else {}."""
    for path in (Path(root or OUT) / key / "run_config.yaml", CONFIGS / f"{key}.yaml"):
        if path.exists():
            return yaml.safe_load(path.read_text()) or {}
    return {}


def model_label(key, cfg=None):
    cfg = read_config(key) if cfg is None else cfg
    if cfg.get("label"):
        return cfg["label"]
    finals = [m["name"] for m in cfg.get("models", []) if m.get("stage") == "instruct"]
    return finals[0].split("/")[-1] if finals else key


class Model:
    """One model's utilities and cell diagnostics, read from outputs/replication/<key>/."""

    def __init__(self, key, root=None):
        self.key, self.dir = key, Path(root or OUT) / key
        cfg = read_config(key, root)
        self.label = model_label(key, cfg)
        self.legacy = bool(cfg.get("legacy"))
        tasks = sorted(read_jsonl(cfg.get("tasks_file", DEFAULT_TASKS)), key=lambda t: t["id"])
        self.tasks = tasks
        self.ids = [t["id"] for t in tasks]
        self.origins = np.array([t["origin"] for t in tasks])
        self.nonharm = np.array([t["topic"] not in HARM_TOPICS and t["origin"] not in HARM_ORIGINS for t in tasks])
        self.U, self.cells = {}, {}
        for path in sorted((self.dir / "results").glob("*.csv.gz")):
            info = cell_info(path, self.ids, self.dir / "analysis" / "cells")
            self.U[info["key"]] = info.pop("u")
            self.cells[info["key"]] = info
        self.pilots = {p.name.removesuffix(".csv.gz").replace("__", "/", 1) for p in (self.dir / "pilot").glob("*.csv.gz")}

    def personas(self):
        """The replication's personas this model has any final persona cell for, in replication.PERSONAS order."""
        have = {k.split("@", 1)[1].split("+")[0] for k in self.U if "@" in k}
        return [p for p in PERSONAS if p in have]


# ----------------------------------------------------------------------------------------------------------- statistics

class _Ctx:
    """Task-level masks for one (possibly resampled) task set, and each cell's within-origin version."""

    def __init__(self, U, origins, nonharm):
        self.origins, self.nonharm = origins, nonharm
        self.bsrc = np.isin(origins, BENIGN_SOURCES)
        self.src = {s: origins == s for s in SOURCES if (origins == s).any()}
        groups = np.unique(origins)
        self.fe = [(origins == o).astype(float) for o in groups[1:]]
        self._masks = [origins == g for g in groups]
        self.W = {k: self.within(u) for k, u in U.items()}

    def within(self, x):
        """openers.within_groups(x, origins), with the masks made once: each source's mean removed."""
        out = x.astype(float).copy()
        for m in self._masks:
            out[m] -= out[m].mean()
        return out


def _r(out, name, x, y, wx, wy, ctx):
    """r(x, y) under every breakdown: all, |within (wx, wy: x and y with each source's mean removed), |benign_sources,
    |non_harmful, |src:<SOURCE>."""
    out[name] = pearson(x, y)
    out[f"{name}|within"] = pearson(wx, wy)
    out[f"{name}|benign_sources"] = pearson(x[ctx.bsrc], y[ctx.bsrc])
    out[f"{name}|non_harmful"] = pearson(x[ctx.nonharm], y[ctx.nonharm])
    for s, m in ctx.src.items():
        out[f"{name}|src:{s}"] = pearson(x[m], y[m])


def _diff(out, name, a, b, suffixes=R_SUFFIXES):
    for s in suffixes:
        if a + s in out and b + s in out:
            out[name + s] = out[a + s] - out[b + s]


def _fit(out, prefix, y, a0, ap, ctx):
    """The persona sections' two-pull fit of y on z(a0), z(ap): over all tasks, with source fixed effects (|within) and over
    the benign sources."""
    b = ctx.bsrc
    for sfx, args in (("", (y, a0, ap)), ("|within", (y, a0, ap, ctx.fe)), ("|benign_sources", (y[b], a0[b], ap[b]))):
        b_def, b_per, w = pulls(*args)
        out[f"{prefix}:b_def{sfx}"], out[f"{prefix}:b_per{sfx}"] = b_def, b_per
        out[f"{prefix}:w{sfx}"], out[f"{prefix}:total{sfx}"] = w, b_def + b_per


def _ratio(out, name, num, den, suffixes=FIT_SUFFIXES):
    for s in suffixes:
        if num + s in out and den + s in out:
            out[name + s] = out[num + s] / out[den + s]


def statistics(U, origins, nonharm, personas=PERSONAS):
    """Every statistic computable from the cells in U ({"stage/condition": utilities}), as {name: value}. Called on the full
    task set and on each resample, so its names depend only on which cells exist."""
    ctx = _Ctx(U, origins, nonharm)
    out = {}
    gap = lambda x: float(x[nonharm].mean() - x[~nonharm].mean())
    plain = {cell_name(x, s): condition_key(x, s) for x in READOUT for s in ("base", "instruct")}
    get = lambda c: U.get(plain[c])
    r = lambda name, ky, kx: _r(out, name, U[ky], U[kx], ctx.W[ky], ctx.W[kx], ctx)

    for key, u in U.items():
        out[f"sd:{short_of(key)}"], out[f"gap:{short_of(key)}"] = float(u.std()), gap(u)
    for y in "CUP":  # every plain readout against the Assistant at the same stage, and post-training's shift
        for s in "bf":
            if get(f"{y}_{s}") is not None and get(f"A_{s}") is not None:
                r(f"r:{y}_{s}~A_{s}", plain[f"{y}_{s}"], plain[f"A_{s}"])
                out[f"slope:{y}_{s}~A_{s}"] = slope_per_sd(get(f"A_{s}"), get(f"{y}_{s}"))
        if all(get(c) is not None for c in (f"{y}_b", f"{y}_f", "A_b", "A_f")):
            dA, dY = get("A_f") - get("A_b"), get(f"{y}_f") - get(f"{y}_b")
            wA = ctx.W[plain["A_f"]] - ctx.W[plain["A_b"]]
            _r(out, f"shift:{y}~A", dA, dY, wA, ctx.W[plain[f"{y}_f"]] - ctx.W[plain[f"{y}_b"]], ctx)
    for y in "UP":  # the base readout against the final Assistant (the user-choice section's fixed target)
        if get(f"{y}_b") is not None and get("A_f") is not None:
            r(f"r:{y}_b~A_f", plain[f"{y}_b"], plain["A_f"])
    if get("U_f") is not None and get("P_f") is not None:
        r("r:U_f~P_f", plain["U_f"], plain["P_f"])
        if get("A_f") is not None:
            out["f3:partial_up_a"] = partial_r(get("U_f"), get("P_f"), get("A_f"))
            out["f3:partial_ua_p"] = partial_r(get("U_f"), get("A_f"), get("P_f"))
    _diff(out, "f1:d", "r:C_f~A_f", "r:C_b~A_b")
    _diff(out, "f2:d", "r:U_f~A_f", "r:U_b~A_f")
    _diff(out, "f3:d_P", "r:P_f~A_f", "r:P_b~A_f")
    _diff(out, "f3:delta", "r:U_f~P_f", "r:U_f~A_f")

    a0_key = plain["A_f"]
    for p in personas:
        key = lambda x, place: condition_key(x, "instruct", place, p)
        cells = {cell_name(x, place=s, persona=p): key(x, s) for s in PLACE for x in PLACE_READOUTS[s]}
        have = {c: k for c, k in cells.items() if k in U}
        ap_key = key("A", "sys")
        rp = lambda name, c, k: r(name, have[c], k)
        if a0_key in U and ap_key in U:
            a0, ap = U[a0_key], U[ap_key]
            out[f"f4:{p}:sep_sys"] = pearson(ap, a0)
            for c, k in have.items():
                _fit(out, f"fit:{c}", U[k], a0, ap, ctx)
                out[f"r0:{c}"] = pearson(U[k], a0)  # how far the placement moves it from the default Assistant
                if c[0] == "C" and "_none@" not in c:  # each coin within each source (the source grids' numbers)
                    rp(f"r:{c}~A_f", c, a0_key)
                    if "_sys@" not in c:
                        rp(f"r:{c}~A_sys@{p}", c, ap_key)
            _ratio(out, f"f4:{p}:rho_user", f"fit:C_user@{p}:b_per", f"fit:C_sys@{p}:b_per")
            _ratio(out, f"f5:{p}:rho_self", f"fit:C_self@{p}:b_per", f"fit:C_sys@{p}:b_per")
            for s in FIT_SUFFIXES:  # as persona_placement.py computes it
                parts = [f"fit:C_{x}@{p}:b_per{s}" for x in ("self", "user", "sys")]
                if all(n in out for n in parts):
                    out[f"f5:{p}:drho{s}"] = (out[parts[0]] - out[parts[1]]) / out[parts[2]]
            for y in "PU":
                _diff(out, f"f7:{p}:dw_{y}", f"fit:{y}_self@{p}:w", f"fit:A_self@{p}:w", FIT_SUFFIXES)
        if f"A_user@{p}" in have and a0_key in U:
            out[f"f4:{p}:sep_user"] = pearson(U[have[f"A_user@{p}"]], U[a0_key])
            if ap_key in U:
                out[f"f4:{p}:r_auser_ap"] = pearson(U[have[f"A_user@{p}"]], U[ap_key])
        if f"A_roleplay@{p}" in have and a0_key in U:
            ar = U[have[f"A_roleplay@{p}"]]
            out[f"f6:{p}:sep_r"] = pearson(ar, U[a0_key])
            if ap_key in U:
                out[f"f6:{p}:r_ar_ap"] = pearson(ar, U[ap_key])
            if f"C_roleplay@{p}" in have:
                _fit(out, f"f6:{p}", U[have[f"C_roleplay@{p}"]], U[a0_key], ar, ctx)
                rp(f"r:C_roleplay@{p}~A_roleplay@{p}", f"C_roleplay@{p}", have[f"A_roleplay@{p}"])
                _diff(out, f"f6:{p}:dw", f"f6:{p}:w", f"fit:C_sys@{p}:w", FIT_SUFFIXES)
        if f"A_self@{p}" in have and f"P_self@{p}" in have:
            a, pp = U[have[f"A_self@{p}"]], U[have[f"P_self@{p}"]]
            out[f"f5:{p}:sep_self"] = pearson(a, pp)
            for y in "CU":
                if f"{y}_self@{p}" in have:
                    c = f"{y}_self@{p}"
                    rp(f"r:{c}~P_self@{p}", c, have[f"P_self@{p}"])
                    rp(f"r:{c}~A_self@{p}", c, have[f"A_self@{p}"])
                    _diff(out, f"f5:{p}:d{y}", f"r:{c}~P_self@{p}", f"r:{c}~A_self@{p}")
                    out[f"f5:{p}:partial_{y}p_a"] = partial_r(U[have[c]], pp, a)
                    out[f"f5:{p}:partial_{y}a_p"] = partial_r(U[have[c]], a, pp)
    return out


def bootstrap(U, origins, nonharm, resamples, personas=PERSONAS):
    """{name: (estimate, ci_low, ci_high)} of every statistic, the CIs from the given task resamples."""
    est = statistics(U, origins, nonharm, personas)
    names = list(est)
    boot = np.full((len(resamples), len(names)), np.nan)
    for k, i in enumerate(resamples):
        s = statistics({c: u[i] for c, u in U.items()}, origins[i], nonharm[i], personas)
        boot[k] = [s.get(n, np.nan) for n in names]
    with np.errstate(invalid="ignore"):
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    return {n: (float(est[n]), float(lo[j]), float(hi[j])) for j, n in enumerate(names)}


def resamples_for(origins, n_boot=N_BOOTSTRAP):
    """The earlier sections' resamples: within origin, seed 0."""
    return stratified_resamples(origins, np.random.default_rng(SEED), n_boot)


# ---------------------------------------------------------------------------------------------------------------- calls

def gate(model, cells):
    """(status, reason) for the cells a call uses ("stage/condition" keys): "not run" if any is missing, "not interpretable"
    if any fails split-half >= 0.9 or median outcome mass >= 0.2, else (None, "")."""
    missing = [c for c in cells if c not in model.U]
    if missing:
        pilot = [c for c in missing if c in model.pilots]
        return "not run", "missing " + ", ".join(short_of(c) + (" (pilot only)" if c in pilot else "") for c in missing)
    bad = []
    for c in cells:
        info = model.cells[c]
        if info["reliability"] < MIN_RELIABILITY:
            bad.append(f"{short_of(c)} split-half {info['reliability']:.3f}")
        if info["median_mass"] < MIN_MASS:
            bad.append(f"{short_of(c)} median mass {info['median_mass']:.2f}")
    return ("not interpretable", "; ".join(bad)) if bad else (None, "")


def _ci(S, name, prec=2):
    est, lo, hi = S[name]
    return f"{est:+.{prec}f} [{lo:+.{prec}f}, {hi:+.{prec}f}]"


def _row(finding, persona, status, reason, call="", section_call="", key_number="", detail="", plot=()):
    return dict(finding=finding, persona=persona or "", call=status or call, section_call=section_call,
                key_number=key_number, detail="; ".join(x for x in (reason, detail) if x), plot=list(plot))


def _A0():
    return condition_key("A", "instruct")


def call_1(S, model):
    need = [condition_key(x, s) for x in "AC" for s in ("base", "instruct")]
    status, reason = gate(model, need)
    if status == "not run":
        return _row(1, None, status, reason)
    r, d, sh = S["r:C_f~A_f"], S["f1:d"], S["shift:C~A"]
    ok = r[1] > 0 and d[0] > 0 and d[1] > 0 and sh[0] >= R_SHIFT and sh[1] > 0
    return _row(1, None, status, reason, "replicates" if ok else "doesn't",
                key_number=f"r {r[0]:.2f} (base {S['r:C_b~A_b'][0]:.2f}), shift {sh[0]:.2f}",
                detail=f"r(A_f, C_f) {_ci(S, 'r:C_f~A_f')}, minus base {_ci(S, 'f1:d')}, r shift {_ci(S, 'shift:C~A')}",
                plot=[("r:C_f~A_f", "r(A_f, C_f)", 0), ("f1:d", "r(A_f, C_f) - r(A_b, C_b)", 0),
                      ("shift:C~A", "r shift", R_SHIFT)])


def call_2(S, model):
    need = [condition_key(x, s) for x in "AU" for s in ("base", "instruct")]
    status, reason = gate(model, need)
    if status == "not run":
        return _row(2, None, status, reason)
    d, sh = S["f2:d"], S["shift:U~A"]
    section = call_from(*d, sh[0], sh[1])
    return _row(2, None, status, reason, "replicates" if section == "more Assistant-like" else "doesn't",
                f"user-choice call: {section}", key_number=f"d {d[0]:+.2f}, r {S['r:U_f~A_f'][0]:.2f}",
                detail=f"d {_ci(S, 'f2:d')}, r shift {_ci(S, 'shift:U~A')}, r(A_f, U_f) {_ci(S, 'r:U_f~A_f')}",
                plot=[("f2:d", "d = r(A_f, U_f) - r(A_f, U_b)", 0.05), ("shift:U~A", "r shift", R_SHIFT)])


def call_3(S, model):
    need = [condition_key("A", "instruct"), condition_key("P", "base"), condition_key("P", "instruct")]
    status, reason = gate(model, need)
    if status == "not run":
        return _row(3, None, status, reason)
    rap, dp = S["r:P_f~A_f"], S["f3:d_P"]
    ok = rap[0] >= R_AP and dp[1] > 0
    section = f"Δ: {delta_call(*S['f3:delta'])}" if "f3:delta" in S else ""
    delta = f"; Δ = r(U_f, P_f) - r(U_f, A_f) {_ci(S, 'f3:delta')}" if "f3:delta" in S else ""
    return _row(3, None, status, reason, "replicates" if ok else "doesn't", section,
                key_number=f"r {rap[0]:.2f}, d_P {dp[0]:+.2f}",
                detail=f"r(A_f, P_f) {_ci(S, 'r:P_f~A_f')}, d_P {_ci(S, 'f3:d_P')}{delta}",
                plot=[("r:P_f~A_f", "r(A_f, P_f)", R_AP), ("f3:d_P", "d_P = r(A_f, P_f) - r(A_f, P_b)", 0)])


def _key(x, place, p):
    return condition_key(x, "instruct", place, p)


def _sep(S, name):
    return name in S and abs(S[name][0]) < SEPARABLE


def call_4(S, model, p):
    need = [_A0(), _key("A", "sys", p), _key("A", "user", p), _key("C", "user", p)]
    status, reason = gate(model, need)
    if status == "not run":
        return _row(4, p, status, reason)
    b = S[f"fit:C_user@{p}:b_per"]
    rho = f"f4:{p}:rho_user"
    section = placement_call(b[1], b[2], *S[rho][1:]) if rho in S else ""
    sys_call = (pull_call(S[f"fit:C_sys@{p}:b_def"][1], *S[f"fit:C_sys@{p}:b_per"][1:])
                if f"fit:C_sys@{p}:b_def" in S else "not run")
    sep = f"|r(A_sys, A_0)| {abs(S[f'f4:{p}:sep_sys'][0]):.2f}, |r(A_user, A_0)| {abs(S[f'f4:{p}:sep_user'][0]):.2f}"
    call = ("not separable" if not (_sep(S, f"f4:{p}:sep_sys") and _sep(S, f"f4:{p}:sep_user"))
            else "replicates" if b[1] > 0 else "doesn't")
    rho_txt = f", ρ_user {_ci(S, rho)}" if rho in S else ""
    sys_txt = (f"; system prompt: b_def {_ci(S, f'fit:C_sys@{p}:b_def')}, b_per {_ci(S, f'fit:C_sys@{p}:b_per')}, "
               f"w {_ci(S, f'fit:C_sys@{p}:w')} ({sys_call})" if f"fit:C_sys@{p}:b_def" in S else "")
    return _row(4, p, status, reason, call, f"placement call: {section}" if section else "",
                key_number=f"b_per {b[0]:.2f}" + (f", ρ {S[rho][0]:.2f}" if rho in S else ""),
                detail=f"b_per(user) {_ci(S, f'fit:C_user@{p}:b_per')}{rho_txt}; {sep}{sys_txt}",
                plot=[(f"fit:C_user@{p}:b_per", "b_per, user instruction", 0)])


def call_5(S, model, p):
    need = [_A0(), _key("A", "sys", p), _key("C", "sys", p), _key("C", "user", p), _key("C", "self", p)]
    status, reason = gate(model, need)
    dc_call, secondary = "", ""
    if f"f5:{p}:dC" in S:  # the secondary call, made only where r(A_self, P_self) < 0.9 (readout_call)
        dc_call = readout_call(S[f"f5:{p}:sep_self"][0], *S[f"f5:{p}:dC"])
        agrees = {"Assistant": "agrees", "not separable": "not called"}.get(dc_call, "doesn't agree")
        secondary = (f"secondary Δ_C {_ci(S, f'f5:{p}:dC')}, r(A_self, P_self) {S[f'f5:{p}:sep_self'][0]:.2f}: "
                     f"{dc_call} ({agrees})")
    if status == "not run":
        return _row(5, p, status, reason, section_call=f"Δ_C: {dc_call}" if dc_call else "", detail=secondary)
    d = S[f"f5:{p}:drho"]
    call = ("not separable" if not _sep(S, f"f4:{p}:sep_sys") else "replicates" if d[0] < 0 and d[2] < 0 else "doesn't")
    return _row(5, p, status, reason, call, f"Δ_C: {dc_call}" if dc_call else "",
                key_number=f"Δρ {d[0]:+.2f}",
                detail=f"ρ_self - ρ_user {_ci(S, f'f5:{p}:drho')} (ρ_self {_ci(S, f'f5:{p}:rho_self')}, ρ_user "
                       f"{_ci(S, f'f4:{p}:rho_user')}); |r(A_sys, A_0)| {abs(S[f'f4:{p}:sep_sys'][0]):.2f}; {secondary}",
                plot=[(f"f5:{p}:drho", "ρ_self - ρ_user", 0)])


def call_6(S, model, p):
    need = [_A0(), _key("A", "roleplay", p), _key("C", "roleplay", p)]
    status, reason = gate(model, need)
    if status == "not run":
        return _row(6, p, status, reason)
    sep = S[f"f6:{p}:sep_r"][0]
    section = ("not separable" if abs(sep) >= SEPARABLE
               else pull_call(S[f"f6:{p}:b_def"][1], *S[f"f6:{p}:b_per"][1:]))
    call = "not separable" if section == "not separable" else "replicates" if section in ("persona", "mix") else "doesn't"
    dw = f"f6:{p}:dw"
    cmp = f"; Δw = w_r - w_s {_ci(S, dw)}: {compare_from(section, *S[dw][1:])}" if dw in S else ""
    return _row(6, p, status, reason, call, f"roleplay call: {section}",
                key_number=f"w_r {S[f'f6:{p}:w'][0]:.2f}" + (f", Δw {S[dw][0]:+.2f}" if dw in S else ""),
                detail=f"b_def {_ci(S, f'f6:{p}:b_def')}, b_per {_ci(S, f'f6:{p}:b_per')}, w_r {_ci(S, f'f6:{p}:w')}; "
                       f"r(A_roleplay, A_0) {sep:+.2f}{cmp}",
                plot=[(f"f6:{p}:b_per", "b_per, roleplay (vs A_roleplay)", 0)])


def _w_call(S, p, y):
    """Finding 7's rule for readout y (P, or U as reported): "unstable", "replicates" or "doesn't"."""
    tot = lambda c: S[f"fit:{c}@{p}:total"]
    if any(tot(c)[1] <= 0 <= tot(c)[2] for c in ("A_self", f"{y}_self")):
        return "unstable"
    d = S[f"f7:{p}:dw_{y}"]
    return "replicates" if d[0] > 0 and d[1] > 0 else "doesn't"


def call_7(S, model, p):
    need = [_A0(), _key("A", "sys", p), _key("A", "self", p), _key("P", "self", p)]
    status, reason = gate(model, need)
    if status == "not run":
        return _row(7, p, status, reason)
    call = "not separable" if not _sep(S, f"f4:{p}:sep_sys") else _w_call(S, p, "P")
    u = f"f7:{p}:dw_U"
    u_txt = f"; reported, U: w(U_self) - w(A_self) {_ci(S, u)}: {_w_call(S, p, 'U')}" if u in S else ""
    w = lambda c: S[f"fit:{c}@{p}:w"][0]
    return _row(7, p, status, reason, call, f"U: {_w_call(S, p, 'U')}" if u in S else "",
                key_number=f"Δw {S[f'f7:{p}:dw_P'][0]:+.2f}",
                detail=f"w(P_self) - w(A_self) {_ci(S, f'f7:{p}:dw_P')} (w {w('P_self'):.2f} vs {w('A_self'):.2f}); "
                       f"|r(A_sys, A_0)| {abs(S[f'f4:{p}:sep_sys'][0]):.2f}{u_txt}",
                plot=[(f"f7:{p}:dw_P", "w(P_self) - w(A_self)", 0)])


def findings(S, model, personas=PERSONAS):
    """One row per finding (per persona for 4-7), in matrix order."""
    def safe(f, *args):
        try:
            return f(S, model, *args)
        except KeyError as e:  # a statistic the call needs wasn't computed (e.g. stats from fewer personas)
            return _row(int(f.__name__[-1]), args[0] if args else None, "not run", f"statistic {e} not computed")

    rows = [safe(call_1), safe(call_2), safe(call_3)]
    for f in (call_4, call_5, call_6, call_7):
        rows += [safe(f, p) for p in personas]
    for r in rows:
        r.update(model=model.key, label=model.label)
    return rows


# --------------------------------------------------------------------------------------------------------------- report

def write_tables(model, S, rows, out):
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "findings.csv", [{k: v for k, v in r.items() if k != "plot"} for r in rows])
    write_csv(out / "stats.csv", [dict(name=n, est=e, lo=lo, hi=hi) for n, (e, lo, hi) in S.items()])
    write_csv(out / "cells.csv", [dict(cell=short_of(k), condition=k, **{c: v for c, v in info.items()
                                                                          if c not in ("sig", "key")})
                                  for k, info in sorted(model.cells.items())])
    if model.U:
        keys = sorted(model.U)
        write_csv(out / "utilities.csv", [dict(id=t["id"], origin=t["origin"], topic=t["topic"],
                                               **{k: float(model.U[k][i]) for k in keys})
                                          for i, t in enumerate(model.tasks)])


def report_text(model, S, rows):
    """The report: calls, then each finding's numbers and breakdowns, then the cells' gates."""
    L = [f"{model.label} ({model.key}), {len(model.U)} conditions scored. CIs: 95% task bootstraps within origin.", ""]
    L.append("Calls:")
    for r in rows:
        p = f", {r['persona']}" if r["persona"] else ""
        sec = f" [{r['section_call']}]" if r["section_call"] else ""
        L.append(f"  {r['finding']}{p:12} {r['call']:18} {r['key_number']}{sec}")
        if r["detail"]:
            L.append(f"  {'':14}{r['detail']}")
    fm = lambda n: f"{S[n][0]:+.3f} [{S[n][1]:+.3f}, {S[n][2]:+.3f}]" if n in S else "-"
    head = ["all", "within origin", "benign sources", "non-harmful"] + [SOURCE_LABEL[s] for s in SOURCES]

    def table(title, names):
        names = [(lab, n) for lab, n in names if n in S]
        if not names:
            return
        L.extend(["", title, f"  {'':34}" + "".join(f"{h:>24}" for h in head)])
        for lab, n in names:
            L.append(f"  {lab:34}" + "".join(f"{fm(n + s):>24}" for s in R_SUFFIXES))

    table("Finding 1, the coin and the Assistant (r):", [("r(A_f, C_f)", "r:C_f~A_f"), ("r(A_b, C_b)", "r:C_b~A_b"),
                                                        ("difference", "f1:d"), ("r shift", "shift:C~A")])
    table("Finding 2, the user's choice (r):", [("r(A_f, U_f)", "r:U_f~A_f"), ("r(A_f, U_b)", "r:U_b~A_f"),
                                                ("d", "f2:d"), ("r(A_b, U_b)", "r:U_b~A_b"), ("r shift", "shift:U~A")])
    table("Finding 3, the Assistant's prediction of the user (r):",
          [("r(A_f, P_f)", "r:P_f~A_f"), ("r(A_f, P_b)", "r:P_b~A_f"), ("d_P", "f3:d_P"), ("r(A_b, P_b)", "r:P_b~A_b"),
           ("r shift", "shift:P~A"), ("r(U_f, P_f)", "r:U_f~P_f"), ("Δ = r(U_f, P_f) - r(U_f, A_f)", "f3:delta")])
    mags = [(f"{lab} {c}", f"{k}:{c}") for c in ("A_b", "A_f", "C_b", "C_f", "U_b", "U_f", "P_b", "P_f")
            for k, lab in (("sd", "SD"), ("gap", "harm gap"))] + \
           [(f"slope {y}_{s} on A_{s}", f"slope:{y}_{s}~A_{s}") for y in "CUP" for s in "bf"]
    mags = [(lab, n) for lab, n in mags if n in S]
    if mags:
        L.extend(["", "Magnitudes (nats; slopes per Assistant SD; gap = non-harmful minus harmful):"])
        L += [f"  {lab:34}{fm(n):>24}" for lab, n in mags]
    for p in model.personas():
        L.extend(["", f"Persona {p}:"])
        for lab, n in (("r(A_sys, A_0)", f"f4:{p}:sep_sys"), ("r(A_user, A_0)", f"f4:{p}:sep_user"),
                       ("r(A_roleplay, A_0)", f"f6:{p}:sep_r"), ("r(A_roleplay, A_sys)", f"f6:{p}:r_ar_ap"),
                       ("r(A_self, P_self)", f"f5:{p}:sep_self")):
            if n in S:
                L.append(f"  {lab:34}{fm(n):>24}")
        fit_head = ("b_def", "b_per", "w", "w within", "w benign sources", "harm gap")
        L.append(f"  fits on z(A_0), z(A_sys) {'':8}" + "".join(f"{h:>24}" for h in fit_head))
        for s in PLACE:
            for x in "ACUP":
                c = f"{x}_{s}@{p}"
                if f"fit:{c}:w" in S:
                    vals = [f"fit:{c}:{k}" for k in ("b_def", "b_per", "w", "w|within", "w|benign_sources")]
                    gap_name = f"gap:{c}" if s != "none" else f"gap:{x}_f"
                    L.append(f"  {c:34}" + "".join(f"{fm(n):>24}" for n in vals + [gap_name]))
        if f"f6:{p}:w" in S:
            L.append(f"  {'C_roleplay on z(A_0), z(A_roleplay)':34}" + "".join(
                f"{fm(f'f6:{p}:{k}'):>24}" for k in ("b_def", "b_per", "w", "w|within", "w|benign_sources")))
        for lab, n in (("ρ_user", f"f4:{p}:rho_user"), ("ρ_self", f"f5:{p}:rho_self"), ("ρ_self - ρ_user", f"f5:{p}:drho"),
                       ("Δw roleplay - system", f"f6:{p}:dw"), ("w(P_self) - w(A_self)", f"f7:{p}:dw_P"),
                       ("w(U_self) - w(A_self)", f"f7:{p}:dw_U")):
            if n in S:
                L.append(f"  {lab:34}" + "".join(f"{fm(n + s):>24}" for s in FIT_SUFFIXES) + "   (all, within, benign)")
        table(f"  {p}: correlations", [(f"r(C_{s}, A_0)", f"r:C_{s}@{p}~A_f") for s in ("sys", "user", "self", "roleplay")]
              + [(f"r(C_{s}, A_sys)", f"r:C_{s}@{p}~A_sys@{p}") for s in ("user", "self", "roleplay")]
              + [("r(C_roleplay, A_roleplay)", f"r:C_roleplay@{p}~A_roleplay@{p}"),
                 ("Δ_C = r(C,P) - r(C,A), self", f"f5:{p}:dC"), ("Δ_U = r(U,P) - r(U,A), self", f"f5:{p}:dU")])
    L.extend(["", "Cells (gates: split-half >= 0.9; median outcome mass >= 0.2 for a call):"])
    for k, info in sorted(model.cells.items()):
        flag = ("" if info["reliability"] >= MIN_RELIABILITY else "  SPLIT-HALF BELOW 0.9") + \
               ("" if info["median_mass"] >= MIN_MASS else "  MASS BELOW 0.2")
        L.append(f"  {k:36} split-half {info['reliability']:.3f}  median mass {info['median_mass']:.2f}  "
                 f"SD {info['sd']:6.2f}  slot bias {info['slot_bias']:+.2f}  pairs {info['n_pairs']}{flag}")
    return "\n".join(L) + "\n"


# -------------------------------------------------------------------------------------------------------------- figures

CALL_COLOR = {"replicates": plotstyle.SERIES[0], "doesn't": plotstyle.SERIES[1]}
NO_CALL = "#898781"  # plotstyle.MUTED


def _r_title(S, ry, prefix=""):
    if ry not in S:
        return prefix
    g = lambda s: S[ry + s][0]
    return f"{prefix}r {g(''):.2f} (within {g('|within'):.2f}, benign src {g('|benign_sources'):.2f})"


def plot_coin(model, S, path):
    """Finding 1: the coin against the Assistant at base and final, and post-training's shift in each."""
    U, o = model.U, model.origins
    k = lambda x, s: condition_key(x, s)
    panels = [("Base", U.get(k("A", "base")), U.get(k("C", "base")), "r:C_b~A_b", "base"),
              ("Final", U.get(k("A", "instruct")), U.get(k("C", "instruct")), "r:C_f~A_f", "final")]
    if all(p[1] is not None and p[2] is not None for p in panels):
        panels.append(("Post-training's shift", panels[1][1] - panels[0][1], panels[1][2] - panels[0][2], "shift:C~A",
                       "final - base"))
    panels = [p for p in panels if p[1] is not None and p[2] is not None]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(4.3 * len(panels), 4.1), squeeze=False)
    for ax, (title, x, y, name, stage) in zip(axes[0], panels):
        origin_scatter(ax, x, y, o)
        ax.set_title(f"{title}\n{_r_title(S, name)}", fontsize=9.5, color=INK, loc="left")
        ax.set_xlabel(f"Assistant utility, {stage} (nats)", fontsize=9, color=INK_2)
        ax.set_ylabel(f"coin utility, {stage} (nats)", fontsize=9, color=INK_2)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, frameon=False, ncol=len(names), loc="lower center", fontsize=9)
    fig.suptitle(f"{model.label}: the coin against the Assistant's own choice (each panel has its own scale)", color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    plotstyle.save(fig, path.stem, path.parent)


def plot_user_predict(model, S, path):
    """Findings 2 and 3: the user's choice (U) and the Assistant's prediction of it (P) against the Assistant, at base, the
    base readout against the final Assistant (the fixed target of d and d_P), and at final."""
    U, o = model.U, model.origins
    k = lambda x, s: U.get(condition_key(x, s))
    rows = [y for y in "UP" if k(y, "base") is not None or k(y, "instruct") is not None]
    if not rows:
        return
    cols = (("Base vs base Assistant", "base", "base"), ("Base vs final Assistant", "instruct", "base"),
            ("Final vs final Assistant", "instruct", "instruct"))
    fig, axes = plt.subplots(len(rows), 3, figsize=(12.6, 3.7 * len(rows)), squeeze=False)
    name = {"U": "user's choice", "P": "prediction of the user"}
    for i, y in enumerate(rows):
        for j, (title, sa, sy) in enumerate(cols):
            ax = axes[i, j]
            x, v = k("A", sa), k(y, sy)
            if x is None or v is None:
                ax.text(0.5, 0.5, "not run", ha="center", va="center", color=INK_2, transform=ax.transAxes)
                ax.set_axis_off()
                continue
            origin_scatter(ax, x, v, o)
            stat = f"r:{y}_{'b' if sy == 'base' else 'f'}~A_{'b' if sa == 'base' else 'f'}"
            ax.set_title(f"{title}\n{_r_title(S, stat)}", fontsize=9, color=INK, loc="left")
            ax.set_xlabel(f"Assistant utility, {'base' if sa == 'base' else 'final'} (nats)", fontsize=9, color=INK_2)
            ax.set_ylabel(f"{name[y]}, {'base' if sy == 'base' else 'final'} (nats)", fontsize=9, color=INK_2)
    handles, names = next((ax.get_legend_handles_labels() for ax in axes.flat if ax.get_legend_handles_labels()[0]),
                          ([], []))
    fig.legend(handles, names, frameon=False, ncol=max(1, len(names)), loc="lower center", fontsize=9)
    d = [f"{lab} {S[n][0]:+.2f} [{S[n][1]:+.2f}, {S[n][2]:+.2f}]" for lab, n in (("d", "f2:d"), ("d_P", "f3:d_P"))
         if n in S]
    fig.suptitle(f"{model.label}: the user's choice and the Assistant's prediction of it, against the Assistant"
                 + (f"\n{'; '.join(d)} (final minus fixed-target r)" if d else ""), color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    plotstyle.save(fig, path.stem, path.parent)


def plot_plain_sources(model, path):
    """Per-source grid of the plain readouts against the Assistant."""
    U, k = model.U, lambda x, s: model.U.get(condition_key(x, s))
    specs = [("Final coin\nvs final Assistant", "A", "C", "instruct"), ("Base coin\nvs base Assistant", "A", "C", "base"),
             ("Final user's choice\nvs final Assistant", "A", "U", "instruct"),
             ("Final prediction\nvs final Assistant", "A", "P", "instruct")]
    rows = [Row(label, k(x, s), k(y, s), f"Assistant, {'final' if s == 'instruct' else 'base'} (nats)",
                f"{READOUT[y]} (nats)", plotstyle.SERIES[j])
            for j, (label, x, y, s) in enumerate(specs) if k(x, s) is not None and k(y, s) is not None]
    if rows:
        plot_source_grid(rows, model.origins, f"{model.label}: the readouts against the Assistant within each task source",
                         path)


def plot_persona_sources(model, p, path):
    """Per-source grid of persona p's coins by placement, against the default Assistant and the persona's Assistant."""
    U = model.U
    a0 = U.get(_A0())
    ap = U.get(_key("A", "sys", p))
    specs = []
    for j, s in enumerate(("sys", "user", "self", "roleplay")):
        c = U.get(_key("C", s, p))
        if c is None or a0 is None:
            continue
        specs.append(Row(f"{PLACE_LABEL[s]} coin\nvs default Assistant", a0, c, "Default Assistant A_0 (nats)",
                         "coin (nats)", plotstyle.SERIES[j]))
        target = U.get(_key("A", "roleplay", p)) if s == "roleplay" else ap
        if target is not None:
            who = "roleplay Assistant" if s == "roleplay" else "persona Assistant"
            specs.append(Row(f"{PLACE_LABEL[s]} coin\nvs {who}", target, c, f"{who[0].upper() + who[1:]} (nats)", "coin (nats)",
                             plotstyle.SERIES[j]))
    if specs:
        plot_source_grid(specs, model.origins, f"{model.label}, {p}: the coin by placement within each task source", path)


PLACE_STYLE = {"sys": (plotstyle.SERIES[0], "o"), "user": (plotstyle.SERIES[1], "o"), "self": (plotstyle.SERIES[2], "o"),
               "roleplay": (plotstyle.SERIES[3], "s")}
# Each placement's w label on its own side of the point, so neighbours (often user and roleplay) don't overprint.
W_LABEL_OFFSET = {"sys": (7, 5, "left"), "user": (-7, 5, "right"), "self": (7, 5, "left"), "roleplay": (7, -13, "left")}


def plot_pulls(model, S, path):
    """Per persona: each placement's coin, pull toward the persona (x) against pull toward the default Assistant (y), 95%
    CIs, like figures/persona_placement_pulls.png. The roleplay coin's persona is its own roleplay Assistant (finding 6);
    the others' is the system-prompt persona Assistant. b_0, the plain coin's slope on A_0, for reference."""
    ps = [p for p in model.personas() if any(f"fit:C_{s}@{p}:b_per" in S for s in ("sys", "user", "self"))
          or f"f6:{p}:b_per" in S]
    if not ps:
        return
    fig, axes = plt.subplots(1, len(ps), figsize=(4.4 * len(ps), 4.7), squeeze=False)
    b0 = S.get("slope:C_f~A_f")
    for ax, p in zip(axes[0], ps):
        top, bottom = 0.0, 0.0
        for s, (color, marker) in PLACE_STYLE.items():
            pre = f"f6:{p}" if s == "roleplay" else f"fit:C_{s}@{p}"
            if f"{pre}:b_per" not in S:
                continue
            (x, xl, xh), (y, yl, yh) = S[f"{pre}:b_per"], S[f"{pre}:b_def"]
            ax.errorbar(x, y, xerr=[[x - xl], [xh - x]], yerr=[[y - yl], [yh - y]], color=color, marker=marker, ms=8, lw=0,
                        elinewidth=1.5, mec="white", mew=0.8, zorder=3)
            dx, dy, ha = W_LABEL_OFFSET[s]
            ax.annotate(f"w {S[f'{pre}:w'][0]:.2f}", (x, y), xytext=(dx, dy), textcoords="offset points", color=INK,
                        fontsize=8.5, ha=ha)
            top, bottom = max(top, xh, yh), min(bottom, xl, yl)
        if b0 is not None:
            ax.errorbar(0, b0[0], yerr=[[b0[0] - b0[1]], [b0[2] - b0[0]]], color=INK_2, marker="D", ms=7, lw=0,
                        elinewidth=1.5, mec="white", mew=0.8, zorder=3)
            top = max(top, b0[2])
        top, bottom = top * 1.15 or 1.0, bottom * 1.15
        ax.plot([0, top], [0, top], color=RULE, ls="--", lw=1, zorder=1)
        ax.axhline(0, color=RULE, lw=0.8, zorder=1)
        ax.axvline(0, color=RULE, lw=0.8, zorder=1)
        ax.set_xlim(bottom - 0.02 * top, top)
        ax.set_ylim(bottom - 0.02 * top, top)
        ax.set_aspect("equal")
        ax.set_xlabel("pull toward the persona, b_per\n(coin nats per SD)", color=INK_2, fontsize=9)
        ax.set_ylabel("pull toward the default Assistant, b_def\n(coin nats per SD of A_0)", color=INK_2, fontsize=9)
        ax.set_title(p, color=INK, fontsize=10, loc="left")
        style(ax)
    handles = [Line2D([], [], ls="none", marker=m, ms=8, color=c, mec="white", label=PLACE_LABEL[s]
                      + (" (vs its own roleplay Assistant)" if s == "roleplay" else ""))
               for s, (c, m) in PLACE_STYLE.items()]
    handles.append(Line2D([], [], ls="none", marker="D", ms=7, color=INK_2, mec="white", label="no persona (b_0)"))
    fig.legend(handles=handles, frameon=False, fontsize=9, loc="lower center", ncol=3)
    fig.suptitle(f"{model.label}: whose preferences the coin follows, by where the persona sits", color=INK)
    fig.tight_layout(rect=(0, 0.11, 1, 0.95))
    plotstyle.save(fig, path.stem, path.parent)


W_MARKS = {"none": ("s", INK_2, False), "sys": ("D", plotstyle.SERIES[0], True), "user": ("o", plotstyle.SERIES[1], True),
           "self": ("o", plotstyle.SERIES[2], True), "roleplay": ("s", plotstyle.SERIES[3], True)}


def plot_w(model, S, path):
    """Per persona: w of each readout under each placement, all against the same target (0 = A_0's ranking, 1 = the
    system-prompt persona Assistant's), like figures/persona_readouts_w.png. Where b_def + b_per is near 0 the ratio means
    nothing, and the cell is marked "w unstable"."""
    ps = [p for p in model.personas() if any(f"fit:{x}_{s}@{p}:w" in S for x in "ACUP" for s in PLACE)]
    if not ps:
        return
    fig, axes = plt.subplots(1, len(ps), figsize=(3.6 * len(ps), 4.4), sharey=True, sharex=True, squeeze=False)
    y = np.arange(4)[::-1]
    dy = dict(zip(PLACE, np.linspace(0.3, -0.3, len(PLACE))))
    from matplotlib.transforms import blended_transform_factory
    for ax, p in zip(axes[0], ps):
        for v in (0, 1):
            ax.axvline(v, color=RULE, lw=0.8, zorder=1)
        for yi, x in zip(y, "ACUP"):
            for s in PLACE:
                n = f"fit:{x}_{s}@{p}:w"
                if n not in S:
                    continue
                est, lo, hi = S[n]
                m, c, filled = W_MARKS[s]
                if hi - lo > 3:
                    ax.text(0.98, yi + dy[s], "w unstable", ha="right", va="center", fontsize=7.5, color=INK_2,
                            transform=blended_transform_factory(ax.transAxes, ax.transData))
                    continue
                ax.plot([max(lo, -1.5), min(hi, 2.5)], [yi + dy[s]] * 2, color=c, lw=1.4, zorder=2,
                        solid_capstyle="round")
                ax.plot(np.clip(est, -1.5, 2.5), yi + dy[s], ls="none", marker=m, ms=6, color=c,
                        mfc=c if filled else "white", mew=1.4, zorder=4)
        ax.set_yticks(y, [READOUT_LABEL[x] for x in "ACUP"], fontsize=8.5)
        ax.set_title(p, loc="left", fontsize=10.5, color=INK)
        ax.set_xlabel("w: 0 default Assistant, 1 persona", fontsize=9, color=INK_2)
        style(ax)
    handles = [Line2D([], [], ls="none", marker=m, ms=6.5, color=c, mfc=c if f else "white", mew=1.4,
                      label=PLACE_LABEL[s]) for s, (m, c, f) in W_MARKS.items()]
    _title_and_legend(fig, f"{model.label}: each readout's place between the default Assistant and the system-prompt "
                      "persona\n(w against the same target in every placement, roleplay included)", handles,
                      ncol=min(5, 2 * len(ps) + 1))
    plotstyle.save(fig, path.stem, path.parent)


def _title_and_legend(fig, title, handles, ncol, right=1.0, legend_x=0.5):
    """Lay the axes out first, then put the title and a legend above them at fixed inch offsets (tight_layout reserves
    space for a multi-line suptitle on top of any rect, which left large gaps)."""
    height = fig.get_size_inches()[1]
    lines = title.count("\n") + 1
    rows = -(-len(handles) // ncol)
    top = 0.12 + 0.24 * lines + 0.24 * rows + 0.15
    fig.tight_layout(rect=(0, 0, right, 1 - top / height))
    fig.suptitle(title, color=INK, y=1 - 0.08 / height, va="top", fontsize=10.5)
    fig.legend(handles=handles, frameon=False, ncol=ncol, loc="upper center", fontsize=8.5,
               bbox_to_anchor=(legend_x, 1 - (0.14 + 0.24 * lines) / height))


def plot_summary(model, S, rows, path):
    """Each finding's call, with the numbers it rests on: estimate, 95% CI and the call's threshold (gray tick), one panel
    per finding, each on its own scale. Color is the call; the call is also written at the right."""
    groups = [(f, [r for r in rows if r["finding"] == f]) for f in FINDINGS]
    heights = [max(1, sum(max(1, len(r["plot"])) for r in rs)) for _, rs in groups]
    height = 0.42 * sum(heights) + 0.55 * len(groups) + 1.4
    fig, axes = plt.subplots(len(groups), 1, figsize=(9.5, height), gridspec_kw=dict(height_ratios=heights), squeeze=False)
    for ax, (f, rs) in zip(axes[:, 0], groups):
        labels, yi = [], 0
        ticks = []
        for r in rs:
            items = [it for it in r["plot"] if it[0] in S] or [(None, "", None)]
            color = CALL_COLOR.get(r["call"], NO_CALL)
            for j, (name, lab, thr) in enumerate(items):
                y = -yi
                who = f"{r['persona']}: " if r["persona"] else ""
                labels.append(f"{who}{lab}" if name else f"{who}(no data)")
                ticks.append(y)
                if name:
                    est, lo, hi = S[name]
                    ax.plot([lo, hi], [y, y], color=color, lw=1.8, solid_capstyle="round", zorder=2)
                    ax.plot(est, y, "o", ms=6.5, color=color, mec="white", mew=0.8, zorder=3)
                    ax.plot([thr, thr], [y - 0.32, y + 0.32], color=INK_2, lw=1.4, zorder=1)
                if j == 0:
                    ax.annotate(r["call"], (1.01, y), xycoords=("axes fraction", "data"), fontsize=8.5, color=INK,
                                va="center")
                yi += 1
        ax.set_yticks(ticks, labels, fontsize=8)
        ax.set_ylim(-yi + 0.4, 0.6)
        ax.axvline(0, color=RULE, lw=0.8, zorder=0)
        ax.set_title(f"{f}. {FINDINGS[f]}", fontsize=9.5, color=INK, loc="left")
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", visible=False)
        style(ax)
    handles = [Line2D([], [], ls="none", marker="o", ms=7, color=c, mec="white", label=lab)
               for lab, c in (("replicates", CALL_COLOR["replicates"]), ("doesn't", CALL_COLOR["doesn't"]),
                              ("no call (not separable, unstable, not interpretable, not run)", NO_CALL))]
    handles.append(Line2D([], [], color=INK_2, lw=1.4, marker="|", ls="none", ms=10, label="the call's threshold"))
    _title_and_legend(fig, f"{model.label}: the pre-registered calls (estimate and 95% CI)", handles, ncol=2,
                      right=0.86, legend_x=0.45)
    plotstyle.save(fig, path.stem, path.parent)


MATRIX_COLOR = {"replicates": plotstyle.BLUE_RAMP[7], "doesn't": plotstyle.SERIES[1], "not separable": "#c3c2b7",
                "unstable": "#c3c2b7", "not interpretable": "#c3c2b7", "not run": plotstyle.SURFACE}


MATRIX_LABEL = {1: "coin tracks A after post-training", 2: "user's choice more A-like", 3: "prediction of the user A-like",
                4: "user instruction", 5: "self-description < instruction", 6: "roleplay", 7: "P follows self-description"}


def matrix_rows(personas=PERSONAS):
    return [(f, None) for f in (1, 2, 3)] + [(f, p) for f in (4, 5, 6, 7) for p in personas]


def plot_matrix(columns, path):
    """Findings (rows; per persona for 4-7) x models (columns). columns: [(label, {(finding, persona): row})]."""
    rows = matrix_rows()
    fig, ax = plt.subplots(figsize=(3.2 + 2.2 * max(1, len(columns)), 0.42 * len(rows) + 1.6))
    for j, (label, cells) in enumerate(columns):
        for i, key in enumerate(rows):
            r = cells.get(key, dict(call="not run", key_number=""))
            call = r["call"]
            ax.add_patch(plt.Rectangle((j + 0.04, i + 0.06), 0.92, 0.88, color=MATRIX_COLOR[call],
                                       ec=RULE if call == "not run" else "none", lw=0.8))
            dark = call == "replicates"
            text = call if call == "not run" or not r.get("key_number") else f"{call}\n{r['key_number']}"
            ax.text(j + 0.5, i + 0.5, text, ha="center", va="center", fontsize=7, linespacing=1.15,
                    color="white" if dark else INK)
    ax.set_xlim(0, max(1, len(columns)))
    ax.set_ylim(len(rows), 0)
    ax.set_xticks(np.arange(len(columns)) + 0.5, [c[0] for c in columns], fontsize=8.5)
    ax.xaxis.tick_top()
    ylabels = [f"{f}. {MATRIX_LABEL[f]}" + (f": {p}" if p else "") for f, p in rows]
    ax.set_yticks(np.arange(len(rows)) + 0.5, ylabels, fontsize=8.5)
    for i, (f, p) in enumerate(rows):  # group separators between findings
        if i and rows[i - 1][0] != f:
            ax.axhline(i, color=INK_2, lw=0.8)
    ax.tick_params(length=0)
    ax.grid(False)
    for side in ax.spines.values():
        side.set_visible(False)
    legend = ("Findings (README, \"Replication across larger models\"): "
              + "; ".join(f"{f}. {FINDINGS[f]}" for f in FINDINGS) + ". A: the Assistant's own choice; P: its prediction "
              "of the user's choice.")
    fig.text(0.01, 0.005, legend, fontsize=7, color=INK_2, wrap=True)
    fig.suptitle("Replication: each finding's pre-registered call by model", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.7 / fig.get_size_inches()[1], 1, 0.97))
    plotstyle.save(fig, path.stem, path.parent)


# ----------------------------------------------------------------------------------------------------------------- main

def analyze_model(key, n_boot=N_BOOTSTRAP, figures=True):
    """Load, compute, call, and write one model's tables, report and figures. Returns (model, stats, finding rows)."""
    model = Model(key)
    print(f"{key}: {model.label}, {len(model.U)} conditions")
    S = bootstrap(model.U, model.origins, model.nonharm, resamples_for(model.origins, n_boot), model.personas()) \
        if model.U else {}
    rows = findings(S, model)
    out = model.dir / "analysis"
    write_tables(model, S, rows, out)
    text = report_text(model, S, rows)
    (out / "report.txt").write_text(text)
    print(text)
    if figures:
        fig_dir = FIGURES / key
        plotstyle.setup()
        plot_summary(model, S, rows, fig_dir / "summary.png")
        if model.U:
            plot_coin(model, S, fig_dir / "coin.png")
            plot_user_predict(model, S, fig_dir / "user_predict.png")
            plot_plain_sources(model, fig_dir / "sources_plain.png")
            plot_pulls(model, S, fig_dir / "persona_pulls.png")
            plot_w(model, S, fig_dir / "persona_w.png")
            for p in model.personas():
                plot_persona_sources(model, p, fig_dir / f"sources_{p}.png")
    return model, S, rows


def configured_keys():
    """The replication's models, configs/replication/*.yaml without validate-*, in file order."""
    keys = []
    for path in sorted(CONFIGS.glob("*.yaml")):
        if path.stem.startswith("validate-"):
            continue
        cfg = yaml.safe_load(path.read_text()) or {}
        keys.append(cfg.get("run_name", path.stem))
    return keys


def read_stats(key):
    """{name: (est, lo, hi)} from a model's stats.csv."""
    with open(OUT / key / "analysis" / "stats.csv", newline="") as fh:
        return {r["name"]: (float(r["est"]), float(r["lo"]), float(r["hi"])) for r in csv.DictReader(fh)}


def read_findings(key):
    path = OUT / key / "analysis" / "findings.csv"
    if not path.exists():
        return {}
    with open(path, newline="") as fh:
        return {(int(r["finding"]), r["persona"] or None): r for r in csv.DictReader(fh)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("keys", nargs="+", help="model keys under outputs/replication/")
    ap.add_argument("--n-boot", type=int, default=N_BOOTSTRAP)
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args(argv)
    for key in a.keys:
        analyze_model(key, a.n_boot, figures=not a.no_figures)
    keys = configured_keys() + [k for k in a.keys if k not in configured_keys()]
    plotstyle.setup()
    plot_matrix([(model_label(k).replace(" (legacy)", "\n(legacy)"), read_findings(k)) for k in keys],
                FIGURES / "replication_matrix.png")


if __name__ == "__main__":
    main()
