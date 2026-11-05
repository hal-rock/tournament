"""The user's hope about the coin: is a preference the user states, but that can't change the outcome, as Assistant-like as
the coin's tilt and the user's own choice?

The measures and calls were pre-registered in the README ("The user's hope about the coin") before any data. Collects the
W&B runs of group "hope" (configs/hope/, one per model and stage). Every pod scores all of a model's conditions, so each
comparison is within one pod and GPU type:

  A   assistant              the Assistant's choice
  C   coinflip               the coin ("Okay, I'm flipping the coin now ... it came up")
  C0  coinflip_no_okay       the same without "Okay, "
  U   user_choice            the user's choice ("Okay, I think I'll choose Task")
  U0  user_choice_no_okay    the same without "Okay, "
  H   hope_before            "I hope it comes up", before the flip (no opener)
  Wh  hope_after_heads       the coinflip, then "heads. I was hoping it would come up"
  Wt  hope_after_tails       the same with "tails."
  W   (Wh + Wt) / 2          the hope after the flip, both outcomes

_b and _f are base and final. Each hope has opener-matched comparators: H against U0 and C0 (no opener), W against U and
C (its line opens with "Okay, ", and its prompt starts with C's, word for word). Per model and hope X, with Y_u and Y_c
its user's choice and coin:

- primary: d_user = r(A_f, X_f) - r(A_f, Y_u,f) and d_coin = r(A_f, X_f) - r(A_f, Y_c,f), paired;
- the same over non-harmful tasks, within origin, and on post-training's change (r(A_f, X_f - X_b), "r change");
- r(A_b, X_b), r(A_f, X_b), d_post = r(A_f, X_f) - r(A_f, X_b), r shift r(A_f - A_b, X_f - X_b);
- gain (slope of X_f - X_b on A_f, nats per Assistant SD) and its ratio to the comparators' gains; slope b, gap
  (non-harmful minus harmful), SD;
- r(X, Y_u), r(X, Y_c), and the partials r(A_f, X_f | Y_u,f), r(A_f, Y_u,f | X_f), r(A_f, Y_c,f | X_f);
- r(X_f, log task length).
Before vs after: r(A_f, H_f) - r(A_f, W_f); the same with each against its own matched user's choice and coin
(difference in differences, which takes out the opener if it acts on the hope as on the choice); r(H, W); r(Wh, Wt).
The opener: r(A_f, U0_f) - r(A_f, U_f) and r(A_f, C0_f) - r(A_f, C_f).

The outcome terms of the after condition (not task utilities): per pair, bias(Wh) - bias(Wt) = 2c + v_x + v_y, where c is
how much more the user is said to have hoped for the side that came up (negative: the other side) and v_t is how much
more the user hoped for task t when it came up than when it didn't (exploratory).

CIs are 95% task bootstraps within origin, one set of resamples for everything, so differences are paired. `call` gives the
pre-registered call per model and hope.

  outputs/hope/analysis/<family>/   the merged fit (analyze.py) of one model's stages
  outputs/hope/hope.csv             one row per model and hope
  outputs/hope/timing.csv           before vs after, per model
  outputs/hope/outcome.csv          c and v per model and stage
  outputs/hope/reproduction.csv     the re-scored references against the earlier runs
  figures/hope_*.png

Usage: uv run python hope.py [--group hope]
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import (N_BOOTSTRAP, ORIGIN_STYLE, analyze, fetch_wandb as fetch_raw, pair_effects, partial_r, pearson,
                     read_jsonl, split_half)
from matskit import plotstyle
from models import MODEL_INFO, PROJECT, olmo_on_shared_pairs
from openers import slope_per_sd, within_groups, write_csv
from report import INK, INK_2, RULE, fetch_wandb, origin_scatter, stratified_resamples, style, task_arrays
from user_choice import SHARED_PAIRS, TASKS_FILE, ci, collect_runs, top_token_shares

OLMO, GEMMA, LLAMA, GLM = ("allenai/Olmo-3.1-32B-Instruct", "google/gemma-4-31B-it", "unsloth/Llama-3.3-70B-Instruct",
                           "zai-org/GLM-4.5-Air")
FAMILIES = {"olmo3.1-32b": OLMO, "gemma4-31b": GEMMA, "llama-70b": LLAMA, "glm-4.5-air": GLM}
STAGES = ("base", "instruct")
SUFFIX = {"base": "b", "instruct": "f"}  # array keys: "<key>_b", "<key>_f"
COND = {"A": "assistant", "C": "coinflip", "C0": "coinflip_no_okay", "U": "user_choice", "U0": "user_choice_no_okay",
        "H": "hope_before", "Wh": "hope_after_heads", "Wt": "hope_after_tails"}
# Each hope's opener-matched (user's choice, coin).
HOPES = {"H": ("U0", "C0"), "W": ("U", "C")}
HOPE_LABEL = {"H": "Hope before the flip", "W": "Hope after the flip"}
R_MEANINGFUL = 0.05  # a difference in r worth calling one
MIN_RELIABILITY, MIN_MASS = 0.9, 0.2
# Reproduction of the earlier runs' references: final-stage cells must match at r >= GATE_SAME_R and SD within GATE_SD on
# the same GPU type, at r >= GATE_OTHER_R on another type. Base cells (tiny-SD coins) are reported only.
GATE_SAME_R, GATE_SD, GATE_OTHER_R = 0.999, 0.02, 0.99

PER = ("r_final", "r_user", "r_coin", "d_user", "d_coin",
       "r_final_benign", "r_user_benign", "r_coin_benign", "d_user_benign", "d_coin_benign",
       "r_final_within", "r_user_within", "r_coin_within", "d_user_within", "d_coin_within",
       "r_change", "r_change_user", "r_change_coin", "dchange_user", "dchange_coin",
       "r_shift", "r_shift_user", "r_shift_coin", "r_base", "r_fixed", "d_post",
       "gain", "gain_user", "gain_coin", "gain_ratio_user", "gain_ratio_coin",
       "b_base", "b_final", "b_user", "b_coin", "gap_base", "gap_final", "gap_user", "gap_coin",
       "sd_base", "sd_final", "sd_user", "sd_coin",
       "r_xu_base", "r_xu_final", "r_xc_base", "r_xc_final", "partial_ax_u", "partial_au_x", "partial_ac_x",
       "r_len", "r_len_user")
TIMING = ("dr_hw", "did_user", "did_coin", "r_hw_base", "r_hw_final", "r_whwt_base", "r_whwt_final",
          "r_wh_final", "r_wt_final", "dr_whwt", "opener_user", "opener_coin")
STATS = tuple(f"{x}:{s}" for x in HOPES for s in PER) + TIMING
K = {name: j for j, name in enumerate(STATS)}


def with_w(arrays):
    """arrays plus W_b, W_f = the mean of the two outcomes' utilities (the fit is linear in the pair effects, so this is
    the fit of their mean on the same pairs)."""
    return {**arrays, **{f"W_{s}": (arrays[f"Wh_{s}"] + arrays[f"Wt_{s}"]) / 2 for s in "bf"}}


def hope_stats(a, x, yu, yc, benign, origins, loglen):
    """PER for hope x (a key of a, e.g. "H") against its user's choice yu and coin yc."""
    w = lambda v: within_groups(v, origins)
    gap = lambda v: v[benign].mean() - v[~benign].mean()
    af, ab = a["A_f"], a["A_b"]
    X, Yu, Yc = ((a[f"{k}_b"], a[f"{k}_f"]) for k in (x, yu, yc))
    rf = lambda v: pearson(af, v)
    rb = lambda v: pearson(af[benign], v[benign])
    rw = lambda v: pearson(w(af), w(v))
    rc = lambda v: pearson(af, v[1] - v[0])
    gain = lambda v: slope_per_sd(af, v[1] - v[0])
    out = [rf(X[1]), rf(Yu[1]), rf(Yc[1]), rf(X[1]) - rf(Yu[1]), rf(X[1]) - rf(Yc[1]),
           rb(X[1]), rb(Yu[1]), rb(Yc[1]), rb(X[1]) - rb(Yu[1]), rb(X[1]) - rb(Yc[1]),
           rw(X[1]), rw(Yu[1]), rw(Yc[1]), rw(X[1]) - rw(Yu[1]), rw(X[1]) - rw(Yc[1]),
           rc(X), rc(Yu), rc(Yc), rc(X) - rc(Yu), rc(X) - rc(Yc),
           pearson(af - ab, X[1] - X[0]), pearson(af - ab, Yu[1] - Yu[0]), pearson(af - ab, Yc[1] - Yc[0]),
           pearson(ab, X[0]), rf(X[0]), rf(X[1]) - rf(X[0]),
           gain(X), gain(Yu), gain(Yc), gain(X) / gain(Yu), gain(X) / gain(Yc),
           slope_per_sd(ab, X[0]), slope_per_sd(af, X[1]), slope_per_sd(af, Yu[1]), slope_per_sd(af, Yc[1]),
           gap(X[0]), gap(X[1]), gap(Yu[1]), gap(Yc[1]), X[0].std(), X[1].std(), Yu[1].std(), Yc[1].std(),
           pearson(X[0], Yu[0]), pearson(X[1], Yu[1]), pearson(X[0], Yc[0]), pearson(X[1], Yc[1]),
           partial_r(af, X[1], Yu[1]), partial_r(af, Yu[1], X[1]), partial_r(af, Yc[1], X[1]),
           pearson(X[1], loglen), pearson(Yu[1], loglen)]
    assert len(out) == len(PER)
    return out


def model_stats(arrays, benign, origins, loglen):
    """STATS of one model. arrays: {"<key>_<b|f>": utilities} for every key of COND."""
    a = with_w(arrays)
    out = []
    for x, (yu, yc) in HOPES.items():
        out += hope_stats(a, x, yu, yc, benign, origins, loglen)
    p = len(PER)
    h, wv = out[:p], out[p:]
    d = lambda vals, name: vals[PER.index(name)]
    rf = lambda v: pearson(a["A_f"], v)
    out += [d(h, "r_final") - d(wv, "r_final"), d(h, "d_user") - d(wv, "d_user"), d(h, "d_coin") - d(wv, "d_coin"),
            pearson(a["H_b"], a["W_b"]), pearson(a["H_f"], a["W_f"]),
            pearson(a["Wh_b"], a["Wt_b"]), pearson(a["Wh_f"], a["Wt_f"]),
            rf(a["Wh_f"]), rf(a["Wt_f"]), rf(a["Wh_f"]) - rf(a["Wt_f"]),
            rf(a["U0_f"]) - rf(a["U_f"]), rf(a["C0_f"]) - rf(a["C_f"])]
    return np.array(out)


def model_cell(arrays, benign, origins, loglen, resamples):
    return dict(est=model_stats(arrays, benign, origins, loglen),
                boot=np.array([model_stats({k: v[i] for k, v in arrays.items()}, benign[i], origins[i], loglen[i])
                               for i in resamples]))


def stat(cell, name):
    """(estimate, ci_low, ci_high)."""
    lo, hi = ci(cell["boot"][:, K[name]])
    return cell["est"][K[name]], lo, hi


def compare(cell, x, against):
    """Is hope x less, more or similarly Assistant-like than its matched "user" choice or "coin", by r with A_f?"""
    d, lo, hi = stat(cell, f"{x}:d_{against}")
    if d <= -R_MEANINGFUL and hi < 0:
        return "less"
    if d >= R_MEANINGFUL and lo > 0:
        return "more"
    return "similar"


def call(cell, x):
    """The pre-registered call for one model and hope: (vs the user's choice, vs the coin, verdict on the choice account).
    The choice account predicts "less" against both."""
    u, c = compare(cell, x, "user"), compare(cell, x, "coin")
    verdict = "supports" if u == c == "less" else "against" if "less" not in (u, c) else "mixed"
    return u, c, verdict


def outcome_terms(heads, tails, ids):
    """The after condition's outcome terms at one stage, from pair_effects entries of hope_after_heads and _tails:
    per pair bias(heads) - bias(tails) = 2c + v_x + v_y (see the module docstring). Returns (c, v (centered, in ids
    order), share of pairs where the user is said to have hoped more for the side that didn't come up)."""
    assert heads["pairs"] == tails["pairs"], "the two outcomes must cover the same pairs"
    q = heads["bias"] - tails["bias"]
    idx = {t: i for i, t in enumerate(ids)}
    ia = np.array([idx[x] for x, _ in heads["pairs"]])
    ib = np.array([idx[y] for _, y in heads["pairs"]])
    n = len(ids)
    # Least squares q ~ 2c + v[ia] + v[ib] with sum(v) = 0: the constraint as an extra row, which leaves c and v exact.
    D = np.zeros((len(q) + 1, n + 1))
    D[np.arange(len(q)), 0] = 2
    np.add.at(D, (np.arange(len(q)), 1 + ia), 1)
    np.add.at(D, (np.arange(len(q)), 1 + ib), 1)
    D[-1, 1:] = 1
    coef = np.linalg.lstsq(D, np.append(q, 0), rcond=None)[0]
    return float(coef[0]), coef[1:], float((q < 0).mean())


def reliabilities(rows, ids, stages=STAGES):
    """Split-half reliability of every hope cell, W included (the mean of the two outcomes), and of A, at each stage."""
    effects = pair_effects(rows)
    measures = {}
    for s in stages:
        for k in ("A", "C", "C0", "U", "U0", "H", "Wh", "Wt"):
            measures[f"{k}_{SUFFIX[s]}"] = {(s, COND[k]): 1.0}
        measures[f"W_{SUFFIX[s]}"] = {(s, COND["Wh"]): 0.5, (s, COND["Wt"]): 0.5}
    out = split_half(ids, effects, measures, np.random.default_rng(0))
    return {name: rel for name, (_, rel) in out.items()}


def disattenuated(r, rel_x, rel_a):
    return r / np.sqrt(rel_x * rel_a)


def gpu_of(run):
    """"<count>x <GPU>" from a W&B run's metadata, or None. d8ysbybo predates the metadata; the README has it on an A100."""
    if run is None:
        return None
    if run.id == "d8ysbybo":
        return "1x NVIDIA A100-SXM4-80GB"
    m = run.metadata or {}
    return f"{m['gpu_count']}x {m['gpu']}" if m.get("gpu") else None


def gpu_type(gpu):
    """The GPU model alone: numerics follow the chip, not how many of them share the model."""
    return gpu.split("x ", 1)[1] if gpu else None


# Earlier runs of the reference conditions, for the reproduction check: family -> [(our key, W&B run id or None for OLMo's
# refit sweep, the reference's measure name, where its utilities come from)].
SWEEP = {"olmo3.1-32b": None, "gemma4-31b": "jrl62npf", "llama-70b": "tvlqj8a0", "glm-4.5-air": "i5vejs3b"}
OPENERS = {"olmo3.1-32b": "gelh6haq", "llama-70b": "1caw4kd1", "glm-4.5-air": "jl7n6tbu"}
USER_CHOICE = {"olmo3.1-32b": ("wtf6085z", "zr5su3zo"), "gemma4-31b": ("oejp2bs9", "rxow1qso"),
               "llama-70b": ("7v41hj1v", "bhv8okeh")}


def reference_fits(family):
    """[(our measure, reference utilities rows, reference measure, reference run id)] for the reproduction check."""
    import wandb
    api = wandb.Api()

    def run_analysis(run_id):
        analysis = Path("outputs") / run_id / "analysis"
        fetch_wandb(api.run(f"{PROJECT}/{run_id}"), analysis)
        with open(analysis / "utilities.csv", newline="") as fh:
            return list(csv.DictReader(fh))

    out = []
    sweep = olmo_on_shared_pairs(SHARED_PAIRS)[0] if SWEEP[family] is None else run_analysis(SWEEP[family])
    for s in STAGES:
        for k in ("A", "C"):
            out.append((f"{s}/{COND[k]}", sweep, f"{s}/{COND[k]}", SWEEP[family] or "d8ysbybo"))
    if family in OPENERS:
        out.append(("instruct/coinflip_no_okay", run_analysis(OPENERS[family]), "instruct/coinflip_no_okay",
                    OPENERS[family]))
    if family in USER_CHOICE:
        with open(Path("outputs/user_choice/analysis") / family / "utilities.csv", newline="") as fh:
            uc = list(csv.DictReader(fh))
        for s, run_id in zip(STAGES, USER_CHOICE[family]):
            out.append((f"{s}/user_choice", uc, f"{s}/user_choice", run_id))
    return out


def reproduction(family, ours, our_gpu):
    """Rows comparing our re-scored references with the earlier runs, and whether the final-stage cells pass."""
    import wandb
    api = wandb.Api()
    rows, passed = [], True
    for measure, ref_rows, ref_measure, run_id in reference_fits(family):
        ref = np.array([float(r[ref_measure]) for r in ref_rows])
        x = ours[measure]
        ref_gpu = gpu_of(api.run(f"{PROJECT}/{run_id}"))
        same = gpu_type(ref_gpu) is not None and gpu_type(ref_gpu) == gpu_type(our_gpu)
        r, dsd = pearson(x, ref), x.std() / ref.std() - 1
        gated = measure.startswith("instruct/")
        ok = (r >= GATE_SAME_R and abs(dsd) <= GATE_SD) if same else r >= GATE_OTHER_R
        passed &= ok or not gated
        rows.append(dict(model=MODEL_INFO[FAMILIES[family]][0], measure=measure, reference=run_id, our_gpu=our_gpu,
                         ref_gpu=ref_gpu, same_gpu_type=same, r=r, sd_change=dsd,
                         result=("pass" if ok else "FAIL") if gated else "reported"))
    return rows, passed


def merged_fit(runs, family, out):
    """(utilities.csv rows, summary, raw rows, {stage: run}) of a model's two stages, fit together; cached in out."""
    rows, by_stage = [], {}
    for s in STAGES:
        run = runs[family, s]
        print(f"{family:12} {s:9} {run.path}")
        rows += fetch_raw(run, Path("outputs") / run.id)
        by_stage[s] = run
    if not (out / "utilities.csv").exists():
        analyze(rows, read_jsonl(TASKS_FILE), out, seed=0, stage_order=list(STAGES))
    with open(out / "utilities.csv", newline="") as fh:
        return list(csv.DictReader(fh)), json.loads((out / "summary.json").read_text()), rows, by_stage


# Figures. Color = who or what decides (coin, the user's choice, the user's hope); filled = the line opens with "Okay, ",
# open = it doesn't. The hope after the flip opens with "Okay, " (its line is the coinflip's, continued).
CELLS = (("C", "Coin"), ("C0", "Coin, no \"Okay, \""), ("U", "User's choice"), ("U0", "User's choice, no \"Okay, \""),
         ("W", "Hope after the flip"), ("H", "Hope before the flip (no \"Okay, \")"))
CELL_STYLE = {"C": (0, "o", True), "C0": (0, "o", False), "U": (1, "s", True), "U0": (1, "s", False),
              "W": (2, "D", True), "H": (2, "D", False)}


def marker(ax, x, y, key, label=None, lo=None, hi=None, horizontal=False):
    c, m, filled = CELL_STYLE[key]
    color = plotstyle.SERIES[c]
    if lo is not None:
        if horizontal:
            ax.plot([lo, hi], [y, y], color=color, lw=1.6, zorder=2, solid_capstyle="round")
        else:
            ax.plot([x, x], [lo, hi], color=color, lw=1.6, zorder=2, solid_capstyle="round")
    ax.plot(x, y, ls="none", marker=m, ms=7.5, color=color, mfc=color if filled else "white", mec=color, mew=1.6,
            zorder=4, label=label)


def legend_handles():
    from matplotlib.lines import Line2D
    out = []
    for key, label in CELLS:
        c, m, filled = CELL_STYLE[key]
        color = plotstyle.SERIES[c]
        out.append(Line2D([], [], ls="none", marker=m, ms=7.5, color=color, mfc=color if filled else "white", mec=color,
                          mew=1.6, label=label))
    return out


def plot_r(cells, labels, path):
    """r with the final Assistant of every final-stage cell, one panel per model, with 95% CIs."""
    # (cell key, stat giving its r(A_f, .)): each comparator is read off the hope it is matched to.
    src = {"C": "W:r_coin", "C0": "H:r_coin", "U": "W:r_user", "U0": "H:r_user", "W": "W:r_final", "H": "H:r_final"}
    fig, axes = plt.subplots(1, len(cells), figsize=(3.3 * len(cells), 3.8), sharey=True, squeeze=False)
    y = np.arange(len(CELLS))[::-1]
    for ax, (family, cell) in zip(axes[0], cells.items()):
        for yi, (key, _) in zip(y, CELLS):
            est, lo, hi = stat(cell, src[key])
            marker(ax, est, yi, key, lo=lo, hi=hi, horizontal=True)
        ax.set_yticks(y, [label for _, label in CELLS], fontsize=8.5)
        ax.set_title(labels[family], loc="left", fontsize=10.5, color=INK)
        ax.set_xlabel("r with the final Assistant", fontsize=9, color=INK_2)
        style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_contrasts(cells, labels, path):
    """The primary differences in r with the final Assistant, each hope against its matched user's choice and coin, and
    before minus after. The band marks |difference| < 0.05, below which no call is made."""
    rows = (("H:d_user", "Before - user's choice", "H"), ("H:d_coin", "Before - coin", "H"),
            ("W:d_user", "After - user's choice", "W"), ("W:d_coin", "After - coin", "W"),
            ("dr_hw", "Before - after", "H"), ("did_user", "Before - after, each\nminus its user's choice", "H"))
    fig, axes = plt.subplots(1, len(cells), figsize=(3.3 * len(cells), 3.6), sharey=True, squeeze=False)
    y = np.arange(len(rows))[::-1]
    for ax, (family, cell) in zip(axes[0], cells.items()):
        ax.axvspan(-R_MEANINGFUL, R_MEANINGFUL, color=RULE, alpha=0.35, lw=0, zorder=0)
        ax.axvline(0, color=INK_2, lw=0.8, zorder=1)
        for yi, (name, _, key) in zip(y, rows):
            est, lo, hi = stat(cell, name)
            marker(ax, est, yi, key, lo=lo, hi=hi, horizontal=True)
        ax.set_yticks(y, [label for _, label, _ in rows], fontsize=8.5)
        ax.set_title(labels[family], loc="left", fontsize=10.5, color=INK)
        ax.set_xlabel("Difference in r with the final Assistant", fontsize=9, color=INK_2)
        style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scatter(data, origins, labels, path):
    """Per model (rows): each hope against the final Assistant, against its matched user's choice, and the two hopes."""
    panels = (("A_f", "H_f", "Final Assistant", "Hope before"), ("A_f", "W_f", "Final Assistant", "Hope after"),
              ("U0_f", "H_f", "User's choice, no \"Okay, \"", "Hope before"), ("U_f", "W_f", "User's choice", "Hope after"),
              ("H_f", "W_f", "Hope before", "Hope after"))
    fig, axes = plt.subplots(len(data), len(panels), figsize=(3.1 * len(panels), 2.9 * len(data)), squeeze=False)
    for row, (family, arrays) in enumerate(data.items()):
        a = with_w(arrays)
        for col, (kx, ky, lx, ly) in enumerate(panels):
            ax = axes[row, col]
            origin_scatter(ax, a[kx], a[ky], origins)
            ax.axhline(0, color=RULE, lw=0.8, zorder=1)
            ax.set_title(f"r = {pearson(a[kx], a[ky]):.2f}", fontsize=9, color=INK_2, loc="left")
            ax.set_xlabel(f"{lx} (nats)", fontsize=8, color=INK_2)
            ax.set_ylabel(f"{labels[family]}\n{ly} (nats)" if col == 0 else f"{ly} (nats)", fontsize=8, color=INK_2)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, frameon=False, ncol=len(ORIGIN_STYLE), loc="upper center", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--group", default="hope")
    args = p.parse_args()
    out = Path("outputs/hope")
    out.mkdir(parents=True, exist_ok=True)
    runs = collect_runs(args.group)
    data, cells, labels, rels, gates, passed = {}, {}, {}, {}, [], True
    table, timing, outcome, repro, tops = [], [], [], [], {}
    ids0 = resamples = origins0 = None
    for family, final in FAMILIES.items():
        if not all((family, s) in runs for s in STAGES):
            print(f"{family}: not run (yet)")
            continue
        rows, summary, raw, by_stage = merged_fit(runs, family, out / "analysis" / family)
        ids = [r["id"] for r in rows]
        U, origins, benign, loglen = task_arrays(rows)
        if ids0 is None:
            ids0, origins0 = ids, origins
            resamples = stratified_resamples(origins, np.random.default_rng(0), N_BOOTSTRAP)
        assert ids == ids0, "task sets differ between models"
        labels[family] = MODEL_INFO[final][0]
        arrays = data[family] = {f"{k}_{SUFFIX[s]}": U[f"{s}/{c}"] for k, c in COND.items() for s in STAGES}

        gpus = {s: gpu_of(by_stage[s]) for s in STAGES}
        if gpu_type(gpus["base"]) != gpu_type(gpus["instruct"]):
            print(f"  note: {family}'s stages ran on different GPU types: {gpus}")
        rels[family] = rel = reliabilities(raw, ids)
        for s in STAGES:
            for k in ("H", "Wh", "Wt", "W"):
                mass = summary[f"{s}/{COND[k]}/outcome_mass_mean"] if k != "W" else None
                ok = rel[f"{k}_{SUFFIX[s]}"] >= MIN_RELIABILITY or s == "base"
                passed &= ok
                gates.append(f"{labels[family]:14} {s:9} {k:3} reliability {rel[f'{k}_{SUFFIX[s]}']:.3f}"
                             + (f", mass {mass:.2f}" if mass is not None else "")
                             + (f": {'pass' if ok else 'FAIL'}" if s == "instruct" else " (reported only)"))
        rep, ok = reproduction(family, {f"{s}/{c}": U[f"{s}/{c}"] for s in STAGES for c in COND.values()}, gpus["instruct"])
        repro += rep
        passed &= ok

        cell = cells[family] = model_cell(arrays, benign, origins, loglen, resamples)
        for x, (yu, yc) in HOPES.items():
            vs_user, vs_coin, verdict = call(cell, x)
            row = dict(model=labels[family], hope=x, vs_user=vs_user, vs_coin=vs_coin, choice_account=verdict)
            masses = {s: [summary[f"{s}/{COND[k]}/outcome_mass_mean"] for k in (("H",) if x == "H" else ("Wh", "Wt"))]
                      for s in STAGES}
            row.update(mass_base=min(masses["base"]), mass_final=min(masses["instruct"]),
                       readable=min(masses["instruct"]) >= MIN_MASS,
                       rel_base=rel[f"{x}_b"], rel_final=rel[f"{x}_f"])
            for name in PER:
                est, lo, hi = stat(cell, f"{x}:{name}")
                row.update({name: est, f"{name}_lo": lo, f"{name}_hi": hi})
            # Disattenuated primary differences (point estimates, from split-half reliabilities).
            ra = rel["A_f"]
            for against, ykey in (("user", yu), ("coin", yc)):
                row[f"d_{against}_disattenuated"] = (disattenuated(row["r_final"], rel[f"{x}_f"], ra)
                                                     - disattenuated(row[f"r_{against}"], rel[f"{ykey}_f"], ra))
            table.append(row)
        trow = dict(model=labels[family])
        for name in TIMING:
            est, lo, hi = stat(cell, name)
            trow.update({name: est, f"{name}_lo": lo, f"{name}_hi": hi})
        timing.append(trow)

        effects = pair_effects(raw)
        for s in STAGES:
            c, v, share = outcome_terms(effects[s, COND["Wh"]], effects[s, COND["Wt"]], ids)
            w = (arrays[f"Wh_{SUFFIX[s]}"] + arrays[f"Wt_{SUFFIX[s]}"]) / 2
            outcome.append(dict(model=labels[family], stage=s, c=c, share_hoped_other_side=share, v_sd=float(v.std()),
                                r_v_assistant=pearson(v, arrays["A_f"]), r_v_hope=pearson(v, w),
                                r_v_loglen=pearson(v, loglen),
                                bias_heads=summary[f"{s}/{COND['Wh']}/slot_bias_mean"],
                                bias_tails=summary[f"{s}/{COND['Wt']}/slot_bias_mean"]))
        tops[family] = {(s, k): top_token_shares(Path("outputs") / by_stage[s].id / "results" / f"{s}__{COND[k]}.csv.gz")
                        for s in STAGES for k in ("H", "Wh", "Wt")}

    print("Gate (split-half reliability of the hope cells, final stage):\n  " + "\n  ".join(gates))
    print("\nReproduction of the earlier runs' references (final stage gated, base reported):")
    for r in repro:
        print(f"  {r['model']:14} {r['measure']:28} vs {r['reference']}: r {r['r']:.4f}, SD {r['sd_change']:+.1%} "
              f"({r['our_gpu']} vs {r['ref_gpu']}{', same type' if r['same_gpu_type'] else ''}): {r['result']}")
    fm = lambda r, s: f"{r[s]:+.2f} [{r[f'{s}_lo']:+.2f}, {r[f'{s}_hi']:+.2f}]"
    for r in table:
        yu, yc = HOPES[r["hope"]]
        print(f"\n{r['model']}, {HOPE_LABEL[r['hope']].lower()}: vs the user's choice ({yu}) {r['vs_user']}, "
              f"vs the coin ({yc}) {r['vs_coin']}; choice account: {r['choice_account']}"
              + ("" if r["readable"] else "  (mass below the floor: reported, not interpreted)"))
        for name, label in (("r_final", "r(A_f, X_f)"), ("r_user", f"r(A_f, {yu}_f)"), ("r_coin", f"r(A_f, {yc}_f)"),
                            ("d_user", "d_user (primary)"), ("d_coin", "d_coin (primary)"),
                            ("d_user_benign", "d_user, non-harmful"), ("d_coin_benign", "d_coin, non-harmful"),
                            ("d_user_within", "d_user, within origin"), ("d_coin_within", "d_coin, within origin"),
                            ("r_change", "r(A_f, X_f - X_b)"), ("dchange_user", "  minus the user's choice's"),
                            ("dchange_coin", "  minus the coin's"),
                            ("r_base", "r(A_b, X_b)"), ("r_fixed", "r(A_f, X_b)"), ("d_post", "d_post"),
                            ("r_shift", "r shift"), ("r_shift_user", "r shift, user's choice"),
                            ("r_shift_coin", "r shift, coin"),
                            ("gain", "gain"), ("gain_ratio_user", "gain / user's choice's"),
                            ("gain_ratio_coin", "gain / coin's"),
                            ("b_final", "b, final"), ("b_user", "b, user's choice"), ("b_coin", "b, coin"),
                            ("gap_base", "gap, base"), ("gap_final", "gap, final"), ("gap_user", "gap, user's choice"),
                            ("gap_coin", "gap, coin"), ("sd_base", "SD, base"), ("sd_final", "SD, final"),
                            ("r_xu_final", "r(X_f, user's choice)"), ("r_xc_final", "r(X_f, coin)"),
                            ("partial_ax_u", "r(A_f, X_f | user's choice)"),
                            ("partial_au_x", "r(A_f, user's choice | X_f)"), ("partial_ac_x", "r(A_f, coin | X_f)"),
                            ("r_len", "r(X_f, log length)"), ("r_len_user", "r(user's choice, log length)")):
            print(f"  {label:32} {fm(r, name)}")
        print(f"  disattenuated d_user {r['d_user_disattenuated']:+.2f}, d_coin {r['d_coin_disattenuated']:+.2f}; "
              f"reliability base {r['rel_base']:.3f}, final {r['rel_final']:.3f}; mass base {r['mass_base']:.2f}, "
              f"final {r['mass_final']:.2f}")
    print("\nBefore vs after:")
    for r in timing:
        print(f"  {r['model']}")
        for name in TIMING:
            print(f"    {name:14} {fm(r, name)}")
    print("\nOutcome terms of the hope after the flip (c < 0: the user is said to have hoped for the side that didn't "
          "come up):")
    for r in outcome:
        print(f"  {r['model']:14} {r['stage']:9} c {r['c']:+.2f}, share of pairs hoping for the other side "
              f"{r['share_hoped_other_side']:.2f}, slot bias heads {r['bias_heads']:+.2f} / tails {r['bias_tails']:+.2f}; "
              f"v (exploratory): SD {r['v_sd']:.2f}, r with A_f {r['r_v_assistant']:+.2f}, with W "
              f"{r['r_v_hope']:+.2f}, with log length {r['r_v_loglen']:+.2f}")
    print("\nWhere the probability goes (mean over prompts, tokens logged in each top 5):")
    for family, by in tops.items():
        for (s, k), top in by.items():
            print(f"  {labels[family]:14} {s:9} {COND[k]:18} " + "  ".join(f"{t!r} {p:.3f}" for t, p in top))

    write_csv(out / "hope.csv", table)
    write_csv(out / "timing.csv", timing)
    write_csv(out / "outcome.csv", outcome)
    write_csv(out / "reproduction.csv", repro)
    plotstyle.setup()
    plot_r(cells, labels, "figures/hope_r.png")
    plot_contrasts(cells, labels, "figures/hope_contrasts.png")
    plot_scatter(data, origins0, labels, "figures/hope_scatter.png")
    print(f"\nGate {'passed' if passed else 'FAILED'}. Wrote {out}/ and figures/hope_*.png")


if __name__ == "__main__":
    main()
