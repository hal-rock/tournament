"""What does the Assistant think the user will choose? Does the user's choice, written in the user's turn, line up better with
the Assistant's prediction of that choice or with the Assistant's own choice?

The measures and calls were pre-registered in the README ("What does the Assistant think the user will choose?") before any
data. Collects the W&B runs of group "predict_user" (configs/predict_user/, one per model and stage). Every pod scores:

  A   assistant       the Assistant's own choice (Assistant prefilled "I'll choose Task")
  U   user_choice     the user's choice (the user turn left open at "Okay, I think I'll choose Task")
  P   predict_user    the Assistant's prediction of the user's choice (the user asks "Before I decide, which one do you
                      think I'll choose?"; Assistant prefilled "I think you'll choose Task")

and, from the hope run's pods (group "hope", same GPU types, secondary measures only):

  C   coinflip              the coin
  U0  user_choice_no_okay   the user's choice without "Okay, "

_b and _f are base and final. Per model:

- primary: delta = r(U_f, P_f) - r(U_f, A_f), with the call "prediction" / "own choice" / "similar";
- the same over non-harmful tasks and within origin;
- the first reading directly: r(A_f, P_f), d_P = r(A_f, P_f) - r(A_f, P_b), r shift r(A_f - A_b, P_f - P_b);
- partials r(U_f, P_f | A_f), r(U_f, A_f | P_f); R^2 of U_f on A_f, on P_f, on both;
- base: delta_b = r(U_b, P_b) - r(U_b, A_b), r(A_b, P_b);
- post-training's change: delta_shift = r(dU, dP) - r(dU, dA), d = final - base;
- scale: SDs, slopes on A_f (nats per Assistant SD), gaps (non-harmful minus harmful);
- r with log task length;
- delta_c = r(C_f, P_f) - r(C_f, A_f) and delta_u0 = r(U0_f, P_f) - r(U0_f, A_f), with the hope run's C and U0;
- exploratory: delta with log length partialled out of U, P and A; mean final utility by origin; top next tokens.

CIs are 95% task bootstraps within origin, one set of resamples for every model and statistic, so differences are paired.

  outputs/predict_user/analysis/<family>/   the merged fit (analyze.py) of one model's stages
  outputs/predict_user/predict_user.csv     one row per model
  outputs/predict_user/reproduction.csv     the re-scored A and U against the hope run's
  outputs/predict_user/by_origin.csv        exploratory: mean final utility by origin
  figures/predict_user_*.png

Usage: uv run python predict_user.py [--group predict_user]
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import N_BOOTSTRAP, ORIGIN_STYLE, pair_effects, partial_r, pearson, split_half
from hope import FAMILIES, gpu_of, gpu_type, merged_fit
from matskit import plotstyle
from models import MODEL_INFO, PROJECT
from openers import slope_per_sd, within_groups, write_csv
from report import INK, INK_2, RULE, origin_scatter, stratified_resamples, style, task_arrays
from user_choice import ci, collect_runs, top_token_shares

STAGES = ("base", "instruct")
SUFFIX = {"base": "b", "instruct": "f"}
COND = {"A": "assistant", "U": "user_choice", "P": "predict_user"}
HOPE_COND = {"C": "coinflip", "U0": "user_choice_no_okay"}  # final stage only, from the hope run
R_MEANINGFUL = 0.05  # a difference in r worth calling one
MIN_RELIABILITY, MIN_MASS = 0.9, 0.2
GATE_SAME_R, GATE_SD, GATE_OTHER_R = 0.999, 0.02, 0.99  # reproduction of the hope run's A and U, final stage

STATS = ("r_up", "r_ua", "delta",
         "r_up_benign", "r_ua_benign", "delta_benign", "r_up_within", "r_ua_within", "delta_within",
         "r_ap", "r_au", "d_ap_au", "r_ap_fixed", "d_P", "shift_ap", "shift_au",
         "partial_up_a", "partial_ua_p", "r2_a", "r2_p", "r2_ap",
         "r_up_b", "r_ua_b", "delta_b", "r_ap_b",
         "shift_up", "shift_ua", "delta_shift",
         "sd_p_b", "sd_p_f", "sd_u_b", "sd_u_f", "sd_a_f", "b_p", "b_u",
         "gap_p_b", "gap_p_f", "gap_u_f", "gap_a_f", "dgap_pu",
         "len_p", "len_u", "len_a",
         "r_cp", "r_ca", "delta_c", "partial_cp_a", "partial_ca_p", "r_u0p", "r_u0a", "delta_u0",
         "delta_len", "r_up_len", "r_ua_len")
K = {name: j for j, name in enumerate(STATS)}


def r_squared(y, *xs):
    """R^2 of an OLS fit of y on xs, with an intercept."""
    X = np.column_stack([np.ones_like(y), *xs])
    resid = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return float(1 - resid.var() / y.var())


def stats(a, benign, origins, loglen):
    """STATS of one model. a: {"<key>_<b|f>": utilities} for A, U, P at both stages and C_f, U0_f."""
    w = lambda v: within_groups(v, origins)
    gap = lambda v: float(v[benign].mean() - v[~benign].mean())
    r = pearson
    rb = lambda x, y: r(x[benign], y[benign])
    rw = lambda x, y: r(w(x), w(y))
    resid = lambda v: v - np.polyval(np.polyfit(loglen, v, 1), loglen)
    Ab, Af, Ub, Uf, Pb, Pf, Cf, U0f = (a[k] for k in ("A_b", "A_f", "U_b", "U_f", "P_b", "P_f", "C_f", "U0_f"))
    dA, dU, dP = Af - Ab, Uf - Ub, Pf - Pb
    v = dict(
        r_up=r(Uf, Pf), r_ua=r(Uf, Af), delta=r(Uf, Pf) - r(Uf, Af),
        r_up_benign=rb(Uf, Pf), r_ua_benign=rb(Uf, Af), delta_benign=rb(Uf, Pf) - rb(Uf, Af),
        r_up_within=rw(Uf, Pf), r_ua_within=rw(Uf, Af), delta_within=rw(Uf, Pf) - rw(Uf, Af),
        r_ap=r(Af, Pf), r_au=r(Af, Uf), d_ap_au=r(Af, Pf) - r(Af, Uf),
        r_ap_fixed=r(Af, Pb), d_P=r(Af, Pf) - r(Af, Pb), shift_ap=r(dA, dP), shift_au=r(dA, dU),
        partial_up_a=partial_r(Uf, Pf, Af), partial_ua_p=partial_r(Uf, Af, Pf),
        r2_a=r_squared(Uf, Af), r2_p=r_squared(Uf, Pf), r2_ap=r_squared(Uf, Af, Pf),
        r_up_b=r(Ub, Pb), r_ua_b=r(Ub, Ab), delta_b=r(Ub, Pb) - r(Ub, Ab), r_ap_b=r(Ab, Pb),
        shift_up=r(dU, dP), shift_ua=r(dU, dA), delta_shift=r(dU, dP) - r(dU, dA),
        sd_p_b=Pb.std(), sd_p_f=Pf.std(), sd_u_b=Ub.std(), sd_u_f=Uf.std(), sd_a_f=Af.std(),
        b_p=slope_per_sd(Af, Pf), b_u=slope_per_sd(Af, Uf),
        gap_p_b=gap(Pb), gap_p_f=gap(Pf), gap_u_f=gap(Uf), gap_a_f=gap(Af) / Af.std(), dgap_pu=gap(Pf) - gap(Uf),
        len_p=r(Pf, loglen), len_u=r(Uf, loglen), len_a=r(Af, loglen),
        r_cp=r(Cf, Pf), r_ca=r(Cf, Af), delta_c=r(Cf, Pf) - r(Cf, Af),
        partial_cp_a=partial_r(Cf, Pf, Af), partial_ca_p=partial_r(Cf, Af, Pf),
        r_u0p=r(U0f, Pf), r_u0a=r(U0f, Af), delta_u0=r(U0f, Pf) - r(U0f, Af),
        delta_len=r(resid(Uf), resid(Pf)) - r(resid(Uf), resid(Af)),
        r_up_len=r(resid(Uf), resid(Pf)), r_ua_len=r(resid(Uf), resid(Af)),
    )
    assert tuple(v) == STATS
    return np.array(list(v.values()), dtype=float)


def model_cell(arrays, benign, origins, loglen, resamples):
    return dict(est=stats(arrays, benign, origins, loglen),
                boot=np.array([stats({k: x[i] for k, x in arrays.items()}, benign[i], origins[i], loglen[i])
                               for i in resamples]))


def stat(cell, name):
    """(estimate, ci_low, ci_high)."""
    lo, hi = ci(cell["boot"][:, K[name]])
    return cell["est"][K[name]], lo, hi


def call(cell):
    """The pre-registered call: does the user's choice line up better with the Assistant's prediction of it ("prediction")
    or with the Assistant's own choice ("own choice"), by r?"""
    d, lo, hi = stat(cell, "delta")
    if d >= R_MEANINGFUL and lo > 0:
        return "prediction"
    if d <= -R_MEANINGFUL and hi < 0:
        return "own choice"
    return "similar"


def disattenuated_delta(cell, rel):
    """delta with each r divided by the square root of its two split-half reliabilities (point estimate)."""
    e = cell["est"]
    return (e[K["r_up"]] / np.sqrt(rel["U_f"] * rel["P_f"])) - (e[K["r_ua"]] / np.sqrt(rel["U_f"] * rel["A_f"]))


def reliabilities(rows, ids):
    """Split-half reliability of A, U and P at each stage."""
    measures = {f"{k}_{SUFFIX[s]}": {(s, c): 1.0} for k, c in COND.items() for s in STAGES}
    out = split_half(ids, pair_effects(rows), measures, np.random.default_rng(0))
    return {name: rel for name, (_, rel) in out.items()}


def hope_fit(hope_runs, family):
    """utilities.csv rows of the hope run's merged fit for one model (cached by hope.py), and its runs by stage."""
    out = Path("outputs/hope/analysis") / family
    path = out / "utilities.csv"
    if path.exists():
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
    else:
        rows = merged_fit(hope_runs, family, out)[0]
    return rows, {s: hope_runs[family, s] for s in STAGES}


def reproduction(family, ours, our_gpus, hope_util, hope_stages):
    """Our re-scored A and U against the hope run's, per stage. Final-stage cells gate; base cells are reported."""
    rows, passed = [], True
    for s in STAGES:
        ref_gpu = gpu_of(hope_stages[s])
        same = gpu_type(ref_gpu) is not None and gpu_type(ref_gpu) == gpu_type(our_gpus[s])
        for k in ("A", "U"):
            m = f"{s}/{COND[k]}"
            x, ref = ours[m], hope_util[m]
            rr, dsd = pearson(x, ref), x.std() / ref.std() - 1
            ok = (rr >= GATE_SAME_R and abs(dsd) <= GATE_SD) if same else rr >= GATE_OTHER_R
            gated = s == "instruct"
            passed &= ok or not gated
            rows.append(dict(model=MODEL_INFO[FAMILIES[family]][0], measure=m, reference=hope_stages[s].id,
                             our_gpu=our_gpus[s], ref_gpu=ref_gpu, same_gpu_type=same, r=rr, sd_change=dsd,
                             result=("pass" if ok else "FAIL") if gated else "reported"))
    return rows, passed


# Figures. Base is open, final filled; one color for every r.
LABEL = {"A": "Assistant's own choice", "U": "User's choice", "P": "Assistant's prediction"}


def plot_r(cells, labels, path):
    """r between the three readouts at each stage, and the two partials at the final stage, one panel per model."""
    rows = (("User's choice ~\nAssistant's own choice", "r_ua_b", "r_ua"),
            ("User's choice ~\nAssistant's prediction", "r_up_b", "r_up"),
            ("Assistant's prediction ~\nAssistant's own choice", "r_ap_b", "r_ap"),
            ("User's choice ~ own choice,\nprediction held fixed", None, "partial_ua_p"),
            ("User's choice ~ prediction,\nown choice held fixed", None, "partial_up_a"))
    color = plotstyle.SERIES[0]
    fig, axes = plt.subplots(1, len(cells), figsize=(3.3 * len(cells), 4.1), sharey=True, sharex=True, squeeze=False)
    y = np.arange(len(rows))[::-1]
    for ax, (family, cell) in zip(axes[0], cells.items()):
        ax.axvline(0, color=INK_2, lw=0.8, zorder=1)
        for yi, (_, base, final) in zip(y, rows):
            for name, dy, filled in ((base, 0.14, False), (final, -0.14, True)):
                if name is None:
                    continue
                est, lo, hi = stat(cell, name)
                ax.plot([lo, hi], [yi + dy] * 2, color=color if filled else INK_2, lw=1.6, zorder=2,
                        solid_capstyle="round")
                ax.plot(est, yi + dy, ls="none", marker="o", ms=7, color=color if filled else INK_2,
                        mfc=color if filled else "white", mew=1.6, zorder=4)
        ax.set_yticks(y, [label for label, _, _ in rows], fontsize=8.5)
        ax.set_title(labels[family], loc="left", fontsize=10.5, color=INK)
        ax.set_xlabel("r over the 300 tasks", fontsize=9, color=INK_2)
        style(ax)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], ls="none", marker="o", ms=7, color=INK_2, mfc="white", mew=1.6, label="Base"),
               Line2D([], [], ls="none", marker="o", ms=7, color=color, mfc=color, mew=1.6, label="Final")]
    fig.legend(handles=handles, frameon=False, ncol=2, loc="upper center", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_contrasts(cells, labels, path):
    """delta = r(user's choice, prediction) - r(user's choice, own choice), and its variants. Right of 0: the user's choice
    is closer to the Assistant's prediction. The band marks |delta| < 0.05, below which no call is made."""
    rows = (("delta", "Final (primary)"), ("delta_benign", "Final, non-harmful"), ("delta_within", "Final, within origin"),
            ("delta_len", "Final, length removed\n(exploratory)"), ("delta_b", "Base"),
            ("delta_shift", "Post-training's change"), ("delta_u0", "Final, user's choice\nwithout \"Okay, \""),
            ("delta_c", "Final, coin instead of\nthe user's choice"))
    color = plotstyle.SERIES[0]
    fig, axes = plt.subplots(1, len(cells), figsize=(3.3 * len(cells), 4.3), sharey=True, sharex=True, squeeze=False)
    y = np.arange(len(rows))[::-1]
    for ax, (family, cell) in zip(axes[0], cells.items()):
        ax.axvspan(-R_MEANINGFUL, R_MEANINGFUL, color=RULE, alpha=0.35, lw=0, zorder=0)
        ax.axvline(0, color=INK_2, lw=0.8, zorder=1)
        for yi, (name, _) in zip(y, rows):
            est, lo, hi = stat(cell, name)
            primary = name == "delta"
            ax.plot([lo, hi], [yi, yi], color=color, lw=1.6, zorder=2, solid_capstyle="round")
            ax.plot(est, yi, ls="none", marker="D" if primary else "o", ms=8 if primary else 6.5, color=color,
                    mfc=color if primary else "white", mew=1.6, zorder=4)
        ax.set_yticks(y, [label for _, label in rows], fontsize=8.5)
        ax.set_title(labels[family], loc="left", fontsize=10.5, color=INK)
        ax.set_xlabel("r with the prediction minus\nr with the own choice", fontsize=9, color=INK_2)
        style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scatter(data, origins, labels, path):
    """Per model (rows): the prediction against the own choice, the user's choice against each, and post-training's change
    in the user's choice against the change in the prediction."""
    d = lambda a, k: a[f"{k}_f"] - a[f"{k}_b"]
    panels = ((lambda a: a["A_f"], lambda a: a["P_f"], "Assistant's own choice", "Assistant's prediction"),
              (lambda a: a["P_f"], lambda a: a["U_f"], "Assistant's prediction", "User's choice"),
              (lambda a: a["A_f"], lambda a: a["U_f"], "Assistant's own choice", "User's choice"),
              (lambda a: d(a, "P"), lambda a: d(a, "U"), "Prediction, final - base", "User's choice, final - base"))
    fig, axes = plt.subplots(len(data), len(panels), figsize=(3.1 * len(panels), 2.9 * len(data)), squeeze=False)
    for row, (family, a) in enumerate(data.items()):
        for col, (fx, fy, lx, ly) in enumerate(panels):
            ax = axes[row, col]
            x, yv = fx(a), fy(a)
            origin_scatter(ax, x, yv, origins)
            ax.axhline(0, color=RULE, lw=0.8, zorder=1)
            ax.set_title(f"r = {pearson(x, yv):.2f}", fontsize=9, color=INK_2, loc="left")
            ax.set_xlabel(f"{lx} (nats)", fontsize=8, color=INK_2)
            ax.set_ylabel(f"{labels[family]}\n{ly} (nats)" if col == 0 else f"{ly} (nats)", fontsize=8, color=INK_2)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, frameon=False, ncol=len(ORIGIN_STYLE), loc="upper center", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


BY_ORIGIN_KEYS = ("A", "U", "P", "C")


def by_origin(data, origins, labels):
    """Exploratory: mean final utility of each readout within each origin; A in units of its own SD, the rest in nats."""
    rows = []
    for family, a in data.items():
        for g in ORIGIN_STYLE:
            k = origins == g
            row = dict(model=labels[family], origin=g, n=int(k.sum()))
            for key in BY_ORIGIN_KEYS:
                v = a[f"{key}_f"] / (a["A_f"].std() if key == "A" else 1)
                row[key] = float(v[k].mean())
            rows.append(row)
    return rows


def score_predictions(table):
    """The pre-registered predictions (README), each as (text, held)."""
    by = {r["model"]: r for r in table}
    n = len(by)
    count = lambda f: sum(bool(f(r)) for r in by.values())
    return [
        ("1. call 'own choice' in >= 3 of 4", count(lambda r: r["call"] == "own choice") >= 3),
        ("2a. r(A_f, P_f) >= 0.5 in all", count(lambda r: r["r_ap"] >= 0.5) == n),
        ("2b. r(A_f, P_f) < r(A_f, U_f) in >= 3", count(lambda r: r["r_ap"] < r["r_au"]) >= 3),
        ("3. r(U_f, A_f | P_f) >= 0.3, CI > 0, in >= 3",
         count(lambda r: r["partial_ua_p"] >= 0.3 and r["partial_ua_p_lo"] > 0) >= 3),
        ("4. r(A_b, P_b) >= 0.7 and delta_b >= 0 in >= 3", count(lambda r: r["r_ap_b"] >= 0.7 and r["delta_b"] >= 0) >= 3),
        ("5. d_P > 0, CI > 0, in >= 3", count(lambda r: r["d_P_lo"] > 0) >= 3),
        ("6a. gap P_f > 0, CI > 0, in >= 3", count(lambda r: r["gap_p_f_lo"] > 0) >= 3),
        ("6b. gap P_f < gap U_f in >= 3", count(lambda r: r["dgap_pu"] < 0) >= 3),
        ("7. P mass at the final stage >= 0.9 in all", count(lambda r: r["mass_final"] >= 0.9) == n),
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--group", default="predict_user")
    args = p.parse_args()
    out = Path("outputs/predict_user")
    out.mkdir(parents=True, exist_ok=True)
    runs, hope_runs = collect_runs(args.group), collect_runs("hope")
    data, cells, labels, table, repro, tops, gates, passed = {}, {}, {}, [], [], {}, [], True
    ids0 = origins0 = resamples = None
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
        h_rows, h_stages = hope_fit(hope_runs, family)
        assert [r["id"] for r in h_rows] == ids, "the hope run's tasks differ"
        HU = task_arrays(h_rows)[0]
        labels[family] = MODEL_INFO[final][0]
        arrays = data[family] = {**{f"{k}_{SUFFIX[s]}": U[f"{s}/{c}"] for k, c in COND.items() for s in STAGES},
                                 **{f"{k}_f": HU[f"instruct/{c}"] for k, c in HOPE_COND.items()}}

        gpus = {s: gpu_of(by_stage[s]) for s in STAGES}
        rep, ok = reproduction(family, U, gpus, HU, h_stages)
        repro += rep
        passed &= ok
        rel = reliabilities(raw, ids)
        masses = {s: summary[f"{s}/predict_user/outcome_mass_mean"] for s in STAGES}
        ok = rel["P_f"] >= MIN_RELIABILITY
        passed &= ok
        gates.append(f"{labels[family]:14} P reliability base {rel['P_b']:.3f}, final {rel['P_f']:.3f} "
                     f"({'pass' if ok else 'FAIL'}); mass base {masses['base']:.2f}, final {masses['instruct']:.2f}"
                     + ("" if masses["instruct"] >= MIN_MASS else " (below the floor: reported, not interpreted)"))

        cell = cells[family] = model_cell(arrays, benign, origins, loglen, resamples)
        row = dict(model=labels[family], call=call(cell), mass_base=masses["base"], mass_final=masses["instruct"],
                   readable=masses["instruct"] >= MIN_MASS, gpu_base=gpus["base"], gpu_final=gpus["instruct"],
                   **{f"rel_{k}": v for k, v in rel.items()},
                   bias_p_base=summary["base/predict_user/slot_bias_mean"],
                   bias_p_final=summary["instruct/predict_user/slot_bias_mean"])
        for name in STATS:
            est, lo, hi = stat(cell, name)
            row.update({name: est, f"{name}_lo": lo, f"{name}_hi": hi})
        row["delta_disattenuated"] = disattenuated_delta(cell, rel)
        row["flag_noise"] = row["call"] != "similar" and abs(row["delta_disattenuated"]) < R_MEANINGFUL
        table.append(row)
        tops[family] = {s: top_token_shares(Path("outputs") / by_stage[s].id / "results" / f"{s}__predict_user.csv.gz")
                        for s in STAGES}

    print("Gates:\n  " + "\n  ".join(gates))
    print("\nReproduction of the hope run's A and U (final stage gated, base reported):")
    for r in repro:
        print(f"  {r['model']:14} {r['measure']:22} vs {r['reference']}: r {r['r']:.4f}, SD {r['sd_change']:+.1%} "
              f"({r['our_gpu']} vs {r['ref_gpu']}{', same type' if r['same_gpu_type'] else ''}): {r['result']}")
    fm = lambda r, s: f"{r[s]:+.3f} [{r[f'{s}_lo']:+.3f}, {r[f'{s}_hi']:+.3f}]"
    groups = (
        ("Primary", ("r_up", "r(U_f, P_f)"), ("r_ua", "r(U_f, A_f)"), ("delta", "delta")),
        ("Non-harmful and within origin", ("r_up_benign", "r(U_f, P_f), non-harmful"),
         ("r_ua_benign", "r(U_f, A_f), non-harmful"), ("delta_benign", "delta, non-harmful"),
         ("r_up_within", "r(U_f, P_f), within origin"), ("r_ua_within", "r(U_f, A_f), within origin"),
         ("delta_within", "delta, within origin")),
        ("The first reading", ("r_ap", "r(A_f, P_f)"), ("r_au", "r(A_f, U_f)"), ("d_ap_au", "r(A_f, P_f) - r(A_f, U_f)"),
         ("r_ap_fixed", "r(A_f, P_b)"), ("d_P", "d_P"), ("shift_ap", "r shift, A and P"), ("shift_au", "r shift, A and U")),
        ("Partials", ("partial_up_a", "r(U_f, P_f | A_f)"), ("partial_ua_p", "r(U_f, A_f | P_f)"),
         ("r2_a", "R^2 of U_f on A_f"), ("r2_p", "R^2 of U_f on P_f"), ("r2_ap", "R^2 of U_f on both")),
        ("Base and post-training's change", ("r_up_b", "r(U_b, P_b)"), ("r_ua_b", "r(U_b, A_b)"), ("delta_b", "delta_b"),
         ("r_ap_b", "r(A_b, P_b)"), ("shift_up", "r(dU, dP)"), ("shift_ua", "r(dU, dA)"), ("delta_shift", "delta_shift")),
        ("Scale", ("sd_p_b", "SD P_b"), ("sd_p_f", "SD P_f"), ("sd_u_b", "SD U_b"), ("sd_u_f", "SD U_f"),
         ("sd_a_f", "SD A_f"), ("b_p", "slope of P_f on A_f"), ("b_u", "slope of U_f on A_f"), ("gap_p_b", "gap P_b"),
         ("gap_p_f", "gap P_f"), ("gap_u_f", "gap U_f"), ("gap_a_f", "gap A_f (in A SDs)"), ("dgap_pu", "gap P_f - gap U_f")),
        ("Length", ("len_p", "r(P_f, log length)"), ("len_u", "r(U_f, log length)"), ("len_a", "r(A_f, log length)"),
         ("r_up_len", "r(U_f, P_f), length removed"), ("r_ua_len", "r(U_f, A_f), length removed"),
         ("delta_len", "delta, length removed (exploratory)")),
        ("The hope run's coin and no-opener choice", ("r_cp", "r(C_f, P_f)"), ("r_ca", "r(C_f, A_f)"),
         ("delta_c", "delta_c"), ("partial_cp_a", "r(C_f, P_f | A_f)"), ("partial_ca_p", "r(C_f, A_f | P_f)"),
         ("r_u0p", "r(U0_f, P_f)"), ("r_u0a", "r(U0_f, A_f)"), ("delta_u0", "delta_u0")),
    )
    for r in table:
        print(f"\n{r['model']}: call {r['call']}" + (" (flagged: disattenuated delta inside +-0.05)" if r["flag_noise"]
                                                      else "") + ("" if r["readable"] else " (P mass below the floor)"))
        print(f"  disattenuated delta {r['delta_disattenuated']:+.3f}; reliability P {r['rel_P_b']:.3f} / "
              f"{r['rel_P_f']:.3f}, U {r['rel_U_b']:.3f} / {r['rel_U_f']:.3f}, A {r['rel_A_b']:.3f} / {r['rel_A_f']:.3f} "
              f"(base / final); P mass {r['mass_base']:.2f} / {r['mass_final']:.2f}, slot bias {r['bias_p_base']:+.2f} / "
              f"{r['bias_p_final']:+.2f}")
        for title, *items in groups:
            print(f"  {title}:")
            for name, label in items:
                print(f"    {label:32} {fm(r, name)}")
    print("\nPredictions, scored:")
    for text, held in score_predictions(table):
        print(f"  {'held  ' if held else 'failed'} {text}")
    origin_rows = by_origin(data, origins0, labels)
    print("\nExploratory: mean final utility by origin (A in Assistant SDs, the rest in nats):")
    print(f"  {'':14} {'origin':12} " + " ".join(f"{k:>6}" for k in BY_ORIGIN_KEYS))
    for r in origin_rows:
        print(f"  {r['model']:14} {r['origin']:12} " + " ".join(f"{r[k]:+6.2f}" for k in BY_ORIGIN_KEYS))
    print("\nWhere the probability goes after \"I think you'll choose Task\" (mean over prompts, tokens in each top 5):")
    for family, by in tops.items():
        for s, top in by.items():
            print(f"  {labels[family]:14} {s:9} " + "  ".join(f"{t!r} {p:.3f}" for t, p in top))

    write_csv(out / "predict_user.csv", table)
    write_csv(out / "reproduction.csv", repro)
    write_csv(out / "by_origin.csv", origin_rows)
    plotstyle.setup()
    plot_r(cells, labels, "figures/predict_user_r.png")
    plot_contrasts(cells, labels, "figures/predict_user_contrasts.png")
    plot_scatter(data, origins0, labels, "figures/predict_user_scatter.png")
    print(f"\nGates {'passed' if passed else 'FAILED'}. Wrote {out}/ and figures/predict_user_*.png")


if __name__ == "__main__":
    main()
