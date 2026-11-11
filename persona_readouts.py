"""The persona as the user: whose preferences do the user's choice and the coin follow, the Assistant's or the model's picture of
the user?

The measures and calls were pre-registered in the README ("The persona as the user: whose preferences do the user's choice and
the coin follow?") before any data. Per model (Llama 70B, Gemma 4 31B) and persona p (evil, slacker), at the final stage:

  A, C, U, P       the Assistant's own choice, the coin, the user's choice, the Assistant's prediction of the user's choice
  X_self           readout X with the persona as the user's first-person self-description (W&B group "persona_readouts",
                   one pod per persona with all four readouts)
  X_sys            with the persona as the system prompt: P and U from "persona_readouts", A and C from "personas"
  X_0              no persona: A_0 and C_0 from the sweep, U_0 and P_0 from "predict_user"
  A_per = A_sys    the persona's own Assistant, the target of the persona sections' fit

Primary (self box): separation r(A_self, P_self); dU = r(U, P) - r(U, A) and dC = r(C, P) - r(C, A), with the calls "user" /
"Assistant" / "similar" ("not separable" when r(A, P) >= 0.9); dU - dC. Secondary: the same for the system box; partials; the
persona sections' fit X = c + b_def z(A_0) + b_per z(A_per) for every readout and placement, w, and rho_X = b_per(X_self) /
b_per(X_sys); r(X_s, X_0); harm gap; r with log length; within-source dU and dC (exploratory).

CIs are 95% task bootstraps within origin, one set of resamples for everything, so differences are paired.

  outputs/persona_readouts/analysis/<family>/   the merged fit (analyze.py) of one model's new runs
  outputs/persona_readouts/persona_readouts.csv one row per model and persona
  outputs/persona_readouts/reproduction.csv     the re-scored cells against earlier runs
  figures/persona_readouts_*.png

Usage: uv run python persona_readouts.py [--models llama-70b gemma4-31b]
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import personas
from analyze import N_BOOTSTRAP, ORIGIN_STYLE, partial_r, pearson
from hope import gpu_of, gpu_type
from matskit import plotstyle
from models import MODEL_INFO, PROJECT
from openers import write_csv
from personas_user import median_mass, pulls
from report import INK, INK_2, RULE, origin_scatter, stratified_resamples, style, task_arrays
from user_choice import FAMILIES, ci, load_sweep, top_token_shares

MODELS = ("llama-70b", "gemma4-31b")
PERSONAS = ("evil", "slacker")
PLACES = ("self", "sys")
READ = ("A", "C", "U", "P")
COND = {"A": "assistant", "C": "coinflip", "U": "user_choice", "P": "predict_user"}
ORIGINS = tuple(ORIGIN_STYLE)
R_MEANINGFUL, SEPARABLE = 0.05, 0.9
MIN_RELIABILITY, MIN_MASS = 0.9, 0.2
GATE_SAME_R, GATE_SD, GATE_OTHER_R = 0.999, 0.02, 0.99
# Earlier runs of the re-scored cells: (our condition, file of the earlier fit, its column, W&B run id of that cell).
REFERENCES = {
    "llama-70b": (("assistant@evil+self", "outputs/persona_placement/llama-70b/analysis/utilities.csv", "rtg1enyk"),
                  ("coinflip@evil+self", "outputs/persona_placement/llama-70b/analysis/utilities.csv", "rtg1enyk"),
                  ("user_choice@evil", "outputs/personas_user/analysis/llama-70b/utilities.csv", "2pb8u0po")),
    "gemma4-31b": (("assistant@evil+self", "outputs/persona_placement/gemma4-31b/analysis/utilities.csv", "1k8fj865"),
                   ("coinflip@evil+self", "outputs/persona_placement/gemma4-31b/analysis/utilities.csv", "1k8fj865"),
                   ("user_choice@evil", "outputs/personas_user/analysis/gemma4-31b/utilities.csv", "i3bwtmzz")),
}

PER_PLACE = ("sep", "r_up", "r_ua", "dU", "r_cp", "r_ca", "dC", "dUC", "pu_p_a", "pu_a_p", "pc_p_a", "pc_a_p")
PER_READ = ("b_def", "b_per", "w", "gap", "r_len", "r_x0")
STATS = (tuple(f"{s}:{n}" for s in PLACES for n in PER_PLACE)
         + tuple(f"{s}:{x}:{n}" for s in PLACES for x in READ for n in PER_READ)
         + tuple(f"none:{x}:{n}" for x in READ for n in ("b_def", "b_per", "w", "gap", "r_len"))
         + tuple(f"rho:{x}" for x in READ)
         + tuple(f"self:{d}:{o}" for d in ("dU", "dC") for o in ORIGINS))
K = {name: j for j, name in enumerate(STATS)}


def stats(V, benign, loglen, origins):
    """STATS of one model and persona. V: {"<X>_<self|sys|0>": utilities} for X in A, C, U, P, and "Aper" (= A_sys)."""
    a0, aper = V["A_0"], V["Aper"]
    gap = lambda x: float(x[benign].mean() - x[~benign].mean())
    r = pearson
    out = []
    for s in PLACES:
        A, C, U, P = (V[f"{x}_{s}"] for x in READ)
        dU, dC = r(U, P) - r(U, A), r(C, P) - r(C, A)
        out += [r(A, P), r(U, P), r(U, A), dU, r(C, P), r(C, A), dC, dU - dC,
                partial_r(U, P, A), partial_r(U, A, P), partial_r(C, P, A), partial_r(C, A, P)]
    for s in PLACES:
        for x in READ:
            X = V[f"{x}_{s}"]
            out += [*pulls(X, a0, aper), gap(X), r(X, loglen), r(X, V[f"{x}_0"])]
    for x in READ:
        X = V[f"{x}_0"]
        out += [*pulls(X, a0, aper), gap(X), r(X, loglen)]
    b_per = lambda s, x: out[K[f"{s}:{x}:b_per"]]
    out += [b_per("self", x) / b_per("sys", x) for x in READ]
    for d in ("U", "C"):
        for o in ORIGINS:
            m = origins == o
            X, A, P = V[f"{d}_self"][m], V["A_self"][m], V["P_self"][m]
            out.append(r(X, P) - r(X, A))
    assert len(out) == len(STATS)
    return np.array(out, dtype=float)


def model_cell(V, benign, loglen, origins, resamples):
    return dict(est=stats(V, benign, loglen, origins),
                boot=np.array([stats({k: v[i] for k, v in V.items()}, benign[i], loglen[i], origins[i])
                               for i in resamples]))


def stat(cell, name):
    """(estimate, ci_low, ci_high)."""
    lo, hi = ci(cell["boot"][:, K[name]])
    return cell["est"][K[name]], lo, hi


def call(cell, s, x):
    """The pre-registered call for readout x (U or C) in placement s: does it follow the model's picture of the user ("user")
    or the Assistant ("Assistant") in that context?"""
    return readout_call(cell["est"][K[f"{s}:sep"]], *stat(cell, f"{s}:d{x}"))


def readout_call(sep, d, lo, hi):
    """call's rule on the separation r(A, P) and the contrast with its CI (also used by replication_analysis.py)."""
    if sep >= SEPARABLE:
        return "not separable"
    if d >= R_MEANINGFUL and lo > 0:
        return "user"
    if d <= -R_MEANINGFUL and hi < 0:
        return "Assistant"
    return "similar"


def arrays_for(persona, N, Pr, B, PU):
    """The utilities of one persona: N the new runs' fit, Pr the persona runs', B the sweep's, PU the predict_user run's."""
    return {"A_self": N[f"instruct/assistant@{persona}+self"], "C_self": N[f"instruct/coinflip@{persona}+self"],
            "U_self": N[f"instruct/user_choice@{persona}+self"], "P_self": N[f"instruct/predict_user@{persona}+self"],
            "A_sys": Pr[f"instruct/assistant@{persona}"], "C_sys": Pr[f"instruct/coinflip@{persona}"],
            "U_sys": N[f"instruct/user_choice@{persona}"], "P_sys": N[f"instruct/predict_user@{persona}"],
            "A_0": B["instruct/assistant"], "C_0": B["instruct/coinflip"],
            "U_0": PU["instruct/user_choice"], "P_0": PU["instruct/predict_user"],
            "Aper": Pr[f"instruct/assistant@{persona}"]}


def read_fit(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def reproduction(family, N, our_gpus, ids):
    """Our re-scored cells against the earlier runs' fits."""
    import wandb
    api = wandb.Api()
    rows, passed = [], True
    for cond, path, run_id in REFERENCES[family]:
        ref_rows = read_fit(path)
        assert [r["id"] for r in ref_rows] == ids, f"{path}: task order differs"
        ref = task_arrays(ref_rows)[0][f"instruct/{cond}"]
        x = N[f"instruct/{cond}"]
        ref_gpu, our_gpu = gpu_of(api.run(f"{PROJECT}/{run_id}")), our_gpus[cond]
        same = gpu_type(ref_gpu) is not None and gpu_type(ref_gpu) == gpu_type(our_gpu)
        rr, dsd = pearson(x, ref), x.std() / ref.std() - 1
        ok = (rr >= GATE_SAME_R and abs(dsd) <= GATE_SD) if same else rr >= GATE_OTHER_R
        passed &= ok
        rows.append(dict(model=MODEL_INFO[FAMILIES[family][0]][0], condition=cond, reference=run_id, our_gpu=our_gpu,
                         ref_gpu=ref_gpu, same_gpu_type=same, r=rr, sd_change=dsd, result="pass" if ok else "FAIL"))
    return rows, passed


# Figures.
PLACE_STYLE = {"self": ("User's self-description", True), "sys": ("System prompt", False), "none": ("No persona", None)}


def plot_r(cells, labels, path):
    """Per model and persona: r of the user's choice and the coin with the prediction and with the Assistant's own choice, and
    the separation r(A, P), under each placement. Filled: self-description; open: system prompt."""
    rows = (("User's choice ~ prediction", "r_up"), ("User's choice ~ Assistant", "r_ua"),
            ("Coin ~ prediction", "r_cp"), ("Coin ~ Assistant", "r_ca"), ("Prediction ~ Assistant\n(separation)", "sep"))
    keys = list(cells)
    fig, axes = plt.subplots(1, len(keys), figsize=(3.4 * len(keys), 3.9), sharey=True, sharex=True, squeeze=False)
    y = np.arange(len(rows))[::-1]
    color = plotstyle.SERIES[0]
    for ax, key in zip(axes[0], keys):
        cell = cells[key]
        ax.axvline(0, color=INK_2, lw=0.8, zorder=1)
        for yi, (_, name) in zip(y, rows):
            for s, dy in (("self", -0.14), ("sys", 0.14)):
                est, lo, hi = stat(cell, f"{s}:{name}")
                filled = s == "self"
                c = color if filled else INK_2
                ax.plot([lo, hi], [yi + dy] * 2, color=c, lw=1.6, zorder=2, solid_capstyle="round")
                ax.plot(est, yi + dy, ls="none", marker="o", ms=7, color=c, mfc=c if filled else "white", mew=1.6, zorder=4)
        ax.set_yticks(y, [label for label, _ in rows], fontsize=8.5)
        ax.set_title(labels[key], loc="left", fontsize=10.5, color=INK)
        ax.set_xlabel("r over the 300 tasks", fontsize=9, color=INK_2)
        style(ax)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], ls="none", marker="o", ms=7, color=color, mfc=color, mew=1.6, label="User's self-description"),
               Line2D([], [], ls="none", marker="o", ms=7, color=INK_2, mfc="white", mew=1.6, label="System prompt")]
    fig.legend(handles=handles, frameon=False, ncol=2, loc="upper center", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_w(cells, labels, path):
    """Per model and persona: w (0 = the default Assistant's ranking, 1 = the persona Assistant's) of each readout with no
    persona, the self-description and the system prompt."""
    keys = list(cells)
    fig, axes = plt.subplots(1, len(keys), figsize=(3.4 * len(keys), 3.4), sharey=True, sharex=True, squeeze=False)
    y = np.arange(len(READ))[::-1]
    label = {"A": "Assistant's own choice", "C": "Coin", "U": "User's choice", "P": "Assistant's prediction\nof the user"}
    marks = {"none": ("s", INK_2, False), "self": ("o", plotstyle.SERIES[0], True), "sys": ("D", plotstyle.SERIES[1], True)}
    for ax, key in zip(axes[0], keys):
        cell = cells[key]
        for v in (0, 1):
            ax.axvline(v, color=RULE, lw=0.8, zorder=1)
        for yi, x in zip(y, READ):
            for s, dy in (("none", 0.2), ("self", 0.0), ("sys", -0.2)):
                est, lo, hi = stat(cell, f"{s}:{x}:w")
                m, c, filled = marks[s]
                if hi - lo > 3:  # b_def + b_per near 0: the ratio means nothing (see b_def, b_per in the report)
                    from matplotlib.transforms import blended_transform_factory
                    ax.text(0.97, yi + dy, "w unstable", ha="right", va="center", fontsize=7.5, color=c,
                            transform=blended_transform_factory(ax.transAxes, ax.transData))
                    continue
                lo_c, hi_c = max(lo, -1.5), min(hi, 2.5)
                ax.plot([lo_c, hi_c], [yi + dy] * 2, color=c, lw=1.4, zorder=2, solid_capstyle="round")
                ax.plot(np.clip(est, -1.5, 2.5), yi + dy, ls="none", marker=m, ms=6.5, color=c, mfc=c if filled else "white",
                        mew=1.4, zorder=4)
        ax.set_yticks(y, [label[x] for x in READ], fontsize=8.5)
        ax.set_title(labels[key], loc="left", fontsize=10.5, color=INK)
        ax.set_xlabel("w: 0 default Assistant, 1 persona", fontsize=9, color=INK_2)
        style(ax)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], ls="none", marker=m, ms=6.5, color=c, mfc=c if f else "white", mew=1.4, label=PLACE_STYLE[s][0])
               for s, (m, c, f) in marks.items()]
    fig.legend(handles=handles, frameon=False, ncol=3, loc="upper center", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scatter(data, origins, labels, path):
    """Rows: model and persona. Columns: under the self-description, the prediction against the Assistant's own choice, and
    the user's choice and the coin against each."""
    panels = (("A_self", "P_self", "Assistant's own choice", "Prediction of the user"),
              ("P_self", "U_self", "Prediction of the user", "User's choice"),
              ("A_self", "U_self", "Assistant's own choice", "User's choice"),
              ("P_self", "C_self", "Prediction of the user", "Coin"),
              ("A_self", "C_self", "Assistant's own choice", "Coin"))
    fig, axes = plt.subplots(len(data), len(panels), figsize=(3.0 * len(panels), 2.8 * len(data)), squeeze=False)
    for row, (key, V) in enumerate(data.items()):
        for col, (kx, ky, lx, ly) in enumerate(panels):
            ax = axes[row, col]
            origin_scatter(ax, V[kx], V[ky], origins)
            ax.axhline(0, color=RULE, lw=0.8, zorder=1)
            ax.set_title(f"r = {pearson(V[kx], V[ky]):.2f}", fontsize=9, color=INK_2, loc="left")
            ax.set_xlabel(f"{lx} (nats)", fontsize=8, color=INK_2)
            ax.set_ylabel(f"{labels[key]}\n{ly} (nats)" if col == 0 else f"{ly} (nats)", fontsize=8, color=INK_2)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, frameon=False, ncol=len(ORIGIN_STYLE), loc="upper center", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def score_predictions(table):
    """The pre-registered predictions (README), each as (text, held or None if not all cells are in)."""
    by = {(r["family"], r["persona"]): r for r in table}
    full = len(by) == 4
    get = lambda f, p: by.get((f, p))
    rows = [r for r in by.values()]
    evil = [r for r in rows if r["persona"] == "evil"]
    slack = [r for r in rows if r["persona"] == "slacker"]
    count = lambda rs, f: sum(bool(f(r)) for r in rs)
    out = [
        ("1. r(A_self, P_self) < 0.6 for evil and < 0.8 for slacker, both models",
         count(evil, lambda r: r["self:sep"] < 0.6) == 2 and count(slack, lambda r: r["self:sep"] < 0.8) == 2),
        ("2. w of P_self >= 0.6 for evil, both models", count(evil, lambda r: r["self:P:w"] >= 0.6) == 2),
        ("3. dU 'user' in >= 3 of 4 cells", count(rows, lambda r: r["call_U_self"] == "user") >= 3),
        ("4a. dC 'Assistant' for both Llama personas",
         all(get("llama-70b", p) and get("llama-70b", p)["call_C_self"] == "Assistant" for p in PERSONAS)),
        ("4b. Gemma evil dC not 'Assistant'", bool(get("gemma4-31b", "evil")) and get("gemma4-31b", "evil")["call_C_self"] != "Assistant"),
        ("5. dU - dC > 0 with CI > 0 in all 4 cells", count(rows, lambda r: r["self:dUC_lo"] > 0) == 4),
        ("6. evil gaps of P_self and U_self < 0, both models",
         count(evil, lambda r: r["self:P:gap"] < 0 and r["self:U:gap"] < 0) == 2),
        ("7. r(A_sys, P_sys) >= 0.5 for evil, both models", count(evil, lambda r: r["sys:sep"] >= 0.5) == 2),
        ("8. median mass >= 0.7 in every new cell (Gemma user-turn cells excepted)",
         all(m >= 0.7 for r in rows for c, m in r["masses"].items()
             if not (r["family"] == "gemma4-31b" and c.split("@")[0] in ("user_choice", "coinflip")))),
    ]
    return [(t, held if full else None) for t, held in out]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=list(MODELS), help="a subset, e.g. while a model is still running")
    args = p.parse_args()
    out = Path("outputs/persona_readouts")
    out.mkdir(parents=True, exist_ok=True)
    new_runs, persona_runs = personas.collect_runs("persona_readouts"), personas.collect_runs("personas")
    data, cells, labels, table, repro, gates, tops = {}, {}, {}, [], [], [], {}
    passed, ids0, resamples = True, None, None
    for family in args.models:
        if family not in new_runs or len(new_runs[family]) < 3:
            print(f"{family}: not run (yet)")
            continue
        label = MODEL_INFO[FAMILIES[family][0]][0]
        print(f"{label}:")
        rows, summary, paths = personas.merged_fit(new_runs[family], out / "analysis" / family)
        gpus = {c: gpu_of(run) for run in new_runs[family] for c in run.config["models"][0]["conditions"]}
        prow = personas.merged_fit(persona_runs[family], Path("outputs/personas/analysis") / family)[0]
        srow = load_sweep(family)[0]
        purow = read_fit(Path("outputs/predict_user/analysis") / family / "utilities.csv")
        ids = [r["id"] for r in rows]
        assert all(ids == [r["id"] for r in x] for x in (prow, srow, purow)), "task sets differ"
        N, origins, benign, loglen = task_arrays(rows)
        Pr, B, PU = (task_arrays(x)[0] for x in (prow, srow, purow))
        if ids0 is None:
            ids0, resamples = ids, stratified_resamples(origins, np.random.default_rng(0), N_BOOTSTRAP)
        assert ids == ids0, "task sets differ between models"

        rep, ok = reproduction(family, N, gpus, ids)
        repro += rep
        passed &= ok
        masses = {}
        for c, path in sorted(paths.items()):
            rel, mass = summary[f"instruct/{c}/reliability"], median_mass(path)
            masses[c] = mass
            ok_cell = rel >= MIN_RELIABILITY
            passed &= ok_cell
            gates.append(f"{label:14} {c:28} {gpus[c]}: reliability {rel:.3f}, median mass {mass:.2f}"
                         f"{'' if mass >= MIN_MASS else ' (below 0.2: not interpreted)'}: {'pass' if ok_cell else 'FAIL'}")
        tops[family] = {c: top_token_shares(path) for c, path in sorted(paths.items())}
        for persona in PERSONAS:
            key = f"{family}/{persona}"
            labels[key] = f"{label}, {persona}"
            V = data[key] = arrays_for(persona, N, Pr, B, PU)
            cell = cells[key] = model_cell(V, benign, loglen, origins, resamples)
            row = dict(model=label, family=family, persona=persona,
                       masses={c: m for c, m in masses.items() if f"@{persona}" in c})
            for s in PLACES:
                for x in ("U", "C"):
                    row[f"call_{x}_{s}"] = call(cell, s, x)
            for name in STATS:
                est, lo, hi = stat(cell, name)
                row.update({name: est, f"{name}_lo": lo, f"{name}_hi": hi})
            table.append(row)

    print("\nGates:\n  " + "\n  ".join(gates))
    print("\nReproduction of re-scored cells:")
    for r in repro:
        print(f"  {r['model']:14} {r['condition']:22} vs {r['reference']}: r {r['r']:.4f}, SD {r['sd_change']:+.1%} "
              f"({r['our_gpu']} vs {r['ref_gpu']}{', same type' if r['same_gpu_type'] else ''}): {r['result']}")
    fm = lambda r, s: f"{r[s]:+.3f} [{r[f'{s}_lo']:+.3f}, {r[f'{s}_hi']:+.3f}]"
    for r in table:
        print(f"\n{r['model']}, {r['persona']}: self-description calls U {r['call_U_self']}, C {r['call_C_self']}; "
              f"system prompt calls U {r['call_U_sys']}, C {r['call_C_sys']}")
        for s in PLACES:
            print(f"  {s}:")
            for n, lab in (("sep", "r(A, P) (separation)"), ("r_up", "r(U, P)"), ("r_ua", "r(U, A)"), ("dU", "dU"),
                           ("r_cp", "r(C, P)"), ("r_ca", "r(C, A)"), ("dC", "dC"), ("dUC", "dU - dC"),
                           ("pu_p_a", "r(U, P | A)"), ("pu_a_p", "r(U, A | P)"), ("pc_p_a", "r(C, P | A)"),
                           ("pc_a_p", "r(C, A | P)")):
                print(f"    {lab:24} {fm(r, f'{s}:{n}')}")
        print("  the persona sections' fit (b_def, b_per, w), gap, r with log length, r with the no-persona readout:")
        for s in ("none", *PLACES):
            for x in READ:
                names = ("b_def", "b_per", "w", "gap", "r_len") + (("r_x0",) if s != "none" else ())
                print(f"    {s:5} {x}: " + ", ".join(f"{n} {fm(r, f'{s}:{x}:{n}')}" for n in names))
        print("  rho (self / system b_per): " + ", ".join(f"{x} {fm(r, f'rho:{x}')}" for x in READ))
        print("  within source (exploratory), dU / dC: " + "; ".join(
            f"{o.lower()} {r[f'self:dU:{o}']:+.2f} / {r[f'self:dC:{o}']:+.2f}" for o in ORIGINS))
    print("\nPredictions, scored:")
    for text, held in score_predictions(table):
        print(f"  {'held  ' if held else 'failed' if held is not None else 'n/a   '} {text}")
    print("\nWhere the probability goes (mean over prompts, tokens in each top 5):")
    for family, by in tops.items():
        for c, top in by.items():
            print(f"  {family:11} {c:28} " + "  ".join(f"{t!r} {pr:.3f}" for t, pr in top))

    write_csv(out / "persona_readouts.csv", [{k: v for k, v in r.items() if k != "masses"} for r in table])
    write_csv(out / "reproduction.csv", repro)
    if cells:
        plotstyle.setup()
        plot_r(cells, labels, "figures/persona_readouts_r.png")
        plot_w(cells, labels, "figures/persona_readouts_w.png")
        plot_scatter(data, origins, labels, "figures/persona_readouts_scatter.png")
    print(f"\nGates {'passed' if passed else 'FAILED'}. Wrote {out}/ and figures/persona_readouts_*.png")


if __name__ == "__main__":
    main()
