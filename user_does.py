"""When the user does the task: the coin and the user's choice, with the Assistant or the user doing the task.

The measures and calls were pre-registered in the README ("When the user does the task") before any data. A 2 x 2
per model and stage: who chooses (coin C, user U) x who does the task (the Assistant: C, U; the user: C', U').
A (the Assistant's utilities), C come from the model sweep; U from the user-choice runs (outputs/user_choice/analysis,
written by user_choice.py); C' and U' from the W&B runs of group "user_does" (configs/user_does/). _b and _f are base
and final.

Per model and chooser, X the Assistant-does-it cell and P = X' its user-does-it counterpart:
- gain(X) = slope of X_f - X_b on A_f in nats per Assistant SD; retention = gain(P) / gain(X), the primary measure,
  and the same over non-harmful tasks;
- the user-choice section's measures for P: r(A_b, P_b), r(A_f, P_f), r(A_f, P_b), d, r shift, non-harmful and
  within-origin r, slope b, gap (non-harmful minus harmful), SD;
- paired with X: r(A_f, P_f) - r(A_f, X_f) (and at base), gap(P_f) - gap(X_f), r(P, X) at each stage;
- r(P_f, log length) and its difference from r(X_f, log length);
and across choosers: the interaction, coin minus user, on retention and on r(A_f, P_f) - r(A_f, X_f).

CIs are 95% task bootstraps within origin, one set of resamples for everything, so all differences are paired.
`call` gives the pre-registered call per model and chooser.

  outputs/user_does/analysis/<family>/   the merged fit (analyze.py) of one model's stages
  outputs/user_does/user_does.csv        one row per model and chooser
  outputs/user_does/interaction.csv      coin minus user, per model
  outputs/user_does/by_origin.csv
  figures/user_does_*.png

Usage: uv run python user_does.py [--group user_does]
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import N_BOOTSTRAP, ORIGIN_STYLE, analyze, fetch_wandb as fetch_raw, pearson, read_jsonl
from matskit import plotstyle
from models import MODEL_INFO
from openers import slope_per_sd, within_groups, write_csv
from report import INK, INK_2, RULE, errorbar, origin_scatter, stratified_resamples, style, task_arrays
from user_choice import FAMILIES, STAGES, TASKS_FILE, ci, collect_runs, load_sweep, top_token_shares

NEW = {"coin": "coinflip_user_does", "user": "user_choice_user_does"}
CHOOSERS = ("coin", "user")
CHOOSER_LABEL = {"coin": "Coin chooses", "user": "User chooses"}
GATE_R, GATE_SD, MIN_RELIABILITY = 0.999, 0.02, 0.9
MIN_MASS = 0.2  # a cell below this median outcome mass is reported but not interpreted

# Per chooser; the full stat vector is these for "coin", then for "user", then INTERACTION.
PER = ("retention", "retention_benign", "gain_new", "gain_old", "gain_new_benign", "gain_old_benign",
       "r_base", "r_final", "r_fixed", "d", "r_shift", "r_final_benign", "r_final_within",
       "b_base", "b_final", "gap_base", "gap_final", "sd_base", "sd_final",
       "r_old_base", "r_old_final", "r_old_final_benign", "b_old_final", "gap_old_final", "sd_old_final",
       "dr_final", "dr_base", "dr_final_benign", "dgap_final", "r_px_base", "r_px_final", "r_px_final_benign",
       "r_len_new", "r_len_old", "dr_len")
INTERACTION = ("retention_coin_minus_user", "dr_final_coin_minus_user")
STATS = tuple(f"{c}:{s}" for c in CHOOSERS for s in PER) + INTERACTION
K = {name: j for j, name in enumerate(STATS)}


def chooser_stats(ab, af, xb, xf, pb, pf, benign, origins, loglen):
    """PER for one chooser: X is the Assistant-does-it cell, P the user-does-it one."""
    w = lambda x: within_groups(x, origins)
    gap = lambda x: x[benign].mean() - x[~benign].mean()
    gain_new, gain_old = slope_per_sd(af, pf - pb), slope_per_sd(af, xf - xb)
    gn_b, go_b = (slope_per_sd(af[benign], (y1 - y0)[benign]) for y0, y1 in ((pb, pf), (xb, xf)))
    r_final, r_fixed, r_old_final = pearson(af, pf), pearson(af, pb), pearson(af, xf)
    r_len_new, r_len_old = pearson(pf, loglen), pearson(xf, loglen)
    return [gain_new / gain_old, gn_b / go_b, gain_new, gain_old, gn_b, go_b,
            pearson(ab, pb), r_final, r_fixed, r_final - r_fixed, pearson(af - ab, pf - pb),
            pearson(af[benign], pf[benign]), pearson(w(af), w(pf)),
            slope_per_sd(ab, pb), slope_per_sd(af, pf), gap(pb), gap(pf), pb.std(), pf.std(),
            pearson(ab, xb), r_old_final, pearson(af[benign], xf[benign]), slope_per_sd(af, xf), gap(xf), xf.std(),
            r_final - r_old_final, pearson(ab, pb) - pearson(ab, xb),
            pearson(af[benign], pf[benign]) - pearson(af[benign], xf[benign]), gap(pf) - gap(xf),
            pearson(pb, xb), pearson(pf, xf), pearson(pf[benign], xf[benign]),
            r_len_new, r_len_old, r_len_new - r_len_old]


def model_stats(arrays, benign, origins, loglen):
    """STATS of one model. arrays: dict of A_b, A_f and, per chooser, X_b, X_f, P_b, P_f."""
    a = arrays
    out = []
    for c in CHOOSERS:
        out += chooser_stats(a["A_b"], a["A_f"], a[f"{c}_b"], a[f"{c}_f"], a[f"{c}'_b"], a[f"{c}'_f"],
                             benign, origins, loglen)
    p = len(PER)
    out += [out[PER.index("retention")] - out[p + PER.index("retention")],
            out[PER.index("dr_final")] - out[p + PER.index("dr_final")]]
    return np.array(out)


def model_cell(arrays, benign, origins, loglen, resamples):
    return dict(est=model_stats(arrays, benign, origins, loglen),
                boot=np.array([model_stats({k: v[i] for k, v in arrays.items()}, benign[i], origins[i], loglen[i])
                               for i in resamples]))


def stat(cell, name):
    """(estimate, ci_low, ci_high)."""
    lo, hi = ci(cell["boot"][:, K[name]])
    return cell["est"][K[name]], lo, hi


def call(cell, chooser):
    """The pre-registered call on retention, first that applies."""
    r, lo, hi = stat(cell, f"{chooser}:retention")
    if r <= -0.25 and hi < 0:
        return "reversed"
    if abs(r) < 0.25 or lo <= 0 <= hi:
        return "absent"
    if hi < 1:
        return "reduced"
    if lo > 1:
        return "enhanced"
    return "doer-independent"


def merged_fit(runs, family, out):
    """(utilities.csv rows, summary, {stage: {chooser: raw results path}}) of a model's user_does stages, fit together."""
    paths, rows = {}, []
    for s in STAGES:
        run = runs[family, s]
        print(f"{family:12} {s:9} {run.path}")
        rows += fetch_raw(run, Path("outputs") / run.id)
        paths[s] = {c: Path("outputs") / run.id / "results" / f"{s}__{NEW[c]}.csv.gz" for c in CHOOSERS}
    if not (out / "utilities.csv").exists():
        analyze(rows, read_jsonl(TASKS_FILE), out, seed=0, stage_order=list(STAGES))
    with open(out / "utilities.csv", newline="") as fh:
        return list(csv.DictReader(fh)), json.loads((out / "summary.json").read_text()), paths


def read_fit(path):
    with open(path / "utilities.csv", newline="") as fh:
        return list(csv.DictReader(fh))


CELLS = (("coin", "C: coin, Assistant does it"), ("coin'", "C': coin, user does it"),
         ("user", "U: user chooses, Assistant does it"), ("user'", "U': user chooses and does it"))
# Color = chooser (coin, user), filled = Assistant does it, open = user does it.
CELL_STYLE = {"coin": (0, "o", True), "coin'": (0, "o", False), "user": (1, "s", True), "user'": (1, "s", False)}


def cell_errorbar(ax, x, est, lo, hi, key, label):
    c, m, filled = CELL_STYLE[key]
    color = plotstyle.SERIES[c]
    errorbar(ax, x, est, np.minimum(lo, est), np.maximum(hi, est), color, m, label, ls="none")
    if not filled:  # open marker: redraw the face in the surface color
        ax.plot(x, est, ls="none", marker=m, ms=7, mfc="white", mec=color, mew=1.6, zorder=4)


def legend_handles(keys_labels):
    """Legend entries that show open markers as open."""
    from matplotlib.lines import Line2D
    out = []
    for key, label in keys_labels:
        c, m, filled = CELL_STYLE[key]
        color = plotstyle.SERIES[c]
        out.append(Line2D([], [], ls="none", marker=m, ms=7, color=color, mfc=color if filled else "white", mec=color,
                          mew=1.6, label=label))
    return out


def plot_summary(cells, labels, path):
    """Small multiples by measure; models along x, the four cells side by side."""
    families = list(cells)
    x = np.arange(len(families))
    # (title, {cell key: stat name}, reference lines)
    panels = (
        ("r with the final Assistant", {"coin": "coin:r_old_final", "coin'": "coin:r_final",
                                        "user": "user:r_old_final", "user'": "user:r_final"}),
        ("Same, non-harmful tasks", {"coin": "coin:r_old_final_benign", "coin'": "coin:r_final_benign",
                                     "user": "user:r_old_final_benign", "user'": "user:r_final_benign"}),
        ("Gain: slope of final - base on A_f (nats/SD)", {"coin": "coin:gain_old", "coin'": "coin:gain_new",
                                                          "user": "user:gain_old", "user'": "user:gain_new"}),
        ("Non-harmful minus harmful, final (nats)", {"coin": "coin:gap_old_final", "coin'": "coin:gap_final",
                                                     "user": "user:gap_old_final", "user'": "user:gap_final"}),
    )
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    for ax, (title, series) in zip(axes.flat, panels):
        for j, (key, label) in enumerate(CELLS):
            est, lo, hi = zip(*(stat(cells[f], series[key]) for f in families))
            cell_errorbar(ax, x + (j - 1.5) * 0.16, np.array(est), np.array(lo), np.array(hi), key, label)
        ax.axhline(0, color=RULE, lw=0.8, zorder=1)
        ax.set_xticks(x, [labels[f] for f in families])
        ax.set_title(title, loc="left", fontsize=10.5, color=INK)
        style(ax)
    ax = axes.flat[len(panels)]
    for j, c in enumerate(CHOOSERS):
        est, lo, hi = zip(*(stat(cells[f], f"{c}:retention") for f in families))
        cell_errorbar(ax, x + (j - 0.5) * 0.2, np.array(est), np.array(lo), np.array(hi), c + "'", CHOOSER_LABEL[c])
    for r in (0, 1):
        ax.axhline(r, color=RULE, lw=0.8, zorder=1)
    ax.set_xticks(x, [labels[f] for f in families])
    ax.set_title("Retention: gain(user does it) / gain(Assistant does it)", loc="left", fontsize=10.5, color=INK)
    style(ax)
    legend_ax = axes.flat[-1]
    legend_ax.axis("off")
    legend_ax.legend(handles=legend_handles(CELLS), frameon=False, fontsize=10, loc="center left",
                     title="Filled: the Assistant does the task\nOpen: the user does it", title_fontsize=9.5,
                     alignment="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scatter(data, origins, labels, path):
    """Each model's four cells (final stage) against the final Assistant. Axes differ between panels."""
    fig, axes = plt.subplots(len(data), len(CELLS), figsize=(3.3 * len(CELLS), 3 * len(data)), squeeze=False)
    for row, (family, arrays) in enumerate(data.items()):
        for col, (key, title) in enumerate(CELLS):
            ax = axes[row, col]
            y = arrays[f"{key}_f"]
            origin_scatter(ax, arrays["A_f"], y, origins)
            ax.axhline(0, color=RULE, lw=0.8, zorder=1)
            ax.set_title(f"r = {pearson(arrays['A_f'], y):.2f}, y SD {y.std():.2f}", fontsize=9, color=INK_2, loc="left")
            if row == 0:
                ax.text(0, 1.14, title, transform=ax.transAxes, fontsize=10, color=INK)
            ax.set_xlabel("Final Assistant utility (nats)", fontsize=8, color=INK_2)
        axes[row, 0].set_ylabel(f"{labels[family]}, final\nutility (nats)")
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, frameon=False, ncol=len(ORIGIN_STYLE), loc="upper center", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def by_origin(data, origins, labels):
    rows = []
    for family, a in data.items():
        for g in ORIGIN_STYLE:
            k = origins == g
            rows.append(dict(model=labels[family], origin=g, n=int(k.sum()),
                             **{f"{key}_{s}": float(a[f"{key}_{s}"][k].mean()) for key, _ in CELLS for s in "bf"}))
    return rows


def plot_by_origin(rows, labels, path):
    """Mean final utility by origin, the four cells, one panel per model."""
    fig, axes = plt.subplots(1, len(labels), figsize=(4.4 * len(labels), 4), squeeze=False)
    for ax, (family, label) in zip(axes[0], labels.items()):
        rs = {r["origin"]: r for r in rows if r["model"] == label}
        y = np.arange(len(ORIGIN_STYLE))
        for j, (key, name) in enumerate(CELLS):
            c, m, filled = CELL_STYLE[key]
            color = plotstyle.SERIES[c]
            ax.plot([rs[g][f"{key}_f"] for g in ORIGIN_STYLE], y + (j - 1.5) * 0.12, ls="none", marker=m, ms=7,
                    color=color, mfc=color if filled else "white", mec=color if not filled else "white",
                    mew=1.4 if not filled else 0.8, label=name, zorder=3)
        ax.axvline(0, color=RULE, lw=0.8, zorder=1)
        ax.set_yticks(y, list(ORIGIN_STYLE))
        ax.set_title(label, loc="left", fontsize=11, color=INK)
        ax.set_xlabel("Mean final utility (nats)", color=INK_2)
        style(ax)
    fig.legend(handles=legend_handles(CELLS), frameon=False, ncol=4, loc="lower center", fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--group", default="user_does")
    args = p.parse_args()
    out = Path("outputs/user_does")
    out.mkdir(parents=True, exist_ok=True)
    runs = collect_runs(args.group)
    data, cells, labels, gates, passed, table, tops, summaries = {}, {}, {}, [], True, [], {}, {}
    ids0 = resamples = None
    for family, (final, _) in FAMILIES.items():
        rows, summary, paths = merged_fit(runs, family, out / "analysis" / family)
        summaries[family] = summary
        sweep_rows, _ = load_sweep(family)
        uc_rows = read_fit(Path("outputs/user_choice/analysis") / family)
        ids = [r["id"] for r in rows]
        assert ids == [r["id"] for r in sweep_rows] == [r["id"] for r in uc_rows], "task sets differ"
        N, origins, benign, loglen = task_arrays(rows)
        B, UC = task_arrays(sweep_rows)[0], task_arrays(uc_rows)[0]
        if ids0 is None:
            ids0, resamples = ids, stratified_resamples(origins, np.random.default_rng(0), N_BOOTSTRAP)
        assert ids == ids0, "task sets differ between models"
        labels[family] = MODEL_INFO[final][0]
        arrays = {"A_b": B["base/assistant"], "A_f": B["instruct/assistant"],
                  "coin_b": B["base/coinflip"], "coin_f": B["instruct/coinflip"],
                  "user_b": UC["base/user_choice"], "user_f": UC["instruct/user_choice"]}
        for c in CHOOSERS:
            arrays[f"{c}'_b"], arrays[f"{c}'_f"] = N[f"base/{NEW[c]}"], N[f"instruct/{NEW[c]}"]
        data[family] = arrays
        # Gate: reliability in every final cell and every user-choice cell; base coin cells are reported only.
        for s in STAGES:
            for c in CHOOSERS:
                rel, mass = summary[f"{s}/{NEW[c]}/reliability"], summary[f"{s}/{NEW[c]}/outcome_mass_mean"]
                gated = s == "instruct" or c == "user"
                ok = rel >= MIN_RELIABILITY or not gated
                passed &= ok
                gates.append(f"{labels[family]:14} {s:9} {NEW[c]:22} reliability {rel:.3f}, mass {mass:.2f}"
                             + (f": {'pass' if ok else 'FAIL'}" if gated else " (reported only)"))
        # A re-scored user-choice condition (only if a pod of a different GPU type re-ran it) must match the original.
        for s in STAGES:
            if f"{s}/user_choice" in N:
                r = pearson(N[f"{s}/user_choice"], UC[f"{s}/user_choice"])
                dsd = N[f"{s}/user_choice"].std() / UC[f"{s}/user_choice"].std() - 1
                ok = r >= GATE_R and abs(dsd) <= GATE_SD
                passed &= ok
                gates.append(f"{labels[family]:14} {s:9} re-scored user_choice vs original: r {r:.4f}, SD {dsd:+.1%}: "
                             f"{'pass' if ok else 'FAIL'}")
        cell = cells[family] = model_cell(arrays, benign, origins, loglen, resamples)
        for c in CHOOSERS:
            row = dict(model=labels[family], chooser=c, call=call(cell, c))
            for s in STAGES:
                for key in ("reliability", "outcome_mass_mean", "outcome_mass_min", "slot_bias_mean", "additive_r2"):
                    row[f"{s}_{key}"] = summary[f"{s}/{NEW[c]}/{key}"]
            for name in PER:
                est, lo, hi = stat(cell, f"{c}:{name}")
                row.update({name: est, f"{name}_lo": lo, f"{name}_hi": hi})
            table.append(row)
        tops[family] = {(s, c): top_token_shares(paths[s][c]) for s in STAGES for c in CHOOSERS}

    print("Gate:\n  " + "\n  ".join(gates))
    fm = lambda r, s: f"{r[s]:+.2f} [{r[f'{s}_lo']:+.2f}, {r[f'{s}_hi']:+.2f}]"
    for r in table:
        print(f"\n{r['model']}, {CHOOSER_LABEL[r['chooser']].lower()}, user does it: {r['call']}")
        for name, label in (("retention", "retention"), ("retention_benign", "retention, non-harmful"),
                            ("gain_new", "gain, user does it"), ("gain_old", "gain, Assistant does it"),
                            ("r_base", "r(A_b, P_b)"), ("r_fixed", "r(A_f, P_b)"), ("r_final", "r(A_f, P_f)"),
                            ("d", "d = r(A_f,P_f) - r(A_f,P_b)"), ("r_shift", "r shift"),
                            ("r_final_benign", "non-harmful, final"), ("r_final_within", "within origin, final"),
                            ("r_old_base", "r(A_b, X_b)"), ("r_old_final", "r(A_f, X_f)"),
                            ("dr_base", "r(A_b,P_b) - r(A_b,X_b)"), ("dr_final", "r(A_f,P_f) - r(A_f,X_f)"),
                            ("dr_final_benign", "same, non-harmful"),
                            ("b_base", "b, base"), ("b_final", "b, final"), ("b_old_final", "b, X final"),
                            ("gap_base", "gap, base"), ("gap_final", "gap, final"), ("gap_old_final", "gap, X final"),
                            ("dgap_final", "gap P_f - gap X_f"), ("sd_base", "SD, base"), ("sd_final", "SD, final"),
                            ("sd_old_final", "SD, X final"), ("r_px_base", "r(P_b, X_b)"), ("r_px_final", "r(P_f, X_f)"),
                            ("r_px_final_benign", "r(P_f, X_f), non-harmful"),
                            ("r_len_new", "r(P_f, log length)"), ("r_len_old", "r(X_f, log length)"),
                            ("dr_len", "difference")):
            print(f"  {label:30} {fm(r, name)}")
        print(f"  mass base {r['base_outcome_mass_mean']:.2f}, final {r['instruct_outcome_mass_mean']:.2f}; slot bias base "
              f"{r['base_slot_bias_mean']:+.2f}, final {r['instruct_slot_bias_mean']:+.2f}; reliability base "
              f"{r['base_reliability']:.3f}, final {r['instruct_reliability']:.3f}")
        if min(r["base_outcome_mass_mean"], r["instruct_outcome_mass_mean"]) < MIN_MASS:
            print("  (a cell is below the readability floor: reported, not interpreted)")
    inter = []
    print("\nInteraction, coin minus user:")
    for family, cell in cells.items():
        for name in INTERACTION:
            est, lo, hi = stat(cell, name)
            inter.append(dict(model=labels[family], stat=name, est=est, lo=lo, hi=hi))
            print(f"  {labels[family]:14} {name:28} {est:+.2f} [{lo:+.2f}, {hi:+.2f}]")
    print("\nWhere the probability goes (mean over prompts, tokens logged in each top 5):")
    for family, by in tops.items():
        for (s, c), top in by.items():
            print(f"  {labels[family]:14} {s:9} {NEW[c]:22} " + "  ".join(f"{t!r} {p:.3f}" for t, p in top))
    origins = task_arrays(read_fit(Path("outputs/user_choice/analysis") / next(iter(FAMILIES))))[1]
    origin_rows = by_origin(data, origins, labels)
    print("\nMean final utility by origin (C, C', U, U'):")
    for r in origin_rows:
        print(f"  {r['model']:14} {r['origin']:12} " + "  ".join(f"{r[f'{k}_f']:+.2f}" for k, _ in CELLS))

    write_csv(out / "user_does.csv", table)
    write_csv(out / "interaction.csv", inter)
    write_csv(out / "by_origin.csv", origin_rows)
    plotstyle.setup()
    plot_summary(cells, labels, "figures/user_does_summary.png")
    plot_scatter(data, origins, labels, "figures/user_does_scatter.png")
    plot_by_origin(origin_rows, labels, "figures/user_does_by_origin.png")
    print(f"\nGate {'passed' if passed else 'FAILED'}. Wrote {out}/ and figures/user_does_*.png")


if __name__ == "__main__":
    main()
