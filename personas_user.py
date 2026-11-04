"""The user's choice under the evil persona: does post-training move it toward the default Assistant or the persona?

The measures and calls were pre-registered in the README ("The user's choice under a persona") before any data. Per model
(Llama 70B, Gemma 4 31B):
- A_0: the final default Assistant (the sweep, `tvlqj8a0`, `jrl62npf`);
- A_e, C_e: the final evil-persona Assistant and coin (W&B group "personas", personas.py's merged fit);
- U_b, U_f: the no-system-prompt user at base and final (W&B group "user_choice", user_choice.py's merged fit);
- U^e_b, U^e_f: the evil-prompted user at base and final (W&B group "personas_user", configs/personas_user/).

Primary: least squares U^e_f - U^e_b = c + b_def z(A_0) + b_per z(A_e), w = b_per / (b_def + b_per), next to the same fit of the
no-prompt change U_f - U_b (dw = w - w_0). CIs are 95% task bootstraps within origin, one set of resamples for everything.

  outputs/personas_user/analysis/<family>/   the merged fit (analyze.py) of one model's new runs
  outputs/personas_user/personas_user.csv    one row per model: every statistic with its CI, and the call
  outputs/personas_user/report.txt           (written by redirecting stdout)
  figures/personas_user_*.png

Usage: uv run python personas_user.py
"""

import csv
import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import personas
import user_choice
from analyze import N_BOOTSTRAP, ORIGIN_STYLE, analyze, fetch_wandb as fetch_raw, pearson, read_jsonl
from matskit import plotstyle
from models import MODEL_INFO
from openers import write_csv
from report import INK, INK_2, RULE, origin_scatter, stratified_resamples, style, task_arrays
from user_choice import FAMILIES, TASKS_FILE, ci, load_sweep, top_token_shares

MODELS = ("llama-70b", "gemma4-31b")
STAGES = ("base", "instruct")
NEW = "user_choice@evil"
GATE_R, GATE_SD, MIN_RELIABILITY, MIN_MASS = 0.999, 0.02, 0.9, 0.2
BENIGN_SOURCES = ("MATH", "ALPACA", "WILDCHAT")
STATS = ("b_def", "b_per", "w", "b_def0", "b_per0", "w0", "dw", "d_bdef", "d_bper", "r_a0_ae",
         "r_a0_ueb", "r_a0_uef", "r_ae_ueb", "r_ae_uef", "d_def", "d_per",
         "lb_def", "lb_per", "lw", "cb_def", "cb_per", "cw", "lw_minus_cw", "r_uef_ce",
         "r_ueb_ub", "bb_def", "bb_per", "gap_ueb", "gap_uef", "gap_ub", "gap_uf", "gap_uef_minus_uf",
         "b_def_l", "b_per_l", "w_l", "b_def_ls", "b_per_ls", "w_ls",
         "sd_ueb", "sd_uef", "sd_ub", "sd_uf", "r_uef_uf") \
    + tuple(f"r_{t}_{o}" for t in ("a0", "ae") for o in BENIGN_SOURCES)
K = {name: j for j, name in enumerate(STATS)}


def pulls(y, a0, ae, extra=()):
    """(b_def, b_per, w) of y = c + b_def z(a0) + b_per z(ae) + extra columns."""
    X = np.column_stack([np.ones_like(y), personas.z(a0), personas.z(ae), *extra])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta[1], beta[2], beta[2] / (beta[1] + beta[2])


def stats(V, benign, loglen, origins):
    """STATS over arrays of the same (possibly resampled) tasks. V: A0, Ae, Ce, Ub, Uf, Ueb, Uef."""
    a0, ae, ce, ub, uf, ueb, uef = (V[k] for k in ("A0", "Ae", "Ce", "Ub", "Uf", "Ueb", "Uef"))
    gap = lambda x: x[benign].mean() - x[~benign].mean()
    sources = [(origins == o).astype(float) for o in np.unique(origins)[1:]]
    de, d0 = uef - ueb, uf - ub
    b_def, b_per, w = pulls(de, a0, ae)
    b_def0, b_per0, w0 = pulls(d0, a0, ae)
    lb_def, lb_per, lw = pulls(uef, a0, ae)
    cb_def, cb_per, cw = pulls(ce, a0, ae)
    bb_def, bb_per, _ = pulls(ueb, a0, ae)
    out = [b_def, b_per, w, b_def0, b_per0, w0, w - w0, b_def - b_def0, b_per - b_per0, pearson(a0, ae),
           pearson(a0, ueb), pearson(a0, uef), pearson(ae, ueb), pearson(ae, uef),
           pearson(a0, uef) - pearson(a0, ueb), pearson(ae, uef) - pearson(ae, ueb),
           lb_def, lb_per, lw, cb_def, cb_per, cw, lw - cw, pearson(uef, ce),
           pearson(ueb, ub), bb_def, bb_per, gap(ueb), gap(uef), gap(ub), gap(uf), gap(uef) - gap(uf),
           *pulls(de, a0, ae, [loglen]), *pulls(de, a0, ae, [loglen, *sources]),
           ueb.std(), uef.std(), ub.std(), uf.std(), pearson(uef, uf)]
    for t in (a0, ae):
        for o in BENIGN_SOURCES:
            m = origins == o
            out.append(pearson(t[m], de[m]))
    return np.array(out)


def call(est, lo, hi):
    """The pre-registered call, the persona section's rules applied to the primary fit."""
    if abs(est[K["r_a0_ae"]]) > personas.SEPARABLE:
        return "not separable"
    if hi[K["b_per"]] < 0:
        return "against the persona"
    def_up, per_up = lo[K["b_def"]] > 0, lo[K["b_per"]] > 0
    if not (def_up or per_up):
        return "neither"
    if def_up and not per_up:
        return "default Assistant"
    if per_up and not def_up:
        return "persona"
    return "mix"


def merged_fit(runs, out):
    """(utilities.csv rows, summary, {(stage, condition): raw results path}) of one model's new runs, fit together."""
    rows, paths = [], {}
    for run in runs:
        print(f"  {run.path} {run.name}")
        rows += fetch_raw(run, Path("outputs") / run.id)
        (m,) = run.config["models"]
        for c in m["conditions"]:
            paths[m["stage"], c] = Path("outputs") / run.id / "results" / f"{m['stage']}__{c}.csv.gz"
    if not (out / "utilities.csv").exists():
        analyze(rows, read_jsonl(TASKS_FILE), out, seed=0, stage_order=list(STAGES))
    with open(out / "utilities.csv", newline="") as fh:
        return list(csv.DictReader(fh)), json.loads((out / "summary.json").read_text()), paths


def median_mass(path):
    with gzip.open(path, "rt", newline="") as fh:
        return float(np.median([float(r["outcome_mass"]) for r in csv.DictReader(fh)]))


def plot_scatter(data, origins, labels, path):
    """Rows: models. Columns: the post-training change in the user's choice, with and without the evil prompt, against the
    default Assistant and against the evil persona's Assistant."""
    panels = (("A0", "dE", "Default Assistant (nats)", "With evil prompt: U^e_f - U^e_b"),
              ("Ae", "dE", "Evil-persona Assistant (nats)", "With evil prompt: U^e_f - U^e_b"),
              ("A0", "d0", "Default Assistant (nats)", "No system prompt: U_f - U_b"),
              ("Ae", "d0", "Evil-persona Assistant (nats)", "No system prompt: U_f - U_b"))
    fig, axes = plt.subplots(len(data), len(panels), figsize=(3.9 * len(panels), 3.6 * len(data)), squeeze=False)
    for i, (family, V) in enumerate(data.items()):
        V = {**V, "dE": V["Uef"] - V["Ueb"], "d0": V["Uf"] - V["Ub"]}
        for j, (xk, yk, xlabel, ylabel) in enumerate(panels):
            ax = axes[i, j]
            origin_scatter(ax, V[xk], V[yk], origins)
            ax.set_title(f"r = {pearson(V[xk], V[yk]):+.2f}", fontsize=10, color=INK)
            ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
            ax.set_ylabel(f"{labels[family]}\n{ylabel} (nats)" if j == 0 else f"{ylabel} (nats)",
                          color=INK if j == 0 else INK_2, fontsize=9)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, frameon=False, fontsize=9, loc="lower center", ncol=len(names))
    fig.suptitle("Post-training's change in the user's choice, against the default and the evil Assistant", color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    plotstyle.save(fig, path.stem, path.parent)


def plot_pulls(cells, labels, path):
    """Per model: pulls toward the evil persona (x) and the default Assistant (y), 95% CIs, for the evil-prompted user's
    post-training change (primary), the no-prompt user's change (reference), and the evil coin (levels)."""
    series = (("b_per", "b_def", "w", "user change, evil prompt (primary)", 0, "o"),
              ("b_per0", "b_def0", "w0", "user change, no system prompt", None, "D"),
              ("cb_per", "cb_def", "cw", "evil coin (levels, persona runs)", 1, "s"))
    fig, axes = plt.subplots(1, len(cells), figsize=(5.0 * len(cells), 4.8), squeeze=False)
    for ax, (family, cell) in zip(axes[0], cells.items()):
        est, (lo, hi) = cell["est"], ci(cell["boot"])
        top, bottom = 0.0, 0.0
        for xk, yk, wk, label, color, marker in series:
            x, y = K[xk], K[yk]
            ax.errorbar(est[x], est[y], xerr=[[est[x] - lo[x]], [hi[x] - est[x]]],
                        yerr=[[est[y] - lo[y]], [hi[y] - est[y]]], color=INK_2 if color is None else plotstyle.SERIES[color],
                        marker=marker, ms=8, lw=0, elinewidth=1.5, mec="white", mew=0.8, zorder=3,
                        label=label if ax is axes[0, 0] else None)
            ax.annotate(f"w = {est[K[wk]]:.2f}", (est[x], est[y]), xytext=(8, 6), textcoords="offset points",
                        color=INK, fontsize=9)
            top, bottom = max(top, hi[x], hi[y]), min(bottom, lo[x], lo[y])
        top, bottom = top * 1.15, bottom * 1.15
        ax.plot([0, top], [0, top], color=RULE, ls="--", lw=1, zorder=1)
        ax.axhline(0, color=RULE, lw=0.8, zorder=1)
        ax.axvline(0, color=RULE, lw=0.8, zorder=1)
        ax.set_xlim(bottom - 0.02 * top, top)
        ax.set_ylim(bottom - 0.02 * top, top)
        ax.set_aspect("equal")
        ax.set_xlabel("pull toward the evil persona's Assistant, b_per\n(nats per SD of A_e)", color=INK_2, fontsize=9)
        ax.set_ylabel("pull toward the default Assistant, b_def\n(nats per SD of A_0)", color=INK_2, fontsize=9)
        ax.set_title(labels[family], color=INK, fontsize=10)
        style(ax)
    fig.legend(frameon=False, fontsize=9, loc="lower center", ncol=3)
    fig.suptitle("Whose preferences does post-training move the user toward under the evil persona?", color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    plotstyle.save(fig, path.stem, path.parent)


def main():
    plotstyle.setup()
    out = Path("outputs/personas_user")
    out.mkdir(parents=True, exist_ok=True)
    new_runs = personas.collect_runs("personas_user")
    persona_runs = personas.collect_runs("personas")
    uc_runs = user_choice.collect_runs("user_choice")
    data, cells, labels, gates, tops, passed = {}, {}, {}, [], {}, {}
    ids0 = resamples = None
    for family in MODELS:
        labels[family] = label = MODEL_INFO[FAMILIES[family][0]][0]
        print(f"{label}:")
        rows, summary, paths = merged_fit(new_runs[family], out / "analysis" / family)
        prow = personas.merged_fit(persona_runs[family], Path("outputs/personas/analysis") / family)[0]
        urow = user_choice.merged_fit(uc_runs, family, Path("outputs/user_choice/analysis") / family)[0]
        srow = load_sweep(family)[0]
        ids = [r["id"] for r in rows]
        assert all(ids == [r["id"] for r in x] for x in (prow, urow, srow)), "task sets differ"
        N, origins, benign, loglen = task_arrays(rows)
        P, Uc, B = task_arrays(prow)[0], task_arrays(urow)[0], task_arrays(srow)[0]
        if ids0 is None:
            ids0, resamples = ids, stratified_resamples(origins, np.random.default_rng(0), N_BOOTSTRAP)
        assert ids == ids0, "task sets differ between models"

        # Gate: the re-scored plain user choice matches the user-choice runs; every new cell is reliable.
        x, y = N["instruct/user_choice"], Uc["instruct/user_choice"]
        r, sd = pearson(x, y), x.std() / y.std() - 1
        ok = r >= GATE_R and abs(sd) <= GATE_SD
        gates.append(f"{label:14} re-scored user_choice vs user-choice runs: r {r:.4f}, SD {sd:+.1%}: {'pass' if ok else 'FAIL'}")
        for (stage, c), path in paths.items():
            rel, mass = summary[f"{stage}/{c}/reliability"], median_mass(path)
            ok_cell = rel >= MIN_RELIABILITY
            ok &= ok_cell
            gates.append(f"{label:14} {stage:9} {c:17} reliability {rel:.3f}, median mass {mass:.2f}"
                         f"{'' if mass >= MIN_MASS else ' (below 0.2: not interpreted)'}: {'pass' if ok_cell else 'FAIL'}")
        passed[family] = ok
        V = {"A0": B["instruct/assistant"], "Ae": P["instruct/assistant@evil"], "Ce": P["instruct/coinflip@evil"],
             "Ub": Uc["base/user_choice"], "Uf": Uc["instruct/user_choice"],
             "Ueb": N[f"base/{NEW}"], "Uef": N[f"instruct/{NEW}"]}
        data[family] = V
        cells[family] = dict(
            est=stats(V, benign, loglen, origins),
            boot=np.array([stats({k: v[i] for k, v in V.items()}, benign[i], loglen[i], origins[i]) for i in resamples]))
        tops[family] = {f"{s}/{c}": top_token_shares(p) for (s, c), p in paths.items()}

    print("\nValidation gate:\n  " + "\n  ".join(gates))
    table = []
    fm = lambda est, lo, hi, s: f"{est[K[s]]:+.3f} [{lo[K[s]]:+.3f}, {hi[K[s]]:+.3f}]"
    for family, cell in cells.items():
        est, (lo, hi) = cell["est"], ci(cell["boot"])
        verdict = call(est, lo, hi) if passed[family] else "no call (gate)"
        print(f"\n{labels[family]}: {verdict}")
        row = dict(model=labels[family], call=verdict)
        for s in STATS:
            print(f"  {s:18} {fm(est, lo, hi, s)}")
            row.update({s: est[K[s]], f"{s}_lo": lo[K[s]], f"{s}_hi": hi[K[s]]})
        row["call_length_source"] = call(*(np.array(v) for v in _swap(est, lo, hi, "_ls")))
        print(f"  call with log length + source fixed effects: {row['call_length_source']}")
        table.append(row)

    print("\nExploratory, where the probability goes (mean over prompts, tokens logged in each top 5):")
    for family, by_cell in tops.items():
        for c, top in by_cell.items():
            print(f"  {labels[family]:14} {c:26} " + "  ".join(f"{t!r} {p:.3f}" for t, p in top))

    print("\nExploratory, mean utility by origin in SDs from the task mean:")
    for family, V in data.items():
        for k in ("Ueb", "Uef", "Ub", "Uf", "Ae", "A0"):
            x = V[k]
            print(f"  {labels[family]:14} {k:4} " + "  ".join(
                f"{o[:8]:8} {(x[origins == o].mean() - x.mean()) / x.std():+.2f}" for o in ORIGIN_STYLE))

    write_csv(out / "personas_user.csv", table)
    plot_scatter(data, origins, labels, Path("figures/personas_user_scatter.png"))
    plot_pulls(cells, labels, Path("figures/personas_user_pulls.png"))
    print("\nWrote outputs/personas_user/ and figures/personas_user_*.png")


def _swap(est, lo, hi, suffix):
    """est, lo, hi with the primary fit's b_def/b_per replaced by a robustness variant's (e.g. "_ls"), for call()."""
    out = []
    for a in (est, lo, hi):
        a = a.copy()
        a[K["b_def"]], a[K["b_per"]] = a[K[f"b_def{suffix}"]], a[K[f"b_per{suffix}"]]
        out.append(a)
    return out


if __name__ == "__main__":
    main()
