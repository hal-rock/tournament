"""Persona placement: the evil persona in the system prompt, as a user instruction, or as the user's self-description.

The measures and calls were pre-registered in the README ("Persona placement: system prompt, user instruction, or user
self-description") before any data. Per model (Llama 70B, Gemma 4 31B):
- A_0, C_0: the sweep's final default Assistant and coin;
- A_e, C_sys: the final system-prompt evil Assistant and coin (W&B group "personas");
- C_user, C_self, A_user, A_self at the final stage, and the three coins at base (W&B group "persona_placement").

Primary, per placement p (user, self): least squares C_p = c + b_def z(A_0) + b_per z(A_e) at the final stage, with the
placement ratio rho_p = b_per(p) / b_per(system). CIs are 95% task bootstraps within origin, one set of resamples for everything.

  outputs/persona_placement/<family>/analysis/   the merged fit (analyze.py) of one model's new runs
  outputs/persona_placement/<family>/gate/<run>/ each pod's re-scored plain coinflip, fit on its own
  outputs/persona_placement/persona_placement.csv  one row per model: every statistic with its CI, and the calls
  outputs/persona_placement/report.txt           (written by redirecting stdout)
  figures/persona_placement_*.png              incl. persona_placement_sources_<family>.png, the per-source grid

Usage: uv run python persona_placement.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import personas
from analyze import N_BOOTSTRAP, ORIGIN_STYLE, pearson
from matskit import plotstyle
from models import MODEL_INFO
from openers import write_csv
from analyze import analyze, fetch_wandb as fetch_raw, read_jsonl
from personas_user import median_mass, pulls
from report import INK, INK_2, RULE, origin_scatter, stratified_resamples, style, task_arrays
from source_grid import Row, plot_source_grid
from user_choice import FAMILIES, TASKS_FILE, ci, load_sweep, top_token_shares
import csv
import json

MODELS = ("llama-70b", "gemma4-31b")
PLACES = ("sys", "user", "self")
PLACE_LABEL = {"sys": "System prompt", "user": "User instruction", "self": "User self-description"}
GATE_R, GATE_SD, MIN_RELIABILITY, MIN_MASS = 0.999, 0.02, 0.9, 0.2
ORIGINS = tuple(ORIGIN_STYLE)
PER = ("b_def", "b_per", "w", "gap", "r_a0", "r_ae", "b_def_bc", "b_per_bc", "bb_def", "bb_per", "rb_ae", "sd_b", "sd",
       "b_per_ls", "w_ls")
STATS = (tuple(f"{p}:{s}" for p in PLACES for s in PER)
         + ("rho_user", "rho_self", "rho_self_minus_user", "rho_user_bc", "rho_self_bc",
            "r_auser_ae", "r_auser_a0", "gap_auser", "r_aself_a0", "r_aself_ae", "gap_aself", "gap_a0", "gap_ae",
            "own_b_def", "own_b_per", "own_w", "r_cuser_csys", "r_cself_csys", "r_cself_c0", "r_cuser_c0", "gap_c0", "b0")
         + tuple(f"{p}:r_{t}_{o}" for p in ("user", "self") for t in ("a0", "ae") for o in ORIGINS))
K = {name: j for j, name in enumerate(STATS)}


def merged_fit(runs, out):
    """(utilities.csv rows, summary, {(stage, condition): raw path}, {run name: (stage, utilities of its plain coinflip)}).
    Every pod re-scores the plain coinflip; the merged fit keeps one copy of it per stage (the first run's), and each run's
    copy is also fit on its own, for that pod's check against the sweep."""
    rows, paths, checks, seen = [], {}, {}, set()
    tasks = read_jsonl(TASKS_FILE)
    for run in sorted(runs, key=lambda r: r.name):
        print(f"  {run.path} {run.name}")
        run_rows = fetch_raw(run, Path("outputs") / run.id)
        (m,) = run.config["models"]
        for c in m["conditions"]:
            paths.setdefault((m["stage"], c), Path("outputs") / run.id / "results" / f"{m['stage']}__{c}.csv.gz")
        plain = [r for r in run_rows if r["condition"] == "coinflip"]
        gate_dir = out / "gate" / run.id
        if not (gate_dir / "utilities.csv").exists():
            analyze(plain, tasks, gate_dir, seed=0, stage_order=[m["stage"]])
        with open(gate_dir / "utilities.csv", newline="") as fh:
            checks[run.name] = (m["stage"], task_arrays(list(csv.DictReader(fh)))[0][f"{m['stage']}/coinflip"])
        for r in run_rows:
            key = (r["stage"], r["condition"])
            if key not in seen or (r["condition"] != "coinflip"):
                rows.append(r)
        seen |= {(r["stage"], r["condition"]) for r in run_rows}
    if not (out / "analysis" / "utilities.csv").exists():
        analyze(rows, tasks, out / "analysis", seed=0, stage_order=["base", "instruct"])
    with open(out / "analysis" / "utilities.csv", newline="") as fh:
        return (list(csv.DictReader(fh)), json.loads((out / "analysis" / "summary.json").read_text()), paths, checks)


def stats(V, benign, loglen, origins):
    """STATS over arrays of the same (possibly resampled) tasks."""
    a0, ae = V["A0"], V["Ae"]
    gap = lambda x: x[benign].mean() - x[~benign].mean()
    sources = [(origins == o).astype(float) for o in np.unique(origins)[1:]]
    out = []
    for p in PLACES:
        c, b = V[f"C{p}"], V[f"B{p}"]
        b_def, b_per, w = pulls(c, a0, ae)
        b_per_ls, w_ls = pulls(c, a0, ae, [loglen, *sources])[1:]
        out += [b_def, b_per, w, gap(c), pearson(c, a0), pearson(c, ae), *pulls(c, a0, ae, [b])[:2],
                *pulls(b, a0, ae)[:2], pearson(b, ae), b.std(), c.std(), b_per_ls, w_ls]
    g = lambda p, s: out[PLACES.index(p) * len(PER) + PER.index(s)]
    out += [g("user", "b_per") / g("sys", "b_per"), g("self", "b_per") / g("sys", "b_per"),
            (g("self", "b_per") - g("user", "b_per")) / g("sys", "b_per"),
            g("user", "b_per_bc") / g("sys", "b_per_bc"), g("self", "b_per_bc") / g("sys", "b_per_bc"),
            pearson(V["Auser"], ae), pearson(V["Auser"], a0), gap(V["Auser"]),
            pearson(V["Aself"], a0), pearson(V["Aself"], ae), gap(V["Aself"]), gap(a0), gap(ae),
            *pulls(V["Cuser"], a0, V["Auser"]), pearson(V["Cuser"], V["Csys"]), pearson(V["Cself"], V["Csys"]),
            pearson(V["Cself"], V["C0"]), pearson(V["Cuser"], V["C0"]), gap(V["C0"]), personas.slope_per_sd(a0, V["C0"])]
    for p in ("user", "self"):
        for t in (a0, ae):
            for o in ORIGINS:
                m = origins == o
                out.append(pearson(t[m], V[f"C{p}"][m]))
    return np.array(out)


def call(est, lo, hi, p):
    """The pre-registered placement call for p (user, self), on rho_p, first that applies."""
    b, rho = K[f"{p}:b_per"], K[f"rho_{p}"]
    if hi[b] < 0:
        return "reversed"
    if lo[b] <= 0 or lo[rho] <= 0:
        return "absent"
    if lo[rho] > 1:
        return "stronger than the system prompt"
    if hi[rho] >= 1:
        return "as strong as the system prompt"
    return "weaker"


def persona_call(est, lo, hi, p):
    """The persona section's call (default Assistant / persona / mix ...) for placement p."""
    sub = {s: K[f"{p}:{s}"] for s in ("b_def", "b_per")}
    if hi[sub["b_per"]] < 0:
        return "against the persona"
    def_up, per_up = lo[sub["b_def"]] > 0, lo[sub["b_per"]] > 0
    return {(False, False): "neither", (True, False): "default Assistant", (False, True): "persona",
            (True, True): "mix"}[def_up, per_up]


def plot_pulls(cells, labels, path):
    """Per model: each placement's pull toward the persona (x) and the default Assistant (y), 95% CIs; b_0 for reference."""
    fig, axes = plt.subplots(1, len(cells), figsize=(5.0 * len(cells), 4.8), squeeze=False)
    for ax, (family, cell) in zip(axes[0], cells.items()):
        est, (lo, hi) = cell["est"], ci(cell["boot"])
        top, bottom = 0.0, 0.0
        for k, p in enumerate(PLACES):
            x, y = K[f"{p}:b_per"], K[f"{p}:b_def"]
            ax.errorbar(est[x], est[y], xerr=[[est[x] - lo[x]], [hi[x] - est[x]]], yerr=[[est[y] - lo[y]], [hi[y] - est[y]]],
                        color=plotstyle.SERIES[k], marker="o", ms=8, lw=0, elinewidth=1.5, mec="white", mew=0.8, zorder=3,
                        label=PLACE_LABEL[p] if ax is axes[0, 0] else None)
            ax.annotate(f"w = {est[K[f'{p}:w']]:.2f}", (est[x], est[y]), xytext=(8, 6), textcoords="offset points",
                        color=INK, fontsize=9)
            top, bottom = max(top, hi[x], hi[y]), min(bottom, lo[x], lo[y])
        b0 = K["b0"]
        ax.errorbar(0, est[b0], yerr=[[est[b0] - lo[b0]], [hi[b0] - est[b0]]], color=INK_2, marker="D", ms=7, lw=0,
                    elinewidth=1.5, mec="white", mew=0.8, zorder=3, label="no persona (b_0)" if ax is axes[0, 0] else None)
        top, bottom = max(top, hi[b0]) * 1.15, bottom * 1.15
        ax.plot([0, top], [0, top], color=RULE, ls="--", lw=1, zorder=1)
        ax.axhline(0, color=RULE, lw=0.8, zorder=1)
        ax.axvline(0, color=RULE, lw=0.8, zorder=1)
        ax.set_xlim(bottom - 0.02 * top, top)
        ax.set_ylim(bottom - 0.02 * top, top)
        ax.set_aspect("equal")
        ax.set_xlabel("pull toward the evil persona's Assistant, b_per\n(coin nats per SD of A_e)", color=INK_2, fontsize=9)
        ax.set_ylabel("pull toward the default Assistant, b_def\n(coin nats per SD of A_0)", color=INK_2, fontsize=9)
        ax.set_title(labels[family], color=INK, fontsize=10)
        style(ax)
    fig.legend(frameon=False, fontsize=9, loc="lower center", ncol=4)
    fig.suptitle("Where the evil persona sits, and whose preferences the coin follows", color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    plotstyle.save(fig, path.stem, path.parent)


def plot_scatter(data, origins, labels, target, path):
    """Rows: models. Columns: placements. Each panel: that placement's final coin against A_e (target "Ae") or A_0 ("A0")."""
    name = {"Ae": "Evil-persona Assistant (nats)", "A0": "Default Assistant (nats)"}[target]
    fig, axes = plt.subplots(len(data), len(PLACES), figsize=(4.0 * len(PLACES), 3.6 * len(data)), squeeze=False)
    for i, (family, V) in enumerate(data.items()):
        for j, p in enumerate(PLACES):
            ax = axes[i, j]
            x, y = V[target], V[f"C{p}"]
            origin_scatter(ax, x, y, origins)
            r = f"r = {pearson(x, y):+.2f}"
            ax.set_title(f"{PLACE_LABEL[p]}\n{r}" if i == 0 else r, fontsize=10, color=INK)
            ax.set_xlabel(name, color=INK_2, fontsize=9)
            ax.set_ylabel(f"{labels[family]}\nevil coinflip (nats)" if j == 0 else "evil coinflip (nats)",
                          color=INK if j == 0 else INK_2, fontsize=9)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, frameon=False, fontsize=9, loc="lower center", ncol=len(names))
    vs = "the evil persona's Assistant" if target == "Ae" else "the default Assistant"
    fig.suptitle(f"The evil coin by where the persona sits, against {vs}", color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    plotstyle.save(fig, path.stem, path.parent)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS), help="a subset, e.g. while a model is still running")
    args = ap.parse_args()
    plotstyle.setup()
    out = Path("outputs/persona_placement")
    out.mkdir(parents=True, exist_ok=True)
    new_runs = personas.collect_runs("persona_placement")
    persona_runs = personas.collect_runs("personas")
    data, cells, labels, gates, tops, passed = {}, {}, {}, [], {}, {}
    ids0 = resamples = None
    for family in args.models:
        labels[family] = label = MODEL_INFO[FAMILIES[family][0]][0]
        print(f"{label}:")
        rows, summary, paths, checks = merged_fit(new_runs[family], out / family)
        prow = personas.merged_fit(persona_runs[family], Path("outputs/personas/analysis") / family)[0]
        srow = load_sweep(family)[0]
        ids = [r["id"] for r in rows]
        assert all(ids == [r["id"] for r in x] for x in (prow, srow)), "task sets differ"
        N, origins, benign, loglen = task_arrays(rows)
        P, B = task_arrays(prow)[0], task_arrays(srow)[0]
        if ids0 is None:
            ids0, resamples = ids, stratified_resamples(origins, np.random.default_rng(0), N_BOOTSTRAP)
        assert ids == ids0, "task sets differ between models"

        ok = True
        for name, (stage, x) in sorted(checks.items()):  # each pod's own re-scored plain coinflip vs the sweep's
            y = B[f"{stage}/coinflip"]
            r, sd = pearson(x, y), x.std() / y.std() - 1
            ok_pod = r >= GATE_R and abs(sd) <= GATE_SD
            ok &= ok_pod
            gates.append(f"{label:14} {name}: re-scored {stage} coinflip vs sweep: r {r:.4f}, SD {sd:+.1%}: "
                         f"{'pass' if ok_pod else 'FAIL'}")
        for (stage, c), path in sorted(paths.items()):
            rel, mass = summary[f"{stage}/{c}/reliability"], median_mass(path)
            ok_cell = rel >= MIN_RELIABILITY
            ok &= ok_cell
            gates.append(f"{label:14} {stage:9} {c:20} reliability {rel:.3f}, median mass {mass:.2f}"
                         f"{'' if mass >= MIN_MASS else ' (below 0.2: not interpreted)'}: {'pass' if ok_cell else 'FAIL'}")
        passed[family] = ok
        V = {"A0": B["instruct/assistant"], "C0": B["instruct/coinflip"], "Ae": P["instruct/assistant@evil"],
             "Csys": P["instruct/coinflip@evil"], "Cuser": N["instruct/coinflip@evil+user"],
             "Cself": N["instruct/coinflip@evil+self"], "Auser": N["instruct/assistant@evil+user"],
             "Aself": N["instruct/assistant@evil+self"], "Bsys": N["base/coinflip@evil"],
             "Buser": N["base/coinflip@evil+user"], "Bself": N["base/coinflip@evil+self"]}
        data[family] = V
        cells[family] = dict(
            est=stats(V, benign, loglen, origins),
            boot=np.array([stats({k: v[i] for k, v in V.items()}, benign[i], loglen[i], origins[i]) for i in resamples]))
        tops[family] = {f"{s}/{c}": top_token_shares(p) for (s, c), p in sorted(paths.items())}

    print("\nValidation gate:\n  " + "\n  ".join(gates))
    table = []
    fm = lambda est, lo, hi, s: f"{est[K[s]]:+.3f} [{lo[K[s]]:+.3f}, {hi[K[s]]:+.3f}]"
    for family, cell in cells.items():
        est, (lo, hi) = cell["est"], ci(cell["boot"])
        row = dict(model=labels[family])
        print(f"\n{labels[family]}")
        for p in ("user", "self"):
            row[f"call_{p}"] = call(est, lo, hi, p) if passed[family] else "no call (gate)"
            print(f"  placement call, {p}: {row[f'call_{p}']}")
        for p in PLACES:
            row[f"persona_call_{p}"] = persona_call(est, lo, hi, p)
            print(f"  persona-section call, {p}: {row[f'persona_call_{p}']}")
        for s in STATS:
            print(f"  {s:22} {fm(est, lo, hi, s)}")
            row.update({s: est[K[s]], f"{s}_lo": lo[K[s]], f"{s}_hi": hi[K[s]]})
        table.append(row)

    print("\nExploratory, where the probability goes (mean over prompts, tokens logged in each top 5):")
    for family, by_cell in tops.items():
        for c, top in by_cell.items():
            print(f"  {labels[family]:14} {c:30} " + "  ".join(f"{t!r} {p:.3f}" for t, p in top))
    print("\nExploratory, mean utility by origin in SDs from the task mean:")
    for family, V in data.items():
        for k in ("C0", "Csys", "Cuser", "Cself", "A0", "Ae", "Auser", "Aself"):
            x = V[k]
            print(f"  {labels[family]:14} {k:6} " + "  ".join(
                f"{o[:8]:8} {(x[origins == o].mean() - x.mean()) / x.std():+.2f}" for o in ORIGINS))

    write_csv(out / "persona_placement.csv", table)
    plot_pulls(cells, labels, Path("figures/persona_placement_pulls.png"))
    plot_scatter(data, origins, labels, "Ae", Path("figures/persona_placement_vs_evil.png"))
    plot_scatter(data, origins, labels, "A0", Path("figures/persona_placement_vs_default.png"))
    for family, V in data.items():  # per-source grids: sources across, one row per (placement, Assistant)
        rows = [Row(f"{PLACE_LABEL[p]} coin\nvs {name} Assistant", V[t], V[f"C{p}"], f"{name.capitalize()} Assistant (nats)",
                    "evil coinflip (nats)", plotstyle.SERIES[k])
                for k, p in enumerate(PLACES) for t, name in (("A0", "default"), ("Ae", "evil"))]
        plot_source_grid(rows, origins, f"{labels[family]}: the evil coin by placement, within each task source",
                         Path(f"figures/persona_placement_sources_{family}.png"))
    print("\nWrote outputs/persona_placement/ and figures/persona_placement_*.png")


if __name__ == "__main__":
    main()
