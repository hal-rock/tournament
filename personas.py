"""Personas in the system prompt: does the coinflip follow the default Assistant, the persona, or a mix?

The measures and calls were pre-registered in the README ("Personas in the system prompt") before any data. Per model
(OLMo 3.1 32B Instruct, Llama 3.3 70B Instruct, Gemma 4 31B-it):
- A_0, C_0: the no-system-prompt Assistant and coinflip utilities, from the sweep (`d8ysbybo` refit on its pairs,
  `tvlqj8a0`, `jrl62npf`), unless the gate fails (OLMo then uses the ones re-scored on the persona pods);
- A_p, C_p for p in helpful (the control), evil and slacker: from the W&B runs of group "personas" (configs/personas/).

Primary, per model and persona (evil, slacker): least squares C_p = c + b_def z(A_0) + b_per z(A_p) over the 300 tasks, z()
standardizing over tasks; persona weight w = b_per / (b_def + b_per). CIs are 95% task bootstraps within origin, one set
of resamples for everything, so differences (w_evil - w_slacker) are paired. `call` gives the pre-registered call.

  outputs/personas/analysis/<family>/   the merged fit (analyze.py) of one model's persona runs
  outputs/personas/personas.csv         one row per model and persona: every statistic with its CI, and the call
  outputs/personas/control.csv          the helpful control, per model
  outputs/personas/by_origin.csv
  figures/personas_*.png

Usage: uv run python personas.py [--group personas]
"""

import argparse
import collections
import csv
import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import N_BOOTSTRAP, ORIGIN_STYLE, analyze, fetch_wandb as fetch_raw, partial_r, pearson, read_jsonl
from matskit import plotstyle
from models import MODEL_INFO, PROJECT
from openers import slope_per_sd, write_csv
from report import INK, INK_2, RULE, errorbar, origin_scatter, stratified_resamples, style, task_arrays
from user_choice import FAMILIES, TASKS_FILE, ci, load_sweep, top_token_shares

MODELS = ("olmo3.1-32b", "llama-70b", "gemma4-31b")
PERSONAS = ("evil", "slacker")
CONDITIONS = ("none", "helpful", "slacker", "evil")  # figure order
COND_LABEL = {"none": "No system prompt", "helpful": "\"You are a helpful assistant.\"", "slacker": "Slacker",
              "evil": "Evil"}
GATE_R, GATE_SD, MIN_RELIABILITY, MIN_MASS = 0.999, 0.02, 0.9, 0.2
SEPARABLE = 0.9  # |r(A_p, A_0)| above this: the two pulls can't be told apart

PER = ("b_def", "b_per", "w", "total", "total_minus_b0", "r_cp_a0", "r_cp_ap", "partial_per", "partial_def", "r_ap_a0",
       "shift_r", "transfer", "b_def_h", "b_per_h", "w_h", "b_def_nh", "b_per_nh", "w_nh",
       "gap_ap", "gap_cp", "gap_cp_minus_c0", "r_ap_len", "r_cp_len", "math_ap", "sd_ap", "sd_cp", "r_cp_c0", "r2_one",
       "r2_two")
CONTROL = ("b0", "gap_a0", "gap_c0", "r_a0_len", "math_a0", "r_ah_a0", "r_ch_c0", "b_h", "b_h_minus_b0", "gap_ch",
           "r_ah_c0")
STATS = tuple(f"{p}:{s}" for p in PERSONAS for s in PER) + CONTROL + ("w_evil_minus_slacker",)
K = {name: j for j, name in enumerate(STATS)}


def z(x):
    return (x - x.mean()) / x.std()


def two_pulls(c, a0, ap):
    """(b_def, b_per, R^2) of the least-squares fit c = const + b_def z(a0) + b_per z(ap)."""
    X = np.column_stack([np.ones_like(c), z(a0), z(ap)])
    beta, *_ = np.linalg.lstsq(X, c, rcond=None)
    r2 = 1 - ((c - X @ beta) ** 2).sum() / ((c - c.mean()) ** 2).sum()
    return beta[1], beta[2], r2


def slope(x, y):
    """Slope of y on x, in nats per nat."""
    return float(np.mean((x - x.mean()) * (y - y.mean())) / x.var())


def stats(U, benign, loglen, math):
    """STATS over arrays of the same (possibly resampled) tasks. U: {"A0", "C0", "Ahelpful", "Chelpful", "Aevil", ...}."""
    a0, c0, ah, ch = U["A0"], U["C0"], U["Ahelpful"], U["Chelpful"]
    gap = lambda x: x[benign].mean() - x[~benign].mean()
    on_math = lambda x: x[math].mean() - x.mean()
    b0 = slope_per_sd(a0, c0)
    out = []
    ws = {}
    for p in PERSONAS:
        ap, cp = U[f"A{p}"], U[f"C{p}"]
        b_def, b_per, r2_two = two_pulls(cp, a0, ap)
        b_def_h, b_per_h, _ = two_pulls(cp, ah, ap)
        b_def_nh, b_per_nh, _ = two_pulls(cp[benign], a0[benign], ap[benign])
        ws[p] = w = b_per / (b_def + b_per)
        out += [b_def, b_per, w, b_def + b_per, b_def + b_per - b0, pearson(cp, a0), pearson(cp, ap),
                partial_r(cp, ap, a0), partial_r(cp, a0, ap), pearson(ap, a0),
                pearson(ap - ah, cp - ch), slope(ap - ah, cp - ch) / slope(ah, ch),
                b_def_h, b_per_h, b_per_h / (b_def_h + b_per_h), b_def_nh, b_per_nh, b_per_nh / (b_def_nh + b_per_nh),
                gap(ap), gap(cp), gap(cp) - gap(c0), pearson(ap, loglen), pearson(cp, loglen), on_math(ap),
                ap.std(), cp.std(), pearson(cp, c0), pearson(cp, a0) ** 2, r2_two]
    b_h = slope_per_sd(ah, ch)
    out += [b0, gap(a0), gap(c0), pearson(a0, loglen), on_math(a0), pearson(ah, a0), pearson(ch, c0), b_h, b_h - b0,
            gap(ch), pearson(ah, c0)]
    out.append(ws["evil"] - ws["slacker"])
    return np.array(out)


def call(est, lo, hi, p):
    """The pre-registered call for one model and persona, first that applies."""
    g = lambda s: K[f"{p}:{s}"]
    if abs(est[g("r_ap_a0")]) > SEPARABLE:
        return "not separable"
    if hi[g("b_per")] < 0:
        return "against the persona"
    def_up, per_up = lo[g("b_def")] > 0, lo[g("b_per")] > 0
    if not (def_up or per_up):
        return "neither"
    if def_up and not per_up:
        return "default Assistant"
    if per_up and not def_up:
        return "persona"
    return "mix"


def collect_runs(group):
    """{family: [runs]}: for each config (run name), its latest run with all its result files. A condition's file
    is written only once it's fully scored, so a run counts once every file is there, whatever its state."""
    import wandb
    latest = {}
    for run in sorted(wandb.Api().runs(PROJECT, filters={"group": group, "jobType": "tournament"}),
                      key=lambda r: r.created_at):
        (m,) = run.config["models"]
        have = {f.name for f in run.files()}
        if all(f"results/{m['stage']}__{c}.csv.gz" in have for c in m["conditions"]):
            latest[run.name] = run
    runs = collections.defaultdict(list)
    for run in latest.values():
        (family,) = run.config["tags"]
        runs[family].append(run)
    return runs


def merged_fit(runs, out):
    """(utilities.csv rows, summary, {condition: raw results path}) of one model's persona runs, fit together;
    cached in out."""
    rows, paths = [], {}
    for run in runs:
        print(f"  {run.path} {run.name}")
        rows += fetch_raw(run, Path("outputs") / run.id)
        (m,) = run.config["models"]
        for c in m["conditions"]:
            paths[c] = Path("outputs") / run.id / "results" / f"instruct__{c}.csv.gz"
    if not (out / "utilities.csv").exists():
        analyze(rows, read_jsonl(TASKS_FILE), out, seed=0, stage_order=["instruct"])
    with open(out / "utilities.csv", newline="") as fh:
        return list(csv.DictReader(fh)), json.loads((out / "summary.json").read_text()), paths


def median_mass(path):
    with gzip.open(path, "rt", newline="") as fh:
        return float(np.median([float(r["outcome_mass"]) for r in csv.DictReader(fh)]))


def plot_scatter(data, origins, labels, path):
    """Rows: models. Columns: system prompt. Each panel: that condition's coinflip against its own Assistant."""
    fig, axes = plt.subplots(len(data), len(CONDITIONS), figsize=(3.3 * len(CONDITIONS), 3.2 * len(data)),
                             squeeze=False)
    for i, (family, U) in enumerate(data.items()):
        for j, cond in enumerate(CONDITIONS):
            ax = axes[i, j]
            a, c = (U["A0"], U["C0"]) if cond == "none" else (U[f"A{cond}"], U[f"C{cond}"])
            origin_scatter(ax, a, c, origins)
            ax.set_title(f"{COND_LABEL[cond]}\nr = {pearson(a, c):.2f}", fontsize=10, color=INK)
            if j == 0:
                ax.set_ylabel(f"{labels[family]}\ncoinflip utility (nats)", color=INK)
            if i == len(data) - 1:
                ax.set_xlabel("Assistant utility, same system prompt (nats)", color=INK_2, fontsize=9)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, frameon=False, fontsize=9, loc="lower center", ncol=len(names))
    fig.suptitle("The coinflip against the Assistant's choice under the same system prompt", color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    plotstyle.save(fig, path.stem, path.parent)


def plot_pulls(cells, labels, path):
    """Per model: the coin's pull toward the persona (x) and toward the default Assistant (y), with 95% CIs. On the
    y axis, the no-system-prompt coinflip's slope b_0 for reference."""
    fig, axes = plt.subplots(1, len(cells), figsize=(4.6 * len(cells), 4.4), squeeze=False)
    for ax, (family, cell) in zip(axes[0], cells.items()):
        est, (lo, hi) = cell["est"], ci(cell["boot"])
        top = 0.0
        for k, p in enumerate(PERSONAS):
            x, y = K[f"{p}:b_per"], K[f"{p}:b_def"]
            color = plotstyle.SERIES[k]
            ax.errorbar(est[x], est[y], xerr=[[est[x] - lo[x]], [hi[x] - est[x]]],
                        yerr=[[est[y] - lo[y]], [hi[y] - est[y]]], color=color, marker="o", ms=8, lw=0,
                        elinewidth=1.5, mec="white", mew=0.8, zorder=3)
            ax.annotate(f"{COND_LABEL[p]}  w = {est[K[f'{p}:w']]:.2f}", (est[x], est[y]), xytext=(8, 6),
                        textcoords="offset points", color=INK, fontsize=9)
            top = max(top, hi[x], hi[y])
        b0 = K["b0"]
        ax.errorbar(0, est[b0], yerr=[[est[b0] - lo[b0]], [hi[b0] - est[b0]]], color=INK_2, marker="D", ms=7,
                    lw=0, elinewidth=1.5, mec="white", mew=0.8, zorder=3, label="coin slope on A_0, no system prompt (b_0)")
        bh = K["b_h"]
        ax.errorbar(0, est[bh], yerr=[[est[bh] - lo[bh]], [hi[bh] - est[bh]]], color=INK_2, marker="s", ms=7,
                    lw=0, elinewidth=1.5, mec=INK_2, mew=1.2, zorder=3, mfc="white",
                    label="coin slope on A_helpful, helpful control (b_h)")
        top = max(top, hi[b0], hi[bh]) * 1.15
        bottom = min(0.0, *(lo[K[f"{p}:{s}"]] for p in PERSONAS for s in ("b_per", "b_def"))) * 1.15
        ax.plot([0, top], [0, top], color=RULE, ls="--", lw=1, zorder=1)
        ax.text(top * 0.98, top * 0.9, "equal pulls (w = 0.5)", color=INK_2, fontsize=8, ha="right", rotation=0)
        ax.axhline(0, color=RULE, lw=0.8, zorder=1)
        ax.axvline(0, color=RULE, lw=0.8, zorder=1)
        ax.set_xlim(bottom - 0.02 * top, top)
        ax.set_ylim(bottom - 0.02 * top, top)
        ax.set_aspect("equal")
        ax.set_xlabel("pull toward the persona, b_per\n(coin nats per SD of the persona's Assistant utilities)",
                      color=INK_2, fontsize=9)
        ax.set_ylabel("pull toward the default Assistant, b_def\n(coin nats per SD of A_0)", color=INK_2, fontsize=9)
        ax.set_title(labels[family], color=INK)
        style(ax)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, frameon=False, fontsize=9, loc="lower center", ncol=len(names))
    fig.suptitle("Whose preferences does the coin follow under a persona?", color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    plotstyle.save(fig, path.stem, path.parent)


def by_origin(data, origins, labels):
    rows = []
    for family, U in data.items():
        for measure in ("A", "C"):
            for cond in CONDITIONS:
                x = U[f"{measure}0"] if cond == "none" else U[f"{measure}{cond}"]
                for o in ORIGIN_STYLE:
                    rows.append(dict(model=labels[family], measure=measure, condition=cond, origin=o,
                                     mean=x[origins == o].mean() - x.mean(), sd_units=(x[origins == o].mean() - x.mean()) / x.std()))
    return rows


def plot_by_origin(rows, labels, path):
    """Mean utility by origin (relative to the task mean, in SDs of that condition), Assistant and coinflip."""
    fams = list(labels.values())
    fig, axes = plt.subplots(2, len(fams), figsize=(5.2 * len(fams), 6.4), squeeze=False, sharey="row")
    origins = list(ORIGIN_STYLE)
    offsets = np.linspace(-0.27, 0.27, len(CONDITIONS))
    for j, fam in enumerate(fams):
        for i, (measure, name) in enumerate((("A", "Assistant's choice"), ("C", "coinflip"))):
            ax = axes[i, j]
            for k, cond in enumerate(CONDITIONS):
                ys = [next(r["sd_units"] for r in rows if (r["model"], r["measure"], r["condition"], r["origin"])
                           == (fam, measure, cond, o)) for o in origins]
                color = INK_2 if cond == "none" else plotstyle.SERIES[{"helpful": 2, "slacker": 1, "evil": 0}[cond]]
                ax.scatter(np.arange(len(origins)) + offsets[k], ys, color=color, s=40, edgecolors="white",
                           linewidths=0.8, zorder=3, label=COND_LABEL[cond])
            ax.axhline(0, color=RULE, lw=0.8, zorder=1)
            ax.set_xticks(range(len(origins)), [o.lower() for o in origins], fontsize=9)
            ax.set_title(f"{fam}: {name}", color=INK, fontsize=10)
            if j == 0:
                ax.set_ylabel("mean utility by origin\n(SDs from the task mean)", color=INK_2, fontsize=9)
            style(ax)
    axes[0, -1].legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    plotstyle.save(fig, path.stem, path.parent)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--group", default="personas")
    p.add_argument("--models", nargs="+", default=list(MODELS), help="a subset, e.g. while a model is still running")
    args = p.parse_args()
    plotstyle.setup()
    out = Path("outputs/personas")
    out.mkdir(parents=True, exist_ok=True)
    runs = collect_runs(args.group)
    data, cells, labels, gates, tops, masses = {}, {}, {}, [], {}, {}
    ids0 = resamples = None
    passed = {}
    for family in args.models:
        final = FAMILIES[family][0]
        labels[family] = label = MODEL_INFO[final][0]
        print(f"{label}:")
        rows, summary, paths = merged_fit(runs[family], out / "analysis" / family)
        sweep_rows, _ = load_sweep(family)
        ids = [r["id"] for r in rows]
        assert ids == [r["id"] for r in sweep_rows], "task sets differ"
        P, origins, benign, loglen = task_arrays(rows)
        B = task_arrays(sweep_rows)[0]
        if ids0 is None:
            ids0, resamples = ids, stratified_resamples(origins, np.random.default_rng(0), N_BOOTSTRAP)
        assert ids == ids0, "task sets differ between models"

        # Gate: the re-scored plain conditions match the sweep; every new cell is reliable.
        rescored = [c for c in ("coinflip", "assistant") if f"instruct/{c}" in P]
        match = True
        for c in rescored:
            r, sd = pearson(P[f"instruct/{c}"], B[f"instruct/{c}"]), P[f"instruct/{c}"].std() / B[f"instruct/{c}"].std() - 1
            ok = r >= GATE_R and abs(sd) <= GATE_SD
            match &= ok
            gates.append(f"{label:14} re-scored {c} vs sweep: r {r:.4f}, SD {sd:+.1%}: {'pass' if ok else 'FAIL'}")
        reliable = True
        for c in paths:
            rel = summary[f"instruct/{c}/reliability"]
            masses[family, c] = median_mass(paths[c])
            ok = rel >= MIN_RELIABILITY
            reliable &= ok
            gates.append(f"{label:14} {c:18} reliability {rel:.3f}, median mass {masses[family, c]:.2f}"
                         f"{'' if masses[family, c] >= MIN_MASS else ' (below 0.2: not interpreted)'}: "
                         f"{'pass' if ok else 'FAIL'}")
        if match:
            A0, C0, source = B["instruct/assistant"], B["instruct/coinflip"], "sweep"
        elif "instruct/assistant" in P:
            A0, C0, source = P["instruct/assistant"], P["instruct/coinflip"], "re-scored on the persona pods"
        else:
            A0, C0, source = B["instruct/assistant"], B["instruct/coinflip"], "sweep (gate failed: no calls)"
        passed[family] = reliable and (match or "instruct/assistant" in P)
        gates.append(f"{label:14} A_0 and C_0 from the {source}")
        U = {"A0": A0, "C0": C0}
        for p in ("helpful",) + PERSONAS:
            U[f"A{p}"], U[f"C{p}"] = P[f"instruct/assistant@{p}"], P[f"instruct/coinflip@{p}"]
        data[family] = U
        math = origins == "MATH"
        keys = list(U)
        cells[family] = dict(
            est=stats(U, benign, loglen, math),
            boot=np.array([stats({k: U[k][i] for k in keys}, benign[i], loglen[i], math[i]) for i in resamples]))
        tops[family] = {c: top_token_shares(paths[c]) for c in paths}

    print("\nValidation gate:\n  " + "\n  ".join(gates))
    table, control = [], []
    fm = lambda est, lo, hi, s: f"{est[K[s]]:+.3f} [{lo[K[s]]:+.3f}, {hi[K[s]]:+.3f}]"
    for family, cell in cells.items():
        est, (lo, hi) = cell["est"], ci(cell["boot"])
        label = labels[family]
        print(f"\n{label}")
        for name, text in (("b0", "b_0, no system prompt"), ("b_h", "b, helpful (C_h on A_h)"),
                           ("b_h_minus_b0", "b_h - b_0"), ("r_ah_a0", "r(A_helpful, A_0)"),
                           ("r_ch_c0", "r(C_helpful, C_0)"), ("r_ah_c0", "r(A_helpful, C_0)"), ("gap_a0", "gap A_0"),
                           ("gap_c0", "gap C_0"), ("gap_ch", "gap C_helpful"), ("r_a0_len", "r(A_0, log length)"),
                           ("math_a0", "A_0 on MATH")):
            print(f"  {text:34} {fm(est, lo, hi, name)}")
        control.append(dict(model=label, **{k: v for name in CONTROL for k, v in
                                             ((name, est[K[name]]), (f"{name}_lo", lo[K[name]]), (f"{name}_hi", hi[K[name]]))}))
        for p in PERSONAS:
            verdict = call(est, lo, hi, p) if passed[family] else "no call (gate)"
            print(f"  -- {p}: {verdict}")
            row = dict(model=label, persona=p, call=verdict)
            for s in PER:
                name = f"{p}:{s}"
                row.update({s: est[K[name]], f"{s}_lo": lo[K[name]], f"{s}_hi": hi[K[name]]})
                print(f"     {s:18} {fm(est, lo, hi, name)}")
            table.append(row)
        print(f"  w_evil - w_slacker {fm(est, lo, hi, 'w_evil_minus_slacker')}")
        table[-1]["w_evil_minus_slacker"] = est[K["w_evil_minus_slacker"]]
        table[-1]["w_evil_minus_slacker_lo"], table[-1]["w_evil_minus_slacker_hi"] = (
            lo[K["w_evil_minus_slacker"]], hi[K["w_evil_minus_slacker"]])

    print("\nExploratory, where the probability goes (mean over prompts, tokens logged in each top 5):")
    for family, by_cond in tops.items():
        for c, top in by_cond.items():
            print(f"  {labels[family]:14} {c:18} " + "  ".join(f"{t!r} {p:.3f}" for t, p in top))

    origins = task_arrays(rows)[1]
    origin_rows = by_origin(data, origins, labels)
    print("\nExploratory, mean utility by origin, in SDs from the task mean (A, then C):")
    for fam in labels.values():
        for measure in ("A", "C"):
            for cond in CONDITIONS:
                vals = [r for r in origin_rows if (r["model"], r["measure"], r["condition"]) == (fam, measure, cond)]
                print(f"  {fam:14} {measure} {cond:8} " + "  ".join(f"{r['origin'][:8]:8} {r['sd_units']:+.2f}" for r in vals))

    write_csv(out / "personas.csv", table)
    write_csv(out / "control.csv", control)
    write_csv(out / "by_origin.csv", origin_rows)
    plot_scatter(data, origins, labels, Path("figures/personas_scatter.png"))
    plot_pulls(cells, labels, Path("figures/personas_pulls.png"))
    plot_by_origin(origin_rows, labels, Path("figures/personas_by_origin.png"))
    print("\nWrote outputs/personas/ and figures/personas_*.png")


if __name__ == "__main__":
    main()
