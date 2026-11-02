"""Does post-training make the user's choice of task more like the Assistant's?

The measures and calls were pre-registered in the README ("Does post-training make the user's choice
more like the Assistant's?") before any data. Collects the W&B runs of group "user_choice"
(configs/user_choice/, one per model and stage), where the model continues a Human turn ending "Okay, I
think I'll choose Task" and we read P(" A") vs P(" B"). A and C, the Assistant and coinflip utilities,
come from the model sweep's runs on the same pairs. U is the user-choice utilities; _b and _f are the
base and final stages. Per model:

- r_base, r_final: r(A_b, U_b) and r(A_f, U_f), also over non-harmful tasks and within origin;
- r_fixed: r(A_f, U_b), the base model's user against the final Assistant. r_final - r_fixed holds the
  target fixed, so it doesn't depend on how far A itself moved: is the final user closer to the final
  Assistant than the base user is?
- r_shift: r(A_f - A_b, U_f - U_b), r_change: r(A_f, U_f - U_b);
- b_base, b_final: slope of U on A in nats per Assistant SD, at each stage; b_fixed: U_b on A_f;
- gap_base, gap_final: mean U, non-harmful minus harmful;
- sd_base, sd_final; resid_base, resid_final: SD of U around its line on A;
- the same for the coinflip (rc_*, bc_*, gapc_*, sdc_*), and r_uc_base, r_uc_final: r(U, C);
- partial_au_c, partial_ac_u: at the final stage, r(A, U) holding C fixed and r(A, C) holding U fixed.

CIs are 95% task bootstraps within origin, with the same resamples for every statistic, so differences
are paired. `classify` gives the pre-registered call per model. Not pre-registered, printed last: mean U
within each origin (`by_origin`), r(U_f, the other models' mean Assistant) and r(A, U_f) holding it fixed
(`consensus`), and where the probability goes (`top_token_shares`).

  outputs/user_choice/analysis/<family>/   the merged fit (analyze.py) of one model's stages
  outputs/user_choice/user_choice.csv      one row per model
  outputs/user_choice/contrasts.csv        the paired differences
  outputs/user_choice/by_origin.csv
  figures/user_choice_*.png

Usage: uv run python user_choice.py [--group user_choice]
       uv run python user_choice.py --sweep-only   (the coinflip's numbers on these models, no new data)
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
from models import MODEL_INFO, PROJECT, collect_runs as collect_sweep, consensus_stats, olmo_on_shared_pairs
from openers import slope_per_sd, within_groups, write_csv
from report import INK, INK_2, RULE, errorbar, fetch_wandb, origin_scatter, stratified_resamples, style, task_arrays

OLMO, GEMMA, LLAMA = "allenai/Olmo-3.1-32B-Instruct", "google/gemma-4-31B-it", "unsloth/Llama-3.3-70B-Instruct"
# family (the runs' tag): (final model, sweep run; OLMo is d8ysbybo refit on the shared pairs).
FAMILIES = {"olmo3.1-32b": (OLMO, None), "gemma4-31b": (GEMMA, "jrl62npf"), "llama-70b": (LLAMA, "tvlqj8a0")}
STAGES = ("base", "instruct")
TASKS_FILE = "data/tasks_n300_max500_seed0.jsonl"
SHARED_PAIRS = {"max_pairs": 15000, "seed": 0, "tasks_file": TASKS_FILE}
STATS = ("r_base", "r_final", "r_fixed", "r_base_benign", "r_final_benign", "r_base_within", "r_final_within",
         "r_shift", "r_change", "b_base", "b_final", "b_fixed", "gap_base", "gap_final", "sd_base", "sd_final",
         "resid_base", "resid_final", "rc_base", "rc_final", "rc_final_benign", "rc_final_within", "rc_shift",
         "bc_base", "bc_final", "gapc_base", "gapc_final", "sdc_base", "sdc_final", "r_uc_base", "r_uc_final",
         "partial_au_c", "partial_ac_u")
K = {name: j for j, name in enumerate(STATS)}
# name: (first stat, second stat, label), each the first minus the second.
CONTRASTS = {
    "target": ("r_final", "r_fixed", "r(A_f, U_f) - r(A_f, U_b)"),
    "r": ("r_final", "r_base", "r final - base"),
    "r_benign": ("r_final_benign", "r_base_benign", "r non-harmful, final - base"),
    "r_within": ("r_final_within", "r_base_within", "r within origin, final - base"),
    "b": ("b_final", "b_base", "b final - base"),
    "gap": ("gap_final", "gap_base", "gap final - base"),
    "user_vs_coin": ("r_final", "rc_final", "r(A_f, U_f) - r(A_f, C_f)"),
    "user_vs_coin_benign": ("r_final_benign", "rc_final_benign", "same, non-harmful"),
    "user_vs_coin_within": ("r_final_within", "rc_final_within", "same, within origin"),
    "shift_vs_coin": ("r_shift", "rc_shift", "r shift, user - coinflip"),
    "b_vs_coin": ("b_final", "bc_final", "b final, user - coinflip"),
    "gap_vs_coin": ("gap_final", "gapc_final", "gap final, user - coinflip"),
    "base_vs_coin": ("r_base", "rc_base", "r base, user - coinflip"),
}
R_MEANINGFUL = 0.05  # a change in r worth calling one (a tight CI alone can flag tiny ones)
R_SHIFT = 0.2  # an aligned shift has r_shift at least this, with a CI above 0
GATE_R, GATE_SD, MIN_RELIABILITY = 0.999, 0.02, 0.9
STAGE_MARKER = {"r_base": "o", "r_fixed": "D", "r_final": "s", "rc_final": "^"}


def model_stats(ab, af, ub, uf, cb, cf, benign, origins):
    """STATS of one model, over arrays of the same (possibly resampled) tasks."""
    w = lambda x: within_groups(x, origins)
    gap = lambda x: x[benign].mean() - x[~benign].mean()
    resid = lambda a, y: (y - np.polyval(np.polyfit(a, y, 1), a)).std()
    return np.array([
        pearson(ab, ub), pearson(af, uf), pearson(af, ub),
        pearson(ab[benign], ub[benign]), pearson(af[benign], uf[benign]), pearson(w(ab), w(ub)), pearson(w(af), w(uf)),
        pearson(af - ab, uf - ub), pearson(af, uf - ub),
        slope_per_sd(ab, ub), slope_per_sd(af, uf), slope_per_sd(af, ub),
        gap(ub), gap(uf), ub.std(), uf.std(), resid(ab, ub), resid(af, uf),
        pearson(ab, cb), pearson(af, cf), pearson(af[benign], cf[benign]), pearson(w(af), w(cf)),
        pearson(af - ab, cf - cb), slope_per_sd(ab, cb), slope_per_sd(af, cf), gap(cb), gap(cf), cb.std(), cf.std(),
        pearson(ub, cb), pearson(uf, cf), partial_r(af, uf, cf), partial_r(af, cf, uf),
    ])


def model_cell(arrays, benign, origins, resamples):
    """{"est": STATS, "boot": (n_resamples, STATS)}. arrays: (A_b, A_f, U_b, U_f, C_b, C_f)."""
    return dict(est=model_stats(*arrays, benign, origins),
                boot=np.array([model_stats(*(x[i] for x in arrays), benign[i], origins[i]) for i in resamples]))


def ci(boot):
    return np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)


def contrast(cell, name):
    """(estimate, ci_low, ci_high) of a CONTRASTS entry, paired."""
    x, y, _ = CONTRASTS[name]
    lo, hi = ci(cell["boot"][:, K[x]] - cell["boot"][:, K[y]])
    return cell["est"][K[x]] - cell["est"][K[y]], lo, hi


def classify(cell):
    """The pre-registered call for one model: did post-training make the user's choice more like the Assistant's?"""
    d, lo, hi = contrast(cell, "target")
    shift = cell["est"][K["r_shift"]]
    aligned = shift >= R_SHIFT and ci(cell["boot"][:, K["r_shift"]])[0] > 0
    if d >= R_MEANINGFUL and lo > 0:
        return "more Assistant-like" if aligned else "closer, shift not aligned"
    if d <= -R_MEANINGFUL and hi < 0:
        return "less Assistant-like"
    return "aligned shift only" if aligned else "no change"


def by_origin(data, origins, labels):
    """Mean U at each stage within each origin, per model."""
    rows = []
    for family, (_, _, ub, uf, _, _) in data.items():
        for g in ORIGIN_STYLE:
            k = origins == g
            rows.append(dict(model=labels[family], origin=g, n=int(k.sum()), base=float(ub[k].mean()),
                             final=float(uf[k].mean()), r_stages=pearson(ub[k], uf[k]), sd_base=float(ub[k].std()),
                             sd_final=float(uf[k].std())))
    return rows


def top_token_shares(path, n=6):
    """Where the probability goes: mean probability per token, over the tokens logged in the top 5."""
    total, count = collections.Counter(), 0
    with gzip.open(path, "rt", newline="") as fh:
        for r in csv.DictReader(fh):
            count += 1
            for token, p in json.loads(r["top_tokens"]):
                total[token] += p
    return [(token, p / count) for token, p in total.most_common(n)]


def plot_summary(cells, labels, path):
    """Small multiples by model. r panels show what the user's choice is compared with; each panel has one scale."""
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7), sharex=True)
    families = list(cells)
    x = np.arange(len(families))
    panels = (
        ("r with the Assistant, all tasks",
         (("r_base", "Base user, base Assistant"), ("r_fixed", "Base user, final Assistant"),
          ("r_final", "Final user, final Assistant"), ("rc_final", "Final coinflip, final Assistant"))),
        ("r with the Assistant, non-harmful tasks",
         (("r_base_benign", "Base user, base Assistant"), ("r_final_benign", "Final user, final Assistant"),
          ("rc_final_benign", "Final coinflip, final Assistant"))),
        ("r with the Assistant, within origin",
         (("r_base_within", "Base user, base Assistant"), ("r_final_within", "Final user, final Assistant"),
          ("rc_final_within", "Final coinflip, final Assistant"))),
        ("Slope on the Assistant (nats per SD)",
         (("b_base", "Base user, base Assistant"), ("b_final", "Final user, final Assistant"),
          ("bc_final", "Final coinflip, final Assistant"))),
        ("Benign minus harmful (nats)",
         (("gap_base", "Base user"), ("gap_final", "Final user"), ("gapc_final", "Final coinflip"))),
        ("Utility SD (nats)", (("sd_base", "Base user"), ("sd_final", "Final user"), ("sdc_final", "Final coinflip"))),
    )
    # Colors follow the series' identity across panels: 0 base user, 1 final user, 2 final coinflip, 3 base user vs final A.
    identity = {"r_base": 0, "r_base_benign": 0, "r_base_within": 0, "b_base": 0, "gap_base": 0, "sd_base": 0,
                "r_final": 1, "r_final_benign": 1, "r_final_within": 1, "b_final": 1, "gap_final": 1, "sd_final": 1,
                "rc_final": 2, "rc_final_benign": 2, "rc_final_within": 2, "bc_final": 2, "gapc_final": 2, "sdc_final": 2,
                "r_fixed": 3}
    markers = ("o", "s", "^", "D")
    for ax, (title, series) in zip(axes.flat, panels):
        for j, (stat, label) in enumerate(series):
            c = identity[stat]
            est = np.array([cells[f]["est"][K[stat]] for f in families])
            lo, hi = zip(*(ci(cells[f]["boot"][:, K[stat]]) for f in families))
            errorbar(ax, x + (j - (len(series) - 1) / 2) * 0.13, est, np.minimum(lo, est), np.maximum(hi, est),
                     plotstyle.SERIES[c], markers[c], label, ls="none")
        ax.axhline(0, color=RULE, lw=0.8, zorder=1)
        ax.set_xticks(x, [labels[f] for f in families])
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        style(ax)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower right")
    axes[1, 1].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scatter(data, origins, labels, path):
    """Each model's user choice and coinflip against the Assistant. Axes differ between panels."""
    columns = (("Base user vs base Assistant", 0, 2), ("Base user vs final Assistant", 1, 2),
               ("Final user vs final Assistant", 1, 3), ("Final coinflip vs final Assistant", 1, 5))
    fig, axes = plt.subplots(len(data), len(columns), figsize=(3.2 * len(columns), 3 * len(data)), squeeze=False)
    for row, (family, arrays) in enumerate(data.items()):
        for col, (title, ix, iy) in enumerate(columns):
            ax = axes[row, col]
            origin_scatter(ax, arrays[ix], arrays[iy], origins)
            ax.set_title(f"r = {pearson(arrays[ix], arrays[iy]):.2f}, y SD {arrays[iy].std():.2f}", fontsize=9, color=INK_2,
                         loc="left")
            if row == 0:
                ax.text(0, 1.14, title, transform=ax.transAxes, fontsize=10, color=INK)
            ax.set_xlabel("Assistant utility (nats)", fontsize=8, color=INK_2)
        axes[row, 0].set_ylabel(f"{labels[family]}\nutility (nats)")
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, frameon=False, ncol=len(ORIGIN_STYLE), loc="upper center", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_by_origin(rows, labels, path):
    """Mean user-choice utility by origin, base and final, one panel per model."""
    fig, axes = plt.subplots(1, len(labels), figsize=(4.2 * len(labels), 3.8), squeeze=False)
    for ax, (family, label) in zip(axes[0], labels.items()):
        rs = {r["origin"]: r for r in rows if r["model"] == label}
        y = np.arange(len(ORIGIN_STYLE))
        for j, (stage, name, marker) in enumerate((("base", "Base", "o"), ("final", "Final", "s"))):
            ax.plot([rs[g][stage] for g in ORIGIN_STYLE], y, ls="none", marker=marker, ms=8, color=plotstyle.SERIES[j],
                    mec="white", mew=0.8, label=name, zorder=3)
        ax.axvline(0, color=RULE, lw=0.8, zorder=1)
        ax.set_yticks(y, list(ORIGIN_STYLE))
        ax.set_title(label, loc="left", fontsize=11, color=INK)
        ax.set_xlabel("Mean user-choice utility (nats)", color=INK_2)
        style(ax)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def load_sweep(family):
    """(utilities.csv rows, summary) of the sweep's run for one model."""
    final, run_id = FAMILIES[family]
    if run_id is None:
        return olmo_on_shared_pairs(SHARED_PAIRS)
    import wandb
    analysis = Path("outputs") / run_id / "analysis"
    fetch_wandb(wandb.Api().run(f"{PROJECT}/{run_id}"), analysis)
    with open(analysis / "utilities.csv", newline="") as fh:
        return list(csv.DictReader(fh)), json.loads((analysis / "summary.json").read_text())


def collect_runs(group):
    """{(family, stage): latest finished run} of the W&B group. A run counts only if it logged utilities."""
    import wandb
    runs = {}
    for run in sorted(wandb.Api().runs(PROJECT, filters={"group": group, "state": "finished", "jobType": "tournament"}),
                      key=lambda r: r.created_at):
        if "utilities" in run.summary.keys():
            (family,) = run.config["tags"]
            (stage,) = [m["stage"] for m in run.config["models"]]
            runs[family, stage] = run
    return runs


def merged_fit(runs, family, out):
    """(utilities.csv rows, summary, raw results paths) of a model's stages, fit together; cached in out."""
    paths = {}
    rows = []
    for s in STAGES:
        run = runs[family, s]
        print(f"{family:12} {s:9} {run.path}")
        rows += fetch_raw(run, Path("outputs") / run.id)
        paths[s] = Path("outputs") / run.id / "results" / f"{s}__user_choice.csv.gz"
    if not (out / "utilities.csv").exists():
        analyze(rows, read_jsonl(TASKS_FILE), out, seed=0, stage_order=list(STAGES))
    with open(out / "utilities.csv", newline="") as fh:
        return list(csv.DictReader(fh)), json.loads((out / "summary.json").read_text()), paths


def sweep_only():
    """The sweep's own numbers on these models, for the pre-registration."""
    rows0 = None
    for family in FAMILIES:
        rows, summary = load_sweep(family)
        rows0 = rows0 or rows
        U, origins, benign, _ = task_arrays(rows)
        ab, af, cb, cf = U["base/assistant"], U["instruct/assistant"], U["base/coinflip"], U["instruct/coinflip"]
        # The coinflip stands in for U, so the r_* columns are the coinflip's.
        s = model_stats(ab, af, cb, cf, cb, cf, benign, origins)
        print(f"{family:12} r base {s[K['rc_base']]:.2f} final {s[K['rc_final']]:.2f} non-harmful {s[K['rc_final_benign']]:.2f} "
              f"within {s[K['rc_final_within']]:.2f} shift {s[K['rc_shift']]:.2f}  C_b vs A_f {pearson(af, cb):.2f}  "
              f"b {s[K['bc_base']]:.3f} -> {s[K['bc_final']]:.3f}  gap {s[K['gapc_base']]:.2f} -> {s[K['gapc_final']]:.2f}  "
              f"SD {s[K['sdc_base']]:.2f} -> {s[K['sdc_final']]:.2f}  A SD {ab.std():.2f} -> {af.std():.2f}  "
              f"r(A_b, A_f) {pearson(ab, af):.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--group", default="user_choice")
    p.add_argument("--sweep-only", action="store_true", help="print the sweep's coinflip numbers and stop")
    args = p.parse_args()
    if args.sweep_only:
        sweep_only()
        return

    out = Path("outputs/user_choice")
    out.mkdir(parents=True, exist_ok=True)
    runs = collect_runs(args.group)
    data, cells, table, gates, passed, labels, sweep, tops = {}, {}, [], [], True, {}, {}, {}
    ids0, resamples = None, None
    for family, (final, _) in FAMILIES.items():
        rows, summary, paths = merged_fit(runs, family, out / "analysis" / family)
        sweep_rows, sweep_summary = load_sweep(family)
        ids = [r["id"] for r in rows]
        assert ids == [r["id"] for r in sweep_rows], "task sets differ"
        U, origins, benign, _ = task_arrays(rows)
        B = task_arrays(sweep_rows)[0]
        if ids0 is None:
            ids0, resamples = ids, stratified_resamples(origins, np.random.default_rng(0), N_BOOTSTRAP)
        assert ids == ids0, "task sets differ between models"
        labels[family] = MODEL_INFO[final][0]
        sweep[final] = B
        arrays = (B["base/assistant"], B["instruct/assistant"], U["base/user_choice"], U["instruct/user_choice"],
                  B["base/coinflip"], B["instruct/coinflip"])
        data[family] = arrays
        # Validation gate: the re-scored Assistant condition matches the sweep, and every U cell is reliable.
        gate_r = pearson(U["instruct/assistant"], B["instruct/assistant"])
        gate_sd = U["instruct/assistant"].std() / B["instruct/assistant"].std() - 1
        ok = gate_r >= GATE_R and abs(gate_sd) <= GATE_SD
        passed &= ok
        gates.append(f"{labels[family]:14} re-scored Assistant vs sweep: r {gate_r:.4f}, SD {gate_sd:+.1%}: {'pass' if ok else 'FAIL'}")
        for s in STAGES:
            rel, mass = summary[f"{s}/user_choice/reliability"], summary[f"{s}/user_choice/outcome_mass_mean"]
            ok = rel >= MIN_RELIABILITY
            passed &= ok
            gates.append(f"{labels[family]:14} {s:9} user choice: reliability {rel:.3f}, mass {mass:.2f}: {'pass' if ok else 'FAIL'}")
        cell = cells[family] = model_cell(arrays, benign, origins, resamples)
        lo, hi = ci(cell["boot"])
        row = dict(model=labels[family], call=classify(cell))
        for s in STAGES:
            for key in ("reliability", "outcome_mass_mean", "outcome_mass_min", "slot_bias_mean", "additive_r2"):
                row[f"{s}_{key}"] = summary[f"{s}/user_choice/{key}"]
        row.update(r_final_disattenuated=cell["est"][K["r_final"]] / np.sqrt(
            summary["instruct/user_choice/reliability"] * sweep_summary["instruct/assistant/reliability"]),
            r_base_disattenuated=cell["est"][K["r_base"]] / np.sqrt(
                summary["base/user_choice/reliability"] * sweep_summary["base/assistant/reliability"]))
        for name in STATS:
            row.update({name: cell["est"][K[name]], f"{name}_lo": lo[K[name]], f"{name}_hi": hi[K[name]]})
        table.append(row)
        tops[family] = {s: top_token_shares(paths[s]) for s in STAGES}

    print("Validation gate:\n  " + "\n  ".join(gates))
    fm = lambda r, s: f"{r[s]:+.2f} [{r[f'{s}_lo']:+.2f}, {r[f'{s}_hi']:+.2f}]"
    for r in table:
        print(f"\n{r['model']}: {r['call']}")
        for name, label in (("r_base", "r(A_b, U_b)"), ("r_fixed", "r(A_f, U_b)"), ("r_final", "r(A_f, U_f)"),
                            ("rc_base", "r(A_b, C_b)"), ("rc_final", "r(A_f, C_f)"),
                            ("r_base_benign", "non-harmful, base"), ("r_final_benign", "non-harmful, final"),
                            ("rc_final_benign", "non-harmful, coinflip"), ("r_base_within", "within origin, base"),
                            ("r_final_within", "within origin, final"), ("rc_final_within", "within origin, coinflip"),
                            ("r_shift", "r shift, user"), ("rc_shift", "r shift, coinflip"), ("r_change", "r(A_f, U_f - U_b)"),
                            ("b_base", "b, base"), ("b_fixed", "b, U_b on A_f"), ("b_final", "b, final"), ("bc_final", "b, coinflip"),
                            ("gap_base", "gap, base"), ("gap_final", "gap, final"), ("gapc_final", "gap, coinflip"),
                            ("sd_base", "SD, base"), ("sd_final", "SD, final"), ("sdc_final", "SD, coinflip"),
                            ("resid_base", "SD around line, base"), ("resid_final", "SD around line, final"),
                            ("r_uc_base", "r(U_b, C_b)"), ("r_uc_final", "r(U_f, C_f)"),
                            ("partial_au_c", "r(A, U | C), final"), ("partial_ac_u", "r(A, C | U), final")):
            print(f"  {label:26} {fm(r, name)}")
        print(f"  disattenuated r base {r['r_base_disattenuated']:+.2f}, final {r['r_final_disattenuated']:+.2f}; "
              f"slot bias base {r['base_slot_bias_mean']:+.2f}, final {r['instruct_slot_bias_mean']:+.2f}; "
              f"additive R2 base {r['base_additive_r2']:.3f}, final {r['instruct_additive_r2']:.3f}")
    contrasts = []
    for family, cell in cells.items():
        for name, (_, _, label) in CONTRASTS.items():
            est, lo, hi = contrast(cell, name)
            contrasts.append(dict(model=labels[family], contrast=name, label=label, est=est, lo=lo, hi=hi))
    print("\nPaired differences:")
    for c in contrasts:
        print(f"  {c['model']:14} {c['label']:32} {c['est']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]")

    origins = task_arrays(rows)[1]
    origin_rows = by_origin(data, origins, labels)
    print("\nExploratory, mean user-choice utility by origin (base, final) and r(U_b, U_f) within it:")
    for r in origin_rows:
        print(f"  {r['model']:14} {r['origin']:12} {r['base']:+.2f} {r['final']:+.2f}  r {r['r_stages']:+.2f}")
    print("\nExploratory, final user choice vs the other sweep models' mean Assistant utilities:")
    sweep_runs = collect_sweep("models")
    utilities = {m: dict(sweep[m]) if m in sweep else None for m in MODEL_INFO}
    for m, run in sweep_runs.items():
        if utilities[m] is None:
            analysis = Path("outputs") / run.id / "analysis"
            fetch_wandb(run, analysis)
            with open(analysis / "utilities.csv", newline="") as fh:
                utilities[m] = task_arrays(list(csv.DictReader(fh)))[0]
    # consensus_stats reads "instruct/coinflip" as the measure compared, so hand it the user's choice.
    standin = {m: {"instruct/assistant": u["instruct/assistant"], "instruct/coinflip": u["instruct/coinflip"]}
               for m, u in utilities.items() if u is not None}
    for family, (final, _) in FAMILIES.items():
        standin[final]["instruct/coinflip"] = data[family][3]
    cons = consensus_stats(standin, resamples)
    for family, (final, _) in FAMILIES.items():
        s = cons[final]
        print("  {:14} r(U_f, others' A) {:+.2f} [{:+.2f}, {:+.2f}]  r(A, U_f | others' A) {:+.2f} [{:+.2f}, {:+.2f}]".format(
            labels[family], *s["r_consensus"], *s["r_own_given_consensus"]))
    print("\nExploratory, where the probability goes (mean over prompts, tokens logged in each top 5):")
    for family, by_stage in tops.items():
        for s, top in by_stage.items():
            print(f"  {labels[family]:14} {s:9} " + "  ".join(f"{t!r} {p:.3f}" for t, p in top))

    write_csv(out / "user_choice.csv", table)
    write_csv(out / "contrasts.csv", contrasts)
    write_csv(out / "by_origin.csv", origin_rows)
    plotstyle.setup()
    plot_summary(cells, labels, "figures/user_choice_summary.png")
    plot_scatter(data, origins, labels, "figures/user_choice_scatter.png")
    plot_by_origin(origin_rows, labels, "figures/user_choice_by_origin.png")
    print(f"\nGate {'passed' if passed else 'FAILED'}. Wrote {out}/ and figures/user_choice_*.png")


if __name__ == "__main__":
    main()
