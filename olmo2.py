"""OLMo 2 32B by post-training stage, next to OLMo 3.1 32B: does OLMo 2's SFT effect hold up over the 300 tasks?

The measures and calls were pre-registered in the README ("OLMo 2 32B by post-training stage") before
any data. Collects the W&B runs of group "olmo2" (configs/olmo2/, one per stage), merges them and fits
utilities together. OLMo 3.1 32B is d8ysbybo refit on the same 15,000 pairs. With A_s and C_s a stage's
Assistant and coinflip utilities, per ladder and stage:

- r, r_benign, r_harmful: r(A_s, C_s) over all, non-harmful and harmful tasks;
- r_within: r(A_s, C_s) after removing each origin's mean from both; r_between: over the five origin
  means. These split the correlation into within-type and cross-type parts;
- r_length: r with log task length partialled out;
- r_shift: r(A_s - A_base, C_s - C_base), which removes what the base already has in both conditions;
- b, b_benign: slope of C_s on A_s, in nats of coinflip per Assistant SD (of the same tasks and stage);
- harm_gap, assistant_gap: mean C_s or A_s, non-harmful minus harmful, the contrast ../coinflip's task
  pairs measure;
- coinflip_sd, assistant_sd; resid_sd: SD of C_s around its line on A_s.

CIs are 95% task bootstraps within origin, with the same resamples for every ladder and stage, so
differences between stages and between ladders are paired. `sft_share` is a stat's rise from base to
SFT as a share of its rise from base to final. Not pre-registered, printed last: r between the two
ladders at each stage (`cross_ladder`), each ladder's stage x stage coinflip correlations and r(A, C)
within each origin (`by_origin`).

  outputs/olmo2/analysis/       the merged fit (analyze.py) with every stage
  outputs/olmo2/report/         report.py on it
  outputs/olmo2/olmo2.csv       one row per ladder and stage
  outputs/olmo2/steps.csv       each stat's stage-to-stage changes and SFT share, per ladder
  outputs/olmo2/ladders.csv     OLMo 2 minus OLMo 3 at SFT, for the pre-registered call
  figures/olmo2_*.png

Usage: uv run python olmo2.py [--group olmo2]
"""

import argparse
import csv
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import N_BOOTSTRAP, ORIGIN_STYLE, analyze, fetch_wandb, partial_r, pearson, read_jsonl
from matskit import plotstyle
from models import PROJECT, olmo_on_shared_pairs
from openers import slope_per_sd, within_groups, write_csv
from report import INK, INK_2, RULE, errorbar, origin_scatter, report, stratified_resamples, style, task_arrays

STAGES = ("base", "sft", "dpo", "instruct")
STAGE_LABELS = {"base": "base", "sft": "SFT", "dpo": "DPO", "instruct": "final"}
LADDERS = {"olmo2": "OLMo 2 32B", "olmo3": "OLMo 3.1 32B"}
LADDER_MARKER = {"olmo2": "o", "olmo3": "s"}
TASKS_FILE = "data/tasks_n300_max500_seed0.jsonl"
SHARED_PAIRS = {"max_pairs": 15000, "seed": 0, "tasks_file": TASKS_FILE}
STATS = ("r", "r_benign", "r_harmful", "r_within", "r_between", "r_length", "r_shift", "b", "b_benign", "harm_gap",
         "assistant_gap", "coinflip_sd", "assistant_sd", "resid_sd")
K = {name: j for j, name in enumerate(STATS)}
R_MEANINGFUL = 0.05  # a difference in r worth calling one (a tight CI alone can flag tiny ones)
# ../coinflip RESULTS.md, "OLMo 2 32B and 13B across post-training stages": the Assistant-framing
# effect on the Sonnet 4.5 system-card pairs, log-odds. Same GPU type, same prompt bodies as here.
COINFLIP_EFFECT = {"olmo2": {"base": 0.15, "sft": 0.97, "dpo": 2.25, "instruct": 2.42},
                   "olmo3": {"base": 0.12, "sft": 0.11, "dpo": 2.24, "instruct": 2.27}}
MIN_RELIABILITY = 0.9  # the gate on every OLMo 2 cell
GATE_R = 0.99  # the OLMo 3 refit against its all-pairs fit


def stage_stats(a, c, a0, c0, benign, origins, loglen):
    """STATS of one stage, over arrays of the same (possibly resampled) tasks. a0, c0: the base stage's."""
    aw, cw = within_groups(a, origins), within_groups(c, origins)
    groups = np.unique(origins)
    am = np.array([a[origins == g].mean() for g in groups])
    cm = np.array([c[origins == g].mean() for g in groups])
    moved = bool(np.ptp(c - c0) > 0)  # the base stage has no shift from itself
    return np.array([
        pearson(a, c), pearson(a[benign], c[benign]), pearson(a[~benign], c[~benign]), pearson(aw, cw),
        pearson(am, cm), partial_r(a, c, loglen), pearson(a - a0, c - c0) if moved else np.nan, slope_per_sd(a, c),
        slope_per_sd(a[benign], c[benign]), c[benign].mean() - c[~benign].mean(), a[benign].mean() - a[~benign].mean(),
        c.std(), a.std(), (c - np.polyval(np.polyfit(a, c, 1), a)).std(),
    ])


def ladder_cells(U, benign, origins, loglen, resamples):
    """{stage: {"est": STATS, "boot": (n_resamples, STATS)}}. U: {stage/condition: utilities}."""
    a0, c0 = U["base/assistant"], U["base/coinflip"]
    out = {}
    for s in STAGES:
        a, c = U[f"{s}/assistant"], U[f"{s}/coinflip"]
        out[s] = dict(est=stage_stats(a, c, a0, c0, benign, origins, loglen),
                      boot=np.array([stage_stats(a[i], c[i], a0[i], c0[i], benign[i], origins[i], loglen[i])
                                     for i in resamples]))
    return out


def step(cells, s, t):
    """(estimate, ci_low, ci_high) over STATS of stage t minus stage s, paired."""
    lo, hi = _nan_ci(cells[t]["boot"] - cells[s]["boot"])
    return cells[t]["est"] - cells[s]["est"], lo, hi


def sft_share(cells):
    """(estimate, ci_low, ci_high) over STATS: the rise from base to SFT as a share of base to final."""
    def share(x):
        return (x["sft"] - x["base"]) / (x["instruct"] - x["base"])
    lo, hi = _nan_ci(share({s: cells[s]["boot"] for s in STAGES}))
    return share({s: cells[s]["est"] for s in STAGES}), lo, hi


def _nan_ci(boot):
    """Percentile CI per column. r_shift is NaN at the base stage, so its columns can be all NaN."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "All-NaN", RuntimeWarning)
        return np.nanpercentile(boot, 2.5, axis=0), np.nanpercentile(boot, 97.5, axis=0)


def ladder_diff(cells2, cells3, stage="sft"):
    """(estimate, ci_low, ci_high) over STATS of OLMo 2 minus OLMo 3 at one stage, paired."""
    lo, hi = _nan_ci(cells2[stage]["boot"] - cells3[stage]["boot"])
    return cells2[stage]["est"] - cells3[stage]["est"], lo, hi


def share_diff(cells2, cells3):
    """As ladder_diff, for sft_share."""
    def share(c, key):
        return (c["sft"][key] - c["base"][key]) / (c["instruct"][key] - c["base"][key])
    lo, hi = _nan_ci(share(cells2, "boot") - share(cells3, "boot"))
    return share(cells2, "est") - share(cells3, "est"), lo, hi


def exceeds(est, lo, stat, margin=R_MEANINGFUL):
    return bool(lo[K[stat]] > 0 and est[K[stat]] >= margin)


def classify(cells2, cells3):
    """The pre-registered outcome: does OLMo 2's SFT effect reach across the task range?"""
    d, lo, _ = ladder_diff(cells2, cells3)
    sd, slo, _ = share_diff(cells2, cells3)
    within = exceeds(d, lo, "r_benign") and exceeds(d, lo, "r_within")
    harm = bool(slo[K["harm_gap"]] > 0)
    if within and harm:
        return "holds across the range"
    if harm:
        return "harm contrast only"
    if within:
        return "within-type only"
    return "does not hold"


def by_origin(U, origins):
    return [dict(origin=g, stage=s, n=int((origins == g).sum()),
                 r=pearson(U[f"{s}/assistant"][origins == g], U[f"{s}/coinflip"][origins == g]),
                 mean_c=float(U[f"{s}/coinflip"][origins == g].mean()),
                 mean_a=float(U[f"{s}/assistant"][origins == g].mean()))
            for g in ORIGIN_STYLE for s in STAGES]


def plot_ladders(cells, path):
    """OLMo 2 and OLMo 3 by stage, one panel per stat. Each panel has one scale."""
    panels = (("r", "r(Assistant, coinflip)"), ("r_benign", "r, non-harmful tasks"), ("r_within", "r, within origin"),
              ("r_shift", "r of shifts from base"), ("b", "Slope on Assistant (nats per SD)"),
              ("harm_gap", "Coinflip, benign minus harmful (nats)"))
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7), sharex=True)
    x = np.arange(len(STAGES))
    for ax, (stat, title) in zip(axes.flat, panels):
        for k, ladder in enumerate(cells):
            est = np.array([cells[ladder][s]["est"][K[stat]] for s in STAGES])
            lo, hi = zip(*(_nan_ci(cells[ladder][s]["boot"][:, K[stat]]) for s in STAGES))
            # A percentile CI can sit just off a biased statistic's estimate; errorbar needs it to include it.
            errorbar(ax, x + (k - 0.5) * 0.1, est, np.minimum(lo, est), np.maximum(hi, est), plotstyle.SERIES[k],
                     LADDER_MARKER[ladder], LADDERS[ladder])
        ax.axhline(0, color=RULE, lw=0.8, zorder=1)
        ax.set_xticks(x, [STAGE_LABELS[s] for s in STAGES])
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        style(ax)
    axes[0, 0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scatter(Us, origins, path):
    """Assistant vs coinflip per stage, one row per ladder, axes shared within a row's coinflip scale."""
    fig, axes = plt.subplots(len(Us), len(STAGES), figsize=(3.3 * len(STAGES), 3.1 * len(Us)), sharey="row",
                             squeeze=False)
    for row, (ladder, U) in enumerate(Us.items()):
        for col, s in enumerate(STAGES):
            ax = axes[row, col]
            origin_scatter(ax, U[f"{s}/assistant"], U[f"{s}/coinflip"], origins)
            ax.set_title(f"{STAGE_LABELS[s]}: r = {pearson(U[f'{s}/assistant'], U[f'{s}/coinflip']):.2f}", fontsize=10,
                         color=INK_2, loc="left")
        axes[row, 0].set_ylabel(f"{LADDERS[ladder]}\ncoinflip utility (nats)")
    for ax in axes[-1]:
        ax.set_xlabel("Assistant utility (nats)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=len(ORIGIN_STYLE), loc="upper center", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_vs_coinflip(cells, path):
    """Tournament benign-harmful gap against ../coinflip's effect, per stage, one line per ladder."""
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for k, ladder in enumerate(cells):
        xs = np.array([COINFLIP_EFFECT[ladder][s] for s in STAGES])
        est = np.array([cells[ladder][s]["est"][K["harm_gap"]] for s in STAGES])
        lo, hi = zip(*(_nan_ci(cells[ladder][s]["boot"][:, K["harm_gap"]]) for s in STAGES))
        errorbar(ax, xs, est, np.minimum(lo, est), np.maximum(hi, est), plotstyle.SERIES[k], LADDER_MARKER[ladder],
                 LADDERS[ladder])
        for xi, yi, s in zip(xs, est, STAGES):
            ax.annotate(STAGE_LABELS[s], (xi, yi), textcoords="offset points", xytext=(6, -11 if k else 5), fontsize=8,
                        color=INK_2)
    ax.axhline(0, color=RULE, lw=0.8, zorder=1)
    ax.set_xlabel("Effect on the system-card pairs (log-odds, ../coinflip)", color=INK_2)
    ax.set_ylabel("Coinflip, benign minus harmful (nats, 300 tasks)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def collect_runs(group):
    """{stage: latest finished run} of the W&B group. A run counts only if it logged utilities."""
    import wandb
    runs = {}
    for run in sorted(wandb.Api().runs(PROJECT, filters={"group": group, "state": "finished", "jobType": "tournament"}),
                      key=lambda r: r.created_at):
        if "utilities" in run.summary.keys():
            (stage,) = [m["stage"] for m in run.config["models"]]
            runs[stage] = run
    return runs


def merged_fit(group, out):
    """(utilities.csv rows, summary) of every OLMo 2 stage, fit together; cached in out."""
    tasks = read_jsonl(TASKS_FILE)
    if not (out / "utilities.csv").exists():
        runs = collect_runs(group)
        missing = set(STAGES) - set(runs)
        if missing:
            raise SystemExit(f"no finished run for stages {sorted(missing)} in group {group}")
        rows = []
        for s in STAGES:
            print(f"{s:9} {runs[s].path}")
            rows += fetch_wandb(runs[s], Path("outputs") / runs[s].id)
        analyze(rows, tasks, out, seed=0, stage_order=list(STAGES))
    with open(out / "utilities.csv", newline="") as fh:
        return list(csv.DictReader(fh)), json.loads((out / "summary.json").read_text())


def gate(summary2, U3, ids3, rows_all):
    """The pre-registered checks: every OLMo 2 cell reliable, and the OLMo 3 refit matches its all-pairs fit."""
    ok, lines = True, []
    for s in STAGES:
        for cond in ("assistant", "coinflip"):
            rel, mass = summary2[f"{s}/{cond}/reliability"], summary2[f"{s}/{cond}/outcome_mass_mean"]
            passed = rel >= MIN_RELIABILITY
            ok &= passed
            lines.append(f"OLMo 2 {s:9} {cond:10} reliability {rel:.3f}  mass {mass:.2f}: {'pass' if passed else 'FAIL'}")
    by_id = {r["id"]: r for r in rows_all}
    for s in STAGES:
        for cond in ("assistant", "coinflip"):
            full = np.array([float(by_id[t][f"{s}/{cond}"]) for t in ids3])
            r = pearson(U3[f"{s}/{cond}"], full)
            passed = r >= GATE_R
            ok &= passed
            lines.append(f"OLMo 3 {s:9} {cond:10} 15k fit vs all pairs r {r:.4f}: {'pass' if passed else 'FAIL'}")
    return ok, lines


def ladder_table(cells, summaries):
    table = []
    for ladder, by_stage in cells.items():
        for s in STAGES:
            est, boot = by_stage[s]["est"], by_stage[s]["boot"]
            lo, hi = _nan_ci(boot)
            rel_a = summaries[ladder][f"{s}/assistant/reliability"]
            rel_c = summaries[ladder][f"{s}/coinflip/reliability"]
            row = dict(ladder=LADDERS[ladder], stage=s, coinflip_effect=COINFLIP_EFFECT[ladder][s],
                       assistant_reliability=rel_a, coinflip_reliability=rel_c,
                       r_disattenuated=est[K["r"]] / np.sqrt(rel_a * rel_c),
                       assistant_mass=summaries[ladder][f"{s}/assistant/outcome_mass_mean"],
                       coinflip_mass=summaries[ladder][f"{s}/coinflip/outcome_mass_mean"],
                       coinflip_slot_bias=summaries[ladder][f"{s}/coinflip/slot_bias_mean"],
                       coinflip_additive_r2=summaries[ladder][f"{s}/coinflip/additive_r2"])
            for name in STATS:
                row.update({name: est[K[name]], f"{name}_lo": lo[K[name]], f"{name}_hi": hi[K[name]]})
            table.append(row)
    return table


def step_table(cells):
    rows = []
    for ladder, by_stage in cells.items():
        for name, f in (("base to SFT", lambda: step(by_stage, "base", "sft")),
                        ("SFT to DPO", lambda: step(by_stage, "sft", "dpo")),
                        ("DPO to final", lambda: step(by_stage, "dpo", "instruct")),
                        ("base to final", lambda: step(by_stage, "base", "instruct")),
                        ("SFT share", lambda: sft_share(by_stage))):
            est, lo, hi = f()
            for stat in STATS:
                rows.append(dict(ladder=LADDERS[ladder], change=name, stat=stat, est=est[K[stat]], lo=lo[K[stat]],
                                 hi=hi[K[stat]]))
    return rows


def print_tables(table, steps, diffs, call):
    fm = lambda r, s: f"{r[s]:+.2f} [{r[f'{s}_lo']:+.2f}, {r[f'{s}_hi']:+.2f}]"
    print(f"\n{'ladder':14} {'stage':9} {'r':>20} {'r non-harmful':>20} {'r within':>20} {'r shift':>20} "
          f"{'b':>20} {'gap':>20} {'c sd':>6} {'a sd':>6} {'mass':>5} {'rel':>5} {'../coinflip':>11}")
    for r in table:
        shift = fm(r, "r_shift") if r["stage"] != "base" else ""
        print(f"{r['ladder']:14} {r['stage']:9} {fm(r, 'r'):>20} {fm(r, 'r_benign'):>20} {fm(r, 'r_within'):>20} "
              f"{shift:>20} {fm(r, 'b'):>20} {fm(r, 'harm_gap'):>20} {r['coinflip_sd']:6.2f} {r['assistant_sd']:6.2f} "
              f"{r['coinflip_mass']:5.2f} {r['coinflip_reliability']:5.3f} {r['coinflip_effect']:11.2f}")
    print("\nOther stats by stage:")
    for r in table:
        print(f"{r['ladder']:14} {r['stage']:9} r|len {r['r_length']:+.2f}  r harmful {r['r_harmful']:+.2f}  "
              f"r origin means {r['r_between']:+.2f}  b non-harmful {r['b_benign']:+.3f}  resid {r['resid_sd']:.3f}  "
              f"A gap {r['assistant_gap']:+.2f}  disatt r {r['r_disattenuated']:+.2f}  heads bias {r['coinflip_slot_bias']:+.2f}  "
              f"add. R2 {r['coinflip_additive_r2']:.3f}  A mass {r['assistant_mass']:.2f}")
    print("\nStage changes and SFT's share of the rise (paired):")
    for stat in ("r", "r_benign", "r_within", "b", "b_benign", "harm_gap", "coinflip_sd"):
        for r in steps:
            if r["stat"] == stat:
                print(f"{r['ladder']:14} {stat:12} {r['change']:14} {r['est']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]")
    print("\nOLMo 2 minus OLMo 3 at SFT, and in SFT's share (paired):")
    for r in diffs:
        print(f"{r['measure']:6} {r['stat']:14} {r['est']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]")
    print(f"\nPre-registered call: {call}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--group", default="olmo2")
    p.add_argument("--olmo3-only", action="store_true", help="print OLMo 3's ladder on the shared pairs and stop")
    args = p.parse_args()
    out = Path("outputs/olmo2")
    out.mkdir(parents=True, exist_ok=True)

    rows3, summary3 = olmo_on_shared_pairs(SHARED_PAIRS, stages=STAGES, cache_suffix="-stages")
    U3, origins, benign, loglen = task_arrays(rows3)
    ids = [r["id"] for r in rows3]
    resamples = stratified_resamples(origins, np.random.default_rng(0), N_BOOTSTRAP)
    cells3 = ladder_cells(U3, benign, origins, loglen, resamples)
    if args.olmo3_only:
        table = ladder_table({"olmo3": cells3}, {"olmo3": summary3})
        print_tables(table, step_table({"olmo3": cells3}), [], "n/a (OLMo 3 only)")
        return

    rows2, summary2 = merged_fit(args.group, out / "analysis")
    assert [r["id"] for r in rows2] == ids, "task sets differ"
    U2 = task_arrays(rows2)[0]
    with open("outputs/d8ysbybo/analysis/utilities.csv", newline="") as fh:
        rows_all = list(csv.DictReader(fh))
    passed, gate_lines = gate(summary2, U3, ids, rows_all)
    print("Validation gate:\n  " + "\n  ".join(gate_lines))

    cells = {"olmo2": ladder_cells(U2, benign, origins, loglen, resamples), "olmo3": cells3}
    table = ladder_table(cells, {"olmo2": summary2, "olmo3": summary3})
    steps = step_table(cells)
    diffs = []
    for measure, (est, lo, hi) in (("at SFT", ladder_diff(cells["olmo2"], cells3)),
                                   ("share", share_diff(cells["olmo2"], cells3))):
        diffs += [dict(measure=measure, stat=s, est=est[K[s]], lo=lo[K[s]], hi=hi[K[s]]) for s in STATS]
    call = classify(cells["olmo2"], cells3)
    print_tables(table, steps, diffs, call)

    print("\nExploratory, r between the ladders at the same stage:")
    cross = [dict(stage=s, condition=c, r=pearson(U2[f"{s}/{c}"], U3[f"{s}/{c}"]))
             for s in STAGES for c in ("assistant", "coinflip")]
    for r in cross:
        print(f"  {r['stage']:9} {r['condition']:10} {r['r']:+.2f}")
    print("\nExploratory, r between stages' coinflip utilities (Assistant in brackets):")
    pairs = (("base", "sft"), ("sft", "dpo"), ("dpo", "instruct"), ("sft", "instruct"))
    matrix = []
    for ladder, U in (("olmo2", U2), ("olmo3", U3)):
        for s, t in pairs:
            matrix.append(dict(ladder=LADDERS[ladder], x=s, y=t, r_coinflip=pearson(U[f"{s}/coinflip"], U[f"{t}/coinflip"]),
                               r_assistant=pearson(U[f"{s}/assistant"], U[f"{t}/assistant"])))
            print(f"  {LADDERS[ladder]:14} {s:4} vs {t:9} {matrix[-1]['r_coinflip']:+.2f} ({matrix[-1]['r_assistant']:+.2f})")
    origin_rows = [dict(ladder=LADDERS[k], **r) for k, U in (("olmo2", U2), ("olmo3", U3)) for r in by_origin(U, origins)]
    print("\nExploratory, within each origin, r(A, C) | mean C, by stage (" + ", ".join(STAGES) + "):")
    for ladder in LADDERS.values():
        for g in ORIGIN_STYLE:
            rs = [r for r in origin_rows if r["ladder"] == ladder and r["origin"] == g]
            print(f"  {ladder:14} {g:12} n={rs[0]['n']:2d}  " + " ".join(f"{r['r']:+.2f}" for r in rs) + "  |  "
                  + " ".join(f"{r['mean_c']:+.2f}" for r in rs))

    write_csv(out / "olmo2.csv", table)
    write_csv(out / "steps.csv", steps)
    write_csv(out / "ladders.csv", diffs)
    write_csv(out / "cross_ladder.csv", cross)
    write_csv(out / "stage_matrix.csv", matrix)
    write_csv(out / "by_origin.csv", origin_rows)
    report(rows2, summary2, out / "report")
    plotstyle.setup()
    plot_ladders(cells, "figures/olmo2_by_stage.png")
    plot_scatter({"olmo2": U2, "olmo3": U3}, origins, "figures/olmo2_scatter.png")
    plot_vs_coinflip(cells, "figures/olmo2_vs_coinflip.png")
    print(f"\nGate {'passed' if passed else 'FAILED'}. Wrote {out}/ and figures/olmo2_*.png")


if __name__ == "__main__":
    main()
