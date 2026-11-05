"""Apertus 1.5 70B by training stage (README, "Apertus 1.5 70B by training stage"): how well each stage's coinflip
follows its own Assistant over the 300 tasks, overall and within each task origin.

Collects the W&B runs of group "apertus" (configs/apertus/, one per stage), merges them and fits utilities together
(analyze.py). With A_s and C_s a stage's Assistant and coinflip utilities, per stage (olmo2.stage_stats):
- r over all, non-harmful and harmful tasks; r within origin (each origin's mean removed from both) and over the five
  origin means; r with log task length partialled out; r shift = r(A_s - A_base, C_s - C_base);
- b, the slope of C_s on A_s in nats per Assistant SD; the benign-minus-harmful gap of C_s and of A_s;
- coinflip and Assistant SD, and C_s's SD around its line on A_s;
- r within each of the five origins (by_origin);
- split-half reliabilities, r disattenuated by them, coin mass and heads bias (analyze's summary).
CIs are 95% task bootstraps within origin, with the same resamples at every stage, so steps between stages are paired.
For context (`context`): the same rs for OLMo 2 32B and OLMo 3.1 32B by stage (as cached by olmo2.py) and for the base
and final stages of Llama 3.3 70B, Gemma 4 31B, GLM-4.5-Air and DeepSeek V4 Flash (the sweep), all on the same pairs.

  outputs/apertus/analysis/       the merged fit (analyze.py)
  outputs/apertus/report/         report.py on it
  outputs/apertus/stages.csv      one row per stage
  outputs/apertus/steps.csv       stage-to-stage changes, paired
  outputs/apertus/by_origin.csv   r(A, C) and means within each origin, per stage
  outputs/apertus/context.csv     r overall, within origin and by origin for the comparison ladders and models
  outputs/apertus/length_harm.csv exploratory: task length and the harm contrast, Apertus and the comparison models
  figures/apertus_{scatter,r_by_stage,context,length}.png

Usage: uv run python apertus.py [--group apertus]
"""

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import N_BOOTSTRAP, ORIGIN_STYLE, analyze, fetch_wandb, partial_r, pearson, read_jsonl
from matskit import plotstyle
from olmo2 import K, STATS, _nan_ci, collect_runs, stage_stats, step
from openers import within_groups, write_csv
from report import INK, INK_2, RULE, origin_scatter, report, stratified_resamples, style, task_arrays

STAGES = ("base", "sft", "rl", "dpo")
STAGE_LABELS = {"base": "base", "sft": "SFT", "rl": "RL", "dpo": "DPO (final)"}
TASKS_FILE = "data/tasks_n300_max500_seed0.jsonl"
ORIGIN_LABELS = {"MATH": "MATH", "ALPACA": "Alpaca", "WILDCHAT": "WildChat", "STRESS_TEST": "STRESS-TEST",
                 "BAILBENCH": "BailBench"}
# The correlations reported per stage: over all tasks, non-harmful tasks, within origin, then within each origin.
GROUPS = ("all", "non-harmful", "within origin") + tuple(ORIGIN_STYLE)
GROUP_LABELS = {"all": "All tasks", "non-harmful": "Non-harmful tasks", "within origin": "Within origin (pooled)",
                **ORIGIN_LABELS}
# ../coinflip RESULTS.md, "Apertus 1.5 70B across training stages": the Assistant-framing effect on its 25 benign vs
# harmful pairs, no system prompt, log-odds.
COINFLIP_EFFECT = {"base": 0.01, "sft": 0.07, "rl": 0.16, "dpo": 0.33}
# Comparison runs on the same 300 tasks and 15,000 pairs: (utilities.csv, {label: stage column}).
OLMO_STAGES = {"base": "base", "SFT": "sft", "DPO": "dpo", "final": "instruct"}
SWEEP_STAGES = {"base": "base", "final": "instruct"}
CONTEXT = {
    "OLMo 3.1 32B": ("outputs/d8ysbybo/analysis-15000pairs-stages/utilities.csv", OLMO_STAGES),
    "OLMo 2 32B": ("outputs/olmo2/analysis/utilities.csv", OLMO_STAGES),
    "Llama 3.3 70B": ("outputs/tvlqj8a0/analysis/utilities.csv", SWEEP_STAGES),
    "Gemma 4 31B": ("outputs/jrl62npf/analysis/utilities.csv", SWEEP_STAGES),
    "GLM-4.5-Air": ("outputs/i5vejs3b/analysis/utilities.csv", SWEEP_STAGES),
    "DeepSeek V4 Flash": ("outputs/lszrxqyq/analysis/utilities.csv", SWEEP_STAGES),
}
ACCENT = plotstyle.SERIES[6]  # Apertus in the context figure; the origins keep SERIES[0:5] everywhere


def group_rs(a, c, origins, benign):
    """r(A, C) for each of GROUPS, over arrays of the same (possibly resampled) tasks."""
    out = [pearson(a, c), pearson(a[benign], c[benign]),
           pearson(within_groups(a, origins), within_groups(c, origins))]
    return np.array(out + [pearson(a[origins == g], c[origins == g]) for g in ORIGIN_STYLE])


def stage_cells(U, benign, origins, loglen, resamples, stages=STAGES):
    """{stage: {"est", "boot": STATS, "groups", "groups_boot": GROUPS}}, one bootstrap draw shared by every stage."""
    a0, c0 = U["base/assistant"], U["base/coinflip"]
    out = {}
    for s in stages:
        a, c = U[f"{s}/assistant"], U[f"{s}/coinflip"]
        out[s] = dict(
            est=stage_stats(a, c, a0, c0, benign, origins, loglen),
            boot=np.array([stage_stats(a[i], c[i], a0[i], c0[i], benign[i], origins[i], loglen[i]) for i in resamples]),
            groups=group_rs(a, c, origins, benign),
            groups_boot=np.array([group_rs(a[i], c[i], origins[i], benign[i]) for i in resamples]))
    return out


def merged_fit(group, out):
    """(utilities.csv rows, summary, {stage: run id}) of every stage, fit together; the fit is cached in out."""
    runs = collect_runs(group)
    missing = set(STAGES) - set(runs)
    if missing:
        raise SystemExit(f"no finished run for stages {sorted(missing)} in group {group}")
    run_ids = {s: runs[s].id for s in STAGES}
    if not (out / "utilities.csv").exists():
        rows = []
        for s in STAGES:
            print(f"{s:5} {runs[s].path}")
            rows += fetch_wandb(runs[s], Path("outputs") / runs[s].id)
        analyze(rows, read_jsonl(TASKS_FILE), out, seed=0, stage_order=list(STAGES))
    with open(out / "utilities.csv", newline="") as fh:
        return list(csv.DictReader(fh)), json.loads((out / "summary.json").read_text()), run_ids


def stage_table(cells, summary):
    table = []
    for s in STAGES:
        est, (lo, hi) = cells[s]["est"], _nan_ci(cells[s]["boot"])
        rel_a, rel_c = summary[f"{s}/assistant/reliability"], summary[f"{s}/coinflip/reliability"]
        row = dict(stage=s, coinflip_effect=COINFLIP_EFFECT[s], assistant_reliability=rel_a,
                   coinflip_reliability=rel_c, r_disattenuated=est[K["r"]] / np.sqrt(rel_a * rel_c),
                   assistant_mass=summary[f"{s}/assistant/outcome_mass_mean"],
                   coinflip_mass=summary[f"{s}/coinflip/outcome_mass_mean"],
                   assistant_slot_bias=summary[f"{s}/assistant/slot_bias_mean"],
                   coinflip_slot_bias=summary[f"{s}/coinflip/slot_bias_mean"],
                   coinflip_additive_r2=summary[f"{s}/coinflip/additive_r2"])
        for name in STATS:
            row.update({name: est[K[name]], f"{name}_lo": lo[K[name]], f"{name}_hi": hi[K[name]]})
        table.append(row)
    return table


def origin_table(cells, U, origins, benign, loglen):
    rows = []
    for s in STAGES:
        lo, hi = _nan_ci(cells[s]["groups_boot"])
        for j, g in enumerate(GROUPS):
            m = origins == g if g in ORIGIN_STYLE else benign if g == "non-harmful" else np.ones(len(origins), bool)
            rows.append(dict(stage=s, group=g, n=int(m.sum()), r=cells[s]["groups"][j], r_lo=lo[j], r_hi=hi[j],
                             mean_a=float(U[f"{s}/assistant"][m].mean()) if g in ORIGIN_STYLE else np.nan,
                             mean_c=float(U[f"{s}/coinflip"][m].mean()) if g in ORIGIN_STYLE else np.nan,
                             # exploratory: within the group, each measure's r with log task length
                             r_c_length=pearson(U[f"{s}/coinflip"][m], loglen[m]),
                             r_a_length=pearson(U[f"{s}/assistant"][m], loglen[m])))
    return rows


def step_table(cells):
    rows = []
    for s, t in (("base", "sft"), ("sft", "rl"), ("rl", "dpo"), ("base", "dpo")):
        est, lo, hi = step(cells, s, t)
        rows += [dict(change=f"{s} to {t}", stat=stat, est=est[K[stat]], lo=lo[K[stat]], hi=hi[K[stat]])
                 for stat in STATS]
    return rows


def context(ids):
    """[{model, stage, group, r}] for the comparison runs, aligned to the Apertus tasks by id."""
    rows = []
    for model, (path, stages) in CONTEXT.items():
        with open(path, newline="") as fh:
            by_id = {r["id"]: r for r in csv.DictReader(fh)}
        assert set(by_id) == set(ids), f"{model}: task sets differ"
        U, origins, benign, _ = task_arrays([by_id[i] for i in ids])
        for label, col in stages.items():
            rs = group_rs(U[f"{col}/assistant"], U[f"{col}/coinflip"], origins, benign)
            rows += [dict(model=model, stage=label, group=g, r=float(rs[j])) for j, g in enumerate(GROUPS)]
    return rows


def length_and_harm(U, origins, loglen, ids):
    """Exploratory, per model and stage: what the coinflip tracks besides the Assistant. r(C, log length) and r(A, log
    length), over all tasks and within origin; r(A, C) with log length partialled out, over all tasks and within origin;
    the slope of C on A in nats per nat; and the harm contrast, BailBench minus the general-chat origins (Alpaca and
    WildChat), of C and of A, and its ratio (the share of the Assistant's harm contrast the coin carries)."""
    models = {"Apertus 1.5 70B": (U, origins, loglen, {STAGE_LABELS[s]: s for s in STAGES})}
    for model, (path, stages) in CONTEXT.items():
        with open(path, newline="") as fh:
            by_id = {r["id"]: r for r in csv.DictReader(fh)}
        Um, om, _, lm = task_arrays([by_id[i] for i in ids])
        models[model] = (Um, om, lm, {k: v for k, v in stages.items() if k in ("base", "SFT", "final")})
    rows = []
    for model, (Um, om, lm, stages) in models.items():
        chat = np.isin(om, ["ALPACA", "WILDCHAT"])
        for label, s in stages.items():
            a, c = Um[f"{s}/assistant"], Um[f"{s}/coinflip"]
            aw, cw, lw = within_groups(a, om), within_groups(c, om), within_groups(lm, om)
            harm_c = c[om == "BAILBENCH"].mean() - c[chat].mean()
            harm_a = a[om == "BAILBENCH"].mean() - a[chat].mean()
            rows.append(dict(model=model, stage=label, r=pearson(a, c), r_given_length=partial_r(a, c, lm),
                             r_within=pearson(aw, cw), r_within_given_length=partial_r(aw, cw, lw),
                             r_c_length=pearson(c, lm), r_a_length=pearson(a, lm), r_c_length_within=pearson(cw, lw),
                             r_a_length_within=pearson(aw, lw), slope=float(np.polyfit(a, c, 1)[0]),
                             harm_c=float(harm_c), harm_a=float(harm_a), harm_ratio=float(harm_c / harm_a)))
    return rows


def other_tokens(run_ids, top=6):
    """{stage: [(token, mean probability)]} of the coinflip's most likely next tokens other than the two outcomes,
    from the top 5 logged per prompt (a lower bound where a token falls outside some prompts' top 5)."""
    out = {}
    for s, run_id in run_ids.items():
        total, n = defaultdict(float), 0
        with gzip.open(Path("outputs") / run_id / "results" / f"{s}__coinflip.csv.gz", "rt", newline="") as fh:
            for r in csv.DictReader(fh):
                n += 1
                for tok, p in json.loads(r["top_tokens"]):
                    if tok not in (" heads", " tails"):
                        total[tok] += p
        out[s] = sorted(((t, v / n) for t, v in total.items()), key=lambda x: -x[1])[:top]
    return out


def expectations(table, origin_rows):
    """The six expectations written in the README before the data, each as (text, held)."""
    t = {r["stage"]: r for r in table}
    rs = [t[s]["r"] for s in STAGES]
    gaps = [t[s]["harm_gap"] for s in STAGES]
    steps = np.diff(gaps)
    return [
        ("1. coinflip SD < 0.15 at every stage, gap < 0.15 at DPO",
         all(t[s]["coinflip_sd"] < 0.15 for s in STAGES) and t["dpo"]["harm_gap"] < 0.15),
        ("2. r at DPO in [0.2, 0.6]", 0.2 <= t["dpo"]["r"] <= 0.6),
        ("3. base r <= 0.5 and |gap| < 0.05", t["base"]["r"] <= 0.5 and abs(t["base"]["harm_gap"]) < 0.05),
        ("4. r and gap rise at each step, DPO the largest step in the gap",
         bool(np.all(np.diff(rs) >= 0) and np.all(steps >= 0) and steps.argmax() == 2)),
        ("5. Assistant reliability >= 0.95 and benign above harmful at every stage",
         all(t[s]["assistant_reliability"] >= 0.95 and t[s]["assistant_gap"] > 0 for s in STAGES)),
        ("6. coin mass in [0.5, 0.8] at every stage", all(0.5 <= t[s]["coinflip_mass"] <= 0.8 for s in STAGES)),
    ]


def plot_scatter(U, origins, cells, path, sharey):
    """Assistant vs coinflip utilities, one panel per stage, with the least-squares line over all tasks."""
    fig, axes = plt.subplots(1, len(STAGES), figsize=(3.4 * len(STAGES), 3.6), sharey=sharey, squeeze=False)
    for ax, s in zip(axes[0], STAGES):
        a, c = U[f"{s}/assistant"], U[f"{s}/coinflip"]
        origin_scatter(ax, a, c, origins)
        xs = np.array([a.min(), a.max()])
        ax.plot(xs, np.polyval(np.polyfit(a, c, 1), xs), color=INK, lw=1.5, zorder=4)
        est = cells[s]["groups"][0]
        lo, hi = _nan_ci(cells[s]["groups_boot"][:, :1])
        ax.set_title(f"{STAGE_LABELS[s]}\nr = {est:.2f} [{lo[0]:.2f}, {hi[0]:.2f}], coinflip SD {c.std():.2f}",
                     fontsize=10, color=INK, loc="left")
        ax.set_xlabel("Assistant utility (nats)", color=INK_2)
    axes[0, 0].set_ylabel("Coinflip utility (nats)", color=INK_2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, [ORIGIN_LABELS[o.upper()] for o in labels], frameon=False, ncol=len(ORIGIN_STYLE),
               loc="lower center", fontsize=9)
    note = "" if sharey else "; each panel has its own axes"
    fig.suptitle(f"Apertus 1.5 70B: Assistant vs coinflip utility per task, by training stage (line: fit over all "
                 f"tasks{note})", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_r_by_stage(cells, path):
    """r(A, C) by stage, one panel per task group, shared y-axis."""
    fig, axes = plt.subplots(2, 4, figsize=(12.5, 6), sharex=True, sharey=True)
    x = np.arange(len(STAGES))
    for j, (ax, g) in enumerate(zip(axes.flat, GROUPS)):
        est = np.array([cells[s]["groups"][j] for s in STAGES])
        lo, hi = zip(*(_nan_ci(cells[s]["groups_boot"][:, j:j + 1]) for s in STAGES))
        lo, hi = np.minimum(np.concatenate(lo), est), np.maximum(np.concatenate(hi), est)
        color = ORIGIN_STYLE[g][0] if g in ORIGIN_STYLE else INK
        marker = ORIGIN_STYLE[g][1] if g in ORIGIN_STYLE else "o"
        ax.vlines(x, lo, hi, color=color, lw=2, zorder=2)
        ax.plot(x, est, color=color, lw=1, alpha=0.5, zorder=2)
        ax.scatter(x, est, s=48, c=color, marker=marker, edgecolors="white", linewidths=1, zorder=3)
        for xi, e in zip(x, est):
            ax.annotate(f"{e:.2f}", (xi, e), textcoords="offset points", xytext=(7, -3), fontsize=8, color=INK_2)
        ax.set_title(GROUP_LABELS[g], loc="left", fontsize=10.5, color=INK)
        ax.axhline(0, color=RULE, lw=0.8, zorder=1)
        ax.set_xticks(x, [STAGE_LABELS[s].replace(" (final)", "\n(final)") for s in STAGES])
        style(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("r (Assistant, coinflip)", color=INK_2)
    axes[0, 0].set_ylim(-0.75, 1.0)
    fig.suptitle("Apertus 1.5 70B: correlation of coinflip with Assistant utilities by stage, 95% task-bootstrap CIs",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_context(cells, ctx, path):
    """Final stage: Apertus's r per task group (with CI) next to the other final models on the same pairs."""
    others = list(CONTEXT)
    markers = ["s", "D", "^", "v", "P", "X"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    y = np.arange(len(GROUPS))[::-1]
    for k, model in enumerate(others):
        rs = [next(r["r"] for r in ctx if r["model"] == model and r["stage"] == "final" and r["group"] == g)
              for g in GROUPS]
        ax.scatter(rs, y, s=34, marker=markers[k], facecolors="none", edgecolors=INK_2, linewidths=1, zorder=2,
                   label=model)
    lo, hi = _nan_ci(cells["dpo"]["groups_boot"])
    est = cells["dpo"]["groups"]
    ax.hlines(y, np.minimum(lo, est), np.maximum(hi, est), color=ACCENT, lw=2, zorder=3)
    ax.scatter(est, y, s=60, c=ACCENT, edgecolors="white", linewidths=1, zorder=4, label="Apertus 1.5 70B (DPO)")
    ax.set_yticks(y, [GROUP_LABELS[g] for g in GROUPS])
    ax.axvline(0, color=RULE, lw=0.8, zorder=1)
    ax.set_xlabel("r (Assistant, coinflip) at the final stage", color=INK_2)
    ax.set_xlim(-0.25, 1.0)
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(1.0, 0.0))
    style(ax)
    ax.set_title("Final models on the same 300 tasks and 15,000 pairs", loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_length(U, origins, loglen, path):
    """Coinflip utility against task length at base and DPO, and the DPO Assistant's, one panel each."""
    panels = (("base", "coinflip"), ("dpo", "coinflip"), ("dpo", "assistant"))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), squeeze=False)
    for ax, (s, cond) in zip(axes[0], panels):
        y = U[f"{s}/{cond}"]
        for origin, (color, marker) in ORIGIN_STYLE.items():  # as report.origin_scatter, without its x = 0 rule
            m = origins == origin
            ax.scatter(np.exp(loglen[m]), y[m], s=22, c=color, marker=marker, label=origin.lower(), edgecolors="white",
                       linewidths=0.6, zorder=3)
        ax.axhline(0, color=RULE, lw=0.8, zorder=1)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([20, 50, 100, 200, 500]))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        style(ax)
        ax.set_title(f"{STAGE_LABELS[s]} {'coinflip' if cond == 'coinflip' else 'Assistant'}: r with log length "
                     f"{pearson(y, loglen):+.2f}", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("Task length (characters, log scale)", color=INK_2)
        ax.set_ylabel(f"{'Coinflip' if cond == 'coinflip' else 'Assistant'} utility (nats)", color=INK_2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, [ORIGIN_LABELS[o.upper()] for o in labels], frameon=False, ncol=len(ORIGIN_STYLE),
               loc="lower center", fontsize=9)
    fig.suptitle("Apertus 1.5 70B: from SFT on, the coin favors longer tasks; the Assistant doesn't (each panel has "
                 "its own y-axis)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def print_tables(table, steps, origin_rows, ctx, exp):
    fm = lambda r, s: f"{r[s]:+.2f} [{r[f'{s}_lo']:+.2f}, {r[f'{s}_hi']:+.2f}]"
    print(f"\n{'stage':5} {'r':>20} {'r non-harmful':>20} {'r within':>20} {'r shift':>20} {'b':>20} {'gap':>20} "
          f"{'c sd':>5} {'a sd':>5} {'rel a':>5} {'rel c':>5} {'r dis':>5} {'mass':>5} {'../coinflip':>11}")
    for r in table:
        shift = fm(r, "r_shift") if r["stage"] != "base" else ""
        print(f"{r['stage']:5} {fm(r, 'r'):>20} {fm(r, 'r_benign'):>20} {fm(r, 'r_within'):>20} {shift:>20} "
              f"{fm(r, 'b'):>20} {fm(r, 'harm_gap'):>20} {r['coinflip_sd']:5.2f} {r['assistant_sd']:5.2f} "
              f"{r['assistant_reliability']:5.3f} {r['coinflip_reliability']:5.3f} {r['r_disattenuated']:+5.2f} "
              f"{r['coinflip_mass']:5.2f} {r['coinflip_effect']:11.2f}")
    print("\nOther stats by stage:")
    for r in table:
        print(f"{r['stage']:5} r|len {r['r_length']:+.2f}  r harmful {r['r_harmful']:+.2f}  r origin means "
              f"{r['r_between']:+.2f}  b non-harmful {r['b_benign']:+.3f}  resid {r['resid_sd']:.3f}  "
              f"A gap {r['assistant_gap']:+.2f}  heads bias {r['coinflip_slot_bias']:+.2f}  A-bias "
              f"{r['assistant_slot_bias']:+.2f}  add. R2 {r['coinflip_additive_r2']:.3f}  A mass {r['assistant_mass']:.2f}")
    print("\nr(A, C) by task group and stage [95% CI]:")
    for g in GROUPS:
        rs = [r for r in origin_rows if r["group"] == g]
        print(f"  {GROUP_LABELS[g]:24} n={rs[0]['n']:3d}  "
              + "  ".join(f"{r['r']:+.2f} [{r['r_lo']:+.2f}, {r['r_hi']:+.2f}]" for r in rs))
    print("\nExploratory, r with log task length within each origin, coinflip | Assistant, by stage:")
    for g in ORIGIN_STYLE:
        rs = [r for r in origin_rows if r["group"] == g]
        print(f"  {ORIGIN_LABELS[g]:12} " + " ".join(f"{r['r_c_length']:+.2f}" for r in rs) + "  |  "
              + " ".join(f"{r['r_a_length']:+.2f}" for r in rs))
    print("\nMean utility by origin and stage, Assistant | coinflip:")
    for g in ORIGIN_STYLE:
        rs = [r for r in origin_rows if r["group"] == g]
        print(f"  {ORIGIN_LABELS[g]:12} " + " ".join(f"{r['mean_a']:+.2f}" for r in rs) + "  |  "
              + " ".join(f"{r['mean_c']:+.2f}" for r in rs))
    print("\nStage steps (paired):")
    for stat in ("r", "r_benign", "r_within", "b", "harm_gap", "coinflip_sd"):
        for r in steps:
            if r["stat"] == stat:
                print(f"  {stat:12} {r['change']:12} {r['est']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]")
    print("\nContext, r over all tasks | within origin, by stage:")
    for model, (_, stages) in CONTEXT.items():
        cells = [(st, {r["group"]: r["r"] for r in ctx if r["model"] == model and r["stage"] == st}) for st in stages]
        print(f"  {model:18} " + "  ".join(f"{st} {d['all']:+.2f} | {d['within origin']:+.2f}" for st, d in cells))
    print("\nExpectations written before the data:")
    for text, held in exp:
        print(f"  {'held  ' if held else 'FAILED'} {text}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--group", default="apertus")
    args = p.parse_args()
    out = Path("outputs/apertus")
    out.mkdir(parents=True, exist_ok=True)

    rows, summary, run_ids = merged_fit(args.group, out / "analysis")
    U, origins, benign, loglen = task_arrays(rows)
    ids = [r["id"] for r in rows]
    resamples = stratified_resamples(origins, np.random.default_rng(0), N_BOOTSTRAP)
    cells = stage_cells(U, benign, origins, loglen, resamples)
    table = stage_table(cells, summary)
    origin_rows = origin_table(cells, U, origins, benign, loglen)
    steps = step_table(cells)
    ctx = context(ids)
    exp = expectations(table, origin_rows)
    print_tables(table, steps, origin_rows, ctx, exp)

    print("\nExploratory, coinflip stage x stage r (Assistant in brackets):")
    for k, s in enumerate(STAGES):
        print(f"  {s:5} " + " ".join(f"{pearson(U[f'{s}/coinflip'], U[f'{t}/coinflip']):+.2f} "
                                     f"({pearson(U[f'{s}/assistant'], U[f'{t}/assistant']):+.2f})" for t in STAGES))
    lh = length_and_harm(U, origins, loglen, ids)
    print("\nExploratory, length and the harm contrast (C, A: coinflip, Assistant; w: within origin; harm: BailBench "
          "minus Alpaca/WildChat):")
    print(f"  {'model':18} {'stage':11} {'r':>5} {'r|len':>6} {'rw':>5} {'rw|len':>6} {'C~len':>6} {'A~len':>6} "
          f"{'Cw~len':>6} {'Aw~len':>6} {'slope':>6} {'harm C':>7} {'harm A':>7} {'ratio':>6}")
    for r in lh:
        print(f"  {r['model']:18} {r['stage']:11} {r['r']:5.2f} {r['r_given_length']:6.2f} {r['r_within']:5.2f} "
              f"{r['r_within_given_length']:6.2f} {r['r_c_length']:6.2f} {r['r_a_length']:6.2f} "
              f"{r['r_c_length_within']:6.2f} {r['r_a_length_within']:6.2f} {r['slope']:6.3f} {r['harm_c']:7.2f} "
              f"{r['harm_a']:7.2f} {r['harm_ratio']:6.3f}")
    print("\nExploratory, the coinflip's other likely next tokens (mean probability, from each prompt's top 5):")
    for s, toks in other_tokens(run_ids).items():
        print(f"  {s:5} " + ", ".join(f"{t!r} {v:.3f}" for t, v in toks))

    write_csv(out / "stages.csv", table)
    write_csv(out / "steps.csv", steps)
    write_csv(out / "by_origin.csv", origin_rows)
    write_csv(out / "context.csv", ctx)
    write_csv(out / "length_harm.csv", lh)
    report(rows, summary, out / "report")
    plotstyle.setup()
    sd = [U[f"{s}/coinflip"].std() for s in STAGES]
    # One y-axis shows each stage's coinflip at its true scale, unless the flattest stage would be unreadable.
    plot_scatter(U, origins, cells, "figures/apertus_scatter.png", sharey=max(sd) / min(sd) < 4)
    plot_r_by_stage(cells, "figures/apertus_r_by_stage.png")
    plot_context(cells, ctx, "figures/apertus_context.png")
    plot_length(U, origins, loglen, "figures/apertus_length.png")
    print(f"\nWrote {out}/ and figures/apertus_*.png")


if __name__ == "__main__":
    main()
