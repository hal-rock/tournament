"""Bleedthrough across model families: one row per (model, stage) from the per-model runs.

Collects the finished tournament runs in a W&B group (one run per model, see
configs/models/), plus the OLMo 3 32B stages run d8ysbybo refit on the same 15k pairs
the other runs scored (its base and instruct stages), and for each run's base and
instruct stages computes the report.py bleedthrough stats:

- r, r_benign, r_partial_length: correlation of coinflip with Assistant utilities
  (all tasks, non-harmful only, log length partialled out), with task-bootstrap CIs;
- r_disattenuated: r over sqrt(product of the two split-half reliabilities);
- slope_pooled, slope_within: coinflip log-odds per Assistant log-odd, pooled and
  within origin (adjusting for length);
- assistant_utility_sd, coinflip_utility_sd: how strong each preference scale is, in
  nats;
- r_shift (instruct rows of runs with a base model): correlation of the post-training
  shifts, instruct-base Assistant vs instruct-base coinflip, which removes structure
  pretraining puts into both conditions;
- r_consensus, r_own_given_consensus (instruct rows): post-trained models' Assistant
  utilities agree closely (r ~0.8-0.95), so a coinflip can track Assistant
  preferences without tracking this model's in particular. The consensus is the mean
  of the other models' z-scored instruct Assistant utilities; r_consensus is its
  correlation with this model's coinflip, and r_own_given_consensus the partial
  correlation of this model's own Assistant utilities with its coinflip, holding the
  consensus fixed (task-bootstrap CIs; own and consensus correlate highly, so CIs
  are wide);
- assistant_harm_gap, coinflip_harm_gap: mean utility of non-harmful tasks minus
  harmful ones (BailBench and harmful_request), in nats: the benign-vs-harmful
  contrast that the ../coinflip task pairs measure;
- coinflip_effect: the Assistant-framing effect on the Sonnet 4.5 system-card task
  pairs from ../coinflip (RESULTS.md), for comparison.

It also writes model x model correlation matrices of the instruct Assistant and
instruct coinflip utilities (do families agree on what they prefer?).

Usage: uv run python models.py [--group models] [--update-wandb]
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import N_BOOTSTRAP, ORIGIN_STYLE, analyze, partial_r, pearson, read_jsonl
from analyze import fetch_wandb as fetch_results
from report import (DIVERGING, INK, INK_2, RULE, SERIES, fetch_wandb, stage_analysis, stratified_resamples, style,
                    task_arrays, with_ci)

PROJECT = "halrockwell-mats/tournament"
OLMO_RUN = "halrockwell-mats/tournament/d8ysbybo"
STAGES = ("base", "instruct")
# Instruct model -> (label, family, total params in B). Nominal sizes. Fixed order: the
# order models are listed in tables and figures, grouped by family.
MODEL_INFO = {
    "allenai/Olmo-3.1-32B-Instruct": ("OLMo 3.1 32B", "OLMo", 32),
    "unsloth/Llama-3.3-70B-Instruct": ("Llama 3.3 70B", "Llama", 70),
    "google/gemma-4-31B-it": ("Gemma 4 31B", "Gemma", 31),
    "zai-org/GLM-4.5-Air": ("GLM-4.5-Air", "GLM", 106),
    "deepseek-ai/DeepSeek-V4-Flash": ("DeepSeek V4 Flash", "DeepSeek", 284),
    "Qwen/Qwen3.5-27B": ("Qwen 3.5 27B", "Qwen", 27),
    "Qwen/Qwen3.5-35B-A3B": ("Qwen 3.5 35B-A3B", "Qwen", 35),
    "Qwen/Qwen3.5-122B-A10B": ("Qwen 3.5 122B-A10B", "Qwen", 122),
}
# ../coinflip RESULTS.md: preference effect (log-odds) with the Assistant doing the task,
# Sonnet 4.5 system-card tasks, no system prompt. Keyed by instruct model, per stage.
COINFLIP_EFFECT = {
    "allenai/Olmo-3.1-32B-Instruct": {"base": 0.12, "instruct": 2.27},
    "unsloth/Llama-3.3-70B-Instruct": {"base": -0.04, "instruct": 4.13},
    "google/gemma-4-31B-it": {"base": 0.10, "instruct": 7.81},
    "zai-org/GLM-4.5-Air": {"base": -0.15, "instruct": 1.89},
    "deepseek-ai/DeepSeek-V4-Flash": {"base": 0.00, "instruct": 1.52},
    "Qwen/Qwen3.5-27B": {"instruct": 0.05},
    "Qwen/Qwen3.5-35B-A3B": {"base": 0.47, "instruct": 0.19},
    "Qwen/Qwen3.5-122B-A10B": {"instruct": -0.11},
}
PANELS = (("r", "r(Assistant, coinflip)"), ("r_benign", "r, non-harmful tasks"),
          ("r_shift", "r of post-training shifts"), ("slope_within", "Within-origin slope"),
          ("coinflip_utility_sd", "Coinflip utility SD (nats)"), ("assistant_utility_sd", "Assistant utility SD (nats)"))
CI_STATS = ("r", "r_benign", "r_partial_length", "slope_pooled", "slope_within")


def run_rows(model, rows, summary, seed=0, n_boot=N_BOOTSTRAP):
    """Rows of the models table for one run: its utilities.csv rows and analyze.py summary."""
    label, family, total = MODEL_INFO[model]
    U, origins, benign, loglen = task_arrays(rows)
    groups = [g for g in ORIGIN_STYLE if g in set(origins)]
    resamples = stratified_resamples(origins, np.random.default_rng(seed), n_boot)
    out = []
    for stage in STAGES:
        a, c = U.get(f"{stage}/assistant"), U.get(f"{stage}/coinflip")
        if a is None or c is None:
            continue
        stats = stage_analysis(a, c, loglen, origins, benign, groups, resamples)
        rel_a, rel_c = (summary.get(f"{stage}/{cond}/reliability", float("nan")) for cond in ("assistant", "coinflip"))
        row = dict(model=model, label=label, family=family, params_total_b=total, stage=stage,
                   n_tasks=len(a), n_pairs=summary.get(f"{stage}/coinflip/n_pairs", float("nan")),
                   assistant_utility_sd=float(a.std()), coinflip_utility_sd=float(c.std()),
                   assistant_reliability=rel_a, coinflip_reliability=rel_c,
                   coinflip_additive_r2=summary.get(f"{stage}/coinflip/additive_r2", float("nan")),
                   coinflip_slot_bias=summary.get(f"{stage}/coinflip/slot_bias_mean", float("nan")),
                   assistant_outcome_mass_mean=summary.get(f"{stage}/assistant/outcome_mass_mean", float("nan")),
                   coinflip_outcome_mass_mean=summary.get(f"{stage}/coinflip/outcome_mass_mean", float("nan")),
                   r_disattenuated=float(stats["r"][0] / np.sqrt(rel_a * rel_c)) if min(rel_a, rel_c) > 0
                   else float("nan"),
                   coinflip_effect=COINFLIP_EFFECT[model].get(stage, float("nan")))
        for k in CI_STATS:
            row.update({k: float(stats[k][0]), f"{k}_ci_low": float(stats[k][1]), f"{k}_ci_high": float(stats[k][2])})
        # Benign minus harmful mean utility: the contrast the ../coinflip task pairs measure.
        gap = with_ci(lambda i: np.array([u[i][benign[i]].mean() - u[i][~benign[i]].mean() for u in (a, c)]),
                      len(a), resamples)
        for j, cond in enumerate(("assistant", "coinflip")):
            row.update({f"{cond}_harm_gap": float(gap[0][j]), f"{cond}_harm_gap_ci_low": float(gap[1][j]),
                        f"{cond}_harm_gap_ci_high": float(gap[2][j])})
        sa, sc = U.get(f"{stage}-base/assistant"), U.get(f"{stage}-base/coinflip")
        shift = (with_ci(lambda i: np.array([pearson(sa[i], sc[i])]), len(a), resamples)
                 if sa is not None and sc is not None else [np.array([np.nan])] * 3)
        row.update(r_shift=float(shift[0][0]), r_shift_ci_low=float(shift[1][0]), r_shift_ci_high=float(shift[2][0]))
        out.append(row)
    return out


def cross_model(utilities, condition):
    """Model x model Pearson r of the instruct utilities in one condition, plus the
    models' labels, in MODEL_INFO order. utilities: {model: {measure: array}}."""
    models = [m for m in MODEL_INFO if m in utilities]
    X = np.array([utilities[m][f"instruct/{condition}"] for m in models])
    return [MODEL_INFO[m][0] for m in models], np.corrcoef(X)


def consensus_stats(utilities, resamples):
    """{model: {stat: (estimate, ci_low, ci_high)}} for r_consensus and r_own_given_consensus
    (see module docstring). utilities: {model: {measure: array}}, all on the same tasks."""
    z = {m: (U["instruct/assistant"] - U["instruct/assistant"].mean()) / U["instruct/assistant"].std()
         for m, U in utilities.items()}
    out = {}
    for m, U in utilities.items():
        consensus = np.mean([z[n] for n in z if n != m], axis=0)
        a, c = U["instruct/assistant"], U["instruct/coinflip"]
        est, lo, hi = with_ci(lambda i: np.array([pearson(consensus[i], c[i]), partial_r(a[i], c[i], consensus[i])]),
                              len(a), resamples)
        out[m] = {name: (float(est[j]), float(lo[j]), float(hi[j]))
                  for j, name in enumerate(("r_consensus", "r_own_given_consensus"))}
    return out


def plot_models(table):
    """Small multiples, one per stat, models on the x axis; base stages hollow."""
    labels = list(dict.fromkeys(r["label"] for r in table))
    x_of = {lab: i for i, lab in enumerate(labels)}
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    for ax, (key, title) in zip(axes.flat, PANELS):
        for stage, color, dx in (("base", SERIES[1], -0.12), ("instruct", SERIES[0], 0.12)):
            pts = [r for r in table if r["stage"] == stage and not np.isnan(r[key])]
            if not pts:
                continue
            x = np.array([x_of[r["label"]] for r in pts]) + dx
            est = np.array([r[key] for r in pts])
            lo = np.array([r.get(f"{key}_ci_low", v) for r, v in zip(pts, est)])
            hi = np.array([r.get(f"{key}_ci_high", v) for r, v in zip(pts, est)])
            hollow = stage == "base"
            ax.errorbar(x, est, yerr=[est - lo, hi - est], color=color, marker="o", ms=7, ls="none",
                        elinewidth=1.5, capsize=0, mfc="white" if hollow else color,
                        mec=color if hollow else "white", mew=1.5 if hollow else 0.8,
                        label="post-trained" if stage == "instruct" else "base", zorder=3)
        ax.axhline(0, color=RULE, lw=1, zorder=0)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        style(ax)
    for ax in axes[-1]:
        ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right")
    handles, names = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper right", ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Assistant preferences in the coinflip, across models (95% CI over tasks)", fontsize=12, x=0.01,
                 ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plot_cross_model(mats):
    """Side-by-side heatmaps of model x model r, one per condition, same model order."""
    fig, axes = plt.subplots(1, len(mats), figsize=(5.4 * len(mats) + 1.6, 5.6))
    for k, (ax, (title, (labels, M))) in enumerate(zip(np.atleast_1d(axes), mats.items())):
        # All r here are positive; a 0-1 range keeps the differences visible.
        im = ax.imshow(M, cmap=DIVERGING, vmin=-1 if M.min() < 0 else 0, vmax=1)
        for i, j in np.ndindex(M.shape):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                    color="white" if M[i, j] > 0.75 else INK)
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(labels)), labels if k == 0 else [], fontsize=9)
        ax.set_title(title, fontsize=11, color=INK)
        for side in ax.spines.values():
            side.set_visible(False)
    fig.colorbar(im, ax=axes, shrink=0.7, label="Pearson r across tasks")
    return fig


def plot_vs_coinflip(table):
    """Tournament bleedthrough against the ../coinflip effect, one point per (model, stage)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, (key, ylabel) in zip(axes, (("r", "r(Assistant, coinflip), 300 tasks"),
                                        ("coinflip_harm_gap", "coinflip utility, benign minus harmful (nats)"))):
        for stage, color in (("base", SERIES[1]), ("instruct", SERIES[0])):
            pts = [r for r in table if r["stage"] == stage and not np.isnan(r["coinflip_effect"])]
            hollow = stage == "base"
            for r in pts:
                ax.errorbar(r["coinflip_effect"], r[key], yerr=[[r[key] - r[f"{key}_ci_low"]],
                                                                 [r[f"{key}_ci_high"] - r[key]]],
                            color=color, marker="o", ms=7, elinewidth=1.2, capsize=0,
                            mfc="white" if hollow else color, mec=color if hollow else "white", zorder=3)
                if stage == "instruct" and r["coinflip_effect"] > 1:  # the Qwens cluster at 0
                    left = r["family"] == "DeepSeek"  # next to GLM in the gap panel
                    ax.annotate(r["label"], (r["coinflip_effect"], r[key]), xytext=(-6, 3) if left else (5, 3),
                                textcoords="offset points", fontsize=8, color=INK_2, ha="right" if left else "left")
        ax.axhline(0, color=RULE, lw=1, zorder=0)
        ax.axvline(0, color=RULE, lw=1, zorder=0)
        ax.set_xlabel("coinflip effect on system-card tasks (log-odds, ../coinflip)", color=INK_2)
        ax.set_ylabel(ylabel)
        style(ax)
    fig.suptitle("Tournament bleedthrough vs. the coinflip experiment (filled: post-trained, hollow: base; "
                 "unlabeled filled points near 0: Qwen 3.5 27B, 35B-A3B, 122B-A10B)",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    return fig


def instruct_model(config):
    (name,) = [m["name"] for m in config["models"] if m["stage"] == "instruct"]
    return name


def collect_runs(group, project=PROJECT):
    """Finished runs in the group, keeping the most recent run per instruct model. Runs
    that exited early (e.g. a model that did not fit on the GPUs) can also show as
    finished, so a run counts only if it logged the utilities table."""
    import wandb
    latest = {}
    for run in sorted(wandb.Api().runs(project, filters={"group": group, "state": "finished",
                                                         "jobType": "tournament"}), key=lambda r: r.created_at):
        if "utilities" not in run.summary.keys():
            continue
        model = instruct_model(run.config)
        if model not in MODEL_INFO:
            print(f"skipping {run.path}: {model} not in MODEL_INFO")
            continue
        if model in latest:
            print(f"{model}: {run.id} replaces {latest[model].id}")
        latest[model] = run
    return latest


def olmo_on_shared_pairs(config):
    """Refit the OLMo stages run's base and instruct stages on the pairs the per-model runs
    scored (select_pairs with their max_pairs and seed). Cached in
    outputs/<run>/analysis-<n>pairs; returns (utilities.csv rows, summary)."""
    import wandb
    from tournament import select_pairs
    run = wandb.Api().run(OLMO_RUN)
    out = Path("outputs") / run.id / f"analysis-{config['max_pairs']}pairs"
    if not (out / "utilities.csv").exists():
        tasks = read_jsonl(config["tasks_file"])
        index = {t["id"]: i for i, t in enumerate(tasks)}
        keep = set(select_pairs(len(tasks), config["max_pairs"], config.get("seed", 0)))
        rows = [r for r in fetch_results(run, Path("outputs") / run.id) if r["stage"] in STAGES
                and tuple(sorted((index[r["first"]], index[r["second"]]))) in keep]
        analyze(rows, tasks, out, seed=config.get("seed", 0), stage_order=list(STAGES))
    with open(out / "utilities.csv", newline="") as fh:
        return list(csv.DictReader(fh)), json.loads((out / "summary.json").read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--group", default="models", help="W&B group of the per-model runs")
    p.add_argument("--update-wandb", action="store_true", help="log the tables and figures as a run in the group")
    args = p.parse_args()

    sources = {}
    runs = collect_runs(args.group)
    for model, run in runs.items():
        analysis = Path("outputs") / run.id / "analysis"
        fetch_wandb(run, analysis)
        with open(analysis / "utilities.csv", newline="") as fh:
            sources[model] = (list(csv.DictReader(fh)), json.loads((analysis / "summary.json").read_text()))
        print(f"{model:34} {run.path}")
    if runs:
        sources["allenai/Olmo-3.1-32B-Instruct"] = olmo_on_shared_pairs(next(iter(runs.values())).config)
        print(f"{'allenai/Olmo-3.1-32B-Instruct':34} {OLMO_RUN} (refit on the shared pairs)")
    ids = {m: [r["id"] for r in rows] for m, (rows, _) in sources.items()}
    assert len({tuple(v) for v in ids.values()}) == 1, "runs differ in their task sets"

    table = []
    for model, (rows, summary) in sources.items():
        table += run_rows(model, rows, summary)
    utilities = {m: task_arrays(rows)[0] for m, (rows, _) in sources.items()}
    origins = task_arrays(next(iter(sources.values()))[0])[1]
    consensus = consensus_stats({m: U for m, U in utilities.items() if "instruct/assistant" in U},
                                stratified_resamples(origins, np.random.default_rng(0), N_BOOTSTRAP))
    for r in table:
        for name in ("r_consensus", "r_own_given_consensus"):
            est, lo, hi = consensus[r["model"]][name] if r["stage"] == "instruct" else (float("nan"),) * 3
            r.update({name: est, f"{name}_ci_low": lo, f"{name}_ci_high": hi})
    table.sort(key=lambda r: (list(MODEL_INFO).index(r["model"]), STAGES.index(r["stage"])))

    out_dir = Path("outputs") / "models" / args.group
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "models.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    mats = {f"Post-trained {cond} utilities": cross_model(utilities, cond) for cond in ("assistant", "coinflip")}
    for title, (labels, M) in mats.items():
        with open(out_dir / f"cross_model_{title.split()[1]}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["model", *labels])
            w.writerows([[lab, *(round(v, 4) for v in row)] for lab, row in zip(labels, M)])
    figures = {"models": plot_models(table), "cross_model": plot_cross_model(mats),
               "vs_coinflip": plot_vs_coinflip(table)}
    for name, fig in figures.items():
        fig.savefig(out_dir / f"{name}.png", dpi=150, bbox_inches="tight")

    print(f"\n{'model':20} {'stage':9} {'r':>6} {'95% CI':>14} {'r benign':>9} {'r shift':>8} {'r cons':>7} {'r own|cons':>16} "
          f"{'slope':>7} {'coin sd':>8} {'asst sd':>8} {'coin rel':>9} {'coin mass':>10} {'coin gap':>9} {'coinflip':>9}")
    for r in table:
        print(f"{r['label']:20} {r['stage']:9} {r['r']:6.2f} [{r['r_ci_low']:5.2f}, {r['r_ci_high']:5.2f}] "
              f"{r['r_benign']:9.2f} {r['r_shift']:8.2f} {r['r_consensus']:7.2f} {r['r_own_given_consensus']:5.2f} "
              f"[{r['r_own_given_consensus_ci_low']:5.2f},{r['r_own_given_consensus_ci_high']:5.2f}] {r['slope_within']:7.3f} "
              f"{r['coinflip_utility_sd']:8.3f} {r['assistant_utility_sd']:8.2f} {r['coinflip_reliability']:9.3f} "
              f"{r['coinflip_outcome_mass_mean']:10.2f} {r['coinflip_harm_gap']:9.2f} {r['coinflip_effect']:9.2f}")
    for title, (labels, M) in mats.items():
        print(f"\n{title}, Pearson r:\n{'':20} " + " ".join(f"{lab[:8]:>8}" for lab in labels))
        for lab, row in zip(labels, M):
            print(f"{lab:20} " + " ".join(f"{v:8.2f}" for v in row))
    print(f"\nWrote {out_dir}/models.csv, cross_model_*.csv and figures")

    if args.update_wandb:
        import wandb
        run = wandb.init(project=PROJECT.split("/")[1], entity=PROJECT.split("/")[0], group=args.group,
                         job_type="models-analysis", name=f"{args.group}-analysis")
        run.log({"models/table": wandb.Table(columns=list(table[0]), data=[list(r.values()) for r in table]),
                 **{f"models/{name}": wandb.Image(str(out_dir / f"{name}.png")) for name in figures}})
        run.finish()


if __name__ == "__main__":
    main()
