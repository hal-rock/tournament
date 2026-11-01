"""How well do models agree on what they prefer as the Assistant?

Pairwise correlations of final-stage Assistant utilities across every model scored on the 300
tasks: over all tasks, non-harmful tasks, within task origin, and for each origin and large topic.
No GPU; reads the per-prompt Assistant log-odds (W&B run files) and takes under a minute.

Models: the ten final checkpoints in MODELS. Not included: base, SFT, DPO and DPO-retraining
checkpoints (not final stage), openers runs (coinflip only), y7pafps7 (100 tasks, another prefill)
and the 10-task smoke tests.

Every model is refit on the same 15,000 pairs the sweep scored, so st9cz0uw (OLMo 3 7B, all 44,850
pairs) and d8ysbybo (all pairs) match the rest. Groups:

- all: the 300 tasks. CIs here and below are task bootstraps within origin.
- non-harmful: without BailBench and harmful_request (report.py's benign mask).
- within origin: all tasks with each origin's mean removed from both models, so only agreement
  on tasks of the same type counts.
- within origin + length: the same, with log task length also regressed out.
- origin:<X>, topic:<X>: tasks of one origin, or one topic with at least MIN_TOPIC tasks.

reliability is split-half (Spearman-Brown, as in analyze.split_half) computed on the same subset:
a group with little spread can have a low ceiling even though the 300-task reliability is ~0.99.
r_disattenuated = r / sqrt(reliability_a * reliability_b).

Writes outputs/cross_model/{pairs,models,summary}.csv, matrix_<group>.csv, and
figures/cross_model_assistant_{matrices,by_group}.png.

Usage: uv run python cross_model.py
"""

import csv
import gzip
import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import N_BOOTSTRAP, N_SPLITS, ORIGIN_STYLE, fit, index_pairs, pair_effects, pearson, read_jsonl, spearman
from matskit import plotstyle
from models import PROJECT
from report import stratified_resamples, task_arrays

TASKS_FILE = "data/tasks_n300_max500_seed0.jsonl"
MAX_PAIRS, PAIR_SEED = 15000, 0
MIN_TOPIC = 18
OUT = Path("outputs/cross_model")
# (label, family, run, stage, saved utilities used as a check). In figure order, grouped by family.
MODELS = [
    ("OLMo 3.1 32B Instruct", "OLMo", "d8ysbybo", "instruct", "analysis-15000pairs"),
    ("OLMo 3.1 32B Think", "OLMo", "z6ldqzq2", "think", "analysis"),  # after an empty think block
    ("OLMo 3 7B Instruct", "OLMo", "st9cz0uw", "instruct", "analysis"),  # saved fit: all pairs
    ("Llama 3.3 70B", "Llama", "tvlqj8a0", "instruct", "analysis"),
    ("Gemma 4 31B", "Gemma", "jrl62npf", "instruct", "analysis"),
    ("GLM-4.5-Air", "GLM", "i5vejs3b", "instruct", "analysis"),
    ("DeepSeek V4 Flash", "DeepSeek", "lszrxqyq", "instruct", "analysis"),
    ("Qwen 3.5 27B", "Qwen", "e2gju7py", "instruct", "analysis"),
    ("Qwen 3.5 35B-A3B", "Qwen", "cqs7umcy", "instruct", "analysis"),
    ("Qwen 3.5 122B-A10B", "Qwen", "2jqibquy", "instruct", "analysis"),
]
ORIGIN_LABEL = {"MATH": "MATH", "ALPACA": "Alpaca", "WILDCHAT": "WildChat", "STRESS_TEST": "STRESS-TEST",
                "BAILBENCH": "BailBench"}


def raw_path(run, stage):
    """The run's Assistant log-odds file: local if the run dir has it, else fetched from W&B into
    outputs/cross_model/raw (not the run dir, where a partial results/ would mislead load_results)."""
    name = f"results/{stage}__assistant.csv.gz"
    local = Path("outputs") / run / name
    if local.exists():
        return local
    root = OUT / "raw" / run
    if not (root / name).exists():
        import wandb
        wandb.Api().run(f"{PROJECT}/{run}").file(name).download(root=str(root), replace=True)
    return root / name


def load_effects(run, stage, keep, index):
    """Pair effects and mean outcome mass of one model's Assistant condition, on the shared pairs."""
    rows, mass = [], []
    with gzip.open(raw_path(run, stage), "rt", newline="") as fh:
        for r in csv.DictReader(fh):
            if tuple(sorted((index[r["first"]], index[r["second"]]))) in keep:
                rows.append(dict(stage=stage, condition="assistant", first=r["first"], second=r["second"],
                                 log_odds=float(r["log_odds"])))
                mass.append(float(r["outcome_mass"]))
    return pair_effects(rows)[(stage, "assistant")], float(np.mean(mass))


def split_half_fits(ids, effects, rng):
    """N_SPLITS x models x tasks x 2: utilities fit on each half of the pairs. The pair set is
    shared, so one split applies to every model."""
    pairs = effects[0]["pairs"]
    assert all(e["pairs"] == pairs for e in effects), "models differ in their pairs"
    ia, ib = index_pairs(ids, pairs)
    out = np.empty((N_SPLITS, len(effects), len(ids), 2))
    for s in range(N_SPLITS):
        first = rng.permutation(len(pairs)) < len(pairs) // 2
        for h, mask in enumerate((first, ~first)):
            for m, e in enumerate(effects):
                out[s, m, :, h] = fit(len(ids), ia[mask], ib[mask], e["effect"][mask])
    return out


def residualize(U, covariates):
    """U (tasks x models) minus its least-squares fit on covariates (tasks x k); U if there are none."""
    if covariates is None:
        return U
    return U - covariates @ np.linalg.lstsq(covariates, U, rcond=None)[0]


def make_groups(origins, topics, benign, loglen):
    """{key: (label, task mask, covariates over all 300 tasks or None)}."""
    n = len(origins)
    dummies = (origins[:, None] == np.array(list(ORIGIN_STYLE))[None, :]).astype(float)
    groups = {"all": ("All 300 tasks", np.ones(n, bool), None),
              "non_harmful": ("Non-harmful", benign, None),
              "within_origin": ("Within origin", np.ones(n, bool), dummies),
              "within_origin_length": ("Within origin + length", np.ones(n, bool), np.column_stack([dummies, loglen]))}
    for o in ORIGIN_STYLE:
        groups[f"origin:{o}"] = (ORIGIN_LABEL[o], origins == o, None)
    for t, c in zip(*np.unique(topics, return_counts=True)):
        if c >= MIN_TOPIC:
            groups[f"topic:{t}"] = (f"topic: {t}", topics == t, None)
    return groups


def group_stats(U, halves, origins, group, resamples):
    """The stats for one group: r, its bootstrap replicates, Spearman, reliabilities, SDs."""
    _, mask, cov = group
    sub = np.flatnonzero(mask)
    def resid(i, V):
        return residualize(V[i], None if cov is None else cov[i])
    A = resid(sub, U)
    reps = []
    rng = np.random.default_rng(0)
    for draw in resamples:
        if cov is None and not mask.all():
            # Same origin-stratified scheme, restricted to the group's tasks.
            members = [sub[origins[sub] == o] for o in np.unique(origins[sub])]
            draw = np.concatenate([rng.choice(m, len(m)) for m in members])
        reps.append(np.corrcoef(resid(draw, U).T))
    half_r = np.empty((N_SPLITS, U.shape[1]))
    for s in range(N_SPLITS):
        h0, h1 = resid(sub, halves[s, :, :, 0].T), resid(sub, halves[s, :, :, 1].T)
        half_r[s] = [pearson(h0[:, m], h1[:, m]) for m in range(U.shape[1])]
    mean_half = half_r.mean(0)
    return dict(n=len(sub), r=np.corrcoef(A.T), reps=np.array(reps), sd=A.std(0),
                reliability=2 * mean_half / (1 + mean_half),
                spearman=np.array([[spearman(A[:, a], A[:, b]) for b in range(A.shape[1])] for a in range(A.shape[1])]))


def plot_matrices(stats, groups, labels, families, keys, path):
    """Heatmaps of model x model r, one per group, on one colour scale."""
    assert min(stats[k]["r"].min() for k in keys) >= 0, "negative r: use plotstyle.DIVERGING centred on 0"
    ncols = 3
    nrows = int(np.ceil(len(keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 5.0 * nrows), squeeze=False)
    cuts = [i - 0.5 for i in range(1, len(families)) if families[i] != families[i - 1]]
    short = [lab.replace("OLMo 3.1 32B", "OLMo 32B").replace("OLMo 3 7B", "OLMo 7B").replace("Qwen 3.5", "Qwen")
             for lab in labels]
    for k, (ax, key) in enumerate(zip(axes.flat, keys)):
        M = stats[key]["r"]
        im = ax.imshow(M, cmap=plotstyle.SEQUENTIAL, vmin=0, vmax=1)
        for i, j in np.ndindex(M.shape):
            if i != j:
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if M[i, j] > 0.6 else plotstyle.INK)
        for c in cuts:
            ax.axhline(c, color=plotstyle.SURFACE, lw=2)
            ax.axvline(c, color=plotstyle.SURFACE, lw=2)
        ax.grid(False)
        ax.set_xticks(range(len(labels)), short, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(labels)), short if k % ncols == 0 else [], fontsize=8)
        ax.tick_params(length=0)
        ax.set_title(f"{groups[key][0]} (n = {stats[key]['n']})", loc="left")
        for side in ax.spines.values():
            side.set_visible(False)
    for ax in axes.flat[len(keys):]:
        ax.set_visible(False)
    fig.suptitle("Post-trained Assistant utilities: Pearson r between models", x=0.01, ha="left", fontsize=13,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 0.93, 0.98))
    cax = fig.add_axes([0.945, 0.3, 0.012, 0.4])
    fig.colorbar(im, cax=cax, label="Pearson r across tasks")
    cax.grid(False)
    fig.savefig(path)
    plt.close(fig)


def plot_by_group(stats, groups, families, keys, path):
    """One dot per model pair in each group, across-family and same-family pairs apart."""
    fig, ax = plt.subplots(figsize=(11, 4.8))
    pairs = list(itertools.combinations(range(len(families)), 2))
    rng = np.random.default_rng(0)
    for g, key in enumerate(keys):
        M = stats[key]["r"]
        for same, color, dx in ((False, plotstyle.SERIES[0], -0.14), (True, plotstyle.SERIES[1], 0.14)):
            v = np.array([M[a, b] for a, b in pairs if (families[a] == families[b]) == same])
            ax.scatter(g + dx + rng.uniform(-0.07, 0.07, len(v)), v, s=22, color=color, alpha=0.55, lw=0, zorder=2)
            ax.hlines(v.mean(), g + dx - 0.12, g + dx + 0.12, color=plotstyle.INK, lw=2, zorder=3)
    ax.scatter([], [], s=30, color=plotstyle.SERIES[0], label="Different families (39 pairs)")
    ax.scatter([], [], s=30, color=plotstyle.SERIES[1], label="Same family (6 pairs: OLMo, Qwen)")
    ax.plot([], [], color=plotstyle.INK, lw=2, label="Mean")
    ax.axhline(0, color=plotstyle.BASELINE, lw=1, zorder=0)
    ax.grid(axis="x", visible=False)
    ax.set_xticks(range(len(keys)), [f"{groups[k][0]}\nn = {stats[k]['n']}" for k in keys], fontsize=8.5)
    ax.set_ylabel("Pearson r between two models")
    ax.set_title("Models agree less within a task type than across all tasks, least on MATH, Alpaca and WildChat", loc="left")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    plotstyle.setup()
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = read_jsonl(TASKS_FILE)
    ids = [t["id"] for t in tasks]
    index = {t: i for i, t in enumerate(ids)}
    from tournament import select_pairs
    keep = set(select_pairs(len(ids), MAX_PAIRS, PAIR_SEED))
    labels = [m[0] for m in MODELS]
    families = [m[1] for m in MODELS]

    effects, masses, saved_r = [], [], []
    U = np.empty((len(ids), len(MODELS)))
    meta = None
    for m, (label, _, run, stage, saved) in enumerate(MODELS):
        eff, mass = load_effects(run, stage, keep, index)
        assert len(eff["pairs"]) == MAX_PAIRS, (label, len(eff["pairs"]))
        U[:, m] = fit(len(ids), *index_pairs(ids, eff["pairs"]), eff["effect"])
        effects.append(eff)
        masses.append(mass)
        with open(Path("outputs") / run / saved / "utilities.csv", newline="") as fh:
            rows = {r["id"]: r for r in csv.DictReader(fh)}
        saved_r.append(pearson(U[:, m], [float(rows[t][f"{stage}/assistant"]) for t in ids]))
        meta = meta or [rows[t] for t in ids]
        print(f"{label:24} {run} refit vs saved utilities r = {saved_r[-1]:.5f}, mean outcome mass {mass:.4f}")
        # Every saved fit on these pairs should come back exactly; st9cz0uw's saved fit used all pairs.
        assert saved_r[-1] > (0.995 if run == "st9cz0uw" else 0.999999), label

    _, origins, benign, loglen = task_arrays(meta)
    topics = np.array([r["topic"] for r in meta])
    groups = make_groups(origins, topics, benign, loglen)
    halves = split_half_fits(ids, effects, np.random.default_rng(0))
    resamples = stratified_resamples(origins, np.random.default_rng(0), N_BOOTSTRAP)
    stats = {k: group_stats(U, halves, origins, g, resamples) for k, g in groups.items()}

    tri = list(itertools.combinations(range(len(MODELS)), 2))
    across = [(a, b) for a, b in tri if families[a] != families[b]]
    same = [(a, b) for a, b in tri if families[a] == families[b]]
    pair_rows, model_rows, summary_rows = [], [], []
    for key, s in stats.items():
        lo, hi = np.percentile(s["reps"], [2.5, 97.5], axis=0)
        rel = s["reliability"]
        for a, b in tri:
            pair_rows.append(dict(group=key, n_tasks=s["n"], model_a=labels[a], model_b=labels[b],
                                  same_family=families[a] == families[b], r=s["r"][a, b], r_ci_low=lo[a, b],
                                  r_ci_high=hi[a, b], spearman=s["spearman"][a, b], reliability_a=rel[a],
                                  reliability_b=rel[b], r_disattenuated=s["r"][a, b] / np.sqrt(rel[a] * rel[b])))
        for m, label in enumerate(labels):
            others = [n for n in range(len(labels)) if n != m and families[n] != families[m]]
            model_rows.append(dict(group=key, n_tasks=s["n"], model=label, utility_sd=s["sd"][m], reliability=rel[m],
                                   mean_r_other_families=s["r"][m, others].mean(), outcome_mass_all_tasks=masses[m]))
        pair_mean = lambda M, ps: np.mean([M[a, b] for a, b in ps], axis=0)
        rep_across = np.array([pair_mean(R, across) for R in s["reps"]])
        lo_a, hi_a = np.percentile(rep_across, [2.5, 97.5])
        disatt = s["r"] / np.sqrt(np.outer(rel, rel))
        low = min(tri, key=lambda p: s["r"][p])
        high = max(tri, key=lambda p: s["r"][p])
        summary_rows.append(dict(group=key, n_tasks=s["n"], mean_r_across_families=pair_mean(s["r"], across),
                                 ci_low=lo_a, ci_high=hi_a, mean_r_same_family=pair_mean(s["r"], same),
                                 mean_spearman_across=pair_mean(s["spearman"], across),
                                 mean_disattenuated_across=pair_mean(disatt, across),
                                 min_reliability=rel.min(), median_reliability=np.median(rel),
                                 min_r=s["r"][low], min_pair=f"{labels[low[0]]} / {labels[low[1]]}",
                                 max_r=s["r"][high], max_pair=f"{labels[high[0]]} / {labels[high[1]]}"))
        with open(OUT / f"matrix_{key.replace(':', '_')}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["model", *labels])
            w.writerows([[lab, *(round(float(v), 4) for v in row)] for lab, row in zip(labels, s["r"])])
    for name, rows in (("pairs", pair_rows), ("models", model_rows), ("summary", summary_rows)):
        with open(OUT / f"{name}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    # The sweep's own matrix (models.py) over the eight models it shares: this should reproduce it.
    sweep = Path("outputs/models/models/cross_model_assistant.csv")
    if sweep.exists():
        with open(sweep, newline="") as fh:
            old = {r["model"]: r for r in csv.DictReader(fh)}
        names = {"OLMo 3.1 32B Instruct": "OLMo 3.1 32B"}
        pos = [(i, names.get(lab, lab)) for i, lab in enumerate(labels) if names.get(lab, lab) in old]
        gap = max(abs(stats["all"]["r"][i, j] - float(old[a][b])) for i, a in pos for j, b in pos)
        print(f"Largest gap from models.py's matrix: {gap:.5f}")
        assert gap < 1e-3

    main_keys = ["all", "non_harmful", "within_origin", *[f"origin:{o}" for o in ORIGIN_STYLE]]
    plot_matrices(stats, groups, labels, families, main_keys, "figures/cross_model_assistant_matrices.png")
    plot_by_group(stats, groups, families, main_keys, "figures/cross_model_assistant_by_group.png")

    print(f"\n{'group':26} {'n':>4} {'across r [95% CI]':>22} {'same':>6} {'rho':>6} {'disatt':>7} {'rel min/med':>12}  "
          "lowest / highest pair")
    for r in summary_rows:
        print(f"{r['group']:26} {r['n_tasks']:4d} {r['mean_r_across_families']:8.2f} [{r['ci_low']:5.2f},{r['ci_high']:5.2f}] "
              f"{r['mean_r_same_family']:6.2f} {r['mean_spearman_across']:6.2f} {r['mean_disattenuated_across']:7.2f} "
              f"{r['min_reliability']:5.2f}/{r['median_reliability']:4.2f}  {r['min_r']:.2f} {r['min_pair']}; "
              f"{r['max_r']:.2f} {r['max_pair']}")
    print(f"\nWrote {OUT}/ and figures/cross_model_assistant_*.png")


if __name__ == "__main__":
    main()
