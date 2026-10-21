"""Relate per-task utilities to the task features that come with the task set.

The features are patchy: only topic and length exist for every task; the rest are
origin-specific metadata (MATH difficulty and subject, Alpaca's NeMo task type and
complexity scores, WildChat type, BailBench harm category, STRESS-TEST nudge direction;
the STRESS-TEST value pairs are unique per task and not used). So every test is within
one origin, which also keeps the origin differences (see report.py) from passing for
feature effects.

Outcomes: the focus stage's Assistant, coinflip and non-Assistant coinflip utilities
(report.py) and the previous stage's coinflip utilities.

Numeric features: Pearson r with a 95% bootstrap CI over tasks. Categorical features,
after dropping levels with fewer than MIN_LEVEL_N tasks: epsilon^2, the between-level
share of variance corrected for the number of levels (about 0 with no effect, and can go
negative). p-values permute the feature within origin; q-values are Benjamini-Hochberg
over every test in the run. With 30-75 tasks per origin this is exploratory.

Usage: uv run python features.py outputs/<run_id> [--tasks data/<tasks>.jsonl]   (after analyze.py)
       uv run python features.py entity/project/<run_id> [--update-wandb]
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from analyze import N_BOOTSTRAP, pearson, read_jsonl, read_run_config
from report import DIVERGING, FOCUS_STAGE, INK, PRE_STAGE, fetch_wandb, group_mean_ci, nonassistant, task_arrays

N_PERMUTATIONS = 5000
MIN_LEVEL_N = 3
Q_STAR = 0.05  # q-values below this are starred in the figure
COMPLEXITY = ("reasoning", "creativity_scope", "domain_knowledge", "contextual_knowledge", "constraints",
              "prompt_complexity_score")
# (origin, feature name, dotted path into the task record, kind). Topic and log length
# are tested within every origin as well.
FEATURES = [
    ("MATH", "difficulty level", "metadata.level", "numeric"),
    ("MATH", "subject", "metadata.type", "categorical"),
    *(("ALPACA", f"nemo {k.replace('_', ' ')}", f"metadata.nemo_analysis.complexity_scores.{k}", "numeric")
      for k in COMPLEXITY),
    ("ALPACA", "nemo task type", "metadata.nemo_analysis.task_type_1", "categorical"),
    ("WILDCHAT", "type", "metadata.type", "categorical"),
    ("BAILBENCH", "harm category", "metadata.category", "categorical"),
    ("STRESS_TEST", "nudge direction", "metadata.nudge_direction", "categorical"),
]
# Sequential blue ramp of the dataviz reference palette, for epsilon^2.
SEQUENTIAL = LinearSegmentedColormap.from_list("sequential", ["#f0efec", "#9ec5f4", "#3987e5", "#184f95"])


def lookup(record, path):
    for key in path.split("."):
        record = record.get(key) if isinstance(record, dict) else None
    return record


def feature_values(tasks, origin):
    """{(feature, kind): values for this origin's tasks, None where missing}."""
    out = {(name, kind): [lookup(t, path) for t in tasks] for o, name, path, kind in FEATURES if o == origin}
    out[("topic", "categorical")] = [t["topic"] or "none" for t in tasks]
    out[("log length", "numeric")] = [float(np.log(len(t["prompt"]))) for t in tasks]
    return out


def epsilon_squared(y, labels):
    """(epsilon^2, between-level sum of squares) of y by level."""
    levels, idx = np.unique(labels, return_inverse=True)
    n, k = len(y), len(levels)
    means = np.bincount(idx, y) / np.bincount(idx)
    ss_between = float((np.bincount(idx) * (means - y.mean()) ** 2).sum())
    ss_total = float(((y - y.mean()) ** 2).sum())
    ms_within = (ss_total - ss_between) / (n - k)
    return (ss_between - (k - 1) * ms_within) / ss_total, ss_between


def permutation_p(stat, x, y, rng):
    """Two-sided permutation p of |stat(x, y)|, shuffling x."""
    observed = abs(stat(x, y))
    hits = sum(abs(stat(rng.permutation(x), y)) >= observed for _ in range(N_PERMUTATIONS))
    return (1 + hits) / (1 + N_PERMUTATIONS)


def bh_q(p):
    """Benjamini-Hochberg q-values."""
    p = np.asarray(p)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1)
    return out


def run_tests(tasks, outcomes, rng, n_boot=N_BOOTSTRAP):
    """Rows of within-origin tests (one per origin x feature x outcome) and categorical level means."""
    origins = np.array([t["origin"] for t in tasks])
    test_rows, level_rows = [], []
    for origin in dict.fromkeys(origins):
        m = origins == origin
        sub = [t for t, keep in zip(tasks, m) if keep]
        for (feature, kind), values in feature_values(sub, origin).items():
            present = np.array([v is not None for v in values])
            if kind == "categorical":
                labels = np.array([str(v) for v in values])
                keep = present & np.isin(labels, [l for l in set(labels) if (labels == l).sum() >= MIN_LEVEL_N])
                if len(set(labels[keep])) < 2:
                    continue
            else:
                x = np.array([float(v) if v is not None else np.nan for v in values])
                keep = present
                if np.unique(x[keep]).size < 3:
                    continue
            for outcome, y_all in outcomes.items():
                y = y_all[m][keep]
                row = dict(origin=origin, feature=feature, kind=kind, n=int(keep.sum()), outcome=outcome)
                if kind == "categorical":
                    lab = labels[keep]
                    eps = epsilon_squared(y, lab)[0]
                    p = permutation_p(lambda l, v: epsilon_squared(v, l)[1], lab, y, rng)
                    row.update(n_levels=len(set(lab)), effect=eps, ci_low=np.nan, ci_high=np.nan, p=p)
                    for level in sorted(set(lab), key=lambda l: -(lab == l).sum()):
                        est, lo, hi = group_mean_ci(y[lab == level], rng, n_boot)
                        level_rows.append(dict(origin=origin, feature=feature, level=level,
                                               n=int((lab == level).sum()), outcome=outcome, mean=est,
                                               ci_low=lo, ci_high=hi))
                else:
                    xk = x[keep]
                    boots = [pearson(xk[i], y[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n_boot))]
                    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
                    row.update(n_levels=np.nan, effect=pearson(xk, y), ci_low=lo, ci_high=hi,
                               p=permutation_p(pearson, xk, y, rng))
                test_rows.append(row)
    for row, q in zip(test_rows, bh_q([r["p"] for r in test_rows])):
        row["q"] = float(q)
    return test_rows, level_rows


def plot_tests(test_rows, outcomes):
    """Effect grid: numeric features (r) on the left, categorical ones (epsilon^2) on the right."""
    fig_rows = {}
    for kind in ("numeric", "categorical"):
        fig_rows[kind] = list(dict.fromkeys((r["origin"], r["feature"]) for r in test_rows if r["kind"] == kind))
    height = 0.42 * max(len(v) for v in fig_rows.values()) + 2.2
    fig, axes = plt.subplots(1, 2, figsize=(14, height))
    lookup_row = {(r["origin"], r["feature"], r["outcome"]): r for r in test_rows}
    for ax, kind in zip(axes, ("numeric", "categorical")):
        keys = fig_rows[kind]
        E = np.array([[lookup_row[(*k, o)]["effect"] for o in outcomes] for k in keys])
        Q = np.array([[lookup_row[(*k, o)]["q"] for o in outcomes] for k in keys])
        if kind == "numeric":
            lim = max(np.abs(E).max(), 0.3)
            im = ax.imshow(E, cmap=DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
            dark = np.abs(E) > 0.6 * lim
        else:
            lim = max(E.max(), 0.2)
            im = ax.imshow(np.clip(E, 0, None), cmap=SEQUENTIAL, vmin=0, vmax=lim, aspect="auto")
            dark = E > 0.6 * lim
        for i, j in np.ndindex(E.shape):
            ax.text(j, i, f"{E[i, j]:.2f}{'*' if Q[i, j] < Q_STAR else ''}", ha="center", va="center", fontsize=9,
                    color="white" if dark[i, j] else INK, fontweight="bold" if Q[i, j] < Q_STAR else "normal")
        ax.set_xticks(range(len(outcomes)), [o.replace("/", "\n") for o in outcomes], fontsize=9)
        ax.set_yticks(range(len(keys)), [f"{o.lower()}: {f} (n={lookup_row[(o, f, outcomes[0])]['n']})"
                                         for o, f in keys], fontsize=9)
        ax.xaxis.tick_top()
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_title("Pearson r (numeric features)" if kind == "numeric" else
                     "epsilon^2: between-level variance share (categorical)", fontsize=11, pad=38)
        fig.colorbar(im, ax=ax, shrink=0.6)
    fig.suptitle(f"Task features vs utilities, within origin; * q < {Q_STAR} (BH over all tests)", fontsize=11)
    fig.tight_layout()
    return fig


def features_report(rows, tasks, out_dir, seed=0, n_boot=N_BOOTSTRAP):
    """Run all tests, write CSVs and the figure to out_dir, and return what to log to W&B."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    U, _, _, _ = task_arrays(rows)
    by_id = {t["id"]: t for t in tasks}
    tasks = [by_id[r["id"]] for r in rows]
    outcomes = {
        f"{FOCUS_STAGE}/assistant": U[f"{FOCUS_STAGE}/assistant"],
        f"{FOCUS_STAGE}/coinflip": U[f"{FOCUS_STAGE}/coinflip"],
        f"{FOCUS_STAGE}/nonassistant": nonassistant(U[f"{FOCUS_STAGE}/coinflip"], U[f"{FOCUS_STAGE}/assistant"]),
        f"{PRE_STAGE}/coinflip": U[f"{PRE_STAGE}/coinflip"],
    }
    test_rows, level_rows = run_tests(tasks, outcomes, np.random.default_rng(seed), n_boot)
    fig = plot_tests(test_rows, list(outcomes))
    path = out_dir / "feature_tests.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    tables = dict(feature_tests=test_rows, feature_levels=level_rows)
    for name, table in tables.items():
        with open(out_dir / f"{name}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(table[0]))
            w.writeheader()
            w.writerows(table)

    print(f"\nEffect (r or epsilon^2) by outcome; * q < {Q_STAR}, . p < 0.05:")
    print(f"{'origin':12} {'feature':30} {'n':>3} " + " ".join(f"{o:>22}" for o in outcomes))
    by_key = {}
    for r in test_rows:
        by_key.setdefault((r["origin"], r["feature"], r["kind"], r["n"]), {})[r["outcome"]] = r
    for (origin, feature, kind, n), cells in by_key.items():
        text = []
        for o in outcomes:
            r = cells[o]
            mark = "*" if r["q"] < Q_STAR else ("." if r["p"] < 0.05 else " ")
            text.append(f"{r['effect']:+.2f}{mark} [{r['ci_low']:+.2f},{r['ci_high']:+.2f}]" if kind == "numeric"
                        else f"eps2 {r['effect']:+.2f}{mark}")
        print(f"{origin.lower():12} {feature:30} {n:3d} " + " ".join(f"{t:>22}" for t in text))
    print(f"\nWrote feature analysis to {out_dir}")
    return dict(tables=tables, figures={"features/feature_tests": path})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source", help="local run dir (outputs/<run_id>, analyzed) or W&B run path entity/project/run_id")
    p.add_argument("--tasks", help="tasks file (default: the run config's tasks_file)")
    p.add_argument("--update-wandb", action="store_true", help="log the feature analysis to the W&B run")
    args = p.parse_args()

    run = None
    if Path(args.source).is_dir():
        run_dir = Path(args.source)
        tasks_file = args.tasks or read_run_config(run_dir)["tasks_file"]
    else:
        import wandb
        run = wandb.Api().run(args.source)
        run_dir = Path("outputs") / run.id
        fetch_wandb(run, run_dir / "analysis")
        tasks_file = args.tasks or run.config["tasks_file"]
    with open(run_dir / "analysis" / "utilities.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    result = features_report(rows, read_jsonl(tasks_file), run_dir / "features")
    if args.update_wandb:
        if run is None:
            raise SystemExit("--update-wandb needs a W&B run path")
        import wandb
        live = wandb.init(entity=run.entity, project=run.project, id=run.id, resume="must")
        log = {f"features/{name}": wandb.Table(columns=list(t[0]), data=[list(r.values()) for r in t])
               for name, t in result["tables"].items()}
        log.update({name: wandb.Image(str(path)) for name, path in result["figures"].items()})
        live.log(log)
        live.finish()


if __name__ == "__main__":
    main()
