"""Fit per-task utilities from a tournament run and compare conditions.

For a task pair (a, b) in one (stage, condition), with L = log P(first) - log P(second)
and both presentation orders:

    effect(a over b) = (L[a first] - L[b first]) / 2
    slot bias        = (L[a first] + L[b first]) / 2

The swap cancels any first-slot bias (position, heads/tails) that is the same
in both orders, even if it varies across pairs. Utilities are least-squares
fits of u_a - u_b = effect, centered at zero (Bradley-Terry on the log-odds
themselves; with a full round-robin, u_a is just a's mean effect). Additive R^2
is the share of the effects' sum of squares that u_a - u_b explains: low means
choices depend on the specific pairing, not one preference scale.

Measures are either one (stage, condition)'s utilities, or a post-training
shift, e.g. instruct-base/coinflip = instruct/coinflip - base/coinflip.
Correlating the shifts across conditions (difference-in-differences) removes
whatever the base model already puts into both, such as length effects.

Reliability is split-half: fit on two random halves of the pairs (the same
halves for every stage and condition, so shifts split consistently), correlate,
and Spearman-Brown correct to the full set. A correlation between measures
can't exceed roughly sqrt(rel_x * rel_y), so it is also reported disattenuated.
Correlations are also given over non-harmful tasks only, in case one extreme
cluster drives the full-set r, and as partial correlations controlling for
log task length (a known nuisance in the coinflip condition).

Usage: uv run python analyze.py outputs/<run_id>            (local run dir)
       uv run python analyze.py entity/project/<run_id> [--update-wandb]
"""

import argparse
import collections
import csv
import glob
import gzip
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

N_SPLITS = 50
N_BOOTSTRAP = 2000
NUMERIC = ("logp_first", "logp_second", "log_odds", "outcome_mass", "prompt_tokens")
BASELINE_STAGE = "base"
# Tasks left out of the "benign" correlations.
HARM_TOPICS = {"harmful_request"}
HARM_ORIGINS = {"BAILBENCH"}
# Categorical slots 1-5 of the dataviz reference palette, in fixed order, plus
# a marker per origin so identity never rests on color alone.
ORIGIN_STYLE = {
    "MATH": ("#2a78d6", "o"), "ALPACA": ("#eb6834", "s"), "WILDCHAT": ("#1baf7a", "^"),
    "STRESS_TEST": ("#eda100", "D"), "BAILBENCH": ("#e87ba4", "v"),
}


def pair_effects(rows):
    """{(stage, condition): {pairs: [(a, b)] with a < b, effect: array, bias: array}},
    keys in order of first appearance (i.e. config order)."""
    L = collections.defaultdict(dict)
    for r in rows:
        L[(r["stage"], r["condition"])][(r["first"], r["second"])] = r["log_odds"]
    out = {}
    for key, d in L.items():
        pairs = sorted(p for p in d if p[0] < p[1])
        fwd = np.array([d[p] for p in pairs])
        rev = np.array([d[(b, a)] for a, b in pairs])  # KeyError: an order is missing
        out[key] = dict(pairs=pairs, effect=(fwd - rev) / 2, bias=(fwd + rev) / 2)
    return out


def index_pairs(ids, pairs):
    idx = {t: i for i, t in enumerate(ids)}
    return np.array([idx[a] for a, _ in pairs]), np.array([idx[b] for _, b in pairs])


def fit(n, ia, ib, effect):
    """Least-squares u (length n, mean zero) with u[ia] - u[ib] ~ effect, solved via the
    n x n normal equations (the comparison graph's Laplacian), not the pairs x n design."""
    lap = np.zeros((n, n))
    np.add.at(lap, (ia, ia), 1)
    np.add.at(lap, (ib, ib), 1)
    np.add.at(lap, (ia, ib), -1)
    np.add.at(lap, (ib, ia), -1)
    rhs = np.zeros(n)
    np.add.at(rhs, ia, effect)
    np.add.at(rhs, ib, -effect)
    u = np.linalg.lstsq(lap, rhs, rcond=None)[0]
    return u - u.mean()


def fit_utilities(ids, eff):
    return fit(len(ids), *index_pairs(ids, eff["pairs"]), eff["effect"])


def additive_r2(ids, eff, u):
    """Share of sum(effect^2) explained by u_a - u_b (uncentered: the fit has no intercept)."""
    ia, ib = index_pairs(ids, eff["pairs"])
    e = eff["effect"]
    return float(1 - ((e - (u[ia] - u[ib])) ** 2).sum() / (e ** 2).sum())


def define_measures(keys, baseline=BASELINE_STAGE):
    """{name: {(stage, condition): coefficient}}: one measure per (stage, condition),
    plus stage-minus-baseline shifts for each condition both were run in."""
    measures = {"/".join(k): {k: 1.0} for k in keys}
    for stage, cond in keys:
        if stage != baseline and (baseline, cond) in keys:
            measures[f"{stage}-{baseline}/{cond}"] = {(stage, cond): 1.0, (baseline, cond): -1.0}
    return measures


def combine(measure, utils_by_key):
    return sum(c * utils_by_key[k] for k, c in measure.items())


def pearson(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    return pearson(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))


def partial_r(x, y, z):
    """Pearson r of x and y with z partialled out of both."""
    rxy, rxz, ryz = pearson(x, y), pearson(x, z), pearson(y, z)
    return (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))


def split_half(ids, effects, measures, rng):
    """{name: (mean split-half r, Spearman-Brown reliability)} for every measure.
    Each split puts every task pair in one half, identically for all (stage, condition)."""
    all_pairs = sorted({p for eff in effects.values() for p in eff["pairs"]})
    pos = {p: k for k, p in enumerate(all_pairs)}
    prepped = {key: (np.array([pos[p] for p in eff["pairs"]]), *index_pairs(ids, eff["pairs"]), eff["effect"])
               for key, eff in effects.items()}
    rs = collections.defaultdict(list)
    for _ in range(N_SPLITS):
        in_first = rng.permutation(len(all_pairs)) < len(all_pairs) // 2
        halves = []
        for h in (True, False):
            halves.append({})
            for key, (g, ia, ib, e) in prepped.items():
                m = in_first[g] == h
                halves[-1][key] = fit(len(ids), ia[m], ib[m], e[m])
        for name, meas in measures.items():
            rs[name].append(pearson(combine(meas, halves[0]), combine(meas, halves[1])))
    return {name: (float(np.mean(r)), 2 * np.mean(r) / (1 + np.mean(r))) for name, r in rs.items()}


def bootstrap_r(x, y, rng):
    """95% CI of Pearson r, resampling tasks."""
    n = len(x)
    rs = [pearson(x[s], y[s]) for s in (rng.integers(0, n, n) for _ in range(N_BOOTSTRAP))]
    return tuple(np.percentile(rs, [2.5, 97.5]))


def scatter(name_x, u_x, name_y, u_y, origins, stats):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for origin, (color, marker) in ORIGIN_STYLE.items():
        m = np.array([o == origin for o in origins])
        if m.any():
            ax.scatter(u_x[m], u_y[m], s=36, c=color, marker=marker, label=origin.lower(),
                       edgecolors="white", linewidths=0.8, zorder=3)
    ax.axhline(0, color="#cccccc", lw=0.8, zorder=1)
    ax.axvline(0, color="#cccccc", lw=0.8, zorder=1)
    ax.set_xlabel(f"{name_x} utility (log-odds)")
    ax.set_ylabel(f"{name_y} utility (log-odds)")
    ax.set_title(f"r = {stats['pearson']:.2f} [{stats['pearson_ci_low']:.2f}, {stats['pearson_ci_high']:.2f}], "
                 f"Spearman {stats['spearman']:.2f}, benign r = {stats['pearson_benign']:.2f}, n = {len(u_x)}",
                 fontsize=10, color="#444444")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def fmt(v, width=8, prec=2):
    return f"{v:{width}.{prec}f}" if v is not None else f"{'-':>{width}}"


def analyze(rows, tasks, out_dir, seed=0, stage_order=None):
    """Compute everything, write CSVs and PNGs to out_dir, and return what to log to W&B.
    stage_order (e.g. the config's model order) sets the order of stages in all outputs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    by_id = {t["id"]: t for t in tasks}
    effects = pair_effects(rows)
    if stage_order:
        effects = dict(sorted(effects.items(), key=lambda kv: (stage_order.index(kv[0][0]), kv[0][1])))
    ids = sorted({t for eff in effects.values() for p in eff["pairs"] for t in p})
    origins = [by_id[t]["origin"] for t in ids]
    benign = np.array([by_id[t]["topic"] not in HARM_TOPICS and by_id[t]["origin"] not in HARM_ORIGINS for t in ids])
    log_len = np.log([len(by_id[t]["prompt"]) for t in ids])

    utils_by_key = {k: fit_utilities(ids, eff) for k, eff in effects.items()}
    measures = define_measures(list(effects))
    utils = {name: combine(m, utils_by_key) for name, m in measures.items()}
    reliability = split_half(ids, effects, measures, rng)

    summary = {}
    print(f"\n{'measure':24} {'pairs':>6} {'util sd':>8} {'slot bias':>10} {'add. R2':>8} "
          f"{'split-half':>11} {'reliab.':>8} {'min mass':>9} {'r(loglen)':>10}")
    for name, m in measures.items():
        u, (r_half, rel) = utils[name], reliability[name]
        s = {"utility_sd": float(u.std()), "split_half_r": r_half, "reliability": rel,
             "length_r": pearson(u, log_len)}
        if len(m) == 1:
            (key,) = m
            eff = effects[key]
            masses = [r["outcome_mass"] for r in rows if (r["stage"], r["condition"]) == key]
            s.update(n_pairs=len(eff["pairs"]), slot_bias_mean=float(eff["bias"].mean()),
                     slot_bias_sd=float(eff["bias"].std()), additive_r2=additive_r2(ids, eff, u),
                     outcome_mass_mean=float(np.mean(masses)), outcome_mass_min=float(np.min(masses)))
        summary.update({f"{name}/{k}": v for k, v in s.items()})
        print(f"{name:24} {s.get('n_pairs', '-'):>6} {fmt(s['utility_sd'])} {fmt(s.get('slot_bias_mean'), 10)} "
              f"{fmt(s.get('additive_r2'))} {fmt(r_half, 11)} {fmt(rel)} {fmt(s.get('outcome_mass_min'), 9, 3)} "
              f"{fmt(s['length_r'], 10)}")

    corr_rows, figures = [], {}
    for nx, ny in itertools.combinations(measures, 2):
        if set(measures[nx]) & set(measures[ny]):
            continue  # a shift vs one of its own components is correlated by construction
        x, y = utils[nx], utils[ny]
        rel_x, rel_y = reliability[nx][1], reliability[ny][1]
        lo, hi = bootstrap_r(x, y, rng)
        stats = dict(
            x=nx, y=ny, pearson=pearson(x, y), pearson_ci_low=lo, pearson_ci_high=hi, spearman=spearman(x, y),
            # Undefined when either reliability is <= 0 (pure noise).
            pearson_disattenuated=pearson(x, y) / np.sqrt(rel_x * rel_y) if min(rel_x, rel_y) > 0 else float("nan"),
            n_benign=int(benign.sum()),
            pearson_benign=pearson(x[benign], y[benign]) if benign.sum() >= 3 else float("nan"),
            spearman_benign=spearman(x[benign], y[benign]) if benign.sum() >= 3 else float("nan"),
            pearson_partial_length=partial_r(x, y, log_len),
            sd_ratio_y_over_x=float(y.std() / x.std()),
        )
        corr_rows.append(stats)
        tag = f"{nx}_vs_{ny}".replace("/", "-")
        fig = scatter(nx, x, ny, y, origins, stats)
        fig.savefig(out_dir / f"scatter_{tag}.png", dpi=150)
        plt.close(fig)
        figures[f"scatter/{tag}"] = out_dir / f"scatter_{tag}.png"
    print(f"\n{'x':24} {'y':24} {'pearson':>8} {'95% CI':>15} {'spearman':>9} {'disatt.':>8} "
          f"{'benign r':>9} {'r|len':>6} {'sd y/x':>7}")
    for c in corr_rows:
        print(f"{c['x']:24} {c['y']:24} {c['pearson']:8.2f} [{c['pearson_ci_low']:5.2f},{c['pearson_ci_high']:5.2f}] "
              f"{c['spearman']:9.2f} {c['pearson_disattenuated']:8.2f} {c['pearson_benign']:9.2f} "
              f"{c['pearson_partial_length']:6.2f} {c['sd_ratio_y_over_x']:7.2f}")
    print(f"(benign = {int(benign.sum())} of {len(ids)} tasks, excluding topics {sorted(HARM_TOPICS)} "
          f"and origins {sorted(HARM_ORIGINS)})")

    util_rows = [
        dict(id=t, origin=by_id[t]["origin"], topic=by_id[t]["topic"], n_chars=len(by_id[t]["prompt"]),
             **{name: float(u[i]) for name, u in utils.items()}, prompt=by_id[t]["prompt"])
        for i, t in enumerate(ids)
    ]
    group_rows = []
    for field in ("origin", "topic"):
        for g in sorted({r[field] or "none" for r in util_rows}):
            members = [r for r in util_rows if (r[field] or "none") == g]
            group_rows.append(dict(field=field, group=g, n=len(members),
                                   **{name: float(np.mean([r[name] for r in members])) for name in utils}))
    primitive = [name for name, m in measures.items() if len(m) == 1]
    print(f"\nMean utility by origin:\n{'origin':12} {'n':>3} " + " ".join(f"{name:>20}" for name in primitive))
    for g in (g for g in group_rows if g["field"] == "origin"):
        print(f"{g['group']:12} {g['n']:3d} " + " ".join(f"{g[name]:20.2f}" for name in primitive))

    for name, table in (("utilities", util_rows), ("correlations", corr_rows), ("group_means", group_rows)):
        with open(out_dir / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(table[0]))
            w.writeheader()
            w.writerows(table)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\nWrote analysis to {out_dir}")
    return dict(summary=summary, utilities=util_rows, correlations=corr_rows, group_means=group_rows, figures=figures)


def log_to_wandb(run, result):
    import wandb
    run.summary.update(result["summary"])
    for c in result["correlations"]:
        for k in ("pearson", "spearman", "pearson_disattenuated", "pearson_ci_low", "pearson_ci_high",
                  "pearson_benign", "spearman_benign", "pearson_partial_length"):
            run.summary[f"corr/{c['x']}_vs_{c['y']}/{k}"] = c[k]
    log = {name: wandb.Table(columns=list(result[name][0]), data=[list(r.values()) for r in result[name]])
           for name in ("utilities", "correlations", "group_means")}
    log.update({name: wandb.Image(str(path)) for name, path in result["figures"].items()})
    run.log(log)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_results(run_dir):
    """Rows from a run dir: results/*.csv.gz (one per stage/condition) or a single results.csv.
    top_tokens is dropped to keep memory down; it is only needed for spot checks."""
    run_dir = Path(run_dir)
    paths = sorted((run_dir / "results").glob("*.csv.gz")) or [run_dir / "results.csv"]
    rows = []
    for path in paths:
        with (gzip.open(path, "rt", newline="") if path.suffix == ".gz" else open(path, newline="")) as f:
            for r in csv.DictReader(f):
                r.pop("top_tokens", None)
                for k in NUMERIC:
                    r[k] = float(r[k])
                rows.append(r)
    return rows


def fetch_wandb(run, out_dir):
    """Download a W&B run's result files into out_dir and return its rows."""
    files = [f for f in run.files() if f.name.startswith("results/") and f.name.endswith(".csv.gz")]
    files = files or [f for f in run.files() if f.name == "results.csv"]
    if files:
        for f in files:
            f.download(root=str(out_dir), replace=True)
        return load_results(out_dir)
    # Runs from before per-condition files logged everything as one W&B table.
    art = next(a for a in run.logged_artifacts() if a.type == "run_table" and "results" in a.name)
    table = json.loads(Path(glob.glob(f"{art.download()}/**/*.table.json", recursive=True)[0]).read_text())
    rows = [dict(zip(table["columns"], r)) for r in table["data"]]
    for r in rows:
        for k in NUMERIC:
            r[k] = float(r[k])
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source", help="local run dir (outputs/<run_id>) or W&B run path entity/project/run_id")
    p.add_argument("--update-wandb", action="store_true", help="log the analysis to the W&B run")
    args = p.parse_args()

    run = None
    if Path(args.source).is_dir():
        src = Path(args.source)
        cfg = yaml.safe_load(open(src / "config.yaml"))
        rows = load_results(src)
    else:
        import wandb
        run = wandb.Api().run(args.source)
        cfg = dict(run.config)
        src = Path("outputs") / run.id
        rows = fetch_wandb(run, src)
    result = analyze(rows, read_jsonl(cfg["tasks_file"]), src / "analysis",
                     stage_order=[m["stage"] for m in cfg["models"]])
    if args.update_wandb:
        if run is None:
            raise SystemExit("--update-wandb needs a W&B run path")
        import wandb
        live = wandb.init(entity=run.entity, project=run.project, id=run.id, resume="must")
        log_to_wandb(live, result)
        live.finish()


if __name__ == "__main__":
    main()
