"""Fit utility probes (PLAN_preference_probe.md, step 1): ridge regression of per-task Bradley-Terry utilities
on residual activations from probe_acts.py, at every block output, for each read site.

Sites (probe_acts.SITES) and how rows are built:
    single_eot, single_last           one row per task
    assistant_task, coinflip_task     two rows per task (slot A and slot B readouts, each averaged over the
                                      fixed partners), grouped by task in every split; a task's prediction is
                                      the mean of its two rows
    coinflip_label_b                  one row per task (" tails" with the task in slot A)

Fits, per (target, train set, site, layer):
    target     dpo (dpo/assistant utilities; the primary) or sft (sft/assistant)
    train set  all (the 300 tournament tasks) or benign (the 257 non-harmful ones; harm vs preference)
Each: features standardized over the fit's rows (unsupervised; see Gram), ridge in kernel form (n << d), alpha chosen by inner
5-fold CV on the training fold; out-of-fold predictions from outer 5-fold CV give r_cv. The final probe is
refit on the whole train set with alpha from 5-fold CV there, and stored in the raw residual basis:
    score(h) = h @ w + b.

Metrics in summary.json, per layer:
    r_cv            Pearson r of out-of-fold predictions with the target (over train-set tasks)
    r_partial_len   the same, partialling log n_chars out of both
    r_loo_origin    r of predictions for each source origin when that origin is held out (mean within-origin r,
                    and pooled r over all out-of-origin predictions)
    r_null          r_cv with shuffled targets (3 permutations; dpo/all fits only)
    benign fits:    harm_auroc (final benign probe's scores, harmful vs out-of-fold benign scores) and harm_d
                    (mean difference in SDs of the benign scores); std_acc = share of the 25 standard pairs whose
                    preferred task scores higher, and std_d (mean pref - dispref score, in benign SDs)

Usage (local CPU): uv run python probe_fit.py outputs/probe/dpo [--layers 0:64:2] [--sites ...]
Writes <dir>/probe<tag>.npz and <dir>/fit_summary<tag>.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from analyze import HARM_ORIGINS, HARM_TOPICS
from probe_acts import SITES, as_float, load_acts
from probe_prompts import load_tasks

ALPHAS = np.logspace(-1, 6, 15)  # in units of the mean kernel diagonal (= d for standardized features)
UTILITIES = "outputs/d8ysbybo/analysis/utilities.csv"
TARGETS = {"dpo": "dpo/assistant", "sft": "sft/assistant"}


def pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def partial_r(a, b, z):
    """r(a, b) after regressing z (with intercept) out of both."""
    Z = np.column_stack([np.ones_like(z), z])
    res = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
    return pearson(res(a), res(b))


def auroc(pos, neg):
    """P(score of a random pos > random neg), ties count half."""
    pos, neg = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float((pos > neg).mean() + 0.5 * (pos == neg).mean())


def kfold(groups, k, seed):
    """k folds over unique groups: list of boolean test masks over rows."""
    uniq = np.unique(groups)
    perm = np.random.default_rng(seed).permutation(len(uniq))
    folds = np.array_split(uniq[perm], k)
    return [np.isin(groups, f) for f in folds]


class Gram:
    """Features standardized over all rows of one fit, and their Gram matrix, so every CV fit is an
    eigendecomposition of a sub-block (ridge in kernel form, n << d). The standardization uses no labels,
    only the rows' feature distribution, so held-out rows don't leak targets."""

    def __init__(self, X):
        self.mu, self.sd = X.mean(0), X.std(0)
        self.sd = np.where(self.sd > 0, self.sd, 1.0)
        self.Z = (X - self.mu) / self.sd
        self.K = self.Z @ self.Z.T
        self.scale = np.trace(self.K) / len(self.K)  # alphas are in units of the mean kernel diagonal

    def fit(self, train, y):
        return Ridge(self, np.flatnonzero(train) if train.dtype == bool else train, y)


class Ridge:
    def __init__(self, gram, train, y):
        self.g, self.train = gram, train
        self.ym = y.mean()
        self.lam, self.Q = np.linalg.eigh(gram.K[np.ix_(train, train)])
        self.Qy = self.Q.T @ (y - self.ym)

    def dual(self, alpha):
        return self.Q @ (self.Qy / (self.lam + alpha * self.g.scale))

    def predict(self, test, alphas):
        """[n_test, len(alphas)] predictions for row indices / mask `test`."""
        A = np.stack([self.dual(a) for a in alphas], 1)
        return self.g.K[np.ix_(np.flatnonzero(test) if test.dtype == bool else test, self.train)] @ A + self.ym

    def weights(self, alpha):
        """(w, b) in the raw basis: prediction for a raw row x = x @ w + b."""
        w = (self.g.Z[self.train].T @ self.dual(alpha)) / self.g.sd
        return w, self.ym - self.g.mu @ w


def choose_alpha(gram, train, y, groups, seed=0, k=5):
    """Alpha with the lowest inner-CV MSE over the rows in `train` (a boolean mask); y, groups are full-length."""
    idx = np.flatnonzero(train)
    err = np.zeros(len(ALPHAS))
    for test in kfold(groups[idx], k, seed):
        pred = gram.fit(idx[~test], y[idx[~test]]).predict(idx[test], ALPHAS)
        err += ((pred - y[idx[test], None]) ** 2).sum(0)
    return float(ALPHAS[np.argmin(err)])


def cv_predict(gram, y, groups, seed=0, k=5):
    """Out-of-fold predictions over all rows (nested alpha choice) and the alphas chosen per fold."""
    pred, alphas = np.zeros(len(y)), []
    for f, test in enumerate(kfold(groups, k, seed)):
        a = choose_alpha(gram, ~test, y, groups, seed=seed + 1 + f)
        pred[test] = gram.fit(~test, y[~test]).predict(test, [a])[:, 0]
        alphas.append(a)
    return pred, alphas


def by_task(pred, groups, task_index):
    """Mean prediction per task (rows grouped by task index), in task_index order."""
    return np.array([pred[groups == t].mean() for t in task_index])


class Data:
    def __init__(self, acts_dir, utilities=UTILITIES):
        self.dir = Path(acts_dir)
        self.task_ids = json.loads((self.dir / "tasks.json").read_text())
        meta = {t["id"]: t for t in load_tasks()}
        u = {r["id"]: r for r in csv.DictReader(open(utilities))}
        self.tour = np.array([t in meta for t in self.task_ids])  # tournament tasks (have utilities)
        self.idx = np.flatnonzero(self.tour)  # rows of the 300 tournament tasks
        self.harm = np.array([meta[t]["origin"] in HARM_ORIGINS or meta[t]["topic"] in HARM_TOPICS
                              for t in np.array(self.task_ids)[self.idx]])
        self.origin = np.array([meta[t]["origin"] for t in np.array(self.task_ids)[self.idx]])
        self.loglen = np.log([meta[t]["n_chars"] for t in np.array(self.task_ids)[self.idx]])
        self.y = {k: np.array([float(u[t][c]) for t in np.array(self.task_ids)[self.idx]]) for k, c in TARGETS.items()}
        self.std_pairs = [(self.task_ids.index(f"std_pref_{i}"), self.task_ids.index(f"std_dispref_{j}"))
                          for i in range(5) for j in range(5)] if "std_pref_0" in self.task_ids else []
        self._mm = {}

    def site(self, name):
        if name not in self._mm:
            self._mm[name] = load_acts(self.dir / f"{name}.npy")
        return self._mm[name]

    def rows(self, name, layer, tasks):
        """(X, groups) for tasks (row indices into task_ids): slotted sites give two rows per task."""
        bits = self.site(name)
        if SITES[name][2]:
            X = as_float(bits[tasks, :, layer]).reshape(2 * len(tasks), -1)  # task-major: (t0 a, t0 b, t1 a, ...)
            return X.astype(np.float64), np.repeat(tasks, 2)
        return as_float(bits[tasks, layer]).astype(np.float64), np.asarray(tasks)

    @property
    def n_layers(self):
        return self.site("single_eot").shape[1]


def fit_one(data, site, layer, target, train, n_null=3):
    sel = data.idx if train == "all" else data.idx[~data.harm]
    keep = np.ones(len(data.idx), bool) if train == "all" else ~data.harm
    y_task = data.y[target][keep]
    X, g = data.rows(site, layer, sel)
    y = np.repeat(y_task, 2) if SITES[site][2] else y_task
    gram = Gram(X)
    pred, alphas = cv_predict(gram, y, g)
    p_task = by_task(pred, g, sel)
    out = dict(r_cv=pearson(p_task, y_task), r_partial_len=partial_r(p_task, y_task, data.loglen[keep]),
               alpha_folds=alphas)
    if SITES[site][2]:  # per slot, from the same out-of-fold predictions
        for s, name in enumerate("ab"):
            out[f"r_cv_slot_{name}"] = pearson(pred[s::2], y_task)
    # leave one origin out
    origins = data.origin[keep]
    og = np.repeat(origins, 2) if SITES[site][2] else origins
    loo = np.zeros(len(y))
    within = {}
    for o in np.unique(origins):
        test = og == o
        a = choose_alpha(gram, ~test, y, g)
        loo[test] = gram.fit(~test, y[~test]).predict(test, [a])[:, 0]
        pt = by_task(loo[test], g[test], np.unique(g[test]))
        yt = y_task[origins == o]
        within[o] = pearson(pt, yt) if yt.std() > 0 else float("nan")
    out["r_loo_origin_within"] = within
    out["r_loo_origin_pooled"] = pearson(by_task(loo, g, sel), y_task)
    # null: shuffled targets (shuffle tasks, keeping both slot rows together)
    rng = np.random.default_rng(layer)
    nulls = []
    for _ in range(n_null):
        yp = rng.permutation(y_task)
        ypr = np.repeat(yp, 2) if SITES[site][2] else yp
        pn, _ = cv_predict(gram, ypr, g)
        nulls.append(pearson(by_task(pn, g, sel), yp))
    out["r_null"] = nulls
    # final probe on the whole train set
    everything = np.ones(len(y), bool)
    alpha = choose_alpha(gram, everything, y, g)
    w, b = gram.fit(everything, y).weights(alpha)
    out["alpha"] = alpha
    if train == "benign":
        Xh, gh = data.rows(site, layer, data.idx[data.harm])
        s_harm = by_task(Xh @ w + b, gh, data.idx[data.harm])
        sd = p_task.std()
        out["harm_auroc"] = auroc(p_task, s_harm)  # P(benign out-of-fold score > harmful score)
        out["harm_d"] = float((p_task.mean() - s_harm.mean()) / sd)
    if data.std_pairs:
        tasks = np.array(sorted({t for p in data.std_pairs for t in p}))
        Xs, gs = data.rows(site, layer, tasks)
        s = dict(zip(tasks, by_task(Xs @ w + b, gs, tasks)))
        diffs = np.array([s[p] - s[d] for p, d in data.std_pairs])
        out["std_acc"] = float((diffs > 0).mean())
        out["std_d"] = float(diffs.mean() / p_task.std())
    return out, w, b, (gram.mu, gram.sd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("probe_dir", help="outputs/probe/<stage> (holds acts/)")
    ap.add_argument("--layers", default="0:64:1", help="start:stop:step over blocks")
    ap.add_argument("--sites", nargs="+", default=list(SITES))
    ap.add_argument("--targets", nargs="+", default=["dpo", "sft"])
    ap.add_argument("--trains", nargs="+", default=["all", "benign"])
    ap.add_argument("--tag", default="", help="suffix for the output files (a quick subset fit beside the full one)")
    a = ap.parse_args()
    d = Path(a.probe_dir)
    data = Data(d / "acts")
    layers = list(range(*map(int, a.layers.split(":"))))
    layers = [l for l in layers if l < data.n_layers]
    D = data.site("single_eot").shape[-1]
    arrays, summary, t0 = {}, {}, time.time()
    for target in a.targets:
        for train in a.trains:
            for site in a.sites:
                key = f"{target}/{train}/{site}"
                W, B = np.full((data.n_layers, D), np.nan, np.float32), np.full(data.n_layers, np.nan)
                MU, SD = np.full((data.n_layers, D), np.nan, np.float32), np.full((data.n_layers, D), np.nan, np.float32)
                summary[key] = {}
                for l in layers:
                    # shuffled-target nulls only for the primary fit (they cost 3 more nested CVs per layer)
                    n_null = 3 if (target, train) == ("dpo", "all") else 0
                    res, w, b, (mu, sd) = fit_one(data, site, l, target, train, n_null=n_null)
                    W[l], B[l], MU[l], SD[l] = w, b, mu, sd
                    summary[key][l] = res
                arrays.update({f"{key}/w": W, f"{key}/b": B, f"{key}/mu": MU, f"{key}/sd": SD,
                               f"{key}/r_cv": np.array([summary[key].get(l, {}).get("r_cv", np.nan)
                                                        for l in range(data.n_layers)])})
                best = max(layers, key=lambda l: summary[key][l]["r_cv"])
                null = summary[key][best]["r_null"]
                print(f"{key}: best block {best} r_cv {summary[key][best]['r_cv']:.3f}"
                      + (f" (null max {max(null):.2f})" if null else "") + f"  [{(time.time() - t0) / 60:.1f} min]",
                      flush=True)
                np.savez(d / f"probe{a.tag}.npz", layers=np.array(layers), **arrays)
                (d / f"fit_summary{a.tag}.json").write_text(json.dumps(summary, indent=0))
    print(f"wrote {d / f'probe{a.tag}.npz'} and {d / f'fit_summary{a.tag}.json'}")


if __name__ == "__main__":
    main()
