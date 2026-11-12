"""Inputs for attributing OLMo 2 32B's broad preference bleedthrough to its SFT data (README, "OLMo 2 32B SFT data
attribution"), fixed before any model runs. No GPU, no model.

The coin's per-pair effect is c_ab = (L(a first) - L(b first)) / 2, L = log P(" heads") - log P(" tails") with the first task
as heads (analyze.pair_effects). With a fixed task score s, the coin's slope on s is linear in the c_ab, so over K sampled pairs

    f = sum_ab w_ab c_ab,   w_ab = (s_a - s_b) / (2 K_m),   K_m the sampled pairs both of whose tasks the metric covers,

and grad f is a weighted sum of per-pair gradients. s is the OLMo 2 SFT's own Assistant utility (as1txqy2), standardized over
the metric's tasks, at both checkpoints scored: the coin at SFT follows it at r 0.89. Metrics:
  benign       primary: the 257 non-harmful tasks (benign_odd / benign_even: alternate sampled pairs, for gradient noise)
  ordinary     ALPACA, MATH and WILDCHAT non-harmful tasks, Assistant utility within origin
  harm_gap     mean c of non-harmful minus harmful tasks (cross pairs only)
  t1, t2       benign on two disjoint halves of the non-harmful tasks
  perm1, perm2 benign with the utilities shuffled across tasks (nulls)
  heads_bias   mean (L(a first) + L(b first)) / 2, the placebo
  asst_benign  benign's weights on the Assistant's own per-pair choice (" A" vs " B" after "I'll choose Task")
  standard     ../coinflip's 25-pair effect (its 5 preferred and 5 harmful tasks), for comparison with its scores

Pairs are drawn from the tournament's 15,000 scored pairs, so each has stored base and SFT values for the load checks.
Writes data/olmo2_attr/ (ids, weights and stored values; no task text except the 10 standard tasks, ../coinflip's own):
  task_pairs.json, standard.json, sft_sample.json (../coinflip's OLMo 2 SFT sample, copied unchanged), sft_stats.json
and prints every metric on the sample against the same slope from the fitted utilities, at base and SFT.

    uv run --with pandas python olmo2_attr_prepare.py
"""

import gzip
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "olmo2_attr"
COINFLIP = ROOT.parent / "coinflip"
TASKS = ROOT / "data" / "tasks_n300_max500_seed0.jsonl"
RUNS = {"base": "5f7ymsdk", "sft": "as1txqy2"}
UTILITY = ("sft", "sft/assistant")  # the fixed task score: the SFT's Assistant
K = 6000
SEED = 20261112
COIN_METRICS = ["benign_odd", "benign_even", "ordinary", "harm_gap", "t1", "t2", "perm1", "perm2"]


def standardize(x):
    return (x - x.mean()) / x.std()


def load_tasks():
    tasks = pd.read_json(TASKS, lines=True).set_index("id")
    tasks["harmful"] = (tasks.origin == "BAILBENCH") | (tasks.topic == "harmful_request")
    return tasks


def stored(stage, condition):
    """{(first, second): log_odds} for one stored tournament condition."""
    path = ROOT / "outputs" / RUNS[stage] / "results" / f"{stage}__{condition}.csv.gz"
    df = pd.read_csv(path, usecols=["first", "second", "log_odds"], dtype={"first": str, "second": str})
    return dict(zip(zip(df["first"], df["second"]), df["log_odds"]))


def task_scores(tasks, A, rng):
    """name -> standardized score Series over the tasks that metric covers."""
    benign = tasks[~tasks.harmful]
    a = A.loc[benign.index]
    ordinary = benign[benign.origin.isin(["ALPACA", "MATH", "WILDCHAT"])]
    ao = A.loc[ordinary.index]
    s = {"benign": standardize(a), "ordinary": standardize(ao - ao.groupby(ordinary.origin).transform("mean"))}
    order = rng.permutation(len(benign))
    for name, idx in (("t1", benign.index[order[:len(benign) // 2]]), ("t2", benign.index[order[len(benign) // 2:]])):
        s[name] = standardize(a.loc[idx])
    for name in ("perm1", "perm2"):
        s[name] = pd.Series(rng.permutation(s["benign"].to_numpy()), index=benign.index)
    return s


def pair_weights(sample, scores, harmful):
    """Per sampled pair (a, b), each metric's weight on c_ab (see the module docstring)."""
    rows, counts = [], {}
    for n, (a, b) in enumerate(sample):
        w = {}
        for m, s in scores.items():
            if a in s.index and b in s.index:
                w[m] = float(s[a] - s[b]) / 2
        if harmful[a] != harmful[b]:
            w["harm_gap"] = 1.0 if harmful[b] else -1.0  # c with the non-harmful task first
        if "benign" in w:
            w["benign_odd" if n % 2 else "benign_even"] = w.pop("benign")
        for m in w:
            counts[m] = counts.get(m, 0) + 1
        rows.append(dict(a=a, b=b, w=w))
    for r in rows:
        r["w"] = {m: v / counts[m] for m, v in r["w"].items()}
    # benign itself is odd + even, each half normalized by the whole benign count, so benign = odd + even exactly
    n_benign = counts["benign_odd"] + counts["benign_even"]
    for r in rows:
        for half in ("benign_odd", "benign_even"):
            if half in r["w"]:
                r["w"][half] *= counts[half] / n_benign
    return rows, counts


def metric_value(rows, effects, metric):
    if metric == "benign":
        return metric_value(rows, effects, "benign_odd") + metric_value(rows, effects, "benign_even")
    return float(sum(r["w"].get(metric, 0.0) * effects[(r["a"], r["b"])] for r in rows))


def fitted_slope(utilities, col, score):
    coin = utilities[col].loc[score.index]
    return float((score * (coin - coin.mean())).sum() / (score * score).sum())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    tasks = load_tasks()
    util = {stage: pd.read_csv(ROOT / "outputs" / run / "analysis" / "utilities.csv").set_index("id") for stage, run in RUNS.items()}
    A = util[UTILITY[0]][UTILITY[1]].loc[tasks.index]
    scores = task_scores(tasks, A, rng)

    coin = {st: stored(st, "coinflip") for st in RUNS}
    asst = {st: stored(st, "assistant") for st in RUNS}
    scored_pairs = sorted((a, b) for (a, b) in coin["sft"] if a < b and (b, a) in coin["sft"])
    assert len(scored_pairs) == 15000, len(scored_pairs)
    pick = rng.choice(len(scored_pairs), K, replace=False)
    sample = [scored_pairs[i] for i in sorted(pick)]
    rows, counts = pair_weights(sample, scores, tasks.harmful.to_dict())

    eff = {}
    for st in RUNS:
        eff[("coin", st)] = {(a, b): (coin[st][(a, b)] - coin[st][(b, a)]) / 2 for a, b in sample}
        eff[("asst", st)] = {(a, b): (asst[st][(a, b)] - asst[st][(b, a)]) / 2 for a, b in sample}
        eff[("bias", st)] = {(a, b): (coin[st][(a, b)] + coin[st][(b, a)]) / 2 for a, b in sample}
    for r in rows:
        key = (r["a"], r["b"])
        r["stored"] = {st: dict(coin_ab=coin[st][key], coin_ba=coin[st][(r["b"], r["a"])],
                                asst_ab=asst[st][key], asst_ba=asst[st][(r["b"], r["a"])]) for st in RUNS}

    values = {}
    print("metric on the 6,000 sampled pairs vs the same slope from the fitted utilities (formula check):")
    for st in RUNS:
        v = {m: metric_value(rows, eff[("coin", st)], m) for m in COIN_METRICS + ["benign"]}
        v["heads_bias"] = float(np.mean(list(eff[("bias", st)].values())))
        v["asst_benign"] = metric_value(rows, eff[("asst", st)], "benign")
        values[st] = v
        col = f"{st}/coinflip"
        for m in ("benign", "ordinary", "t1", "t2", "perm1", "perm2"):
            print(f"  {st:4s} {m:11s} sample {v[m]:+.4f}  fitted {fitted_slope(util[st], col, scores[m]):+.4f}")
        u = util[st][col]
        print(f"  {st:4s} harm_gap    sample {v['harm_gap']:+.4f}  fitted {u[~tasks.harmful].mean() - u[tasks.harmful].mean():+.4f}")
        print(f"  {st:4s} asst_benign sample {v['asst_benign']:+.4f}  fitted {fitted_slope(util[st], f'{st}/assistant', scores['benign']):+.4f}")
        print(f"  {st:4s} heads_bias  sample {v['heads_bias']:+.4f}")
    print("change from base to SFT on the sample:", {m: round(values["sft"][m] - values["base"][m], 4) for m in values["sft"]})

    json.dump(dict(utility=list(UTILITY), seed=SEED, counts=counts, values=values, pairs=rows), open(OUT / "task_pairs.json", "w"))

    # ../coinflip's standard 25-pair effect: its task files, read as coinflip.load_tasks reads them.
    def load(path):
        text = open(path).read().replace("’", "'")
        return [" ".join(p.split()) for p in text.split("\n\n") if p.strip()]
    standard = dict(preferred=load(COINFLIP / "pref_tasks.txt"), dispreferred=load(COINFLIP / "nonpref_tasks.txt"),
                    expect_f={"base": 0.156, "sft": 0.979})  # ../coinflip RESULTS.md, OLMo 2 attribution load checks
    assert len(standard["preferred"]) == len(standard["dispreferred"]) == 5
    json.dump(standard, open(OUT / "standard.json", "w"), indent=1)

    shutil.copy(COINFLIP / "results" / "olmo2_attr" / "sft_sample.json", OUT / "sft_sample.json")
    stats = json.load(open(COINFLIP / "results" / "olmo2_attr" / "data_stats.json"))["sft"]
    json.dump(stats, open(OUT / "sft_stats.json", "w"), indent=1)
    print("pairs per metric:", counts)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
