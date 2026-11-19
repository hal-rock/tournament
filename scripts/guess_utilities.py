"""Per-task utilities of the Assistant's guess at the withheld coin, against its own choice (Hal, 2026-11-19; README,
"Shorter questions", "The guess's outcome lean").

The guess (coin_predict_think_bold) was scored only on the report runs' 1,000 pilot pairs (both orders), so its utilities are
fit from those pairs alone, the way analyze.py fits the sweep's: least squares on each pair's effect (half the difference of
the two orders' log-odds, which cancels the outcome lean). Each task sits in about 7 of the pairs, so the fit is much noisier
than the sweep's 15,000-pair utilities; the split-half reliability (pairs halved, Spearman-Brown) is its ceiling. For
comparison, the user-turn coin at " up", read in the same forward passes, fit from the same pairs. The Assistant's choice
utilities are the model's sweep ones (15,000 pairs).

    uv run python scripts/guess_utilities.py [model key, default deepseek-v4-flash]
      -> figures/<key>/guess_utilities.png
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze import HARM_ORIGINS, ORIGIN_STYLE, bootstrap_r, fit, pearson, spearman  # noqa: E402
from matskit import plotstyle  # noqa: E402

GUESS_RUNS = {"gemma4-31b": "7vblk8tv", "llama-70b": "ly2ng7l0", "glm-4.5-air": "or7e7o6a", "llama-405b": "01la3hzo",
              "nemotron4-340b": "a22g64uf", "deepseek-v3": "r7zjpyo0", "kimi-k2": "081zwk4v",
              "deepseek-v4-flash": "kkde34ba"}
SWEEP = {"gemma4-31b": "outputs/jrl62npf", "llama-70b": "outputs/tvlqj8a0", "glm-4.5-air": "outputs/i5vejs3b"}
LABELS = {"gemma4-31b": "Gemma 4 31B", "llama-70b": "Llama 3.3 70B", "glm-4.5-air": "GLM-4.5-Air",
          "llama-405b": "Llama 3.1 405B", "nemotron4-340b": "Nemotron-4 340B", "deepseek-v3": "DeepSeek V3",
          "kimi-k2": "Kimi K2", "deepseek-v4-flash": "DeepSeek V4 Flash"}
N_SPLITS = 200


def pair_effects(key, field):
    """[(a, b, effect toward a)] for every pilot pair, from the guess run's unpatched pass ("report") or its coin at " up"."""
    rows = {(r["a"], r["b"]): r for r in map(json.loads, open(Path("outputs/coin_report") / key / GUESS_RUNS[key] /
                                                                "rows.jsonl"))}
    get = (lambda r: r["coin"]) if field == "coin" else (lambda r: r["report"]["none"])
    return [(a, b, (get(rows[a, b]) - get(rows[b, a])) / 2) for (a, b) in rows if a < b]


def fit_pairs(ids, effects):
    idx = {t: i for i, t in enumerate(ids)}
    ia, ib, e = (np.array(v) for v in zip(*[(idx[a], idx[b], d) for a, b, d in effects]))
    return fit(len(ids), ia, ib, e)


def split_half(ids, effects, rng):
    """Spearman-Brown reliability of utilities fit from random halves of the pairs."""
    rs = []
    for _ in range(N_SPLITS):
        first = rng.permutation(len(effects)) < len(effects) // 2
        halves = [fit_pairs(ids, [e for e, f in zip(effects, first) if f == h]) for h in (True, False)]
        rs.append(pearson(*halves))
    r = float(np.mean(rs))
    return 2 * r / (1 + r)


def main(key):
    plotstyle.setup()
    sweep = Path(SWEEP.get(key, f"outputs/replication/{key}")) / "analysis" / "utilities.csv"
    srows = list(csv.DictReader(open(sweep)))
    ids = [r["id"] for r in srows]
    origins = [r["origin"] for r in srows]
    A = np.array([float(r["instruct/assistant"]) for r in srows])
    sweep_coin = np.array([float(r["instruct/coinflip"]) for r in srows])
    benign = np.array([o not in HARM_ORIGINS for o in origins])

    guess, coin = pair_effects(key, "report"), pair_effects(key, "coin")
    degree = np.zeros(len(ids), int)
    idx = {t: i for i, t in enumerate(ids)}
    for a, b, _ in guess:
        degree[idx[a]] += 1
        degree[idx[b]] += 1
    lap = np.diag(degree).astype(float)
    for a, b, _ in guess:
        lap[idx[a], idx[b]] -= 1
        lap[idx[b], idx[a]] -= 1
    n_components = int(np.sum(np.linalg.eigvalsh(lap) < 1e-8))
    print(f"{LABELS[key]}: {len(guess)} pairs over {len(ids)} tasks, comparisons per task min {degree.min()} / median "
          f"{int(np.median(degree))} / max {degree.max()}, connected components {n_components}")
    keep = degree > 0  # a task in no pilot pair gets a fitted 0, not a measurement
    assert n_components == 1 + int(np.sum(~keep)), "the compared tasks aren't one connected graph"
    print(f"  dropped (in no pilot pair): {[t for t, k in zip(ids, keep) if not k]}")
    ids = [t for t, k in zip(ids, keep) if k]
    origins = [o for o, k in zip(origins, keep) if k]
    A, sweep_coin, benign = A[keep], sweep_coin[keep], benign[keep]

    rng = np.random.default_rng(0)
    panels = []
    for name, effects in (("the Assistant's guess at the coin", guess), ("the user-turn coin at \" up\"", coin)):
        u = fit_pairs(ids, effects)
        lo, hi = bootstrap_r(A, u, rng)
        rel = split_half(ids, effects, rng)
        st = dict(r=pearson(A, u), lo=lo, hi=hi, rho=spearman(A, u), benign=pearson(A[benign], u[benign]), rel=rel,
                  slope=float(np.polyfit(A, u, 1)[0]))
        print(f"  {name:36s} r {st['r']:.2f} [{lo:.2f}, {hi:.2f}]  Spearman {st['rho']:.2f}  benign r {st['benign']:.2f}  "
              f"split-half reliability {rel:.2f}  slope on A {st['slope']:.3f}")
        panels.append((name, u, st))
    print(f"  for reference, the sweep's user-turn coin (15,000 pairs): r with A {pearson(A, sweep_coin):.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9), gridspec_kw={"wspace": 0.28})
    for ax, (name, u, st) in zip(axes, panels):
        for origin, (color, marker) in ORIGIN_STYLE.items():
            m = np.array([o == origin for o in origins])
            ax.scatter(A[m], u[m], s=30, c=color, marker=marker, label=origin.lower().replace("_", " "),
                       edgecolors="white", linewidths=0.7, zorder=3)
        ax.axhline(0, color=plotstyle.BASELINE, lw=0.8, zorder=1)
        ax.axvline(0, color=plotstyle.BASELINE, lw=0.8, zorder=1)
        ax.set_xlabel(f"the Assistant's choice: utility (sweep, 15,000 pairs; {len(ids)} tasks)")
        ax.set_ylabel(f"utility from {len(guess):,} pairs (nats)")
        ax.set_title(f"{name[0].upper()}{name[1:]}", loc="left", fontsize=10.5, weight="bold")
        ax.text(0.02, 0.98, f"r = {st['r']:.2f} [{st['lo']:.2f}, {st['hi']:.2f}]\nSpearman {st['rho']:.2f}, "
                            f"benign r {st['benign']:.2f}\nsplit-half reliability {st['rel']:.2f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=8.5, color=plotstyle.INK2)
        ax.grid(False)
    axes[1].legend(loc="lower right", fontsize=8, frameon=False, title="task origin", title_fontsize=8)
    fig.suptitle(f"{LABELS[key]}: per-task utilities against the Assistant's own choice", x=0.01, ha="left", fontsize=11.5)
    out = Path("figures") / key / "guess_utilities.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out.resolve()}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "deepseek-v4-flash")
