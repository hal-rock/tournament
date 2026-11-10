"""Analysis of report_patch.py runs: does the Assistant's reported preference read the coin's " up"? (README, "Does the
Assistant read the coin's ' up'?")

Per pair {i, j} (order 0: heads = Task A = i; order 1: swapped), in log-odds of " heads" - " tails":
    c_o       the coin at " up" in order o (unpatched);  c = (c_0 - c_1) / 2, positive = the coin favors task i
    r_o^X     the Assistant's reported preferred outcome with arm X (none = unpatched);  r^X = (r^X_0 - r^X_1) / 2
    d^X       r^X - r^none, the arm's shift in the report
Averaging the two orders cancels any heads/tails bias, as everywhere in this repo.

If the report reads a preference stored at " up" in the same code as the coin (heads/tails), swapping " up" with the other
order's moves the report by -2 kappa c, where kappa is how strongly the report reads it:
    share = kappa / slope(r^none on c) = -slope(d^swap on c) / (2 slope(r^none on c))
is the part of the report's coin-aligned component that runs through " up": 0 none, 0.5 swapping cancels it, 1 swapping
reverses it. A patch that just disrupts the report would shrink it instead, d = -lambda r^none, which also gives a negative
slope on c. To separate the two, the per-prompt regression over the " up" arms (swap and ctrl)
    d_o = alpha_arm + kappa (c_src,o - c_o) + lambda r^none_o
asks whether the shift follows the coin difference the patch brings in (kappa > 0), beyond shrinkage (lambda < 0). For
swap, c_src - c is -2c; for ctrl ("OK, " opener, same mapping) it's the opener's small change in the coin.
CIs: 95% bootstrap over pairs (pilot); task-level fits come with the full run.

Usage: uv run python coin_report.py outputs/coin_report/<model>/<run>/rows.jsonl [...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import pearson
from matskit import io, plotstyle
from report import INK, INK_2, RULE, style
from report_patch import ARMS
from user_choice import FAMILIES, load_sweep

GLM_SWEEP = "i5vejs3b"  # GLM-4.5-Air's sweep run (the model-families sweep); the others come from user_choice.load_sweep


def sweep_rows(family):
    """utilities.csv rows of the model's sweep (the default Assistant and coin, final stage)."""
    if family in FAMILIES:
        return load_sweep(family)[0]
    assert family == "glm-4.5-air", family
    import csv
    import wandb
    from analyze import fetch_wandb
    analysis = Path("outputs") / GLM_SWEEP / "analysis"
    fetch_wandb(wandb.Api().run(f"halrockwell-mats/tournament/{GLM_SWEEP}"), analysis)
    with open(analysis / "utilities.csv", newline="") as fh:
        return list(csv.DictReader(fh))

N_BOOT = 2000


def slope(x, y):
    return float(np.mean((x - x.mean()) * (y - y.mean())) / x.var())


def load(path):
    """Arrays per pair (pairs with both orders only): c, c_ok, r[arm], and per order c_o, c_ok_o, r_o[arm]."""
    rows = io.read_jsonl(path)
    by = {}
    for r in rows:
        by.setdefault(r["pair"], {})[r["order"]] = r
    pairs = sorted(p for p, o in by.items() if set(o) == {0, 1})
    arms = ["none", *ARMS]
    o0, o1 = [by[p][0] for p in pairs], [by[p][1] for p in pairs]
    get = lambda rs, f: np.array([f(r) for r in rs])
    D = dict(pair=np.array(pairs), a=get(o0, lambda r: r["a"]), b=get(o0, lambda r: r["b"]))
    for k in ("coin", "coin_ok"):
        D[k + "_o"] = np.stack([get(o0, lambda r: r[k]), get(o1, lambda r: r[k])], 1)  # [N, 2]
    D["r_o"] = {x: np.stack([get(o0, lambda r: r["report"][x]), get(o1, lambda r: r["report"][x])], 1) for x in arms}
    D["mass"] = {x: np.concatenate([get(o0, lambda r: r["mass"][x]), get(o1, lambda r: r["mass"][x])]) for x in arms}
    D["coin_mass"] = np.concatenate([get(o0, lambda r: r["coin_mass"]), get(o1, lambda r: r["coin_mass"])])
    pair_eff = lambda m: (m[:, 0] - m[:, 1]) / 2
    D["c"], D["c_ok"] = pair_eff(D["coin_o"]), pair_eff(D["coin_ok_o"])
    D["r"] = {x: pair_eff(m) for x, m in D["r_o"].items()}
    return D


def regression(D, idx, arms=("swap", "ctrl")):
    """kappa, lambda of d_o = alpha_arm + kappa (c_src,o - c_o) + lambda r_o over orders and the given arms."""
    ys, dc, rr, dummies = [], [], [], []
    for k, arm in enumerate(arms):
        source = ARMS[arm][0]
        c = D["coin_o"][idx]
        c_src = c[:, ::-1] if source == "other" else D["coin_ok_o"][idx] if source == "ok" else c
        ys.append((D["r_o"][arm][idx] - D["r_o"]["none"][idx]).ravel())
        dc.append((c_src - c).ravel())
        rr.append(D["r_o"]["none"][idx].ravel())
        dummies.append(np.full(c.size, k))
    y, dc, rr, g = map(np.concatenate, (ys, dc, rr, dummies))
    X = np.column_stack([*(g == k for k in range(len(arms))), dc, rr]).astype(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta[-2], beta[-1]


def stats(D, idx, extra):
    """Named statistics over the pairs idx (a bootstrap resample or all)."""
    c, r0 = D["c"][idx], D["r"]["none"][idx]
    s = {"r(report, coin)": pearson(r0, c), "slope(report on coin)": slope(c, r0)}
    for name, v in extra.items():
        s[f"r(report, {name})"] = pearson(r0, v[idx])
        s[f"r(coin, {name})"] = pearson(c, v[idx])
    for arm in ARMS:
        d = D["r"][arm][idx] - r0
        s[f"{arm}: mean |d|"] = float(np.abs(d).mean())
        s[f"{arm}: r(d, coin)"] = pearson(d, c)
        s[f"{arm}: slope(d on coin)"] = slope(c, d)
        s[f"{arm}: share"] = -slope(c, d) / (2 * slope(c, r0))
        s[f"{arm}: r(report_patched, coin)"] = pearson(D["r"][arm][idx], c)
    s["ctrl: r(d, c_ok - c)"] = pearson(D["r"]["ctrl"][idx] - r0, D["c_ok"][idx] - c)
    s["kappa (swap+ctrl)"], s["lambda (swap+ctrl)"] = regression(D, idx)
    s["kappa (swap_line+ctrl_line)"], s["lambda (swap_line+ctrl_line)"] = regression(D, idx, ("swap_line", "ctrl_line"))
    return s


def plot(D, label, path):
    c = D["c"]
    panels = [("none", "Reported preference, unpatched"), ("swap", "Shift from swapping \" up\""),
              ("ctrl", "Shift from the \"OK, \" control"), ("swap_line", "Shift from swapping the whole line")]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 3.8))
    for ax, (arm, title) in zip(axes, panels):
        y = D["r"]["none"] if arm == "none" else D["r"][arm] - D["r"]["none"]
        ax.scatter(c, y, s=8, alpha=0.5, color=plotstyle.SERIES[0], edgecolors="none")
        xs = np.array([c.min(), c.max()])
        ax.plot(xs, np.polyval(np.polyfit(c, y, 1), xs), color=INK_2, lw=1.2)
        ax.axhline(0, color=RULE, lw=0.8)
        ax.axvline(0, color=RULE, lw=0.8)
        ax.set_title(f"{title}\nr = {pearson(c, y):+.2f}", fontsize=10, color=INK)
        ax.set_xlabel("coin at \" up\" (nats, pair effect)", fontsize=9, color=INK_2)
        ax.set_ylabel("report (nats)" if arm == "none" else "report shift (nats)", fontsize=9, color=INK_2)
        style(ax)
    fig.suptitle(f"{label}: the Assistant's reported preferred outcome, and what patching \" up\" does to it", color=INK)
    fig.tight_layout()
    plotstyle.save(fig, path.stem, path.parent)


def main():
    plotstyle.setup()
    for p in sys.argv[1:]:
        path = Path(p)
        family, run_id = path.parts[-3], path.parts[-2]
        import wandb
        report = wandb.Api().run(f"halrockwell-mats/tournament/{run_id}").config.get("report", "coin_report")
        D = load(path)
        srows = sweep_rows(family)
        U = {k: np.array([float(r[k]) for r in srows]) for k in ("instruct/assistant", "instruct/coinflip")}
        ids = [r["id"] for r in srows]
        pos = {t: k for k, t in enumerate(ids)}
        ia, ib = np.array([pos[t] for t in D["a"]]), np.array([pos[t] for t in D["b"]])
        extra = {"Assistant (sweep A_i - A_j)": U["instruct/assistant"][ia] - U["instruct/assistant"][ib],
                 "sweep coin (C_i - C_j)": U["instruct/coinflip"][ia] - U["instruct/coinflip"][ib]}
        n = len(D["c"])
        est = stats(D, np.arange(n), extra)
        rng = np.random.default_rng(0)
        boot = [stats(D, rng.integers(0, n, n), extra) for _ in range(N_BOOT)]
        print(f"\n{path}: {family}, {report} ({n} pairs)")
        print(f"  median mass: coin {np.median(D['coin_mass']):.2f}, " +
              ", ".join(f"{x} {np.median(m):.2f}" for x, m in D["mass"].items()))
        print(f"  heads bias (mean log-odds over prompts): coin {D['coin_o'].mean():+.2f}, report "
              + ", ".join(f"{x} {m.mean():+.2f}" for x, m in D["r_o"].items()))
        print(f"  self-patch |d|: median {np.median(np.abs(D['r_o']['self'] - D['r_o']['none'])):.2g}, "
              f"max {np.abs(D['r_o']['self'] - D['r_o']['none']).max():.2g}")
        print(f"  SD: coin {D['c'].std():.2f}, report {D['r']['none'].std():.2f}")
        for k, v in est.items():
            lo, hi = np.percentile([b[k] for b in boot], [2.5, 97.5])
            print(f"  {k:42} {v:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        plot(D, f"{family}, {report}", Path(f"figures/coin_report_pilot_{family}_{report}.png"))


if __name__ == "__main__":
    main()
