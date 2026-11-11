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
With runs of two or more models, also writes outputs/coin_report/summary.csv and figures/coin_report_models.png.
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
    from report import fetch_wandb  # the analysis table (analyze.fetch_wandb fetches raw results instead)
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


LABEL = {"gemma4-31b": "Gemma 4 31B", "llama-70b": "Llama 3.3 70B", "glm-4.5-air": "GLM-4.5-Air", "olmo3.1-32b": "OLMo 3.1 32B"}
SUMMARY = ("r(report, coin)", "r(report, Assistant (sweep A_i - A_j))", "r(coin, sweep coin (C_i - C_j))", "swap: r(d, coin)",
           "swap: share", "kappa (swap+ctrl)", "lambda (swap+ctrl)", "swap_early: r(d, coin)", "swap_late: r(d, coin)",
           "ctrl: r(d, coin)", "swap_line: r(d, coin)", "swap_line: share", "kappa (swap_line+ctrl_line)")


def analyze_run(path):
    """(family, report, D, est, lo, hi) of one rows.jsonl; prints the run's full report."""
    import wandb
    family, run_id = path.parts[-3], path.parts[-2]
    report = wandb.Api().run(f"halrockwell-mats/tournament/{run_id}").config.get("report", "coin_report")
    D = load(path)
    srows = sweep_rows(family)
    U = {k: np.array([float(r[k]) for r in srows]) for k in ("instruct/assistant", "instruct/coinflip")}
    pos = {r["id"]: k for k, r in enumerate(srows)}
    ia, ib = np.array([pos[t] for t in D["a"]]), np.array([pos[t] for t in D["b"]])
    extra = {"Assistant (sweep A_i - A_j)": U["instruct/assistant"][ia] - U["instruct/assistant"][ib],
             "sweep coin (C_i - C_j)": U["instruct/coinflip"][ia] - U["instruct/coinflip"][ib]}
    n = len(D["c"])
    est = stats(D, np.arange(n), extra)
    rng = np.random.default_rng(0)
    boot = [stats(D, rng.integers(0, n, n), extra) for _ in range(N_BOOT)]
    lo = {k: np.percentile([b[k] for b in boot], 2.5) for k in est}
    hi = {k: np.percentile([b[k] for b in boot], 97.5) for k in est}
    print(f"\n{path}: {family}, {report} ({n} pairs)")
    print(f"  median mass: coin {np.median(D['coin_mass']):.2f}, " +
          ", ".join(f"{x} {np.median(m):.2f}" for x, m in D["mass"].items()))
    print(f"  heads bias (mean log-odds over prompts): coin {D['coin_o'].mean():+.2f}, report "
          + ", ".join(f"{x} {m.mean():+.2f}" for x, m in D["r_o"].items()))
    print(f"  self-patch |d|: median {np.median(np.abs(D['r_o']['self'] - D['r_o']['none'])):.2g}, "
          f"max {np.abs(D['r_o']['self'] - D['r_o']['none']).max():.2g}")
    print(f"  SD: coin {D['c'].std():.2f}, report {D['r']['none'].std():.2f}")
    for k, v in est.items():
        print(f"  {k:42} {v:+.3f} [{lo[k]:+.3f}, {hi[k]:+.3f}]")
    plot(D, f"{family}, {report}", Path(f"figures/coin_report_pilot_{family}_{report}.png"))
    return family, report, D, est, lo, hi


def plot_models(rows, path):
    """Per answer (heads/tails, the primary wording; Task A/B): each model's share through " up" and through the whole line."""
    panels = (("heads/tails answer", "primary"), ("Task A / B answer", "coin_report_ab"))
    families = [f for f in LABEL if any(r["family"] == f for r in rows)]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, (title, which) in zip(axes, panels):
        for k, (stat, name) in enumerate((("swap: share", '" up"'), ("swap_line: share", "the whole coin line"))):
            for i, f in enumerate(families):
                (r,) = [r for r in rows if r["family"] == f and (r["primary"] and question(r["report"]) == "coin_report"
                                                                  if which == "primary" else r["report"] == which)]
                e, lo, hi = 100 * r[stat], 100 * r[stat + "_lo"], 100 * r[stat + "_hi"]
                ax.errorbar(i + (k - 0.5) * 0.25, e, yerr=[[e - lo], [hi - e]], color=plotstyle.SERIES[k], marker="os"[k],
                            ms=7, lw=0, elinewidth=1.5, mec="white", mew=0.8, label=f"swap {name}" if i == 0 else None)
        ax.axhline(0, color=RULE, lw=0.8)
        ax.set_xticks(range(len(families)), [LABEL[f] for f in families], fontsize=9)
        ax.set_title(title, color=INK, fontsize=10)
        style(ax)
    axes[0].set_ylabel("share of the answer's coin-aligned part reversed (%)", color=INK_2, fontsize=9)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, frameon=False, fontsize=9, loc="lower center", ncol=2)
    fig.suptitle("How much of the Assistant's reported preference runs through the coin's \" up\"", color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    plotstyle.save(fig, path.stem, path.parent)


def question(report):
    """The question a report condition asks, whatever its prefill: coin_report_short_bold -> coin_report_short."""
    return report.removesuffix("_bold")


def mark_primary(rows, group_of=question):
    """Per model and question (group_of(report)), the primary wording: the prefill with the highest median outcome mass, if
    that's at least 0.2 (pre-registered)."""
    for key in {(r["family"], group_of(r["report"])) for r in rows}:
        group = [r for r in rows if (r["family"], group_of(r["report"])) == key]
        best = max(group, key=lambda r: r["mass"])
        for r in group:
            r["primary"] = r is best and r["mass"] >= 0.2


def main():
    plotstyle.setup()
    results = [analyze_run(Path(p)) for p in sys.argv[1:]]
    if len({r[0] for r in results}) < 2:
        return
    rows = []
    for family, report, D, est, lo, hi in results:
        mass = float(np.median(D["mass"]["none"]))
        row = dict(family=family, report=report, mass=mass, coin_sd=float(D["c"].std()))
        for k in SUMMARY:
            row |= {k: est[k], k + "_lo": lo[k], k + "_hi": hi[k]}
        rows.append(row)
    mark_primary(rows)
    from openers import write_csv
    write_csv(Path("outputs/coin_report/summary.csv"), rows)
    print("\nSummary (primary heads/tails wording and A/B):")
    for r in sorted(rows, key=lambda r: (list(LABEL).index(r["family"]), r["report"])):
        if r["primary"] or r["report"] == "coin_report_ab":
            print(f"  {LABEL[r['family']]:14} {r['report']:17} mass {r['mass']:.2f}  coin SD {r['coin_sd']:.2f}  " + "  ".join(
                f"{k} {r[k]:+.3f} [{r[k + '_lo']:+.3f}, {r[k + '_hi']:+.3f}]" for k in
                ("swap: r(d, coin)", "swap: share", "kappa (swap+ctrl)", "swap_early: r(d, coin)", "swap_late: r(d, coin)",
                 "swap_line: share")))
    plot_models(rows, Path("figures/coin_report_models.png"))


if __name__ == "__main__":
    main()
