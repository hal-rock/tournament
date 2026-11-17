"""Analysis of "Pushing the heads/tails lean at ' up' and around it" (README): reading gains, random-direction controls, the
transplant, the checks and the pre-registered predictions.

Runs: report_patch.py runs tagged up_inject (group coin_report), rows under outputs/coin_report/<family>/<run id>/.
Per prompt (both orders), the answer r is the preference report's heads - tails log-odds; per arm X with a +/- pair,
  G_X = mean over prompts of (r(X_pos) - r(X_neg)) / (2 dose)      (the reading gain, nats per nat of lean pushed)
and the coin at " up" moves by (c(X_pos) - c(X_neg)) / (2 dose) for pushes that include " up". The transplant: d = r(transplant)
- r(none) as a pair effect, its slope on the final's A_i - A_j, and that slope over the final's own answer slope on A_i - A_j.
CIs: 95% bootstrap over pairs, the same resamples for every arm of a run.

Usage: uv run python up_inject.py  -> outputs/coin_report/up_inject/{summary.txt, gains.csv}, figures/up_inject_gains.png
"""

from __future__ import annotations

import io as _io
import json
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from coin_report import load, slope
from matskit import io, plotstyle as ps
from up_followups import FINALS, LABEL, boot_indices, fmt, with_ci, write_csv

OUT = Path("outputs/coin_report/up_inject")
GROUPS = ("up", "ctx", "all")


def runs():
    import wandb
    out = []
    for r in wandb.Api().runs("halrockwell-mats/tournament", filters={"tags": "up_inject", "group": "coin_report"}):
        if r.state != "finished":
            continue
        path = Path("outputs/coin_report") / r.config["family"] / r.id
        if not (path / "rows.jsonl").exists():
            print(f"skipping {r.config['family']} {r.id}: no rows.jsonl here yet")
            continue
        out.append(dict(id=r.id, family=r.config["family"], config=r.config, path=path))
    return out


def run_stats(D, rows, A, dose, idxs):
    """Per resample (pairs idx): every push arm's gain, the random seeds' gains, the coin's gain, the transplant slope."""
    arms = D["arms"]
    pairs = sorted({a.rsplit("_", 1)[0] for a in arms if a.endswith("_pos")})
    out = []
    for idx in idxs:
        s = {}
        for p in pairs:
            g = (D["r_o"][p + "_pos"][idx] - D["r_o"][p + "_neg"][idx]) / (2 * dose)  # [n, 2]
            s[f"{p}: G"] = float(g.mean())
            gc = (D["coin_arm_o"][p + "_pos"][idx] - D["coin_arm_o"][p + "_neg"][idx]) / (2 * dose)
            s[f"{p}: coin gain"] = float(gc.mean())
        a, r0 = A[idx], D["r"]["none"][idx]
        s["answer slope on A"] = slope(a, r0)
        if "transplant_all" in arms:
            d = D["r"]["transplant_all"][idx] - r0
            s["transplant: slope on A"] = slope(a, d)
        out.append(s)
    return {k: with_ci([o[k] for o in out]) for k in out[0]}


def main():
    ps.setup()
    OUT.mkdir(parents=True, exist_ok=True)
    stats, info = {}, {}
    for run in runs():
        fam, cfg = run["family"], run["config"]
        tp = json.load(open(cfg["inject"]["transplant"]))
        D = load(run["path"] / "rows.jsonl")
        rows = io.read_jsonl(run["path"] / "rows.jsonl")
        A = np.array([tp["A"][a] - tp["A"][b] for a, b in zip(D["a"], D["b"])])
        S = run_stats(D, rows, A, cfg["inject"]["dose"], boot_indices(len(D["c"])))
        checks = {}
        for r in rows:
            for arm, v in r.get("push_check", {}).items():
                checks.setdefault(arm, []).append(v)
        stats[fam], info[fam] = S, dict(run=run, D=D, checks={a: float(np.median(v)) for a, v in checks.items()},
                                       mass=float(np.median(D["mass"]["none"])),
                                       self_max=float(np.abs(D["r_o"]["self"] - D["r_o"]["none"]).max()))
    buf = _io.StringIO()
    with redirect_stdout(buf):
        report(stats, info)
    (OUT / "summary.txt").write_text(buf.getvalue())
    print(buf.getvalue())


def interpretable(info, fam, arm):
    """The pre-registered gate: the answer's median mass >= 0.2 and the arm's push_check median in [0.5, 2] (a random arm
    takes push_all's check, the same step size)."""
    check = info[fam]["checks"].get(arm if not arm.startswith("rand") else "push_all_pos")
    return info[fam]["mass"] >= 0.2 and (check is None or 0.5 <= check <= 2)


def report(stats, info):
    fams = [f for f in [*FINALS, *(f + "-base" for f in FINALS)] if f in stats]
    rows = []
    for fam in fams:
        S, I = stats[fam], info[fam]
        print(f"\n{LABEL[fam]} ({I['run']['id']}): mass {I['mass']:.2f}, self-patch max |d| {I['self_max']:.2g}, push_check "
              + ", ".join(f"{a} {v:.2f}" for a, v in sorted(I["checks"].items())))
        print(f"  answer slope on the final's A_i - A_j: {fmt(S['answer slope on A'])}")
        for p in sorted(k.split(":")[0] for k in S if k.endswith(": G")):
            ok = interpretable(info, fam, p + "_pos")
            print(f"  {p:14} G {fmt(S[p + ': G'])}   coin at ' up' moves {fmt(S[p + ': coin gain'])}"
                  + ("" if ok else "   [fails the gate: not interpreted]"))
            rows.append(dict(family=fam, arm=p, G=S[p + ": G"][0], G_lo=S[p + ": G"][1], G_hi=S[p + ": G"][2],
                             coin_gain=S[p + ": coin gain"][0], interpretable=ok))
        if "transplant: slope on A" in S:
            t, base_slope = S["transplant: slope on A"], S["answer slope on A"]
            final = fam.removesuffix("-base")
            fs = stats[final]["answer slope on A"][0] if final in stats else float("nan")
            print(f"  transplant: answer shift's slope on A {fmt(t)} = {t[0] / fs:+.3f} x the final's answer slope "
                  f"({fs:.3f})")
            rows.append(dict(family=fam, arm="transplant_all", G=t[0], G_lo=t[1], G_hi=t[2], coin_gain=np.nan,
                             interpretable=interpretable(info, fam, "push_all_pos")))
    write_csv(OUT / "gains.csv", rows)

    print("\n=== Pre-registered predictions")
    G = lambda fam, p, i=0: stats[fam][f"{p}: G"][i]
    finals = [f for f in FINALS if f in stats]
    bases = [f + "-base" for f in FINALS if f + "-base" in stats]
    missing = [f for f in FINALS if f not in stats] + [f + "-base" for f in FINALS if f + "-base" not in stats]
    note = f" (missing: {', '.join(missing)})" if missing else ""
    say = lambda ok: ("holds" if ok else "fails") + note
    print("  1 finals: G_up and G_ctx > 0, CIs above 0: " + say(all(G(f, "push_up", 1) > 0 and G(f, "push_ctx", 1) > 0
                                                                   for f in finals)))
    print("  2 finals: every random seed's |G| < G_all / 4: " + say(all(
        abs(G(f, f"rand{s}_all")) < G(f, "push_all") / 4 for f in finals for s in (0, 1, 2))))
    print("  3 bases: G_up > 0, CI above 0: " + say(all(G(b, "push_up", 1) > 0 for b in bases)))
    print("  4 each base's G_all below its final's: " + say(all(G(b, "push_all") < G(b[:-5], "push_all") for b in bases
                                                               if b[:-5] in stats)))
    print("  5 the transplant moves each base toward its final's preferred task (slope CI above 0): " + say(all(
        stats[b]["transplant: slope on A"][1] > 0 for b in bases)))
    plot(stats, info, fams)


def plot(stats, info, fams):
    families = [f for f in FINALS if f in stats or f + "-base" in stats]
    fig, axes = plt.subplots(1, len(families), figsize=(4.6 * len(families), 4.2), sharey=False, squeeze=False)
    for ax, fam in zip(axes[0], families):
        for j, (f, color, marker) in enumerate(((fam, ps.SERIES[0], "o"), (fam + "-base", ps.SERIES[1], "s"))):
            if f not in stats:
                continue
            S = stats[f]
            for x, p in enumerate(("push_up", "push_ctx", "push_all")):
                e, lo, hi = S[p + ": G"]
                ax.errorbar(x + (j - 0.5) * 0.25, e, yerr=[[max(e - lo, 0)], [max(hi - e, 0)]], fmt=marker, color=color,
                            ms=7, elinewidth=1.5, mec=ps.SURFACE, mew=0.8, label=("final" if j == 0 else "base") if x == 0 else None)
            for s in (0, 1, 2):
                ax.scatter(3 + (j - 0.5) * 0.25, S[f"rand{s}_all: G"][0], marker="x", color=color, s=30, zorder=3,
                           label=f"random directions ({'final' if j == 0 else 'base'})" if s == 0 else None)
        ax.axhline(0, color=ps.BASELINE, lw=0.8)
        ax.set_xticks(range(4), ["\" up\"", "the 6 around it", "all 7", "all 7, random\ndirections"], fontsize=8)
        ax.set_title(FINALS[fam], fontsize=10)
        ax.set_ylabel("reading gain (answer nats per nat of lean pushed)" if fam == families[0] else "")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8, loc="lower center", ncol=4)
    fig.suptitle("Pushing the heads/tails lean from the final's onset block on: does the later answer follow?", fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    ps.save(fig, "up_inject_gains")


if __name__ == "__main__":
    main()
