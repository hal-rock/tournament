"""Preference probe, step 3 analysis (PLAN_preference_probe.md): steering and swap results from probe_tests.py.

Per pair (HP and TP prompts of the same two tasks), with L the outcome log-odds at the last token (coinflip:
" heads" - " tails"; choice: " A" - " B"; HP puts the preferred task first):
    effect   (L_HP - L_TP) / 2 from the unbatched forwards: how much the prompt favours the preferred task
    steer    dL = mean over HP and TP of L(arm) - L(unsteered row of the same batch). Positive = toward task A,
             which the arms push up along the probe (A's tokens +, B's -).
    swap     ((L_TP' - L_TP) - (L_HP' - L_HP)) / 4, summed over pairs and divided by the summed effect: the share of
             the effect that the swap along the direction removes and reverses (1 = HP fully becomes TP)
CIs: 95% bootstrap over pairs.

Writes outputs/probe/effects.json and figures/probe_steering*.png, figures/probe_swap.png.
Usage (local): uv run python probe_effects.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from matskit import plotstyle as ps
from matskit.stats import bootstrap_ci, ratio_ci

ROOT = Path("outputs/probe")
STAGES = ["sft", "dpo"]
LABEL = {"sft": "SFT", "dpo": "DPO", "choice": "Assistant's direct choice", "coinflip": "coinflip"}
DIR_STYLE = {"probe_dpo": (ps.SERIES[0], "-", "DPO probe"), "probe_sft": (ps.SERIES[1], "-", "SFT probe"),
             "probe_dpo_benign": (ps.SERIES[2], "--", "DPO probe, non-harmful fit")}
RAND = ps.MUTED


def load_rows(stage):
    """rows.jsonl, with the all-blocks swap arms taken from rows_swap_together.jsonl (the rerun with acts.clamp;
    the first run's version stacked its shifts across blocks) and dropped if there is no rerun."""
    rows = [json.loads(line) for line in open(ROOT / stage / "tests/rows.jsonl")]
    rows = [r for r in rows if not (r["arm"] == "swap" and r["blocks"] == "together")]
    rerun = ROOT / stage / "tests/rows_swap_together.jsonl"
    if rerun.exists():
        rows += [json.loads(line) for line in open(rerun)]
    return rows


def by_pair(rows):
    """{(set, format, pair): {order: unbatched L}} and per arm {(set, format, arm key): {pair: {order: (L, L_base)}}}."""
    base, arms = defaultdict(dict), defaultdict(lambda: defaultdict(dict))
    for r in rows:
        base[r["set"], r["format"], r["pair"]][r["order"]] = r["L_unbatched"]
        key = (r["set"], r["format"], r["arm"], r["direction"], r.get("c"), r["blocks"])
        arms[key][r["pair"]][r["order"]] = (r["L"], r["L_base"])
    return base, arms


def summarize(stage):
    rows = load_rows(stage)
    base, arms = by_pair(rows)
    out = {"effect": {}, "steer": {}, "swap": {}, "batch_drift": {}}
    for (s, f) in {(k[0], k[1]) for k in base}:
        eff = np.array([(v["HP"] - v["TP"]) / 2 for (s2, f2, _), v in sorted(base.items()) if (s2, f2) == (s, f)])
        out["effect"][f"{s}/{f}"] = dict(mean=float(eff.mean()), ci=bootstrap_ci(eff), n=len(eff))
    drift = defaultdict(list)
    for r in rows:
        drift[r["set"], r["format"]].append(abs(r["L_base"] - r["L_unbatched"]))
    out["batch_drift"] = {f"{s}/{f}": dict(median=float(np.median(v)), max=float(np.max(v))) for (s, f), v in drift.items()}
    for key, per_pair in arms.items():
        s, f, arm, d, c, blocks = key
        pairs = sorted(per_pair)
        if arm == "steer":
            dl = np.array([np.mean([per_pair[p][o][0] - per_pair[p][o][1] for o in ("HP", "TP")]) for p in pairs])
            out["steer"][f"{s}/{f}/{d}/{c}/{blocks}"] = dict(mean=float(dl.mean()), ci=bootstrap_ci(dl), n=len(dl))
        else:
            num = np.array([((per_pair[p]["TP"][0] - per_pair[p]["TP"][1])
                             - (per_pair[p]["HP"][0] - per_pair[p]["HP"][1])) / 4 for p in pairs])
            eff = np.array([(per_pair[p]["HP"][1] - per_pair[p]["TP"][1]) / 2 for p in pairs])
            out["swap"][f"{s}/{f}/{d}/{blocks}"] = dict(share=float(num.sum() / eff.sum()), ci=ratio_ci(num, eff),
                                                        moved=float(num.mean()), effect=float(eff.mean()))
    return out


def block_axis(keys):
    order = [b for b in keys if b != "together"]
    return sorted(order, key=int) + (["together"] if "together" in keys else [])


def fig_steer(res, which="standard", c=0.2):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), sharex=True)
    for j, st in enumerate(STAGES):
        for i, fmt in enumerate(["choice", "coinflip"]):
            ax = axes[i, j]
            steer = res[st]["steer"]
            keys = [k.split("/") for k in steer if k.startswith(f"{which}/{fmt}/") and float(k.split("/")[3]) == c]
            blocks = block_axis({k[4] for k in keys})
            x = np.arange(len(blocks))
            for d in sorted({k[2] for k in keys}):
                m = np.array([steer[f"{which}/{fmt}/{d}/{c}/{b}"]["mean"] for b in blocks])
                lo, hi = np.array([steer[f"{which}/{fmt}/{d}/{c}/{b}"]["ci"] for b in blocks]).T
                if d.startswith("rand"):
                    ax.plot(x, m, color=RAND, lw=0.9, label="random directions" if d == "rand0" else None)
                else:
                    color, ls, lab = DIR_STYLE[d]
                    ax.plot(x, m, color=color, ls=ls, lw=1.6, label=lab)
                    ax.fill_between(x, lo, hi, color=color, alpha=0.12, lw=0)
            ax.axhline(0, color=ps.BASELINE, lw=0.8)
            eff = res[st]["effect"][f"{which}/{fmt}"]["mean"]
            ax.set_title(f"{LABEL[st]}: {LABEL[fmt]} (effect {eff:+.2f} nats)", fontsize=9.5)
            ax.set_xticks(x, [str(b) if b != "together" else "15-23" for b in blocks], fontsize=8)
            if j == 0:
                ax.set_ylabel("dL toward task A (nats)")
            if i == 1:
                ax.set_xlabel("steered block (output)")
    axes[0, 0].legend(fontsize=7.5, frameon=False)
    name = "standard pairs (benign vs extremely harmful)" if which == "standard" else "non-harmful pairs (gap >= 2)"
    fig.suptitle(f"Steering +c on task A's tokens, -c on task B's, c = {c} x residual norm; {name}", fontsize=10.5)
    fig.tight_layout()
    return ps.save(fig, f"probe_steering_{which}")


def fig_dose(res, which="standard", blocks="together"):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, fmt in zip(axes, ["choice", "coinflip"]):
        for st in STAGES:
            steer = res[st]["steer"]
            for d, ls in (("probe_dpo", "-"), ("rand0", ":")):
                cs = sorted(float(k.split("/")[3]) for k in steer
                            if k.startswith(f"{which}/{fmt}/{d}/") and k.endswith(f"/{blocks}"))
                m = [steer[f"{which}/{fmt}/{d}/{c}/{blocks}"]["mean"] for c in cs]
                ax.plot(cs, m, color=ps.SERIES[0] if st == "dpo" else ps.SERIES[1], ls=ls, marker="o", ms=3,
                        label=f"{LABEL[st]}, {'DPO probe' if d == 'probe_dpo' else 'random'}")
        ax.axhline(0, color=ps.BASELINE, lw=0.8)
        ax.set_title(LABEL[fmt], fontsize=9.5)
        ax.set_xlabel("c (x residual norm)")
    axes[0].set_ylabel("dL toward task A (nats)")
    axes[0].legend(fontsize=7.5, frameon=False)
    fig.suptitle(f"Dose-response, steering at {'blocks 15-23' if blocks == 'together' else 'block ' + blocks}, "
                 f"{which} pairs", fontsize=10.5)
    fig.tight_layout()
    return ps.save(fig, f"probe_steering_dose_{which}")


def fig_swap(res):
    """The swap's change in nats, ((L_TP' - L_TP) - (L_HP' - L_HP)) / 4, against the effect it would remove at share 1
    (dashed, when it fits the axis). Shares themselves are unstable where the effect is ~0 (SFT coinflip)."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.6), sharex=True)
    for j, st in enumerate(STAGES):
        for i, fmt in enumerate(["choice", "coinflip"]):
            ax = axes[i, j]
            swap = res[st]["swap"]
            allb = block_axis({k.split("/")[3] for k in swap if k.startswith(f"standard/{fmt}/")})
            for which, marker, ls in (("standard", "o", "-"), ("benign", "s", ":")):
                keys = [k.split("/") for k in swap if k.startswith(f"{which}/{fmt}/")]
                blocks = block_axis({k[3] for k in keys})
                x = np.array([allb.index(b) for b in blocks])
                for d in sorted({k[2] for k in keys}):
                    moved = np.array([swap[f"{which}/{fmt}/{d}/{b}"]["moved"] for b in blocks])
                    if d.startswith("rand"):
                        ax.plot(x, moved, color=RAND, lw=0.8, ls=ls,
                                label="random" if (d == "rand0" and which == "standard") else None)
                    else:
                        color, _, lab = DIR_STYLE[d]
                        ax.plot(x, moved, color=color, ls=ls, marker=marker, ms=3, lw=1.4, label=f"{lab}, {which}")
            ax.set_xticks(range(len(allb)), [b if b != "together" else "15-23" for b in allb], fontsize=8)
            ax.axhline(0, color=ps.BASELINE, lw=0.8)
            effs = {w: res[st]["effect"][f"{w}/{fmt}"]["mean"] for w in ("standard", "benign")}
            top = ax.get_ylim()[1]
            for w, ls in (("standard", "--"), ("benign", (0, (1, 2)))):
                if effs[w] <= 1.5 * top:
                    ax.axhline(effs[w], color=ps.INK2, lw=0.8, ls=ls)
            ax.set_title(f"{LABEL[st]}: {LABEL[fmt]}\neffect: standard {effs['standard']:+.2f}, "
                         f"non-harmful {effs['benign']:+.2f} nats", fontsize=9)
            if j == 0:
                ax.set_ylabel("swapped (nats)")
            if i == 1:
                ax.set_xlabel("block (output)")
    axes[0, 0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle("Swapping only the tasks' projections on the probe direction between HP and TP\n"
                 "(grey dashed: the effect, i.e. a full swap, where it fits)", fontsize=10.5)
    fig.tight_layout()
    return ps.save(fig, "probe_swap")


def main():
    ps.setup()
    res = {st: summarize(st) for st in STAGES if (ROOT / st / "tests/rows.jsonl").exists()}
    (ROOT / "effects.json").write_text(json.dumps(res, indent=0))
    for st, r in res.items():
        print(st, "effects:", {k: round(v["mean"], 2) for k, v in r["effect"].items()})
        print(st, "batch drift (median |L_base - L_unbatched|):",
              {k: round(v["median"], 3) for k, v in r["batch_drift"].items()})
    if set(res) == set(STAGES):
        for f in (fig_steer(res, "standard"), fig_steer(res, "benign", 0.1), fig_dose(res, "standard"),
                  fig_dose(res, "benign", "together"), fig_swap(res)):
            print(f"wrote {f}")


if __name__ == "__main__":
    main()
