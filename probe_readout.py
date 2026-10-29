"""Preference probe, steps 1-2 analysis (PLAN_preference_probe.md): how well the probes fit, whether they are a harm
detector, and whether they read the tasks' values inside the coinflip prompt in the SFT and DPO models.

Inputs: outputs/probe/{sft,dpo}/acts/ (probe_acts.py) and probe.npz / fit_summary.json (probe_fit.py).

Step 2 readout: a probe fit at one site of one model (`from`) is applied, as h @ w + b, to another site's
activations of a model (`to`), and correlated across the 300 tournament tasks with the DPO Assistant utilities
(and within the 257 non-harmful ones). Cross-fitted (crossfit): each task is scored by a probe fit without it. Slotted sites average the slot A and B scores per task. Next to it, the
"decodable" r: a probe fit at that site itself (probe_fit's out-of-fold r_cv).
    (a)  the SFT's coinflip task tokens carry the value about as well as DPO's
    (b)  they carry much less

Writes outputs/probe/readout.json, outputs/probe/directions.npz (the unit steering directions probe_tests.py
uses: probe_dpo, probe_sft, probe_dpo_benign, from the dpo-target pairwise fits) and figures/probe_*.png.

Usage (local): uv run python probe_readout.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from matskit import plotstyle as ps
from probe_acts import SITES
from probe_fit import Data, Gram, kfold, pearson

ROOT = Path("outputs/probe")
STAGES = ["sft", "dpo"]
LABEL = {"sft": "SFT", "dpo": "DPO"}
COLOR = {"sft": ps.SERIES[1], "dpo": ps.SERIES[0]}
SITE_LABEL = {"single_eot": "task alone, <|im_end|>", "single_last": "task alone, last token",
              "assistant_task": "Assistant pair prompt, task end", "coinflip_task": "coinflip prompt, task end",
              "coinflip_label_b": "coinflip prompt, \" tails\""}
PROBE_KEY = "dpo/all/assistant_task"


def load(stage):
    d = ROOT / stage
    data = Data(d / "acts")
    probe = dict(np.load(d / "probe.npz"))
    summary = json.loads((d / "fit_summary.json").read_text())
    return data, probe, summary


def scores(data, site, layer, w, b, tasks):
    """Probe score per task (slotted sites: mean of the two slots)."""
    X, g = data.rows(site, layer, tasks)
    s = X @ w + b
    return s.reshape(len(tasks), 2).mean(1) if SITES[site][2] else s


def curve(summary, key, field="r_cv"):
    rows = summary.get(key, {})
    layers = sorted(int(l) for l in rows)
    return np.array(layers), np.array([rows[str(l)][field] if isinstance(rows[str(l)][field], float)
                                       else np.nan for l in layers])


def crossfit(src, src_site, layer, alpha, targets, k=5, seed=0):
    """Out-of-fold probe scores: for each of k folds over the 300 tournament tasks, fit the probe (DPO utilities,
    fixed alpha) on the other folds' rows at src_site of model `src` (a Data), and score the held-out tasks at
    every (dst Data, dst site) in `targets`. A task is never scored by a probe that saw its utility (applying the
    full-data probe to the same tasks' activations elsewhere would carry its in-sample fit along).
    Returns [len(targets), 300] per-task scores (slotted sites: mean of the two slots)."""
    tasks = src.idx
    X, g = src.rows(src_site, layer, tasks)
    y = np.repeat(src.y["dpo"], 2) if SITES[src_site][2] else src.y["dpo"]
    gram = Gram(X)
    out = np.zeros((len(targets), len(tasks)))
    for test_tasks in kfold(np.arange(len(tasks)), k, seed):
        train = ~np.isin(g, tasks[test_tasks])
        w, b = gram.fit(train, y[train]).weights(alpha)
        for j, (dst, site) in enumerate(targets):
            out[j, test_tasks] = scores(dst, site, layer, w, b, tasks[test_tasks])
    return out


def transfer(loaded, src_site="assistant_task", sites=("assistant_task", "coinflip_task", "coinflip_label_b")):
    """{(from, to, site): (layers, r_all, r_benign)}: the pairwise probe of model `from`, cross-fitted, read at
    model `to`'s site, correlated with the DPO Assistant utilities."""
    out = {}
    for src in STAGES:
        data, _, summ = loaded[src]
        layers = sorted(int(l) for l in summ[f"dpo/all/{src_site}"])
        targets = [(loaded[dst][0], site) for dst in STAGES for site in sites]
        r = np.zeros((len(targets), len(layers), 2))
        for i, l in enumerate(layers):
            s = crossfit(data, src_site, l, summ[f"dpo/all/{src_site}"][str(l)]["alpha"], targets)
            y, harm = data.y["dpo"], data.harm
            for j in range(len(targets)):
                r[j, i] = pearson(s[j], y), pearson(s[j][~harm], y[~harm])
        for j, (dst, site) in enumerate((d, si) for d in STAGES for si in sites):
            out[src, dst, site] = (np.array(layers), r[j, :, 0], r[j, :, 1])
    return out


def probe_agreement(loaded):
    """Per model and layer: cosine between the single-task (eot) and pairwise probes (full-data fits), and their
    cross-prediction r (cross-fitted: each site's probe read at the other site for held-out tasks)."""
    out = {}
    for st in STAGES:
        data, probe, summ = loaded[st]
        y = data.y["dpo"]
        layers = sorted(int(l) for l in summ["dpo/all/single_eot"])
        cos, cross_sp, cross_ps = [], [], []
        for l in layers:
            ws, wp = probe["dpo/all/single_eot/w"][l], probe["dpo/all/assistant_task/w"][l]
            cos.append(float(ws @ wp / np.linalg.norm(ws) / np.linalg.norm(wp)))
            s = crossfit(data, "single_eot", l, summ["dpo/all/single_eot"][str(l)]["alpha"],
                         [(data, "assistant_task")])[0]
            cross_sp.append(pearson(s, y))
            s = crossfit(data, "assistant_task", l, summ["dpo/all/assistant_task"][str(l)]["alpha"],
                         [(data, "single_eot")])[0]
            cross_ps.append(pearson(s, y))
        out[st] = dict(layers=layers, cos=cos, single_on_pair=cross_sp, pair_on_single=cross_ps)
    return out


def write_directions(probes):
    """probes: {stage: probe.npz contents}."""
    dirs = {"probe_dpo": probes["dpo"][f"{PROBE_KEY}/w"], "probe_sft": probes["sft"][f"{PROBE_KEY}/w"],
            "probe_dpo_benign": probes["dpo"]["dpo/benign/assistant_task/w"]}
    for k, v in dirs.items():
        assert np.isfinite(v[15:24]).all(), f"{k}: steering blocks 15-23 weren't fit"
    np.savez(ROOT / "directions.npz", **{k: np.nan_to_num(v).astype(np.float32) for k, v in dirs.items()})


def fig_fit(loaded):
    sites = ["single_eot", "assistant_task", "coinflip_task", "coinflip_label_b"]
    fig, axes = plt.subplots(1, len(sites), figsize=(13, 3.4), sharey=True)
    for ax, site in zip(axes, sites):
        for st in STAGES:
            summ = loaded[st][2]
            x, r = curve(summ, f"dpo/all/{site}")
            ax.plot(x, r, color=COLOR[st], lw=1.6, label=f"{LABEL[st]}, all tasks")
            x, rb = curve(summ, f"dpo/benign/{site}")
            ax.plot(x, rb, color=COLOR[st], lw=1.2, ls="--", label=f"{LABEL[st]}, non-harmful only")
            nulls = [max(summ[f"dpo/all/{site}"][str(l)]["r_null"]) for l in x]
            ax.plot(x, nulls, color=COLOR[st], lw=0.8, ls=":", alpha=0.8)
        ax.axhline(0, color=ps.BASELINE, lw=0.8)
        ax.set_title(SITE_LABEL[site], fontsize=9.5)
        ax.set_xlabel("block (output)")
    axes[0].set_ylabel("cross-validated r with\nDPO Assistant utility")
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle("Utility probes by depth (dotted: shuffled-utility null, max of 3)", fontsize=10.5)
    fig.tight_layout()
    return ps.save(fig, "probe_r_by_layer")


def fig_harm(loaded):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for st in STAGES:
        summ = loaded[st][2]
        for site, ls in (("assistant_task", "-"), ("single_eot", "--")):
            x, auc = curve(summ, f"dpo/benign/{site}", "harm_auroc")
            axes[0].plot(x, auc, color=COLOR[st], ls=ls, lw=1.4, label=f"{LABEL[st]}, {SITE_LABEL[site]}")
            x, acc = curve(summ, f"dpo/benign/{site}", "std_acc")
            axes[1].plot(x, acc, color=COLOR[st], ls=ls, lw=1.4)
    axes[0].axhline(0.5, color=ps.BASELINE, lw=0.8)
    axes[1].axhline(0.5, color=ps.BASELINE, lw=0.8)
    axes[0].set_ylabel("AUROC, non-harmful above harmful")
    axes[1].set_ylabel("standard pairs: preferred scores higher")
    for ax in axes:
        ax.set_xlabel("block (output)")
        ax.set_ylim(0, 1.02)
    axes[0].legend(fontsize=7.5, frameon=False, loc="lower right")
    fig.suptitle("Probe fit on the 257 non-harmful tasks only: does it rank harmful tasks low?", fontsize=10.5)
    fig.tight_layout()
    return ps.save(fig, "probe_harm_vs_benign")


def fig_readout(loaded, tr):
    sites = ["assistant_task", "coinflip_task", "coinflip_label_b"]
    fig, axes = plt.subplots(2, len(sites), figsize=(12, 6.2), sharey="row", sharex=True)
    for j, site in enumerate(sites):
        for st in STAGES:
            x, r_all, r_ben = tr["dpo", st, site]
            axes[0, j].plot(x, r_all, color=COLOR[st], lw=1.6, label=f"{LABEL[st]} activations, DPO probe")
            axes[1, j].plot(x, r_ben, color=COLOR[st], lw=1.6)
            x, r_all, r_ben = tr[st, st, site]
            if st == "sft":
                axes[0, j].plot(x, r_all, color=COLOR[st], lw=1.2, ls="--", label="SFT activations, SFT probe")
                axes[1, j].plot(x, r_ben, color=COLOR[st], lw=1.2, ls="--")
            xd, rd = curve(loaded[st][2], f"dpo/all/{site}")
            axes[0, j].plot(xd, rd, color=COLOR[st], lw=0.9, ls=":", label=f"{LABEL[st]}, probe fit at this site")
        axes[0, j].set_title(SITE_LABEL[site], fontsize=9.5)
        axes[1, j].set_xlabel("block (output)")
        for ax in axes[:, j]:
            ax.axhline(0, color=ps.BASELINE, lw=0.8)
    axes[0, 0].set_ylabel("r with DPO Assistant utility\n(300 tasks)")
    axes[1, 0].set_ylabel("r, 257 non-harmful tasks")
    axes[0, 0].legend(fontsize=7.5, frameon=False, loc="lower right")
    fig.suptitle("Pairwise Assistant-prompt probe (cross-fitted) read at each prompt's task tokens", fontsize=10.5)
    fig.tight_layout()
    return ps.save(fig, "probe_coinflip_readout")


def main():
    import sys
    if sys.argv[1:2] == ["--directions-only"]:  # from the quick blocks 15-23 fits (probe_fit.py --tag _steer)
        write_directions({st: dict(np.load(ROOT / st / "probe_steer.npz")) for st in STAGES})
        print(f"wrote {ROOT / 'directions.npz'}")
        return
    ps.setup()
    loaded = {st: load(st) for st in STAGES}
    tr = transfer(loaded)
    agree = probe_agreement(loaded)
    write_directions({st: loaded[st][1] for st in STAGES})
    out = dict(transfer={f"{a}->{b}/{s}": dict(layers=v[0].tolist(), r_all=v[1].tolist(), r_benign=v[2].tolist())
                         for (a, b, s), v in tr.items()},
               agreement=agree)
    (ROOT / "readout.json").write_text(json.dumps(out, indent=0))
    for f in (fig_fit(loaded), fig_harm(loaded), fig_readout(loaded, tr)):
        print(f"wrote {f}")


if __name__ == "__main__":
    main()
