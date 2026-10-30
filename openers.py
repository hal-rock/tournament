"""Does the opener of the coinflip's last line change bleedthrough over the 300 tasks?

The measures and predictions were pre-registered in the README ("Last-line openers") before
any data. Collects the W&B runs of group "openers" (configs/openers/, one per model), each
scoring the coinflip with ../coinflip's five openers. The Assistant prompt has no such line,
so A comes from the model sweep's run on the same pairs. With C the coinflip utilities at one
opener and C_okay those at "Okay, ", per model and opener:

- r, r_benign, r_harmful: r(A, C) over all, non-harmful and harmful tasks;
- r_within: r(A, C) after removing each origin's mean from both; r_between: r over the five
  origin means. These split the correlation into within-type and cross-type parts;
- b, b_benign, b_within: slope of C on A in nats of coinflip per Assistant SD (of the same
  tasks). A is fixed per model, so a change in b is a change in the Assistant-aligned scale;
- harm_gap: mean C, non-harmful minus harmful, the contrast ../coinflip's task pairs measure;
- coinflip_sd; resid_sd: SD of C around its line on A (what isn't Assistant-aligned);
- r_okay, scale_okay: r(C_okay, C) and the slope of C on C_okay. r_okay near 1 with scale_okay
  below 1 means the same coinflip, turned down.

CIs are 95% task bootstraps within origin, with the same resamples for every opener of a model,
so changes and retention (value / value at "Okay, ") are paired. Each non-"Okay" cell gets the
pre-registered call from `classify`, and the 12 cells are correlated with their retention in
../coinflip (x fixed, y bootstrapped). Not pre-registered, printed last: L by which side the
benign task is on (`orientation`), and how OLMo's coinflips relate to its SFT stage (`sft_stage`).

  outputs/openers/openers.csv      one row per model and opener
  outputs/openers/across_cells.csv the across-cell correlations
  outputs/openers/orientation.csv  mean L by side of the benign task
  figures/openers_*.png

Usage: uv run python openers.py [--group openers]
"""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import ORIGIN_STYLE, N_BOOTSTRAP, fetch_wandb as fetch_raw, load_results, pair_effects, pearson, spearman
from matskit import plotstyle
from models import MODEL_INFO, PROJECT, collect_runs, olmo_on_shared_pairs
from report import INK, INK_2, RULE, fetch_wandb, origin_scatter, stratified_resamples, style, task_arrays

OLMO, LLAMA, GLM = "allenai/Olmo-3.1-32B-Instruct", "unsloth/Llama-3.3-70B-Instruct", "zai-org/GLM-4.5-Air"
OLMO_SFT_ANALYSIS = Path("outputs/d8ysbybo/analysis")
SWEEP_RUNS = {LLAMA: "tvlqj8a0", GLM: "i5vejs3b"}  # OLMo: d8ysbybo refit on the same pairs (models.py)
OPENERS = ("okay", "alright", "ok", "no_okay", "so")
LABELS = {"okay": '"Okay,"', "alright": '"Alright,"', "ok": '"OK,"', "no_okay": "none", "so": '"So,"'}
# ../coinflip RESULTS.md, "Follow-up: other openers": preference effect (log-odds) on its 25 pairs.
COINFLIP_EFFECT = {
    OLMO: {"okay": 2.27, "no_okay": 0.59, "alright": 0.96, "ok": 0.68, "so": 0.32},
    LLAMA: {"okay": 4.13, "no_okay": 4.12, "alright": 4.19, "ok": 4.05, "so": 3.56},
    GLM: {"okay": 1.88, "no_okay": 0.90, "alright": 1.83, "ok": 0.95, "so": 1.24},
}
STATS = ("r", "r_benign", "r_harmful", "r_within", "r_between", "b", "b_benign", "b_within", "harm_gap",
         "coinflip_sd", "resid_sd", "r_okay", "scale_okay")
K = {name: j for j, name in enumerate(STATS)}
RETAINED = ("b", "b_benign", "b_within", "harm_gap", "coinflip_sd", "resid_sd")
R_MEANINGFUL = 0.05  # |change in r| worth calling a change (a tight CI alone can flag tiny ones)
SCALE_FALLS = 0.9    # "scale falls": the CI of b's retention is wholly below this
MODEL_MARKER = {OLMO: "o", LLAMA: "s", GLM: "^"}
GATE_R, GATE_SD = 0.999, 0.02
READABLE_MASS = 0.2


def slope_per_sd(a, c):
    """Slope of c on a, per SD of a."""
    return float(np.mean((a - a.mean()) * (c - c.mean())) / a.std())


def within_groups(x, groups):
    out = x.astype(float).copy()
    for g in np.unique(groups):
        out[groups == g] -= out[groups == g].mean()
    return out


def cell_stats(a, c, c_okay, benign, origins):
    """STATS of one opener, over arrays of the same (possibly resampled) tasks."""
    aw, cw = within_groups(a, origins), within_groups(c, origins)
    groups = np.unique(origins)
    am = np.array([a[origins == g].mean() for g in groups])
    cm = np.array([c[origins == g].mean() for g in groups])
    fit = np.polyfit(a, c, 1)
    scale = np.mean((c_okay - c_okay.mean()) * (c - c.mean())) / c_okay.var()
    return np.array([
        pearson(a, c), pearson(a[benign], c[benign]), pearson(a[~benign], c[~benign]), pearson(aw, cw),
        pearson(am, cm), slope_per_sd(a, c), slope_per_sd(a[benign], c[benign]), slope_per_sd(aw, cw),
        c[benign].mean() - c[~benign].mean(), c.std(), (c - np.polyval(fit, a)).std(), pearson(c_okay, c), scale,
    ])


def model_cells(a, C, benign, origins, resamples):
    """{opener: {"est": STATS, "boot": (n_resamples, STATS)}}, the same resamples for every opener."""
    return {o: dict(est=cell_stats(a, C[o], C["okay"], benign, origins),
                    boot=np.array([cell_stats(a[i], C[o][i], C["okay"][i], benign[i], origins[i])
                                   for i in resamples])) for o in C}


def ci(boot):
    return np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)


def change(cells, opener):
    """(estimate, ci_low, ci_high) over STATS of opener minus "Okay,", paired."""
    lo, hi = ci(cells[opener]["boot"] - cells["okay"]["boot"])
    return cells[opener]["est"] - cells["okay"]["est"], lo, hi


def retention(cells, opener):
    """(estimate, ci_low, ci_high) over STATS of opener / "Okay,", paired."""
    lo, hi = ci(cells[opener]["boot"] / cells["okay"]["boot"])
    return cells[opener]["est"] / cells["okay"]["est"], lo, hi


def classify(cells, opener):
    """The pre-registered call for one non-"Okay" cell: "r falls", "scale only", "r rises" or "unchanged"."""
    dr, dr_lo, dr_hi = (v[K["r"]] for v in change(cells, opener))
    _, _, keep_hi = (v[K["b"]] for v in retention(cells, opener))
    if dr_hi < 0 and dr <= -R_MEANINGFUL:
        return "r falls"
    if dr_lo > 0 and dr >= R_MEANINGFUL:
        return "r rises"
    if keep_hi < SCALE_FALLS:
        return "scale only"
    return "unchanged"


def across_cells(all_cells, models):
    """Spearman and Pearson, over the non-"Okay" cells of `models`, of ../coinflip's retention
    with the change in r and the retention of b and harm_gap; task bootstrap on the y side."""
    keys = [(m, o) for m in models for o in OPENERS if o != "okay"]
    x = np.array([COINFLIP_EFFECT[m][o] / COINFLIP_EFFECT[m]["okay"] for m, o in keys])
    out = []
    for name, f in (("change in r", lambda v, ok: v[..., K["r"]] - ok[..., K["r"]]),
                    ("change in r_benign", lambda v, ok: v[..., K["r_benign"]] - ok[..., K["r_benign"]]),
                    ("retention of b", lambda v, ok: v[..., K["b"]] / ok[..., K["b"]]),
                    ("retention of harm_gap", lambda v, ok: v[..., K["harm_gap"]] / ok[..., K["harm_gap"]])):
        y = np.array([f(all_cells[m][o]["est"], all_cells[m]["okay"]["est"]) for m, o in keys])
        boot = np.array([f(all_cells[m][o]["boot"], all_cells[m]["okay"]["boot"]) for m, o in keys]).T
        sp, pe = zip(*((spearman(x, b), pearson(x, b)) for b in boot))
        out.append(dict(y=name, n_cells=len(keys), spearman=spearman(x, y), spearman_lo=np.percentile(sp, 2.5),
                        spearman_hi=np.percentile(sp, 97.5), pearson=pearson(x, y), pearson_lo=np.percentile(pe, 2.5),
                        pearson_hi=np.percentile(pe, 97.5)))
    return out


def orientation(rows, benign_ids):
    """Mean L = log P(heads) / P(tails) over benign vs harmful pairs, by which side the benign
    task is on, per condition, as in ../coinflip's table. P(benign) is at the mean L."""
    out = {}
    for r in rows:
        heads_benign, tails_benign = r["first"] in benign_ids, r["second"] in benign_ids
        if heads_benign != tails_benign:
            out.setdefault((r["condition"], "heads" if heads_benign else "tails"), []).append(r["log_odds"])
    return {k: float(np.mean(v)) for k, v in out.items()}


def plot_summary(all_cells, models, path):
    fig, axes = plt.subplots(2, len(models), figsize=(4.2 * len(models), 7.4), sharey="row", squeeze=False)
    x = np.arange(len(OPENERS))
    r_series = (("r", "All tasks", 0), ("r_benign", "Non-harmful", 1), ("r_within", "Within origin", 2))
    keep_series = (("b", "Slope on Assistant", 0), ("harm_gap", "Benign-harmful gap", 1), ("coinflip_sd", "Coinflip SD", 2))
    markers = ("o", "s", "^")
    for col, m in enumerate(models):
        cells = all_cells[m]
        ax = axes[0, col]
        for s, (stat, label, j) in enumerate(r_series):
            est = np.array([cells[o]["est"][K[stat]] for o in OPENERS])
            lo, hi = zip(*(ci(cells[o]["boot"][:, K[stat]]) for o in OPENERS))
            ax.errorbar(x + (s - 1) * 0.16, est, yerr=[est - np.array(lo), np.array(hi) - est], fmt=markers[s],
                        color=plotstyle.SERIES[j], ms=7, mec="white", mew=0.8, elinewidth=1.5, label=label, zorder=3)
        ax.set_title(MODEL_INFO[m][0], color=INK, fontsize=11, loc="left")
        ax = axes[1, col]
        ax.axhline(1, color=RULE, lw=0.8, zorder=1)
        ax.plot(x, [COINFLIP_EFFECT[m][o] / COINFLIP_EFFECT[m]["okay"] for o in OPENERS], color=INK_2, lw=2,
                marker="D", ms=6, mec="white", mew=0.8, label="../coinflip effect", zorder=2)
        for s, (stat, label, j) in enumerate(keep_series):
            est, lo, hi = zip(*((*(v[K[stat]] for v in retention(cells, o)),) for o in OPENERS))
            est, lo, hi = map(np.array, (est, lo, hi))
            ax.errorbar(x + (s - 1) * 0.16, est, yerr=[est - lo, hi - est], fmt=markers[s], color=plotstyle.SERIES[j],
                        ms=7, mec="white", mew=0.8, elinewidth=1.5, label=label, zorder=3)
        for ax in axes[:, col]:
            ax.set_xticks(x, [LABELS[o] for o in OPENERS])
            style(ax)
    axes[0, 0].set_ylabel("r(Assistant, coinflip)")
    axes[1, 0].set_ylabel('Share of the "Okay," value kept')
    axes[0, 0].legend(frameon=False, fontsize=9, loc="lower left")
    axes[1, 0].legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_vs_coinflip(all_cells, models, path):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.3))
    panels = (("Change in r from \"Okay,\"", lambda c, o: change(c, o), "r", 0.0),
              ("Slope on Assistant, share kept", lambda c, o: retention(c, o), "b", 1.0),
              ("Benign-harmful gap, share kept", lambda c, o: retention(c, o), "harm_gap", 1.0))
    for ax, (title, f, stat, ref) in zip(axes, panels):
        ax.axhline(ref, color=RULE, lw=0.8, zorder=1)
        ax.axvline(1, color=RULE, lw=0.8, zorder=1)
        for j, m in enumerate(models):
            os_ = [o for o in OPENERS if o != "okay"]
            xs = np.array([COINFLIP_EFFECT[m][o] / COINFLIP_EFFECT[m]["okay"] for o in os_])
            est, lo, hi = map(np.array, zip(*(tuple(v[K[stat]] for v in f(all_cells[m], o)) for o in os_)))
            ax.errorbar(xs, est, yerr=[est - lo, hi - est], fmt=MODEL_MARKER[m], color=plotstyle.SERIES[j], ms=7,
                        mec="white", mew=0.8, elinewidth=1.5, label=MODEL_INFO[m][0], zorder=3)
            for xi, yi, o in zip(xs, est, os_):
                ax.annotate(LABELS[o].strip('"').rstrip(","), (xi, yi), xytext=(5, 4), textcoords="offset points",
                            fontsize=8, color=INK_2)
        ax.set_xlabel('Share of the "Okay," effect kept in ../coinflip')
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        style(ax)
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scatter(data, models, path):
    fig, axes = plt.subplots(len(models), len(OPENERS), figsize=(3 * len(OPENERS), 2.8 * len(models)),
                             sharex="row", sharey="row", squeeze=False)
    for row, m in enumerate(models):
        a, C, origins = data[m]
        for col, o in enumerate(OPENERS):
            ax = axes[row, col]
            origin_scatter(ax, a, C[o], origins)
            ax.set_title(f"{LABELS[o]}  r = {pearson(a, C[o]):.2f}, SD {C[o].std():.2f}", fontsize=9, color=INK_2, loc="left")
        axes[row, 0].set_ylabel(f"{MODEL_INFO[m][0]}\ncoinflip utility (nats)")
    for ax in axes[-1]:
        ax.set_xlabel("Assistant utility (nats)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=len(ORIGIN_STYLE), loc="upper center", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def sft_stage(a, C, ids):
    """OLMo only, exploratory: is the coinflip without "Okay," like the SFT's? {opener: stats}."""
    with open(OLMO_SFT_ANALYSIS / "utilities.csv", newline="") as fh:
        by_id = {r["id"]: r for r in csv.DictReader(fh)}
    sft = np.array([float(by_id[t]["sft/coinflip"]) for t in ids])
    dpo = np.array([float(by_id[t]["dpo/coinflip"]) for t in ids])
    return {o: dict(r_sft=pearson(sft, c), r_dpo=pearson(dpo, c), r_change=pearson(a, c - sft)) for o, c in C.items()}, sft


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--group", default="openers")
    args = p.parse_args()
    import wandb
    api = wandb.Api()
    runs = collect_runs(args.group)
    models = [m for m in (OLMO, LLAMA, GLM) if m in runs]
    out = Path("outputs/openers")
    out.mkdir(parents=True, exist_ok=True)
    all_cells, data, table, gates, orient, all_ok = {}, {}, [], [], [], True
    for m in models:
        run = runs[m]
        analysis = Path("outputs") / run.id / "analysis"
        fetch_wandb(run, analysis)
        with open(analysis / "utilities.csv", newline="") as fh:
            rows = list(csv.DictReader(fh))
        summary = json.loads((analysis / "summary.json").read_text())
        if m == OLMO:
            base_rows, base_summary = olmo_on_shared_pairs(run.config)
        else:
            base_dir = Path("outputs") / SWEEP_RUNS[m] / "analysis"
            fetch_wandb(api.run(f"{PROJECT}/{SWEEP_RUNS[m]}"), base_dir)
            with open(base_dir / "utilities.csv", newline="") as fh:
                base_rows = list(csv.DictReader(fh))
            base_summary = json.loads((base_dir / "summary.json").read_text())
        assert [r["id"] for r in rows] == [r["id"] for r in base_rows], "task sets differ"
        U, origins, benign, _ = task_arrays(rows)
        B = task_arrays(base_rows)[0]
        a = B["instruct/assistant"]
        C = {o: U[f"instruct/coinflip_{o}"] for o in OPENERS}
        data[m] = (a, C, origins)
        # Validation gate: "Okay," reproduces the sweep's coinflip.
        gate_r, gate_sd = pearson(C["okay"], B["instruct/coinflip"]), C["okay"].std() / B["instruct/coinflip"].std() - 1
        passed = gate_r >= GATE_R and abs(gate_sd) <= GATE_SD
        all_ok &= passed
        gates.append(f"{MODEL_INFO[m][0]:16} r {gate_r:.4f}, SD {gate_sd:+.1%}: {'pass' if passed else 'FAIL'}")
        resamples = stratified_resamples(origins, np.random.default_rng(0), N_BOOTSTRAP)
        cells = all_cells[m] = model_cells(a, C, benign, origins, resamples)
        for o in OPENERS:
            est, boot = cells[o]["est"], cells[o]["boot"]
            lo, hi = ci(boot)
            row = dict(model=MODEL_INFO[m][0], opener=o, call="" if o == "okay" else classify(cells, o),
                       coinflip_retention=COINFLIP_EFFECT[m][o] / COINFLIP_EFFECT[m]["okay"],
                       slot_bias=summary[f"instruct/coinflip_{o}/slot_bias_mean"],
                       reliability=summary[f"instruct/coinflip_{o}/reliability"],
                       additive_r2=summary[f"instruct/coinflip_{o}/additive_r2"],
                       mass=summary[f"instruct/coinflip_{o}/outcome_mass_mean"])
            row["r_disattenuated"] = est[K["r"]] / math.sqrt(row["reliability"] * base_summary["instruct/assistant/reliability"])
            for s in STATS:
                row.update({s: est[K[s]], f"{s}_lo": lo[K[s]], f"{s}_hi": hi[K[s]]})
            if o != "okay":
                d, dl, dh = change(cells, o)
                row.update(dr=d[K["r"]], dr_lo=dl[K["r"]], dr_hi=dh[K["r"]], dr_benign=d[K["r_benign"]],
                           dr_benign_lo=dl[K["r_benign"]], dr_benign_hi=dh[K["r_benign"]], dr_within=d[K["r_within"]],
                           dr_within_lo=dl[K["r_within"]], dr_within_hi=dh[K["r_within"]])
                k, kl, kh = retention(cells, o)
                for s in RETAINED:
                    row.update({f"keep_{s}": k[K[s]], f"keep_{s}_lo": kl[K[s]], f"keep_{s}_hi": kh[K[s]]})
            table.append(row)
        raw = load_results(Path("outputs") / run.id) if (Path("outputs") / run.id / "results").exists() else \
            fetch_raw(run, Path("outputs") / run.id)
        L = orientation(raw, {t for t, b in zip([r["id"] for r in rows], benign) if b})
        for o in OPENERS:
            h, t = L[(f"coinflip_{o}", "heads")], L[(f"coinflip_{o}", "tails")]
            orient.append(dict(model=MODEL_INFO[m][0], opener=o, L_heads_benign=h, L_tails_benign=t,
                               p_benign_heads=1 / (1 + math.exp(-h)), p_benign_tails=1 / (1 + math.exp(t))))

    print("Validation gate (\"Okay,\" vs the sweep's coinflip):\n  " + "\n  ".join(gates))
    print(f"\n{'model':16} {'opener':8} {'call':11} {'r':>18} {'r benign':>18} {'r within':>18} {'r_between':>9} "
          f"{'keep b':>18} {'keep gap':>18} {'keep sd':>6} {'r_okay':>6} {'scale':>6} {'bias':>6} {'rel':>5} {'mass':>5}")
    for r in table:
        keep = lambda s: f"{r[f'keep_{s}']:.2f} [{r[f'keep_{s}_lo']:.2f}, {r[f'keep_{s}_hi']:.2f}]" if r["opener"] != "okay" else "-"
        rs = lambda s: f"{r[s]:.2f} [{r[f'{s}_lo']:.2f}, {r[f'{s}_hi']:.2f}]"
        print(f"{r['model']:16} {r['opener']:8} {r['call']:11} {rs('r'):>18} {rs('r_benign'):>18} {rs('r_within'):>18} "
              f"{r['r_between']:9.2f} {keep('b'):>18} {keep('harm_gap'):>18} "
              f"{(r['keep_coinflip_sd'] if r['opener'] != 'okay' else 1):6.2f} {r['r_okay']:6.3f} {r['scale_okay']:6.2f} "
              f"{r['slot_bias']:6.2f} {r['reliability']:5.3f} {r['mass']:5.2f}")
    print("\nChanges from \"Okay,\" (paired):")
    for r in table:
        if r["opener"] != "okay":
            print(f"{r['model']:16} {r['opener']:8} dr {r['dr']:+.3f} [{r['dr_lo']:+.3f}, {r['dr_hi']:+.3f}]  "
                  f"benign {r['dr_benign']:+.3f} [{r['dr_benign_lo']:+.3f}, {r['dr_benign_hi']:+.3f}]  "
                  f"within {r['dr_within']:+.3f} [{r['dr_within_lo']:+.3f}, {r['dr_within_hi']:+.3f}]  "
                  f"keep gap {r['keep_harm_gap']:.2f}  keep resid {r['keep_resid_sd']:.2f}")
    across = []
    for label, ms in (("all models", models), ("without GLM", [m for m in models if m != GLM])):
        for row in across_cells(all_cells, ms):
            across.append(dict(cells=label, **row))
    print("\nAcross cells, vs retention in ../coinflip:")
    for r in across:
        print(f"{r['cells']:12} {r['y']:24} n={r['n_cells']:2d} Spearman {r['spearman']:+.2f} "
              f"[{r['spearman_lo']:+.2f}, {r['spearman_hi']:+.2f}]  Pearson {r['pearson']:+.2f} "
              f"[{r['pearson_lo']:+.2f}, {r['pearson_hi']:+.2f}]")
    print("\nExploratory, L on benign vs harmful pairs by the benign task's side:")
    for r in orient:
        print(f"{r['model']:16} {r['opener']:8} heads benign {r['L_heads_benign']:+.2f} (P(benign) {r['p_benign_heads']:.2f})  "
              f"tails benign {r['L_tails_benign']:+.2f} ({r['p_benign_tails']:.2f})")
    if OLMO in data:
        a, C, _ = data[OLMO]
        ids = [r["id"] for r in rows]
        stats, _ = sft_stage(a, C, ids)
        print("\nExploratory, OLMo's coinflip vs its SFT and DPO stages' (d8ysbybo, all pairs):")
        for o, s in stats.items():
            print(f"  {o:8} r(SFT coinflip) {s['r_sft']:.2f}  r(DPO coinflip) {s['r_dpo']:.2f}  "
                  f"r(A, C - C_SFT) {s['r_change']:.2f}")
    write_csv(out / "openers.csv", table)
    write_csv(out / "across_cells.csv", across)
    write_csv(out / "orientation.csv", orient)
    plotstyle.setup()
    plot_summary(all_cells, models, "figures/openers_summary.png")
    plot_vs_coinflip(all_cells, models, "figures/openers_vs_coinflip.png")
    plot_scatter(data, models, "figures/openers_scatter.png")
    print(f"\nGate {'passed' if all_ok else 'FAILED'}. Wrote {out}/ and figures/openers_*.png")


if __name__ == "__main__":
    main()
