"""The coinflip as raw text: does it still track the chat Assistant's preferences?

The measures and calls were pre-registered in the README ("The coinflip as raw text") before any data.
Collects the W&B runs of group "raw_text" (configs/raw_text/, one per model and stage). Each scores the
coinflip prompt as a bare document (the model's document-start token(s), no chat template) in four
conditions (prompts.py): its own first-person words with and without "Okay, " (raw_okay, raw_no_okay), and
../coinflip's third-person stories about an AI assistant or a freelancer (raw_story_ai, raw_story_person).
A and C, the chat Assistant and chat coinflip utilities, come from the model sweep's runs on the same
pairs. X is one raw-text condition's utilities; _b and _f are the base and final stages. Per model and X:

- r_final = r(A_f, X_f), the headline; r_base = r(A_b, X_b); r_fixed = r(A_f, X_b), the base model's X
  against the final Assistant, so d = r_final - r_fixed asks whether post-training brought X closer to a
  fixed target;
- r_shift = r(A_f - A_b, X_f - X_b);
- gain = slope of X_f - X_b on A_f, in nats per Assistant SD: how much post-training tilted X toward the
  final Assistant's preferences. retention = gain / the chat coinflip's gain (C_f - C_b on A_f);
- r_final over non-harmful tasks and within origin; b_final, the slope of X_f on A_f; gap (non-harmful
  minus harmful mean) and SD at each stage; r(X_f, C_f).

CIs are 95% task bootstraps within origin, with one set of resamples for every model and condition, so
differences are paired. `classify` gives the pre-registered call per model and condition.

  outputs/raw_text/analysis/<family>/   the merged fit (analyze.py) of one model's stages
  outputs/raw_text/raw_text.csv         one row per model and condition
  outputs/raw_text/contrasts.csv        the paired differences
  figures/raw_text_*.png

Usage: uv run python raw_text.py [--group raw_text]
       uv run python raw_text.py --sweep-only   (the chat coinflip's numbers on these models, no new data)
"""

import argparse
import csv
import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import N_BOOTSTRAP, ORIGIN_STYLE, analyze, fetch_wandb as fetch_raw, pearson, read_jsonl
from matskit import plotstyle
from models import MODEL_INFO, PROJECT, olmo_on_shared_pairs
from openers import slope_per_sd, within_groups, write_csv
from report import INK, INK_2, RULE, errorbar, fetch_wandb, origin_scatter, stratified_resamples, style, task_arrays
from user_choice import top_token_shares

# family (the runs' tag): (final model, sweep run; OLMo is d8ysbybo refit on the shared pairs).
FAMILIES = {
    "olmo3.1-32b": ("allenai/Olmo-3.1-32B-Instruct", None),
    "gemma4-31b": ("google/gemma-4-31B-it", "jrl62npf"),
    "llama-70b": ("unsloth/Llama-3.3-70B-Instruct", "tvlqj8a0"),
    "glm-4.5-air": ("zai-org/GLM-4.5-Air", "i5vejs3b"),
    "deepseek-v4-flash": ("deepseek-ai/DeepSeek-V4-Flash", "lszrxqyq"),
}
RAW = ("raw_okay", "raw_no_okay", "raw_story_ai", "raw_story_person")
LABEL = {"coinflip": "chat coinflip", "raw_okay": 'raw, "Okay, I\'m flipping"', "raw_no_okay": 'raw, "I\'m flipping"',
         "raw_story_ai": "story: AI assistant", "raw_story_person": "story: freelancer"}
STAGES = ("base", "instruct")
TASKS_FILE = "data/tasks_n300_max500_seed0.jsonl"
SHARED_PAIRS = {"max_pairs": 15000, "seed": 0, "tasks_file": TASKS_FILE}
STATS = ("r_final", "r_base", "r_fixed", "r_shift", "r_final_benign", "r_final_within", "b_final", "gain", "gain_c",
         "retention", "gap_base", "gap_final", "sd_base", "sd_final", "r_xc", "rc_final", "rc_fixed")
K = {name: j for j, name in enumerate(STATS)}
R_SHIFT = 0.2  # "tracks" needs r_shift at least this, with a CI above 0
RETAIN = 0.25  # retention of the chat coinflip's gain that counts as tracking (or reversing)
MIN_RELIABILITY, MIN_MASS = 0.9, 0.2


def cell_stats(ab, af, xb, xf, cb, cf, benign, origins):
    """STATS of one model and condition, over arrays of the same (possibly resampled) tasks."""
    w = lambda x: within_groups(x, origins)
    gap = lambda x: x[benign].mean() - x[~benign].mean()
    gain, gain_c = slope_per_sd(af, xf - xb), slope_per_sd(af, cf - cb)
    return np.array([
        pearson(af, xf), pearson(ab, xb), pearson(af, xb), pearson(af - ab, xf - xb),
        pearson(af[benign], xf[benign]), pearson(w(af), w(xf)), slope_per_sd(af, xf), gain, gain_c, gain / gain_c,
        gap(xb), gap(xf), xb.std(), xf.std(), pearson(xf, cf), pearson(af, cf), pearson(af, cb),
    ])


def boot_cell(arrays, benign, origins, resamples):
    """{"est": STATS, "boot": (n_resamples, STATS)}. arrays: (A_b, A_f, X_b, X_f, C_b, C_f)."""
    return dict(est=cell_stats(*arrays, benign, origins),
                boot=np.array([cell_stats(*(x[i] for x in arrays), benign[i], origins[i]) for i in resamples]))


def ci(boot):
    return np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)


def stat_ci(cell, name):
    lo, hi = ci(cell["boot"][:, K[name]])
    return cell["est"][K[name]], lo, hi


def classify(cell):
    """The pre-registered call for one model and raw-text condition."""
    ret, ret_lo, ret_hi = stat_ci(cell, "retention")
    shift, shift_lo, _ = stat_ci(cell, "r_shift")
    if ret >= RETAIN and ret_lo > 0 and shift >= R_SHIFT and shift_lo > 0:
        return "tracks"
    if ret <= -RETAIN and ret_hi < 0:
        return "reversed"
    if abs(ret) < RETAIN:
        return "absent"
    return "unclear"


def paired(cells, family, x, y, stat):
    """(estimate, lo, hi) of stat in condition x minus condition y, paired over resamples."""
    a, b = cells[family, x], cells[family, y]
    lo, hi = ci(a["boot"][:, K[stat]] - b["boot"][:, K[stat]])
    return a["est"][K[stat]] - b["est"][K[stat]], lo, hi


def load_sweep(family):
    """(utilities.csv rows, summary) of the sweep's run for one model."""
    _, run_id = FAMILIES[family]
    if run_id is None:
        return olmo_on_shared_pairs(SHARED_PAIRS)
    import wandb
    analysis = Path("outputs") / run_id / "analysis"
    fetch_wandb(wandb.Api().run(f"{PROJECT}/{run_id}"), analysis)
    with open(analysis / "utilities.csv", newline="") as fh:
        return list(csv.DictReader(fh)), json.loads((analysis / "summary.json").read_text())


def collect_runs(group):
    """{(family, stage): latest run with all its result files} of the W&B group."""
    import wandb
    runs = {}
    for run in sorted(wandb.Api().runs(PROJECT, filters={"group": group, "jobType": "tournament"}),
                      key=lambda r: r.created_at):
        (family,) = run.config["tags"]
        (m,) = run.config["models"]
        have = {f.name for f in run.files()}
        if all(f"results/{m['stage']}__{c}.csv.gz" in have for c in m["conditions"]):
            runs[family, m["stage"]] = run
    return runs


def merged_fit(runs, family, out):
    """(utilities.csv rows, summary, {(stage, condition): raw results path}) of a model's stages, fit together."""
    rows, paths = [], {}
    for s in STAGES:
        run = runs[family, s]
        print(f"{family:18} {s:9} {run.path}")
        rows += fetch_raw(run, Path("outputs") / run.id)
        for c in RAW:
            paths[s, c] = Path("outputs") / run.id / "results" / f"{s}__{c}.csv.gz"
    if not (out / "utilities.csv").exists():
        analyze(rows, read_jsonl(TASKS_FILE), out, seed=0, stage_order=list(STAGES))
    with open(out / "utilities.csv", newline="") as fh:
        return list(csv.DictReader(fh)), json.loads((out / "summary.json").read_text()), paths


def median_mass(path):
    with gzip.open(path, "rt", newline="") as fh:
        return float(np.median([float(r["outcome_mass"]) for r in csv.DictReader(fh)]))


def sweep_only():
    """The chat coinflip's numbers on these models, for the pre-registration."""
    for family in FAMILIES:
        rows, summary = load_sweep(family)
        U, origins, benign, _ = task_arrays(rows)
        ab, af, cb, cf = U["base/assistant"], U["instruct/assistant"], U["base/coinflip"], U["instruct/coinflip"]
        s = cell_stats(ab, af, cb, cf, cb, cf, benign, origins)
        print(f"{family:18} r(A_f, C_f) {s[K['r_final']]:.2f}  r(A_f, C_b) {s[K['r_fixed']]:.2f}  "
              f"r(A_b, C_b) {s[K['r_base']]:.2f}  r shift {s[K['r_shift']]:.2f}  non-harmful {s[K['r_final_benign']]:.2f}  "
              f"within {s[K['r_final_within']]:.2f}  gain {s[K['gain']]:.3f}  b_f {s[K['b_final']]:.3f}  "
              f"gap {s[K['gap_base']]:.2f} -> {s[K['gap_final']]:.2f}  SD {s[K['sd_base']]:.2f} -> {s[K['sd_final']]:.2f}  "
              f"mass {summary['instruct/coinflip/outcome_mass_mean']:.2f}")


def plot_r(cells, labels, path):
    """r with the final Assistant per model and condition: the final model's (filled) and the base model's (open)."""
    conds = ("coinflip",) + RAW
    families = list(labels)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    x = np.arange(len(families))
    width = 0.15
    for j, c in enumerate(conds):
        col = plotstyle.SERIES[j]
        off = x + (j - (len(conds) - 1) / 2) * width
        for ax, (stat, ylab) in zip(axes, (("r_final", "r with the final Assistant"),
                                           ("retention", "Post-training tilt toward the final Assistant,\nshare of the chat coinflip's"))):
            if c == "coinflip":
                est = np.array([cells[f, RAW[0]]["est"][K["rc_final" if stat == "r_final" else "retention"]] for f in families])
                if stat == "retention":
                    est = np.ones(len(families))
                lo = hi = est
            else:
                est = np.array([cells[f, c]["est"][K[stat]] for f in families])
                lo, hi = (np.array(v) for v in zip(*(ci(cells[f, c]["boot"][:, K[stat]]) for f in families)))
            errorbar(ax, off, est, np.minimum(lo, est), np.maximum(hi, est), col, "o", LABEL[c], ls="none")
            if stat == "r_final":
                fixed = "rc_fixed" if c == "coinflip" else "r_fixed"
                cc = RAW[0] if c == "coinflip" else c
                ax.plot(off, [cells[f, cc]["est"][K[fixed]] for f in families], ls="none", marker="o", ms=6,
                        mfc="white", mec=col, mew=1.2, zorder=3)
            ax.set_ylabel(ylab, color=INK_2)
    for ax in axes:
        ax.axhline(0, color=RULE, lw=0.8, zorder=1)
        ax.set_xticks(x, [labels[f] for f in families])
        style(ax)
    axes[0].set_title("Filled: final model's utilities. Open: base model's, against the same final Assistant",
                      loc="left", fontsize=9, color=INK_2)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, frameon=False, fontsize=9, ncol=len(conds), loc="lower center")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scatter(data, origins, labels, path):
    """Each final model's chat coinflip and raw-text utilities against its chat Assistant. Axes differ by panel."""
    conds = ("coinflip",) + RAW
    fig, axes = plt.subplots(len(labels), len(conds), figsize=(3.0 * len(conds), 2.8 * len(labels)), squeeze=False)
    for row, family in enumerate(labels):
        af = data[family]["A_f"]
        for col, c in enumerate(conds):
            ax = axes[row, col]
            y = data[family]["C_f"] if c == "coinflip" else data[family][c][1]
            origin_scatter(ax, af, y, origins)
            ax.set_title(f"r = {pearson(af, y):.2f}, SD {y.std():.2f}", fontsize=9, color=INK_2, loc="left")
            if row == 0:
                ax.text(0, 1.16, LABEL[c], transform=ax.transAxes, fontsize=10, color=INK)
            if row == len(labels) - 1:
                ax.set_xlabel("Chat Assistant utility (nats)", fontsize=8, color=INK_2)
        axes[row, 0].set_ylabel(f"{labels[family]}\nutility (nats)")
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, frameon=False, ncol=len(ORIGIN_STYLE), loc="upper center", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--group", default="raw_text")
    p.add_argument("--sweep-only", action="store_true", help="print the chat coinflip's numbers and stop")
    args = p.parse_args()
    if args.sweep_only:
        sweep_only()
        return

    out = Path("outputs/raw_text")
    out.mkdir(parents=True, exist_ok=True)
    runs = collect_runs(args.group)
    families = [f for f in FAMILIES if (f, "base") in runs and (f, "instruct") in runs]
    missing = [f for f in FAMILIES if f not in families]
    if missing:
        print(f"Not yet complete, skipped: {missing}")
    data, cells, table, gates, labels = {}, {}, [], [], {}
    ids0, resamples, origins0 = None, None, None
    for family in families:
        final, _ = FAMILIES[family]
        rows, summary, paths = merged_fit(runs, family, out / "analysis" / family)
        sweep_rows, _ = load_sweep(family)
        ids = [r["id"] for r in rows]
        assert ids == [r["id"] for r in sweep_rows], "task sets differ"
        U, origins, benign, _ = task_arrays(rows)
        B = task_arrays(sweep_rows)[0]
        if ids0 is None:
            ids0, origins0 = ids, origins
            resamples = stratified_resamples(origins, np.random.default_rng(0), N_BOOTSTRAP)
        assert ids == ids0, "task sets differ between models"
        labels[family] = MODEL_INFO[final][0]
        data[family] = {"A_b": B["base/assistant"], "A_f": B["instruct/assistant"],
                        "C_b": B["base/coinflip"], "C_f": B["instruct/coinflip"]}
        for c in RAW:
            xb, xf = U[f"base/{c}"], U[f"instruct/{c}"]
            data[family][c] = (xb, xf)
            arrays = (B["base/assistant"], B["instruct/assistant"], xb, xf, B["base/coinflip"], B["instruct/coinflip"])
            cell = cells[family, c] = boot_cell(arrays, benign, origins, resamples)
            row = dict(model=labels[family], condition=c, call=classify(cell))
            for s in STAGES:
                rel, mass = summary[f"{s}/{c}/reliability"], median_mass(paths[s, c])
                row.update({f"{s}_reliability": rel, f"{s}_mass_median": mass,
                            f"{s}_slot_bias": summary[f"{s}/{c}/slot_bias_mean"]})
                gates.append(f"{labels[family]:18} {s:9} {c:17} reliability {rel:.3f}, median mass {mass:.2f}"
                             f"{'' if mass >= MIN_MASS else '  UNREADABLE'}{'' if rel >= MIN_RELIABILITY else '  LOW RELIABILITY'}")
            row["r_final_disattenuated"] = cell["est"][K["r_final"]] / np.sqrt(summary[f"instruct/{c}/reliability"])
            for name in STATS:
                est, lo, hi = stat_ci(cell, name)
                row.update({name: est, f"{name}_lo": lo, f"{name}_hi": hi})
            table.append(row)
            tops = {s: top_token_shares(paths[s, c]) for s in STAGES}
            row["top_tokens_final"] = json.dumps([[t, round(p, 3)] for t, p in tops["instruct"]])

    print("Reliability and readability (median P(heads) + P(tails) >= 0.2):\n  " + "\n  ".join(gates))
    fm = lambda r, s: f"{r[s]:+.2f} [{r[f'{s}_lo']:+.2f}, {r[f'{s}_hi']:+.2f}]"
    for family in families:
        print(f"\n{labels[family]}: chat coinflip r(A_f, C_f) {cells[family, RAW[0]]['est'][K['rc_final']]:.2f}, "
              f"gain {cells[family, RAW[0]]['est'][K['gain_c']]:.3f}")
        for r in (r for r in table if r["model"] == labels[family]):
            print(f"  {LABEL[r['condition']]:26} call: {r['call']}")
            for name, label in (("r_final", "r(A_f, X_f)"), ("r_fixed", "r(A_f, X_b)"), ("r_base", "r(A_b, X_b)"),
                                ("r_shift", "r shift"), ("r_final_benign", "r non-harmful"), ("r_final_within", "r within origin"),
                                ("gain", "gain (nats/SD)"), ("retention", "retention"), ("b_final", "b, X_f on A_f"),
                                ("gap_base", "gap, base"), ("gap_final", "gap, final"), ("sd_base", "SD, base"),
                                ("sd_final", "SD, final"), ("r_xc", "r(X_f, C_f)")):
                print(f"    {label:18} {fm(r, name)}")
            print(f"    disattenuated r {r['r_final_disattenuated']:+.2f}; slot bias {r['base_slot_bias']:+.2f} / "
                  f"{r['instruct_slot_bias']:+.2f}; top tokens (final) {r['top_tokens_final']}")

    contrasts = []
    pairs = [(c, "coinflip_ref") for c in RAW] + [("raw_story_ai", "raw_no_okay"), ("raw_story_person", "raw_no_okay"),
                                                   ("raw_story_ai", "raw_story_person"), ("raw_no_okay", "raw_okay")]
    for family in families:
        for x, y in pairs:
            for stat in ("r_final", "gain"):
                if y == "coinflip_ref":
                    # Against the chat coinflip: r(A_f, X_f) - r(A_f, C_f), and gain - the coinflip's gain.
                    ref = "rc_final" if stat == "r_final" else "gain_c"
                    cell = cells[family, x]
                    lo, hi = ci(cell["boot"][:, K[stat]] - cell["boot"][:, K[ref]])
                    est = cell["est"][K[stat]] - cell["est"][K[ref]]
                    name = f"{x} - chat coinflip"
                else:
                    est, lo, hi = paired(cells, family, x, y, stat)
                    name = f"{x} - {y}"
                contrasts.append(dict(model=labels[family], contrast=name, stat=stat, est=est, lo=lo, hi=hi))
    print("\nPaired differences:")
    for c in contrasts:
        print(f"  {c['model']:18} {c['contrast']:36} {c['stat']:8} {c['est']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]")

    write_csv(out / "raw_text.csv", table)
    write_csv(out / "contrasts.csv", contrasts)
    plotstyle.setup()
    plot_r(cells, labels, "figures/raw_text_r.png")
    plot_scatter(data, origins0, labels, "figures/raw_text_scatter.png")
    print(f"\nWrote {out}/ and figures/raw_text_*.png")


if __name__ == "__main__":
    main()
