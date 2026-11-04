"""Other languages: does the coinflip still track the Assistant when the whole prompt is in Spanish or Chinese?

The measures and calls were pre-registered in the README ("Other languages") before any data. Collects the
W&B runs of group "languages" (configs/languages/, one per model, language and stage). Each scores the
Assistant and coinflip conditions written in one language (prompts.LANGUAGES) on the 300 tasks translated into
it (translate.py). English utilities come from the model sweep's runs on the same pairs.

A and C are Assistant and coinflip utilities; _L is the language, _en English (the sweep); _b and _f the base
and final stages. Per model and language:

- r_final = r(A_L_f, C_L_f), the headline: the coinflip against the Assistant in the same language;
  r_en = r(A_en_f, C_en_f), the sweep's; r_fixed = r(A_L_f, C_L_b) and r_base = r(A_L_b, C_L_b);
- r_shift = r(A_L_f - A_L_b, C_L_f - C_L_b);
- gain = slope of C_L_f - C_L_b on A_L_f, in nats per Assistant SD (how far post-training tilted the coin toward
  the final Assistant); retention = gain / gain_en, gain_en the same slope in English;
- r_final over non-harmful tasks and within origin;
- the Assistant across languages: r_aa = r(A_en_f, A_L_f), r_aa_base = r(A_en_b, A_L_b), the SD ratio and the
  benign-harmful gap in each;
- the coin across languages: r_cc = r(C_en_f, C_L_f), r_cross = r(A_en_f, C_L_f), and gain_cross, the slope of
  C_L_f - C_L_b on A_en_f, as a share of gain_en (retention_cross);
- gap (non-harmful minus harmful mean) and SD of C_L at each stage.

CIs are 95% task bootstraps within origin, with one set of resamples for every model and language, so
differences are paired. `classify` gives the pre-registered call.

  outputs/languages/analysis/<family>-<lang>/   the merged fit (analyze.py) of one model's stages in one language
  outputs/languages/languages.csv                one row per model and language
  outputs/languages/contrasts.csv                paired differences
  figures/languages_*.png

Usage: uv run python languages.py [--group languages] [--english sweep]
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
from models import MODEL_INFO, PROJECT
from openers import slope_per_sd, within_groups, write_csv
from raw_text import ci, load_sweep as _load_sweep, median_mass
from report import INK, INK_2, RULE, errorbar, origin_scatter, stratified_resamples, style, task_arrays
from user_choice import top_token_shares

# family (the runs' first tag): (final model, sweep run, languages run on it). Families whose language runs were on another
# GPU type than the sweep (H200) also re-score English there (tag "en", configs/languages/*-en-*.yaml; amendment of
# 2026-11-04), and that is the English reference; the sweep is then a reproduction check.
FAMILIES = {
    "llama-70b": ("unsloth/Llama-3.3-70B-Instruct", "tvlqj8a0", ("es",)),
    "gemma4-31b": ("google/gemma-4-31B-it", "jrl62npf", ("es",)),
    "glm-4.5-air": ("zai-org/GLM-4.5-Air", "i5vejs3b", ("es", "zh")),
    "deepseek-v4-flash": ("deepseek-ai/DeepSeek-V4-Flash", "lszrxqyq", ("es", "zh")),
}
LANG_NAME = {"es": "Spanish", "zh": "Chinese", "en": "English"}
REPRO_MIN_R = 0.99  # expected r between the sweep's utilities and the same-GPU English re-score
STAGES = ("base", "instruct")
STATS = ("r_final", "r_en", "r_fixed", "r_base", "r_shift", "r_final_benign", "r_final_within", "gain", "gain_en",
         "retention", "r_aa", "r_aa_base", "sd_ratio_a", "gap_a", "gap_a_en", "r_cc", "r_cross", "gain_cross",
         "retention_cross", "gap_base", "gap_final", "gap_en", "sd_base", "sd_final", "sd_en")
K = {name: j for j, name in enumerate(STATS)}
R_SHIFT = 0.2  # "tracks" needs r_shift at least this, with a CI above 0
RETAIN = 0.25  # retention of the English coinflip's gain that counts as tracking (or reversing)
MIN_RELIABILITY, MIN_MASS = 0.9, 0.2


def cell_stats(a_en_b, a_en_f, c_en_b, c_en_f, a_b, a_f, c_b, c_f, benign, origins):
    """STATS of one model and language, over arrays of the same (possibly resampled) tasks."""
    w = lambda x: within_groups(x, origins)
    gap = lambda x: x[benign].mean() - x[~benign].mean()
    gain, gain_en, gain_cross = slope_per_sd(a_f, c_f - c_b), slope_per_sd(a_en_f, c_en_f - c_en_b), \
        slope_per_sd(a_en_f, c_f - c_b)
    return np.array([
        pearson(a_f, c_f), pearson(a_en_f, c_en_f), pearson(a_f, c_b), pearson(a_b, c_b), pearson(a_f - a_b, c_f - c_b),
        pearson(a_f[benign], c_f[benign]), pearson(w(a_f), w(c_f)), gain, gain_en, gain / gain_en,
        pearson(a_en_f, a_f), pearson(a_en_b, a_b), a_f.std() / a_en_f.std(), gap(a_f), gap(a_en_f),
        pearson(c_en_f, c_f), pearson(a_en_f, c_f), gain_cross, gain_cross / gain_en,
        gap(c_b), gap(c_f), gap(c_en_f), c_b.std(), c_f.std(), c_en_f.std(),
    ])


def boot_cell(arrays, benign, origins, resamples):
    return dict(est=cell_stats(*arrays, benign, origins),
                boot=np.array([cell_stats(*(x[i] for x in arrays), benign[i], origins[i]) for i in resamples]))


def stat_ci(cell, name):
    lo, hi = ci(cell["boot"][:, K[name]])
    return cell["est"][K[name]], lo, hi


def classify(cell):
    """The pre-registered call for one model and language (the raw-text rule, against the English coinflip)."""
    ret, ret_lo, ret_hi = stat_ci(cell, "retention")
    shift, shift_lo, _ = stat_ci(cell, "r_shift")
    if ret >= RETAIN and ret_lo > 0 and shift >= R_SHIFT and shift_lo > 0:
        return "tracks"
    if ret <= -RETAIN and ret_hi < 0:
        return "reversed"
    if abs(ret) < RETAIN:
        return "absent"
    return "unclear"


def load_sweep(family):
    import raw_text
    raw_text.FAMILIES.setdefault(family, FAMILIES[family][:2])
    return _load_sweep(family)


def collect_runs(group):
    """{(family, lang, stage): latest run with all its result files} of the W&B group."""
    import wandb
    runs = {}
    for run in sorted(wandb.Api().runs(PROJECT, filters={"group": group, "jobType": "tournament"}),
                      key=lambda r: r.created_at):
        family, lang = run.config["tags"]  # lang "en": the same-GPU English re-score
        (m,) = run.config["models"]
        have = {f.name for f in run.files()}
        if all(f"results/{m['stage']}__{c}.csv.gz" in have for c in m["conditions"]):
            runs[family, lang, m["stage"]] = run
    return runs


def cond(c, lang):
    return c if lang == "en" else f"{c}_{lang}"


def tasks_file(lang):
    return "data/tasks_n300_max500_seed0.jsonl" if lang == "en" else f"data/tasks_n300_max500_seed0_{lang}.jsonl"


def merged_fit(runs, family, lang, out):
    """(utilities.csv rows, summary, {(stage, condition): raw results path}) of a model's stages in one language."""
    rows, paths = [], {}
    for s in STAGES:
        run = runs[family, lang, s]
        print(f"{family:18} {lang} {s:9} {run.path}")
        rows += fetch_raw(run, Path("outputs") / run.id)
        for c in ("assistant", "coinflip"):
            paths[s, c] = Path("outputs") / run.id / "results" / f"{s}__{cond(c, lang)}.csv.gz"
    if not (out / "utilities.csv").exists():
        analyze(rows, read_jsonl(tasks_file(lang)), out, seed=0, stage_order=list(STAGES))
    with open(out / "utilities.csv", newline="") as fh:
        return list(csv.DictReader(fh)), json.loads((out / "summary.json").read_text()), paths


def plot_summary(cells, labels, path, unreadable=None):
    """Per model and language: r with the Assistant in the same language (English for reference), the share of
    the English coinflip's post-training tilt it keeps, and how well the Assistant's ranking survives translation.
    unreadable: {(family, lang): median outcome mass} of cells whose final coinflip fails the readability rule; their
    coinflip statistics are drawn faded and labelled, since the pre-registration doesn't interpret them."""
    unreadable = unreadable or {}
    keys = list(cells)
    x = np.arange(len(keys))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    panels = (
        (axes[0], (("r_en", "English", 0), ("r_final", "this language", 1)), "r(Assistant, coinflip), final model", True),
        (axes[1], (("retention", "same-language Assistant", 1), ("retention_cross", "English Assistant", 2)),
         "Post-training tilt toward the final Assistant,\nshare of the English coinflip's", True),
        (axes[2], (("r_aa", "final", 1), ("r_aa_base", "base", 3)), "r(English Assistant, this language's Assistant)", False),
    )
    for ax, series, ylab, uses_coin in panels:
        for j, (stat, name, color) in enumerate(series):
            off = x + (j - (len(series) - 1) / 2) * 0.22
            est = np.array([cells[k]["est"][K[stat]] for k in keys])
            lo, hi = (np.array(v) for v in zip(*(ci(cells[k]["boot"][:, K[stat]]) for k in keys)))
            lo, hi = np.minimum(lo, est), np.maximum(hi, est)
            faded = np.array([uses_coin and stat != "r_en" and k in unreadable for k in keys])
            ok = ~faded
            errorbar(ax, off[ok], est[ok], lo[ok], hi[ok], plotstyle.SERIES[color], "o", name, ls="none")
            if faded.any():
                ax.errorbar(off[faded], est[faded], yerr=[est[faded] - lo[faded], hi[faded] - est[faded]], ls="none",
                            marker="o", ms=7, mfc="white", mec=plotstyle.SERIES[color], color=plotstyle.SERIES[color],
                            alpha=0.45, elinewidth=1.2, zorder=3)
        if uses_coin:
            for i, k in enumerate(keys):
                if k in unreadable:
                    stats = [st for st, _, _ in series if st != "r_en"]
                    y = min(ci(cells[k]["boot"][:, K[st]])[0] for st in stats) - 0.03
                    ax.annotate(f"coinflip unreadable\n(median mass {unreadable[k]:.2f})", (x[i], y), ha="center",
                                va="top", fontsize=7, color=INK_2)
        ax.axhline(0, color=RULE, lw=0.8, zorder=1)
        ax.set_ylabel(ylab, color=INK_2)
        ax.set_xticks(x, [f"{labels[f]}\n{LANG_NAME[lang]}" for f, lang in keys], fontsize=8)
        ax.legend(frameon=False, fontsize=8, loc="lower left")
        style(ax)
    axes[1].axhline(1, color=RULE, lw=0.8, ls="--", zorder=1)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scatter(data, origins, labels, path):
    """Per model and language: the Assistant across languages, and the coinflip against the Assistant in the same
    language and in English. Axes differ by panel."""
    keys = list(data)
    cols = (("A_en_f", "A_f", "English Assistant", "this language's Assistant"),
            ("A_f", "C_f", "this language's Assistant", "this language's coinflip"),
            ("C_en_f", "C_f", "English coinflip", "this language's coinflip"))
    fig, axes = plt.subplots(len(keys), len(cols), figsize=(3.4 * len(cols), 2.9 * len(keys)), squeeze=False)
    for row, k in enumerate(keys):
        f, lang = k
        for col, (xk, yk, xl, yl) in enumerate(cols):
            ax = axes[row, col]
            xv, yv = data[k][xk], data[k][yk]
            origin_scatter(ax, xv, yv, origins)
            ax.set_title(f"r = {pearson(xv, yv):.2f}", fontsize=9, color=INK_2, loc="left")
            ax.set_xlabel(f"{xl} (nats)", fontsize=8, color=INK_2)
            if col == 0:
                ax.set_ylabel(f"{labels[f]}, {LANG_NAME[lang]}\n{yl} (nats)", fontsize=8)
            else:
                ax.set_ylabel(f"{yl} (nats)", fontsize=8, color=INK_2)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, frameon=False, ncol=len(ORIGIN_STYLE), loc="upper center", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--group", default="languages")
    p.add_argument("--english", choices=["same-gpu", "sweep"], default="same-gpu",
                   help="English reference: the same-GPU re-score where there is one (pre-registered), or the H200 sweep")
    args = p.parse_args()
    out = Path("outputs/languages" if args.english == "same-gpu" else "outputs/languages_sweep_ref")
    fig_prefix = "figures/languages" if args.english == "same-gpu" else "figures/languages_sweep_ref"
    out.mkdir(parents=True, exist_ok=True)
    runs = collect_runs(args.group)
    keys = [(f, lang) for f, (_, _, langs) in FAMILIES.items() for lang in langs
            if all((f, lang, s) in runs for s in STAGES)]
    missing = [(f, lang) for f, (_, _, langs) in FAMILIES.items() for lang in langs if (f, lang) not in keys]
    if missing:
        print(f"Not yet complete, skipped: {missing}")

    data, cells, table, gates, labels = {}, {}, [], [], {}
    ids0 = resamples = origins0 = None
    for f, lang in keys:
        final = FAMILIES[f][0]
        labels[f] = MODEL_INFO[final][0]
        rows, summary, paths = merged_fit(runs, f, lang, Path("outputs/languages/analysis") / f"{f}-{lang}")
        sweep_rows, sweep_summary = load_sweep(f)
        ids = [r["id"] for r in rows]
        assert ids == [r["id"] for r in sweep_rows], "task sets differ"
        U, origins, benign, _ = task_arrays(rows)
        B = task_arrays(sweep_rows)[0]
        if args.english == "same-gpu" and all((f, "en", s) in runs for s in STAGES):
            # The same-GPU English re-score is the reference; the sweep is a reproduction check.
            en_rows, en_summary, _ = merged_fit(runs, f, "en", Path("outputs/languages/analysis") / f"{f}-en")
            assert [r["id"] for r in en_rows] == ids, "task sets differ"
            E = task_arrays(en_rows)[0]
            for m in ("base/assistant", "instruct/assistant", "base/coinflip", "instruct/coinflip"):
                r_repro = pearson(B[m], E[m])
                gates.append(f"{labels[f]:18} English re-score vs sweep, {m:18} r {r_repro:.4f}, SD {E[m].std():.2f} vs "
                             f"{B[m].std():.2f}{'' if r_repro >= REPRO_MIN_R else '  BELOW ' + str(REPRO_MIN_R)}")
            B, sweep_summary = E, en_summary
        else:
            gates.append(f"{labels[f]:18} English reference: the sweep (no same-GPU re-score)")
        if ids0 is None:
            ids0, origins0 = ids, origins
            resamples = stratified_resamples(origins, np.random.default_rng(0), N_BOOTSTRAP)
        assert ids == ids0, "task sets differ between models"
        d = data[f, lang] = {"A_en_b": B["base/assistant"], "A_en_f": B["instruct/assistant"],
                             "C_en_b": B["base/coinflip"], "C_en_f": B["instruct/coinflip"],
                             "A_b": U[f"base/assistant_{lang}"], "A_f": U[f"instruct/assistant_{lang}"],
                             "C_b": U[f"base/coinflip_{lang}"], "C_f": U[f"instruct/coinflip_{lang}"]}
        arrays = tuple(d[k] for k in ("A_en_b", "A_en_f", "C_en_b", "C_en_f", "A_b", "A_f", "C_b", "C_f"))
        cell = cells[f, lang] = boot_cell(arrays, benign, origins, resamples)
        row = dict(model=labels[f], lang=lang, call=classify(cell))
        for s in STAGES:
            for c in ("assistant", "coinflip"):
                cond = f"{c}_{lang}"
                rel, mass = summary[f"{s}/{cond}/reliability"], median_mass(paths[s, c])
                row.update({f"{s}_{c}_reliability": rel, f"{s}_{c}_mass_median": mass,
                            f"{s}_{c}_slot_bias": summary[f"{s}/{cond}/slot_bias_mean"]})
                gates.append(f"{labels[f]:18} {lang} {s:9} {c:10} reliability {rel:.3f}, median mass {mass:.2f}"
                             f"{'' if mass >= MIN_MASS else '  UNREADABLE'}{'' if rel >= MIN_RELIABILITY else '  LOW RELIABILITY'}")
        # Disattenuated by the split-half reliabilities of the two utilities correlated.
        rel = lambda s, c: summary[f"{s}/{c}_{lang}/reliability"]
        row["r_final_disattenuated"] = cell["est"][K["r_final"]] / np.sqrt(rel("instruct", "assistant") * rel("instruct", "coinflip"))
        row["r_aa_disattenuated"] = cell["est"][K["r_aa"]] / np.sqrt(
            rel("instruct", "assistant") * sweep_summary["instruct/assistant/reliability"])
        for name in STATS:
            est, lo, hi = stat_ci(cell, name)
            row.update({name: est, f"{name}_lo": lo, f"{name}_hi": hi})
        for c in ("assistant", "coinflip"):
            row[f"top_tokens_final_{c}"] = json.dumps([[t, round(p, 3)] for t, p in top_token_shares(paths["instruct", c])])
        table.append(row)

    print("Reliability and readability (median outcome mass >= 0.2):\n  " + "\n  ".join(gates))
    fm = lambda r, s: f"{r[s]:+.2f} [{r[f'{s}_lo']:+.2f}, {r[f'{s}_hi']:+.2f}]"
    for r in table:
        print(f"\n{r['model']}, {LANG_NAME[r['lang']]}: call {r['call']}")
        for name, label in (("r_final", "r(A_L_f, C_L_f)"), ("r_en", "r(A_en_f, C_en_f)"), ("r_fixed", "r(A_L_f, C_L_b)"),
                            ("r_base", "r(A_L_b, C_L_b)"), ("r_shift", "r shift"), ("r_final_benign", "r non-harmful"),
                            ("r_final_within", "r within origin"), ("gain", "gain (nats/SD)"), ("gain_en", "gain, English"),
                            ("retention", "retention"), ("r_aa", "r(A_en_f, A_L_f)"), ("r_aa_base", "r(A_en_b, A_L_b)"),
                            ("sd_ratio_a", "SD A_L_f / A_en_f"), ("gap_a", "Assistant gap, L"), ("gap_a_en", "Assistant gap, en"),
                            ("r_cc", "r(C_en_f, C_L_f)"), ("r_cross", "r(A_en_f, C_L_f)"), ("retention_cross", "retention, A_en"),
                            ("gap_base", "coin gap, base"), ("gap_final", "coin gap, final"), ("gap_en", "coin gap, en final"),
                            ("sd_base", "coin SD, base"), ("sd_final", "coin SD, final"), ("sd_en", "coin SD, en final")):
            print(f"    {label:20} {fm(r, name)}")
        print(f"    disattenuated r(A_L_f, C_L_f) {r['r_final_disattenuated']:+.2f}, r(A_en_f, A_L_f) {r['r_aa_disattenuated']:+.2f}; "
              f"slot bias (coin) {r['base_coinflip_slot_bias']:+.2f} / {r['instruct_coinflip_slot_bias']:+.2f}")
        print(f"    top tokens (final): assistant {r['top_tokens_final_assistant']}; coinflip {r['top_tokens_final_coinflip']}")

    contrasts = []
    for f, lang in keys:
        cell = cells[f, lang]
        for stat, ref in (("r_final", "r_en"),):
            lo, hi = ci(cell["boot"][:, K[stat]] - cell["boot"][:, K[ref]])
            contrasts.append(dict(model=labels[f], contrast=f"{lang} - English", stat=stat,
                                  est=cell["est"][K[stat]] - cell["est"][K[ref]], lo=lo, hi=hi))
    for f in FAMILIES:
        if (f, "es") in cells and (f, "zh") in cells:
            for stat in ("r_final", "retention", "r_aa"):
                a, b = cells[f, "zh"], cells[f, "es"]
                lo, hi = ci(a["boot"][:, K[stat]] - b["boot"][:, K[stat]])
                contrasts.append(dict(model=labels[f], contrast="zh - es", stat=stat,
                                      est=a["est"][K[stat]] - b["est"][K[stat]], lo=lo, hi=hi))
    print("\nPaired differences:")
    for c in contrasts:
        print(f"  {c['model']:18} {c['contrast']:14} {c['stat']:10} {c['est']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]")

    write_csv(out / "languages.csv", table)
    write_csv(out / "contrasts.csv", contrasts)
    plotstyle.setup()
    unreadable = {(f, r["lang"]): r["instruct_coinflip_mass_median"] for r in table for f in FAMILIES
                  if labels.get(f) == r["model"] and r["instruct_coinflip_mass_median"] < MIN_MASS}
    plot_summary(cells, labels, f"{fig_prefix}_summary.png", unreadable)
    plot_scatter(data, origins0, labels, f"{fig_prefix}_scatter.png")
    print(f"\nWrote {out}/ and {fig_prefix}_*.png")


if __name__ == "__main__":
    main()
