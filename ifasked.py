"""Shallow instructions: "If asked what you prefer in the next message, please respond according to <rule>." at the start of
the user's message, with an arbitrary ranking rule in place of a persona.

The measures and calls were pre-registered in the README ("Shallow instructions: 'if asked what you prefer in the next
message'") before any data. Per model (Llama 3.3 70B, Gemma 4 31B, GLM-4.5-Air) and rule k (shorter, longer, alpha,
enjoy_less; prompts.IFASKED_RULES), all final-stage utilities on the sweep's 300 tasks and 15,000 pairs:
- A_0, C_0: the default Assistant and coin (the sweep);
- A_k, C_k: the Assistant's choice and the coinflip with rule k's instruction (W&B group "ifasked");
- S_k: the rule's own score per task, higher = the rule picks it: shorter -log length, longer +log length, alpha minus the
  alphabetical rank (rule_scores), enjoy_less -A_0.

Primary: the persona section's two-pull fit C_k = c + b_def z(A_0) + b_ins z(A_k), w = b_ins / (b_def + b_ins), and its call
(call); with |r(A_k, A_0)| > 0.9 the pulls can't be separated, and the call is on q, the coin's slope on A_0 relative to the
default coin's (q_call). The length pair also gets the mirrored contrast (C_shorter - C_longer) / 2 against -log length
(length_call). CIs are 95% task bootstraps within origin, one set of resamples for every model and statistic.

  outputs/ifasked/<family>/analysis/   the merged fit (analyze.py) of one model's runs
  outputs/ifasked/<family>/gate/<run>/ each pod's re-scored plain coinflip, fit on its own
  outputs/ifasked/ifasked.csv          one row per model and rule: every statistic with its CI, and the calls
  outputs/ifasked/report.txt           (written by redirecting stdout)
  figures/ifasked_*.png

Usage: uv run python ifasked.py [--models llama-70b] > outputs/ifasked/report.txt
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import personas
import prompts
from analyze import N_BOOTSTRAP, pearson, read_jsonl
from matskit import plotstyle
from models import MODEL_INFO
from openers import slope_per_sd, write_csv
from personas import z
from personas_user import median_mass, pulls
from report import INK, INK_2, RULE, origin_scatter, stratified_resamples, style, task_arrays
from source_grid import Row, plot_source_grid
from user_choice import TASKS_FILE, ci, top_token_shares

MODELS = {"llama-70b": "unsloth/Llama-3.3-70B-Instruct", "gemma4-31b": "google/gemma-4-31B-it",
          "glm-4.5-air": "zai-org/GLM-4.5-Air"}
RULES = tuple(prompts.IFASKED_RULES)
RULE_LABEL = {"shorter": "Shorter description", "longer": "Longer description", "alpha": "First alphabetically",
              "enjoy_less": "Task you'd enjoy less"}
SCORE_LABEL = {"shorter": "-log length", "longer": "log length", "alpha": "alphabetical position (earlier = higher)",
               "enjoy_less": "-A_0"}
GATE_R, GATE_SD, MIN_RELIABILITY, MIN_MASS = 0.999, 0.02, 0.9, 0.2
SEPARABLE = personas.SEPARABLE  # |r(A_k, A_0)| above this: the two pulls can't be told apart
PER = ("b_def", "b_ins", "w", "total",  # primary: C_k on z(A_0), z(A_k)
       "r_ak_a0", "q",  # separability; q = slope of C_k on A_0 / b_0, the call when not separable
       "r_ak_s", "agree_a", "agree_a0", "agree_c", "agree_c0",  # manipulation check and the rule's order in each utility
       "dagree_a", "dagree_c",  # agree_a - agree_a0, agree_c - agree_c0: how far each moves toward the rule
       "r_ak_a0_nh",  # r(A_k, A_0) over non-harmful tasks
       "b_def_s", "b_s",  # C_k on z(A_0), z(S_k): the pull toward the rule itself (not for enjoy_less, where S = -A_0)
       "r_dc_da", "tau",  # what the instruction changes: r(C_k - C_0, A_k - A_0), and its slope over the default coin's
       "b_def_fe", "b_ins_fe", "w_fe",  # with source fixed effects (plus log length for alpha and enjoy_less)
       "w_nh",  # non-harmful tasks only
       "gap_ak", "gap_ck", "r_ck_c0", "r_ck_ak", "r_ck_a0", "sd_ratio")
LENGTH = ("b_dc", "b_da", "r_dcda", "agree_dc")  # the mirrored length contrast
CONTROL = ("b0", "gap_a0", "gap_c0", "r_a0_len", "r_c0_len")
STATS = tuple(f"{k}:{s}" for k in RULES for s in PER) + tuple(f"len:{s}" for s in LENGTH) + CONTROL
K = {name: j for j, name in enumerate(STATS)}


def rule_scores(descriptions, loglen, a0):
    """{rule: per-task score}, higher = the rule picks that task over a lower one. loglen: log length of each description as
    the model read it; alpha: minus the rank of the description's letters, lower-cased, everything else dropped (ties share
    a rank)."""
    keys = ["".join(ch for ch in d.lower() if ch.isalpha()) for d in descriptions]
    rank = {key: i for i, key in enumerate(sorted(set(keys)))}
    return {"shorter": -np.asarray(loglen, float), "longer": np.asarray(loglen, float),
            "alpha": -np.array([rank[key] for key in keys], float), "enjoy_less": -np.asarray(a0, float)}


def agreement(u, s):
    """Share of task pairs the rule orders (s differs) whose utilities u order them the same way; ties in u count half."""
    du, ds = np.sign(u[:, None] - u[None, :]), np.sign(s[:, None] - s[None, :])
    m = ds > 0
    return float(((du[m] > 0) + 0.5 * (du[m] == 0)).mean())


def fe_coef(y, x, extra):
    """Coefficient of z(x) in y = c + b z(x) + extra columns."""
    X = np.column_stack([np.ones_like(y), z(x), *extra])
    return float(np.linalg.lstsq(X, y, rcond=None)[0][1])


def stats(V, S, benign, loglen, origins):
    """STATS over arrays of the same (possibly resampled) tasks. V: A0, C0, A_k, C_k per rule; S: rule_scores."""
    a0, c0 = V["A0"], V["C0"]
    gap = lambda x: x[benign].mean() - x[~benign].mean()
    sources = [(origins == o).astype(float) for o in np.unique(origins)[1:]]
    b0 = slope_per_sd(a0, c0)
    out = []
    for k in RULES:
        ak, ck, s = V[f"A_{k}"], V[f"C_{k}"], S[k]
        b_def, b_ins, w = pulls(ck, a0, ak)
        b_def_s, b_s = pulls(ck, a0, s)[:2] if k != "enjoy_less" else (np.nan, np.nan)
        da, dc = ak - a0, ck - c0
        fe = sources + ([loglen] if k in ("alpha", "enjoy_less") else [])  # length is the length rules' own score
        b_def_fe, b_ins_fe, w_fe = pulls(ck, a0, ak, fe)
        out += [b_def, b_ins, w, b_def + b_ins,
                pearson(ak, a0), slope_per_sd(a0, ck) / b0,
                pearson(ak, s), agreement(ak, s), agreement(a0, s), agreement(ck, s), agreement(c0, s),
                agreement(ak, s) - agreement(a0, s), agreement(ck, s) - agreement(c0, s),
                pearson(ak[benign], a0[benign]),
                b_def_s, b_s,
                pearson(dc, da), personas.slope(da, dc) / personas.slope(a0, c0),
                b_def_fe, b_ins_fe, w_fe,
                pulls(ck[benign], a0[benign], ak[benign])[2],
                gap(ak), gap(ck), pearson(ck, c0), pearson(ck, ak), pearson(ck, a0), ck.std() / c0.std()]
    dc = (V["C_shorter"] - V["C_longer"]) / 2
    da = (V["A_shorter"] - V["A_longer"]) / 2
    out += [fe_coef(dc, -loglen, sources), fe_coef(da, -loglen, sources), pearson(dc, da), agreement(dc, -loglen)]
    out += [b0, gap(a0), gap(c0), pearson(a0, loglen), pearson(c0, loglen)]
    return np.array(out, float)


def q_call(lo, hi):
    """When the pulls can't be separated: the coin's slope on the default Assistant, as a share q of the default coin's."""
    if hi < 0:
        return "reversed"
    if lo > 1:
        return "strengthened"
    if lo <= 1 <= hi:
        return "unchanged" if lo > 0 else "undetermined"
    return "weakened" if lo > 0 else "cancelled"


def call(est, lo, hi, rule):
    """The pre-registered call for one model and rule, first that applies."""
    g = lambda s: K[f"{rule}:{s}"]
    if abs(est[g("r_ak_a0")]) > SEPARABLE:
        return "not separable: " + q_call(lo[g("q")], hi[g("q")])
    c = personas.pull_call(lo[g("b_def")], lo[g("b_ins")], hi[g("b_ins")])
    return {"persona": "instructed choices", "against the persona": "against the instructed choices"}.get(c, c)


def length_call(lo, hi):
    """The mirrored length contrast: (C_shorter - C_longer) / 2 on z(-log length), within source."""
    if lo[K["len:b_dc"]] > 0:
        return "follows the length instruction"
    if hi[K["len:b_dc"]] < 0:
        return "against the length instruction"
    return "no length effect"


MARKS = {"A0": ("default Assistant", "o", 0.0), "A": ("instructed Assistant", "o", 1.0),
         "C0": ("default coin", "D", 0.0), "C": ("instructed coin", "D", 1.0)}


def plot_agreement(cells, labels, path):
    """Per model: for each rule, the share of task pairs ordered the rule's way by the default and instructed Assistant
    (circles) and coin (diamonds); hollow = default, filled = instructed. 95% CIs."""
    fig, axes = plt.subplots(1, len(cells), figsize=(4.6 * len(cells), 4.2), sharey=True, squeeze=False)
    stat = {"A0": "agree_a0", "A": "agree_a", "C0": "agree_c0", "C": "agree_c"}
    for ax, (family, cell) in zip(axes[0], cells.items()):
        est, (lo, hi) = cell["est"], ci(cell["boot"])
        for i, rule in enumerate(RULES):
            color = plotstyle.SERIES[i]
            for j, (m, (name, marker, fill)) in enumerate(MARKS.items()):
                y = len(RULES) - 1 - i + (0.18 if m.startswith("A") else -0.18)
                k = K[f"{rule}:{stat[m]}"]
                ax.errorbar(est[k], y, xerr=[[est[k] - lo[k]], [hi[k] - est[k]]], color=color, marker=marker, ms=7,
                            lw=0, elinewidth=1.5, mfc=color if fill else "white", mec=color, mew=1.4, zorder=3)
            for m0, m1 in (("A0", "A"), ("C0", "C")):  # a thin line from default to instructed
                y = len(RULES) - 1 - i + (0.18 if m0 == "A0" else -0.18)
                ax.plot([est[K[f"{rule}:{stat[m0]}"]], est[K[f"{rule}:{stat[m1]}"]]], [y, y], color=color, lw=1, zorder=2)
        ax.axvline(0.5, color=RULE, ls="--", lw=1, zorder=1)
        ax.set_yticks(range(len(RULES)), [RULE_LABEL[r] for r in RULES][::-1], color=INK)
        ax.set_xlim(0, 1)
        ax.set_xlabel("share of task pairs ordered the rule's way", color=INK_2, fontsize=9)
        ax.set_title(labels[family], color=INK, fontsize=10)
        style(ax)
    handles = [plt.Line2D([], [], color=INK_2, marker=marker, lw=0, ms=7, mfc=INK_2 if fill else "white", mec=INK_2,
                          mew=1.4, label=name) for name, marker, fill in MARKS.values()]
    fig.legend(handles=handles, frameon=False, fontsize=9, loc="lower center", ncol=4)
    fig.suptitle("Does the coin follow a shallow instruction for what to answer if asked?\n"
                 "upper row of each pair: Assistant's choice; lower: coinflip", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    plotstyle.save(fig, path.stem, path.parent)


def plot_pulls(cells, labels, path):
    """Per model: each rule's coin pulls toward the instructed Assistant (x) and the default Assistant (y), with b_0."""
    fig, axes = plt.subplots(1, len(cells), figsize=(4.8 * len(cells), 4.8), squeeze=False)
    for ax, (family, cell) in zip(axes[0], cells.items()):
        est, (lo, hi) = cell["est"], ci(cell["boot"])
        top, bottom = 0.0, 0.0
        for i, rule in enumerate(RULES):
            x, y = K[f"{rule}:b_ins"], K[f"{rule}:b_def"]
            ax.errorbar(est[x], est[y], xerr=[[est[x] - lo[x]], [hi[x] - est[x]]], yerr=[[est[y] - lo[y]], [hi[y] - est[y]]],
                        color=plotstyle.SERIES[i], marker="s", ms=8, lw=0, elinewidth=1.5, mec="white", mew=0.8, zorder=3,
                        label=RULE_LABEL[rule] if ax is axes[0, 0] else None)
            top, bottom = max(top, hi[x], hi[y]), min(bottom, lo[x], lo[y])
        b0 = K["b0"]
        ax.errorbar(0, est[b0], yerr=[[est[b0] - lo[b0]], [hi[b0] - est[b0]]], color=INK_2, marker="D", ms=7, lw=0,
                    elinewidth=1.5, mec="white", mew=0.8, zorder=3, label="no instruction (b_0)" if ax is axes[0, 0] else None)
        top, bottom = max(top, hi[b0]) * 1.15, min(bottom * 1.15, -0.05 * top)
        ax.plot([0, top], [0, top], color=RULE, ls="--", lw=1, zorder=1)
        ax.axhline(0, color=RULE, lw=0.8, zorder=1)
        ax.axvline(0, color=RULE, lw=0.8, zorder=1)
        ax.set_xlim(bottom, top)
        ax.set_ylim(bottom, top)
        ax.set_aspect("equal")
        ax.set_xlabel("pull toward the instructed Assistant, b_ins\n(coin nats per SD)", color=INK_2, fontsize=9)
        ax.set_ylabel("pull toward the default Assistant, b_def\n(coin nats per SD)", color=INK_2, fontsize=9)
        ax.set_title(labels[family], color=INK, fontsize=10)
        style(ax)
    fig.legend(frameon=False, fontsize=9, loc="lower center", ncol=5)
    fig.suptitle("The coin's two pulls under each instruction", color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    plotstyle.save(fig, path.stem, path.parent)


def plot_rule(data, scores, origins, labels, rule, path):
    """Rows: models. Columns: the instructed Assistant against the rule's score (manipulation check), the instructed coin
    against the instructed and the default Assistant, and the coin's change against the Assistant's change."""
    panels = (("S", f"A_{rule}", f"rule score: {SCORE_LABEL[rule]}", "instructed Assistant (nats)",
               "Instructed Assistant\nvs the rule"),
              (f"A_{rule}", f"C_{rule}", "instructed Assistant (nats)", "instructed coinflip (nats)",
               "Instructed coin\nvs instructed Assistant"),
              ("A0", f"C_{rule}", "default Assistant (nats)", "instructed coinflip (nats)",
               "Instructed coin\nvs default Assistant"),
              ("dA", "dC", "Assistant: instructed - default (nats)", "coin: instructed - default (nats)",
               "What the instruction changes:\ncoin vs Assistant"))
    fig, axes = plt.subplots(len(data), len(panels), figsize=(3.9 * len(panels), 3.7 * len(data)), squeeze=False)
    for i, (family, V) in enumerate(data.items()):
        V = V | {"S": scores[family][rule], "dA": V[f"A_{rule}"] - V["A0"], "dC": V[f"C_{rule}"] - V["C0"]}
        for j, (xk, yk, xlabel, ylabel, title) in enumerate(panels):
            ax = axes[i, j]
            origin_scatter(ax, V[xk], V[yk], origins)
            r = f"r = {pearson(V[xk], V[yk]):+.2f}"
            ax.set_title(f"{title}\n{r}" if i == 0 else r, fontsize=10, color=INK)
            ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
            ax.set_ylabel(f"{labels[family]}\n{ylabel}" if j == 0 else ylabel, color=INK if j == 0 else INK_2, fontsize=9)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, frameon=False, fontsize=9, loc="lower center", ncol=len(names))
    fig.suptitle(f"\"...please respond according to {prompts.IFASKED_RULES[rule]}.\"", color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    plotstyle.save(fig, path.stem, path.parent)


def plot_length(data, loglen, origins, labels, path):
    """The mirrored length contrast. Rows: models. Columns: (A_shorter - A_longer) / 2 and (C_shorter - C_longer) / 2
    against log length, and the coin's contrast against the Assistant's."""
    fig, axes = plt.subplots(len(data), 3, figsize=(11.7, 3.7 * len(data)), squeeze=False)
    for i, (family, V) in enumerate(data.items()):
        da, dc = (V["A_shorter"] - V["A_longer"]) / 2, (V["C_shorter"] - V["C_longer"]) / 2
        for j, (x, y, xlabel, ylabel, title) in enumerate((
                (loglen, da, "log task length", "Assistant: (shorter - longer) / 2", "Assistant's contrast\nvs length"),
                (loglen, dc, "log task length", "coin: (shorter - longer) / 2", "Coin's contrast\nvs length"),
                (da, dc, "Assistant: (shorter - longer) / 2", "coin: (shorter - longer) / 2",
                 "Coin's contrast\nvs the Assistant's"))):
            ax = axes[i, j]
            origin_scatter(ax, x, y, origins, fit_lines=j < 2)
            r = f"r = {pearson(x, y):+.2f}"
            ax.set_title(f"{title}\n{r}" if i == 0 else r, fontsize=10, color=INK)
            ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
            ax.set_ylabel(f"{labels[family]}\n{ylabel} (nats)" if j == 0 else f"{ylabel} (nats)",
                          color=INK if j == 0 else INK_2, fontsize=9)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, frameon=False, fontsize=9, loc="lower center", ncol=len(names))
    fig.suptitle("Shorter vs longer: what the two instructions change in opposite directions (lines: fit within source)",
                 color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    plotstyle.save(fig, path.stem, path.parent)


def main():
    import argparse
    from coin_report import sweep_rows
    from persona_placement import merged_fit
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS), help="a subset, e.g. while a model is still running")
    args = ap.parse_args()
    plotstyle.setup()
    out = Path("outputs/ifasked")
    out.mkdir(parents=True, exist_ok=True)
    runs = personas.collect_runs("ifasked")
    described = {t["id"]: t["prompt"].strip() for t in read_jsonl(TASKS_FILE)}  # as build_prompt puts them in the prompt
    data, scores, cells, labels, gates, tops, passed = {}, {}, {}, {}, [], {}, {}
    ids0 = resamples = None
    for family in args.models:
        labels[family] = label = MODEL_INFO[MODELS[family]][0]
        print(f"{label}:")
        rows, summary, paths, checks = merged_fit(runs[family], out / family)
        srow = sweep_rows(family)
        ids = [r["id"] for r in rows]
        assert ids == [r["id"] for r in srow], "task sets differ"
        N, origins, benign, loglen = task_arrays(rows)
        B = task_arrays(srow)[0]
        if ids0 is None:
            ids0, resamples = ids, stratified_resamples(origins, np.random.default_rng(0), N_BOOTSTRAP)
        assert ids == ids0, "task sets differ between models"

        ok = True
        for name, (stage, x) in sorted(checks.items()):  # each pod's own re-scored plain coinflip vs the sweep's
            y = B[f"{stage}/coinflip"]
            r, sd = pearson(x, y), x.std() / y.std() - 1
            ok_pod = r >= GATE_R and abs(sd) <= GATE_SD
            ok &= ok_pod
            gates.append(f"{label:14} {name}: re-scored {stage} coinflip vs sweep: r {r:.4f}, SD {sd:+.1%}: "
                         f"{'pass' if ok_pod else 'FAIL'}")
        for (stage, c), path in sorted(paths.items()):
            rel, mass = summary[f"{stage}/{c}/reliability"], median_mass(path)
            ok_cell = rel >= MIN_RELIABILITY
            ok &= ok_cell
            gates.append(f"{label:14} {stage:9} {c:26} reliability {rel:.3f}, median mass {mass:.2f}"
                         f"{'' if mass >= MIN_MASS else ' (below 0.2: not interpreted)'}: {'pass' if ok_cell else 'FAIL'}")
        passed[family] = ok
        V = {"A0": B["instruct/assistant"], "C0": B["instruct/coinflip"]}
        for k in RULES:
            V |= {f"A_{k}": N[f"instruct/assistant_ifasked_{k}"], f"C_{k}": N[f"instruct/coinflip_ifasked_{k}"]}
        desc_len = np.log([len(described[i]) for i in ids])
        S = rule_scores([described[i] for i in ids], desc_len, V["A0"])
        data[family], scores[family] = V, S
        cells[family] = dict(
            est=stats(V, S, benign, desc_len, origins),
            boot=np.array([stats({k: v[i] for k, v in V.items()}, {k: v[i] for k, v in S.items()}, benign[i], desc_len[i],
                                 origins[i]) for i in resamples]))
        tops[family] = {f"{s}/{c}": top_token_shares(p) for (s, c), p in sorted(paths.items())}

    print("\nValidation gate:\n  " + "\n  ".join(gates))
    table = []
    fm = lambda est, lo, hi, s: f"{est[K[s]]:+.3f} [{lo[K[s]]:+.3f}, {hi[K[s]]:+.3f}]"
    for family, cell in cells.items():
        est, (lo, hi) = cell["est"], ci(cell["boot"])
        print(f"\n{labels[family]}")
        for k in RULES:
            row = dict(model=labels[family], rule=k, call=call(est, lo, hi, k) if passed[family] else "no call (gate)")
            print(f"  {k}: call {row['call']}")
            for s in PER:
                print(f"    {s:12} {fm(est, lo, hi, f'{k}:{s}')}")
                row |= {s: est[K[f"{k}:{s}"]], f"{s}_lo": lo[K[f"{k}:{s}"]], f"{s}_hi": hi[K[f"{k}:{s}"]]}
            table.append(row)
        row = dict(model=labels[family], rule="length contrast",
                   call=length_call(lo, hi) if passed[family] else "no call (gate)")
        print(f"  length contrast: {row['call']}")
        for s in [f"len:{x}" for x in LENGTH] + list(CONTROL):
            print(f"    {s:12} {fm(est, lo, hi, s)}")
            row |= {s: est[K[s]], f"{s}_lo": lo[K[s]], f"{s}_hi": hi[K[s]]}
        table.append(row)

    print("\nExploratory, where the probability goes (mean over prompts, tokens logged in each top 5):")
    for family, by_cell in tops.items():
        for c, top in by_cell.items():
            print(f"  {labels[family]:14} {c:36} " + "  ".join(f"{t!r} {p:.3f}" for t, p in top))

    write_csv(out / "ifasked.csv", table)
    plot_agreement(cells, labels, Path("figures/ifasked_agreement.png"))
    plot_pulls(cells, labels, Path("figures/ifasked_pulls.png"))
    origins = task_arrays(rows)[1]
    for k in RULES:
        plot_rule(data, scores, origins, labels, k, Path(f"figures/ifasked_{k}.png"))
    plot_length(data, np.log([len(described[i]) for i in ids0]), origins, labels, Path("figures/ifasked_length.png"))
    for family, V in data.items():  # per-source grids: sources across, one row per rule
        grid = [Row(f"{RULE_LABEL[k]}:\ncoin vs instructed Assistant", V[f"A_{k}"], V[f"C_{k}"],
                    "instructed Assistant (nats)", "instructed coinflip (nats)", plotstyle.SERIES[i])
                for i, k in enumerate(RULES)]
        plot_source_grid(grid, origins, f"{labels[family]}: the instructed coin within each task source",
                         Path(f"figures/ifasked_sources_{family}.png"))
    print("\nWrote outputs/ifasked/ and figures/ifasked_*.png")


if __name__ == "__main__":
    main()
