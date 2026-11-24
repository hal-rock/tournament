"""Replication round 2's length rules (README, "Replication round 2", C), drawn as the reworded run's per-rule figures
(ifasked.plot_rule, e.g. figures/ifchoose_longer.png), for the large models only (Hal, 2026-11-24). Exploratory figures of
the pre-registered measures' data.

Per model and rule ("If I ask you to choose between tasks in this message, choose the one with the shorter / longer
description."), four panels:
  1. the instructed Assistant against the rule's score (log length, or minus it for "shorter"): does it follow the rule?
  2. the instructed coin against the instructed Assistant;
  3. the instructed coin against the default Assistant;
  4. what the instruction changes: the coin's change against the Assistant's change.
Once with the user-reported coin (C) and once with the tool's coin (T), where the model has readable tool cells.

Plus how often each pairwise pick follows the rule, by length ratio (ifasked_pairs.py's measure on the raw pairs: the
order-averaged log-odds of each pair, so no utility fit), for the instructed Assistant and each coin.

Readouts come from round2_analysis.readouts (round 1's A_0 / C_0 / T where the gate passes, else this run's).

  uv run python round2_length_figures.py [keys ...]
Writes figures/<key>/length_<rule>[_tool].png, figures/replication2_length_<rule>[_tool].png,
figures/replication2_pairs_length.png and outputs/replication/round2/pairs_length.csv.
"""

import argparse
import csv
import gzip
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import round2_analysis as r2  # noqa: E402
from analyze import pearson  # noqa: E402
from ifasked import SCORE_LABEL  # noqa: E402
from ifasked_pairs import BIN_LABEL, BINS  # noqa: E402
from matskit import plotstyle  # noqa: E402
from prompts import IFCHOOSE_LEAD, IFCHOOSE_RULES  # noqa: E402
from report import INK, INK_2, RULE, origin_scatter, style  # noqa: E402


def rule_text(rule):
    """The instruction as the prompts state it (prompts.IFCHOOSE_LEAD), without its blank line."""
    return IFCHOOSE_LEAD.format(rule=IFCHOOSE_RULES[rule]).strip()

RULES = ("shorter", "longer")
COIN = {"C": ("coinflip", "user-reported coin"), "T": ("tool coin", "tool's coin")}


def model_data(key):
    """{name: utilities}, origins, log length (n_chars), and which cells the gates exclude, for one model."""
    U, info, origins, _, loglen, _ = r2.readouts(key)
    bad = {n for n, i in info.items() if i["reliability"] < 0.9 or i["median_mass"] < 0.2}
    return U, origins, loglen, bad


def panels_for(coin):
    word, _ = COIN[coin]
    return (("S", "A_k", "rule score: {score}", "instructed Assistant (nats)", "Instructed Assistant\nvs the rule"),
            ("A_k", "X_k", "instructed Assistant (nats)", f"instructed {word} (nats)",
             f"Instructed {'coin' if coin == 'C' else 'tool coin'}\nvs instructed Assistant"),
            ("A0", "X_k", "default Assistant (nats)", f"instructed {word} (nats)",
             f"Instructed {'coin' if coin == 'C' else 'tool coin'}\nvs default Assistant"),
            ("dA", "dX", "Assistant: instructed - default (nats)", f"{word}: instructed - default (nats)",
             f"What the instruction changes:\n{'coin' if coin == 'C' else 'tool coin'} vs Assistant"))


def plot_rows(rows, rule, coin, path, title_extra=""):
    """rows: [(label, V, origins)] with V holding S, A0, A_k, X_k, dA, dX. One row of four panels per model."""
    panels = panels_for(coin)
    height = 3.7 * len(rows) + 1.2
    fig, axes = plt.subplots(len(rows), 4, figsize=(15.6, height), squeeze=False)
    for i, (label, V, origins) in enumerate(rows):
        for j, (xk, yk, xlabel, ylabel, title) in enumerate(panels):
            ax = axes[i, j]
            origin_scatter(ax, V[xk], V[yk], origins)
            r = f"r = {pearson(V[xk], V[yk]):+.2f}"
            ax.set_title(f"{title}\n{r}" if i == 0 else r, fontsize=10, color=INK)
            ax.set_xlabel(xlabel.format(score=SCORE_LABEL[rule]), color=INK_2, fontsize=9)
            ax.set_ylabel(f"{label}\n{ylabel}" if j == 0 else ylabel, color=INK if j == 0 else INK_2, fontsize=9)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, frameon=False, fontsize=9, loc="lower center", ncol=len(names))
    fig.suptitle(f"\"{rule_text(rule)}\"" + (f"  ({title_extra})" if title_extra else ""), color=INK)
    fig.tight_layout(rect=(0, 0.6 / height, 1, 1 - 0.55 / height))
    plotstyle.save(fig, path.stem, path.parent)


def raw_path(key, condition):
    """The raw results file a readout came from (round2_analysis.readouts's sources)."""
    return r2.OUT / condition.split("/")[0] / "results" / f"instruct__{condition.split('/', 1)[1]}.csv.gz"


def pair_picks(path):
    """{(a, b): d}, a < b, d > 0 = a picked: the order-averaged log-odds of each pair scored in both orders."""
    lo = {}
    with gzip.open(path, "rt", newline="") as fh:
        for row in csv.DictReader(fh):
            lo[row["first"], row["second"]] = float(row["log_odds"])
    return {(a, b): (lo[a, b] - lo[b, a]) / 2 for (a, b) in lo if a < b and (b, a) in lo}


def follow_shares(picks, length, rule):
    """Share of untied pairs whose pick follows the rule, overall and per length-ratio bin, and the bins' counts."""
    d = np.array(list(picks.values()))
    la = np.array([length[a] for a, _ in picks])
    lb = np.array([length[b] for _, b in picks])
    keep = la != lb
    d, la, lb = d[keep], la[keep], lb[keep]
    a_wanted = la < lb if rule == "shorter" else la > lb
    follows = (d > 0) == a_wanted
    ratio = np.maximum(la, lb) / np.minimum(la, lb)
    bins = [(ratio >= lo) & (ratio < hi) for lo, hi in BINS]
    return float(follows.mean()), [float(follows[m].mean()) for m in bins], [int(m.sum()) for m in bins]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("keys", nargs="*", default=r2.FIRST_ROUND + r2.NEW)
    a = ap.parse_args(argv)
    plotstyle.setup()
    length = {t["id"]: len(t["prompt"]) for t in r2.ra.read_jsonl(r2.ra.DEFAULT_TASKS)}
    summary = {(rule, coin): [] for rule in RULES for coin in COIN}
    pair_rows = []
    for key in a.keys:
        U, origins, loglen, bad = model_data(key)
        _, info, *_ = r2.readouts(key)
        label = r2.LABEL.get(key, key)
        for rule in RULES:
            S = loglen if rule == "longer" else -loglen
            for coin, default in (("C", "C0"), ("T", "T")):
                need = ["A0", f"A_{rule}", default, f"{coin}_{rule}"]
                if any(n not in U or n in bad for n in need):
                    continue
                V = {"S": S, "A0": U["A0"], "A_k": U[f"A_{rule}"], "X_k": U[f"{coin}_{rule}"],
                     "dA": U[f"A_{rule}"] - U["A0"], "dX": U[f"{coin}_{rule}"] - U[default]}
                suffix = "" if coin == "C" else "_tool"
                plot_rows([(label, V, origins)], rule, coin, Path("figures") / key / f"length_{rule}{suffix}.png",
                          COIN[coin][1] if coin == "T" else "")
                summary[rule, coin].append((label, V, origins))
            for name in (f"A_{rule}", f"C_{rule}", f"T_{rule}", "A0", "C0", "T"):
                if name not in info or name in bad:
                    continue
                share, per_bin, counts = follow_shares(pair_picks(raw_path(key, info[name]["source"])), length, rule)
                pair_rows.append(dict(model=key, rule=rule, readout=name, overall=round(share, 4),
                                      **{BIN_LABEL[b]: round(per_bin[b], 4) for b in range(len(BINS))},
                                      **{f"n {BIN_LABEL[b]}": counts[b] for b in range(len(BINS))}))
    for (rule, coin), rows in summary.items():
        if rows:
            plot_rows(rows, rule, coin, Path("figures") / f"replication2_length_{rule}{'' if coin == 'C' else '_tool'}.png",
                      COIN[coin][1] if coin == "T" else "")
    out = r2.OUT / "round2" / "pairs_length.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pair_rows[0]))
        w.writeheader()
        w.writerows(pair_rows)
    plot_pairs(pair_rows, Path("figures/replication2_pairs_length.png"))
    return pair_rows


def plot_pairs(rows, path):
    """Per model (panels) and rule (line style): the share of pairwise picks following the rule by length ratio, for the
    instructed Assistant, the instructed user-reported coin and the instructed tool's coin, against the defaults'."""
    keys = list(dict.fromkeys(r["model"] for r in rows))
    blue, orange, green = plotstyle.SERIES[:3]
    readouts = (("A", green, "Assistant's own choice"), ("C", orange, "User reports the flip"),
                ("T", blue, "Tool reports the flip"))
    ncol = 4
    nrow = -(-len(keys) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.1 * nrow + 1.0), sharey=True, squeeze=False)
    x = np.arange(len(BINS))
    for ax, key in zip(axes.flat, keys):
        for rule, ls in (("shorter", "-"), ("longer", (0, (4, 2)))):
            for prefix, color, _ in readouts:
                inst = next((r for r in rows if r["model"] == key and r["rule"] == rule and r["readout"] == f"{prefix}_{rule}"),
                            None)
                if inst is None:
                    continue
                ax.plot(x, [inst[b] for b in BIN_LABEL], ls=ls, color=color, lw=2, marker="o", ms=5,
                        mec=plotstyle.SURFACE, mew=1)
        ax.axhline(0.5, color=RULE, lw=1, zorder=0)
        ax.set_xticks(x, BIN_LABEL, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(r2.LABEL.get(key, key), fontsize=10, color=INK)
        ax.set_xlabel("length ratio (longer / shorter)", fontsize=8.5, color=INK_2)
        style(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("share of picks following the rule", fontsize=9, color=INK_2)
    for ax in list(axes.flat)[len(keys):]:
        ax.set_visible(False)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, lw=2, marker="o", ms=5, label=lab) for _, c, lab in readouts]
    handles += [Line2D([], [], color=INK_2, lw=2, ls="-", label='"shorter"'),
                Line2D([], [], color=INK_2, lw=2, ls=(0, (4, 2)), label='"longer"')]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=9)
    fig.suptitle("Length rules: how often each pairwise pick follows the rule, by how different the two lengths are",
                 color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    plotstyle.save(fig, path.stem, path.parent)


if __name__ == "__main__":
    main()
