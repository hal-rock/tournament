"""Personas as roleplay: the persona runs' system-prompt personas, framed as roleplay with the persona in the third person.

The measures and calls were pre-registered in the README ("Personas as roleplay") before any data. Per model (Llama 70B,
Gemma 4 31B) and persona p (evil, slacker), all final-stage utilities on the sweep's 300 tasks and 15,000 pairs:
- A_0, C_0: the default Assistant and coin, no system prompt (the sweep);
- A_s, C_s: the Assistant and coin with the persona as the system prompt (W&B group "personas");
- A_r, C_r: the same with the roleplay system prompt, prompts.ROLEPLAY[p] (W&B group "roleplay").

Primary: least squares C_r = c + b_def z(A_0) + b_per z(A_r) and w_r = b_per / (b_def + b_per), the roleplay coin's place
between the default Assistant and the roleplay Assistant's own choices; next to it the same fit of C_s on its own Assistant
(w_s), and dw = w_r - w_s. CIs are 95% task bootstraps within origin, one set of resamples for everything, so differences
are paired.

  outputs/roleplay/<family>/analysis/   the merged fit (analyze.py) of one model's roleplay runs
  outputs/roleplay/<family>/gate/<run>/ each pod's re-scored plain coinflip, fit on its own
  outputs/roleplay/roleplay.csv         one row per model and persona: every statistic with its CI, and the calls
  outputs/roleplay/report.txt           (written by redirecting stdout)
  figures/roleplay_*.png

Usage: uv run python roleplay.py [--models llama-70b]
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import personas
from analyze import N_BOOTSTRAP, ORIGIN_STYLE, pearson
from matskit import plotstyle
from models import MODEL_INFO
from openers import write_csv
from persona_placement import merged_fit
from personas_user import median_mass, pulls
from report import INK, INK_2, RULE, origin_scatter, stratified_resamples, style, task_arrays
from source_grid import Row, plot_source_grid
from user_choice import FAMILIES, ci, load_sweep, top_token_shares

MODELS = ("llama-70b", "gemma4-31b")
PERSONAS = ("evil", "slacker")
GATE_R, GATE_SD, MIN_RELIABILITY, MIN_MASS = 0.999, 0.02, 0.9, 0.2
SEPARABLE = personas.SEPARABLE  # |r(A_r, A_0)| above this: the two pulls can't be told apart
ORIGINS = tuple(ORIGIN_STYLE)
PER = ("b_def_r", "b_per_r", "w_r", "total_r",  # primary: C_r on z(A_0), z(A_r)
       "b_def_s", "b_per_s", "w_s", "total_s",  # the system-prompt persona: C_s on z(A_0), z(A_s)
       "dw",  # w_r - w_s
       "r_ar_as", "r_ar_a0", "r_as_a0", "gap_ar", "gap_as", "r_ar_len", "r_as_len",  # manipulation checks
       "b_def_fix", "b_per_fix", "rho",  # C_r on z(A_0), z(A_s): the system persona's Assistant as a fixed target
       "gap_cr", "gap_cs", "r_cr_cs", "r_cr_c0", "r_cr_ar", "r_cr_a0", "r_cs_as", "sd_cr", "sd_cs",
       "b_def_r_ls", "b_per_r_ls", "w_r_ls", "w_s_ls", "dw_ls",  # with log length and source fixed effects
       "w_r_nh", "w_s_nh", "dw_nh",  # over non-harmful tasks only
       "r_dc_da")  # exploratory: r(C_r - C_s, A_r - A_s), does the coin follow the frame's change in the choices?
CONTROL = ("b0", "gap_c0", "gap_a0", "r_a0_len")
STATS = (tuple(f"{p}:{s}" for p in PERSONAS for s in PER) + CONTROL
         + tuple(f"{p}:r_{t}_{o}" for p in PERSONAS for t in ("a0", "ar") for o in ORIGINS))
K = {name: j for j, name in enumerate(STATS)}


def stats(V, benign, loglen, origins):
    """STATS over arrays of the same (possibly resampled) tasks. V: A0, C0, and As_p, Cs_p, Ar_p, Cr_p per persona p."""
    a0, c0 = V["A0"], V["C0"]
    gap = lambda x: x[benign].mean() - x[~benign].mean()
    sources = [(origins == o).astype(float) for o in np.unique(origins)[1:]]
    out = []
    for p in PERSONAS:
        a_s, c_s, a_r, c_r = V[f"As_{p}"], V[f"Cs_{p}"], V[f"Ar_{p}"], V[f"Cr_{p}"]
        b_def_r, b_per_r, w_r = pulls(c_r, a0, a_r)
        b_def_s, b_per_s, w_s = pulls(c_s, a0, a_s)
        b_def_fix, b_per_fix, _ = pulls(c_r, a0, a_s)
        b_def_r_ls, b_per_r_ls, w_r_ls = pulls(c_r, a0, a_r, [loglen, *sources])
        w_s_ls = pulls(c_s, a0, a_s, [loglen, *sources])[2]
        w_r_nh = pulls(c_r[benign], a0[benign], a_r[benign])[2]
        w_s_nh = pulls(c_s[benign], a0[benign], a_s[benign])[2]
        out += [b_def_r, b_per_r, w_r, b_def_r + b_per_r, b_def_s, b_per_s, w_s, b_def_s + b_per_s, w_r - w_s,
                pearson(a_r, a_s), pearson(a_r, a0), pearson(a_s, a0), gap(a_r), gap(a_s), pearson(a_r, loglen),
                pearson(a_s, loglen),
                b_def_fix, b_per_fix, b_per_fix / pulls(c_s, a0, a_s)[1],
                gap(c_r), gap(c_s), pearson(c_r, c_s), pearson(c_r, c0), pearson(c_r, a_r), pearson(c_r, a0),
                pearson(c_s, a_s), c_r.std(), c_s.std(),
                b_def_r_ls, b_per_r_ls, w_r_ls, w_s_ls, w_r_ls - w_s_ls,
                w_r_nh, w_s_nh, w_r_nh - w_s_nh,
                pearson(c_r - c_s, a_r - a_s)]
    out += [personas.slope_per_sd(a0, c0), gap(c0), gap(a0), pearson(a0, loglen)]
    for p in PERSONAS:
        for t in (a0, V[f"Ar_{p}"]):
            for o in ORIGINS:
                m = origins == o
                out.append(pearson(t[m], V[f"Cr_{p}"][m]))
    return np.array(out)


def call(est, lo, hi, p):
    """The persona section's call for the roleplay coin, with the roleplay Assistant as the persona, first that applies."""
    g = lambda s: K[f"{p}:{s}"]
    if abs(est[g("r_ar_a0")]) > SEPARABLE:
        return "not separable"
    if hi[g("b_per_r")] < 0:
        return "against the persona"
    def_up, per_up = lo[g("b_def_r")] > 0, lo[g("b_per_r")] > 0
    return {(False, False): "neither", (True, False): "default Assistant", (False, True): "persona",
            (True, True): "mix"}[def_up, per_up]


def compare(est, lo, hi, p):
    """The pre-registered comparison with the system-prompt persona, on dw = w_r - w_s."""
    if call(est, lo, hi, p) in ("not separable", "neither"):
        return "no comparison"
    d = K[f"{p}:dw"]
    if hi[d] < 0:
        return "less than the system prompt"
    if lo[d] > 0:
        return "more than the system prompt"
    return "as much as the system prompt"


FRAMES = {"s": ("system prompt", "o"), "r": ("roleplay", "s")}


def plot_pulls(cells, labels, path):
    """Per model: each persona's coin pulls under the system prompt (circles) and as roleplay (squares), 95% CIs, with an
    arrow from one to the other. x: the pull toward that framing's persona Assistant; y: toward the default Assistant."""
    fig, axes = plt.subplots(1, len(cells), figsize=(5.2 * len(cells), 5.0), squeeze=False)
    for ax, (family, cell) in zip(axes[0], cells.items()):
        est, (lo, hi) = cell["est"], ci(cell["boot"])
        top, bottom = 0.0, 0.0
        for k, p in enumerate(PERSONAS):
            color = plotstyle.SERIES[k]
            pts = {}
            for f, (name, marker) in FRAMES.items():
                x, y = K[f"{p}:b_per_{f}"], K[f"{p}:b_def_{f}"]
                ax.errorbar(est[x], est[y], xerr=[[est[x] - lo[x]], [hi[x] - est[x]]],
                            yerr=[[est[y] - lo[y]], [hi[y] - est[y]]], color=color, marker=marker, ms=8, lw=0,
                            elinewidth=1.5, mec="white", mew=0.8, zorder=3,
                            label=f"{p}, {name}" if ax is axes[0, 0] else None)
                pts[f] = (est[x], est[y])
                top, bottom = max(top, hi[x], hi[y]), min(bottom, lo[x], lo[y])
            for f in FRAMES:  # each label on the side away from the other framing's point
                right = (pts[f][0] >= pts["s" if f == "r" else "r"][0])
                ax.annotate(f"w = {est[K[f'{p}:w_{f}']]:.2f}", pts[f], xytext=(9, -3) if right else (-9, -3),
                            ha="left" if right else "right", textcoords="offset points", color=INK, fontsize=9)
            ax.annotate("", xy=pts["r"], xytext=pts["s"], zorder=2,
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.2, shrinkA=6, shrinkB=6))
        b0 = K["b0"]
        ax.errorbar(0, est[b0], yerr=[[est[b0] - lo[b0]], [hi[b0] - est[b0]]], color=INK_2, marker="D", ms=7, lw=0,
                    elinewidth=1.5, mec="white", mew=0.8, zorder=3, label="no persona (b_0)" if ax is axes[0, 0] else None)
        top, bottom = max(top, hi[b0]) * 1.15, bottom * 1.15
        ax.plot([0, top], [0, top], color=RULE, ls="--", lw=1, zorder=1)
        ax.axhline(0, color=RULE, lw=0.8, zorder=1)
        ax.axvline(0, color=RULE, lw=0.8, zorder=1)
        ax.set_xlim(bottom - 0.02 * top, top)
        ax.set_ylim(bottom - 0.02 * top, top)
        ax.set_aspect("equal")
        ax.set_xlabel("pull toward the persona's Assistant under the same framing, b_per\n(coin nats per SD)",
                      color=INK_2, fontsize=9)
        ax.set_ylabel("pull toward the default Assistant, b_def\n(coin nats per SD)", color=INK_2, fontsize=9)
        ax.set_title(labels[family], color=INK, fontsize=10)
        style(ax)
    fig.legend(frameon=False, fontsize=9, loc="lower center", ncol=5)
    fig.suptitle("Does the coin follow the persona when the persona is framed as roleplay?", color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    plotstyle.save(fig, path.stem, path.parent)


def plot_scatter(data, origins, labels, persona, path):
    """Rows: models. Columns: the roleplay Assistant against the system-prompt persona's (manipulation check); the
    roleplay coin against the roleplay Assistant, the default Assistant and the system-prompt persona's coin; and
    (exploratory) the coin's change from the system prompt to roleplay against the Assistant's change."""
    data = {f: V | {f"dA_{persona}": V[f"Ar_{persona}"] - V[f"As_{persona}"],
                    f"dC_{persona}": V[f"Cr_{persona}"] - V[f"Cs_{persona}"]} for f, V in data.items()}
    panels = ((f"As_{persona}", f"Ar_{persona}", "System-prompt persona Assistant (nats)", "roleplay Assistant (nats)",
               "Roleplay Assistant\nvs system-prompt Assistant"),
              (f"Ar_{persona}", f"Cr_{persona}", "Roleplay Assistant (nats)", "roleplay coinflip (nats)",
               "Roleplay coin\nvs roleplay Assistant"),
              ("A0", f"Cr_{persona}", "Default Assistant (nats)", "roleplay coinflip (nats)",
               "Roleplay coin\nvs default Assistant"),
              (f"Cs_{persona}", f"Cr_{persona}", "System-prompt persona coinflip (nats)", "roleplay coinflip (nats)",
               "Roleplay coin\nvs system-prompt coin"),
              (f"dA_{persona}", f"dC_{persona}", "Assistant: roleplay - system prompt (nats)",
               "coin: roleplay - system prompt (nats)", "What roleplay changes:\ncoin vs Assistant"))
    fig, axes = plt.subplots(len(data), len(panels), figsize=(3.9 * len(panels), 3.7 * len(data)), squeeze=False)
    for i, (family, V) in enumerate(data.items()):
        for j, (xk, yk, xlabel, ylabel, title) in enumerate(panels):
            ax = axes[i, j]
            origin_scatter(ax, V[xk], V[yk], origins)
            r = f"r = {pearson(V[xk], V[yk]):+.2f}"
            ax.set_title(f"{title}\n{r}" if i == 0 else r, fontsize=10, color=INK)
            ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
            ax.set_ylabel(f"{labels[family]}\n{ylabel}" if j == 0 else ylabel, color=INK if j == 0 else INK_2, fontsize=9)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, frameon=False, fontsize=9, loc="lower center", ncol=len(names))
    fig.suptitle(f"The {persona} persona as roleplay", color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    plotstyle.save(fig, path.stem, path.parent)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS), help="a subset, e.g. while a model is still running")
    args = ap.parse_args()
    plotstyle.setup()
    out = Path("outputs/roleplay")
    out.mkdir(parents=True, exist_ok=True)
    new_runs = personas.collect_runs("roleplay")
    persona_runs = personas.collect_runs("personas")
    data, cells, labels, gates, tops, passed = {}, {}, {}, [], {}, {}
    ids0 = resamples = None
    for family in args.models:
        labels[family] = label = MODEL_INFO[FAMILIES[family][0]][0]
        print(f"{label}:")
        rows, summary, paths, checks = merged_fit(new_runs[family], out / family)
        prow = personas.merged_fit(persona_runs[family], Path("outputs/personas/analysis") / family)[0]
        srow = load_sweep(family)[0]
        ids = [r["id"] for r in rows]
        assert all(ids == [r["id"] for r in x] for x in (prow, srow)), "task sets differ"
        N, origins, benign, loglen = task_arrays(rows)
        P, B = task_arrays(prow)[0], task_arrays(srow)[0]
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
            gates.append(f"{label:14} {stage:9} {c:24} reliability {rel:.3f}, median mass {mass:.2f}"
                         f"{'' if mass >= MIN_MASS else ' (below 0.2: not interpreted)'}: {'pass' if ok_cell else 'FAIL'}")
        passed[family] = ok
        V = {"A0": B["instruct/assistant"], "C0": B["instruct/coinflip"]}
        for p in PERSONAS:
            V |= {f"As_{p}": P[f"instruct/assistant@{p}"], f"Cs_{p}": P[f"instruct/coinflip@{p}"],
                  f"Ar_{p}": N[f"instruct/assistant@{p}+roleplay"], f"Cr_{p}": N[f"instruct/coinflip@{p}+roleplay"]}
        data[family] = V
        cells[family] = dict(
            est=stats(V, benign, loglen, origins),
            boot=np.array([stats({k: v[i] for k, v in V.items()}, benign[i], loglen[i], origins[i]) for i in resamples]))
        tops[family] = {f"{s}/{c}": top_token_shares(p) for (s, c), p in sorted(paths.items())}

    print("\nValidation gate:\n  " + "\n  ".join(gates))
    table = []
    fm = lambda est, lo, hi, s: f"{est[K[s]]:+.3f} [{lo[K[s]]:+.3f}, {hi[K[s]]:+.3f}]"
    for family, cell in cells.items():
        est, (lo, hi) = cell["est"], ci(cell["boot"])
        print(f"\n{labels[family]}")
        for p in PERSONAS:
            row = dict(model=labels[family], persona=p)
            row["call"] = call(est, lo, hi, p) if passed[family] else "no call (gate)"
            row["comparison"] = compare(est, lo, hi, p) if passed[family] else "no call (gate)"
            print(f"  {p}: call {row['call']}; vs the system prompt: {row['comparison']}")
            for s in [f"{p}:{x}" for x in PER] + list(CONTROL) + [s for s in STATS if s.startswith(f"{p}:r_a")]:
                print(f"    {s:28} {fm(est, lo, hi, s)}")
                row.update({s.removeprefix(f"{p}:"): est[K[s]], f"{s.removeprefix(f'{p}:')}_lo": lo[K[s]],
                            f"{s.removeprefix(f'{p}:')}_hi": hi[K[s]]})
            table.append(row)

    print("\nExploratory, where the probability goes (mean over prompts, tokens logged in each top 5):")
    for family, by_cell in tops.items():
        for c, top in by_cell.items():
            print(f"  {labels[family]:14} {c:32} " + "  ".join(f"{t!r} {p:.3f}" for t, p in top))
    print("\nExploratory, mean utility by origin in SDs from the task mean:")
    for family, V in data.items():
        for k, x in V.items():
            print(f"  {labels[family]:14} {k:10} " + "  ".join(
                f"{o[:8]:8} {(x[origins == o].mean() - x.mean()) / x.std():+.2f}" for o in ORIGINS))

    write_csv(out / "roleplay.csv", table)
    plot_pulls(cells, labels, Path("figures/roleplay_pulls.png"))
    for p in PERSONAS:
        plot_scatter(data, origins, labels, p, Path(f"figures/roleplay_{p}.png"))
    for family, V in data.items():  # per-source grids: sources across, one row per (persona, Assistant)
        rows = [Row(f"{p.capitalize()} roleplay coin\nvs {name} Assistant", V[t], V[f"Cr_{p}"],
                    f"{name.capitalize()} Assistant (nats)", "roleplay coinflip (nats)", plotstyle.SERIES[k])
                for k, p in enumerate(PERSONAS) for t, name in (("A0", "default"), (f"Ar_{p}", "roleplay"))]
        plot_source_grid(rows, origins, f"{labels[family]}: the roleplay coin within each task source",
                         Path(f"figures/roleplay_sources_{family}.png"))
    print("\nWrote outputs/roleplay/ and figures/roleplay_*.png")


if __name__ == "__main__":
    main()
