"""Lazy personas: "You are a pirate." or "In your next message, answer as a pirate." at the start of the user's message.

The measures and calls were pre-registered in the README ("Lazy personas") before any data. Per model (OLMo 3.1 32B Instruct,
Llama 3.3 70B Instruct, Gemma 4 31B-it, GLM-4.5-Air), persona p (prompts.LAZY_PERSONAS) and phrasing f (prompts.LAZY_LEADS:
youare, answeras), final-stage utilities on the sweep's 300 tasks and 15,000 pairs:
- A_0, C_0: the default Assistant and coin (the sweep);
- A_pf, C_pf: the Assistant's choice and the coinflip with the persona's lead (W&B group "lazy").

Everything primary is over the 225 "arbitrary" tasks (WildChat, Alpaca, MATH; Hal's focus), with utilities as fit over all 300:
- the Assistant's movement: r(A_pf, A_0), and s_A, the reliable share of the persona's change dA = resid(A_pf ~ A_0) (its
  variance minus the noise variance the two sets of utilities put in it, from their split-half reliabilities);
- the coin's movement: r(C_pf, C_0), and s_C, the same for dC = resid(C_pf - C_0 ~ A_0);
- transfer: rho = r(dC, dA), whether the coin's change goes the way of the Assistant's change, with A_0 partialled out of
  both so that both merely flattening (or sharpening) along the default doesn't count; tau = slope of dC on dA over b_0,
  the default coin's slope on A_0 (1: the coin passes the persona's change on as fully as it passes on the default);
- the call per cell (cell_call), and per model the phrasing contrasts, averaged over the five personas (phrasing_call).
CIs are 95% task bootstraps within origin, one set of resamples for every model and statistic.

  outputs/lazy/<family>/analysis/   the merged fit (analyze.py) of one model's runs
  outputs/lazy/<family>/gate/<run>/ each pod's re-scored plain coinflip, fit on its own
  outputs/lazy/lazy.csv             one row per model, persona and phrasing: every statistic with its CI, and the calls
  outputs/lazy/pooled.csv           one row per model: the phrasing contrasts over personas
  outputs/lazy/report.txt           (written by redirecting stdout)
  figures/lazy_*.png

Usage: uv run python lazy.py [--models llama-70b] > outputs/lazy/report.txt
"""

import json
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
from openers import write_csv
from personas_user import median_mass, pulls
from report import INK, INK_2, RULE, stratified_resamples, style, task_arrays
from source_grid import SOURCE_LABEL, Row, plot_source_grid
from user_choice import TASKS_FILE, ci, top_token_shares

MODELS = {"olmo3.1-32b": "allenai/Olmo-3.1-32B-Instruct", "llama-70b": "unsloth/Llama-3.3-70B-Instruct",
          "gemma4-31b": "google/gemma-4-31B-it", "glm-4.5-air": "zai-org/GLM-4.5-Air"}
PERSONAS = tuple(prompts.LAZY_PERSONAS)
PHRASINGS = tuple(prompts.LAZY_LEADS)
PHRASING_LABEL = {"youare": '"You are a ___."', "answeras": '"In your next message, answer as a ___."'}
ARB = ("WILDCHAT", "ALPACA", "MATH")
GATE_R, GATE_SD, MIN_RELIABILITY, MIN_MASS = 0.999, 0.02, 0.9, 0.2
MOVED = 0.5  # s_A below this: most of the persona's change in the Assistant is noise, so there's nothing for the coin to follow
PER = ("r_ap_a0", "r_cp_c0", "s_a", "s_c",  # movement
       "rho", "tau",  # primary transfer
       "tau_corr", "rho_corr",  # tau / s_A and rho / sqrt(s_A s_C): corrected for the noise in the changes
       "rho_w", "tau_w",  # within source (source dummies in both residualizations)
       *(f"sh_a_{s}" for s in ARB), *(f"sh_c_{s}" for s in ARB),  # mean change by source, in default SDs
       "r_da_len", "r_dc_len", "sd_a", "sd_c",  # length; spread over the default's
       "b_def", "b_per", "w",  # the persona section's two-pull fit, over the same tasks (context)
       "gap_ap", "gap_cp")  # harm gaps over all 300 tasks
PAIR = ("r_a_phr", "r_da_phr", "r_c_phr", "d_ma", "d_mc", "d_rho", "d_tau")  # per persona, youare vs answeras
POOLED = tuple(f"{m}_{x}" for m in ("ma", "mc", "rho", "tau") for x in (*PHRASINGS, "diff"))
CONTROL = ("b0", "r_a0_c0", "gap_a0", "gap_c0")
STATS = (tuple(f"{p}:{f}:{s}" for p in PERSONAS for f in PHRASINGS for s in PER) + tuple(f"{p}:{s}" for p in PERSONAS for s in PAIR)
         + POOLED + CONTROL)
K = {name: j for j, name in enumerate(STATS)}


def resid(y, *cols):
    """Residual of y on an intercept and cols."""
    X = np.column_stack([np.ones_like(y), *cols])
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


def coef(y, *cols):
    """The coefficient of the first column in y = c + cols."""
    X = np.column_stack([np.ones_like(y), *cols])
    return float(np.linalg.lstsq(X, y, rcond=None)[0][1])


def noise_share(d, noise_var):
    """Reliable share of d's variance: 1 - noise variance / variance."""
    return 1 - noise_var / d.var()


def stats(V, noise, origins, benign):
    """STATS over arrays of the same (possibly resampled) tasks. V: A0, C0 and A:<p>:<f>, C:<p>:<f> over all tasks; noise:
    the per-task noise variance of each key of V (fixed, from split-half reliability)."""
    arb = np.isin(origins, ARB)
    src = origins[arb]
    dummies = [(src == s).astype(float) for s in ARB[1:]]
    a0, c0 = V["A0"][arb], V["C0"][arb]
    gap = lambda x: x[benign].mean() - x[~benign].mean()
    b0, b0_w = personas.slope(a0, c0), coef(c0, a0, *dummies)
    out, cell = [], {}
    for p in PERSONAS:
        for f in PHRASINGS:
            ka, kc = f"A:{p}:{f}", f"C:{p}:{f}"
            ap, cp = V[ka][arb], V[kc][arb]
            da, dc = resid(ap, a0), resid(cp - c0, a0)
            lam, kap = personas.slope(a0, ap), personas.slope(a0, cp - c0)
            s_a = noise_share(da, noise[ka] + lam ** 2 * noise["A0"])
            s_c = noise_share(dc, noise[kc] + noise["C0"] + kap ** 2 * noise["A0"])
            rho, tau = pearson(da, dc), personas.slope(da, dc) / b0
            da_w, dc_w = resid(ap, a0, *dummies), resid(cp - c0, a0, *dummies)
            d_a, d_c = ap - a0, cp - c0
            shifts = [(d_a[src == s].mean() - d_a.mean()) / a0.std() for s in ARB]
            shifts += [(d_c[src == s].mean() - d_c.mean()) / c0.std() for s in ARB]
            loglen = V["loglen"][arb]
            out += [pearson(ap, a0), pearson(cp, c0), s_a, s_c,
                    rho, tau,
                    tau / s_a if s_a > 0 else np.nan, rho / np.sqrt(s_a * s_c) if min(s_a, s_c) > 0 else np.nan,
                    pearson(da_w, dc_w), personas.slope(da_w, dc_w) / b0_w,
                    *shifts,
                    pearson(da, loglen), pearson(dc, loglen), ap.std() / a0.std(), cp.std() / c0.std(),
                    *pulls(cp, a0, ap),
                    gap(V[ka]), gap(V[kc])]
            cell[p, f] = dict(ap=ap, cp=cp, da=da, dc=dc, ma=1 - pearson(ap, a0), mc=1 - pearson(cp, c0), rho=rho, tau=tau)
    for p in PERSONAS:
        y, n = cell[p, "youare"], cell[p, "answeras"]
        out += [pearson(y["ap"], n["ap"]), pearson(y["da"], n["da"]), pearson(y["cp"], n["cp"]),
                y["ma"] - n["ma"], y["mc"] - n["mc"], y["rho"] - n["rho"], y["tau"] - n["tau"]]
    for m in ("ma", "mc", "rho", "tau"):
        means = [np.mean([cell[p, f][m] for p in PERSONAS]) for f in PHRASINGS]
        out += [*means, means[0] - means[1]]
    out += [b0, pearson(a0, c0), gap(V["A0"]), gap(V["C0"])]
    return np.array(out, float)


def cell_call(est, lo, hi, p, f):
    """The pre-registered call for one model, persona and phrasing, first that applies."""
    g = lambda s: K[f"{p}:{f}:{s}"]
    if est[g("s_a")] < MOVED:
        return "Assistant unmoved"
    if lo[g("rho")] > 0:
        return "coin follows the persona"
    if hi[g("rho")] < 0:
        return "coin against the persona"
    return "no transfer"


def phrasing_call(lo, hi, m):
    """Per model, on the mean over personas of youare - answeras: m "ma" (the Assistant's movement, 1 - r(A_pf, A_0)) or "mc"
    (the coin's, 1 - r(C_pf, C_0))."""
    k = K[f"{m}_diff"]
    if lo[k] > 0:
        return '"you are" moves it more'
    if hi[k] < 0:
        return '"answer as" moves it more'
    return "no difference"


FILL = {"youare": True, "answeras": False}
OFFSET = {"youare": -0.16, "answeras": 0.16}


def _mark(ax, x, est, lo, hi, k, color, filled, marker="o"):
    ax.errorbar(x, est[k], yerr=[[est[k] - lo[k]], [hi[k] - est[k]]], color=color, marker=marker, ms=7, lw=0,
                elinewidth=1.4, mfc=color if filled else "white", mec=color, mew=1.4, zorder=3)


def plot_overview(cells, labels, path):
    """Rows: the Assistant's r(A_pf, A_0), the coin's r(C_pf, C_0), and the transfer rho; columns: models. Per persona, filled =
    "You are", hollow = "answer as"; 95% CIs. Transfer is drawn faint where the Assistant didn't move (s_A < 0.5)."""
    rows = (("r_ap_a0", "Assistant: r(persona, default)"), ("r_cp_c0", "Coin: r(persona, default)"),
            ("rho", "Transfer: r(coin's change,\nAssistant's change)"))
    fig, axes = plt.subplots(len(rows), len(cells), figsize=(3.9 * len(cells), 3.0 * len(rows)), sharey="row",
                             squeeze=False)
    for j, (family, cell) in enumerate(cells.items()):
        est, (lo, hi) = cell["est"], ci(cell["boot"])
        for i, (stat, ylabel) in enumerate(rows):
            ax = axes[i, j]
            for x, p in enumerate(PERSONAS):
                for f in PHRASINGS:
                    color = plotstyle.SERIES[x]
                    if stat == "rho" and est[K[f"{p}:{f}:s_a"]] < MOVED:
                        color = RULE
                    _mark(ax, x + OFFSET[f], est, lo, hi, K[f"{p}:{f}:{stat}"], color, FILL[f])
            ax.axhline(1 if stat != "rho" else 0, color=RULE, ls="--", lw=1, zorder=1)
            ax.set_xticks(range(len(PERSONAS)), PERSONAS, color=INK, fontsize=9)
            if j == 0:
                ax.set_ylabel(ylabel, color=INK, fontsize=9)
            if i == 0:
                ax.set_title(labels[family], color=INK, fontsize=10)
            style(ax)
    handles = [plt.Line2D([], [], color=INK_2, marker="o", lw=0, ms=7, mfc=INK_2 if FILL[f] else "white", mec=INK_2,
                          mew=1.4, label=PHRASING_LABEL[f]) for f in PHRASINGS]
    handles.append(plt.Line2D([], [], color=RULE, marker="o", lw=0, ms=7, label="transfer where the Assistant didn't move"))
    fig.legend(handles=handles, frameon=False, fontsize=9, loc="lower center", ncol=3)
    fig.suptitle("Lazy personas over WildChat, Alpaca and MATH tasks: how far each moves the Assistant and the coin, and\n"
                 "whether the coin moves the way the Assistant does (95% CIs)", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    plotstyle.save(fig, path.stem, path.parent)


def plot_shifts(cells, labels, path):
    """Rows: models; columns: personas. In each panel, per source, the mean change of the Assistant (circles) and the coin
    (diamonds) relative to the 225 tasks' mean, in SDs of the default; filled = "You are", hollow = "answer as"."""
    fig, axes = plt.subplots(len(cells), len(PERSONAS), figsize=(3.0 * len(PERSONAS), 2.7 * len(cells)), sharey="row",
                             squeeze=False)
    colors = {"a": plotstyle.SERIES[0], "c": plotstyle.SERIES[1]}
    for i, (family, cell) in enumerate(cells.items()):
        est, (lo, hi) = cell["est"], ci(cell["boot"])
        for j, p in enumerate(PERSONAS):
            ax = axes[i, j]
            for x, s in enumerate(ARB):
                for who, dx in (("a", -0.24), ("c", 0.08)):
                    for f in PHRASINGS:
                        _mark(ax, x + dx + (0 if f == "youare" else 0.14), est, lo, hi, K[f"{p}:{f}:sh_{who}_{s}"],
                              colors[who], FILL[f], "o" if who == "a" else "D")
            ax.axhline(0, color=RULE, lw=0.8, zorder=1)
            ax.set_xticks(range(len(ARB)), [SOURCE_LABEL[s] for s in ARB], color=INK, fontsize=8)
            if i == 0:
                ax.set_title(p, color=INK, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{labels[family]}\nchange (default SDs)", color=INK, fontsize=9)
            style(ax)
    handles = [plt.Line2D([], [], color=colors[w], marker=m, lw=0, ms=7, mfc=colors[w] if FILL[f] else "white",
                          mec=colors[w], mew=1.4, label=f"{name}, {PHRASING_LABEL[f]}")
               for w, m, name in (("a", "o", "Assistant"), ("c", "D", "coin")) for f in PHRASINGS]
    fig.legend(handles=handles, frameon=False, fontsize=8.5, loc="lower center", ncol=2)
    fig.suptitle("Which sources each persona moves: mean change of the persona's utilities from the default, by source,\n"
                 "relative to the mean change over the three (in SDs of the default's utilities; 95% CIs)", color=INK,
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    plotstyle.save(fig, path.stem, path.parent)


def sweep_summary(family):
    """summary.json of the model's sweep (for the default utilities' reliabilities)."""
    from coin_report import GLM_SWEEP
    from user_choice import FAMILIES, SHARED_PAIRS, load_sweep
    if family == "glm-4.5-air":
        return json.loads((Path("outputs") / GLM_SWEEP / "analysis" / "summary.json").read_text())
    if FAMILIES[family][1] is None:
        from models import olmo_on_shared_pairs
        return olmo_on_shared_pairs(SHARED_PAIRS)[1]
    return load_sweep(family)[1]


def main():
    import argparse
    from coin_report import sweep_rows
    from persona_placement import merged_fit
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS), help="a subset, e.g. while a model is still running")
    args = ap.parse_args()
    plotstyle.setup()
    out = Path("outputs/lazy")
    out.mkdir(parents=True, exist_ok=True)
    runs = personas.collect_runs("lazy")
    data, cells, labels, gates, tops, passed = {}, {}, {}, [], {}, {}
    ids0 = resamples = None
    for family in args.models:
        labels[family] = label = MODEL_INFO[MODELS[family]][0]
        print(f"{label}:")
        rows, summary, paths, checks = merged_fit(runs[family], out / family)
        srow, ssum = sweep_rows(family), sweep_summary(family)
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

        V = {"A0": B["instruct/assistant"], "C0": B["instruct/coinflip"], "loglen": loglen}
        noise = {"A0": (1 - ssum["instruct/assistant/reliability"]) * V["A0"].var(),
                 "C0": (1 - ssum["instruct/coinflip/reliability"]) * V["C0"].var()}
        for p in PERSONAS:
            for f in PHRASINGS:
                for who, base in (("A", "assistant"), ("C", "coinflip")):
                    c = f"{base}_{f}_{p}"
                    V[f"{who}:{p}:{f}"] = N[f"instruct/{c}"]
                    noise[f"{who}:{p}:{f}"] = (1 - summary[f"instruct/{c}/reliability"]) * N[f"instruct/{c}"].var()
        data[family] = V
        cells[family] = dict(est=stats(V, noise, origins, benign),
                             boot=np.array([stats({k: v[i] for k, v in V.items()}, noise, origins[i], benign[i])
                                            for i in resamples]))
        tops[family] = {f"{s}/{c}": top_token_shares(p) for (s, c), p in sorted(paths.items())}

    print("\nValidation gate:\n  " + "\n  ".join(gates))
    fm = lambda est, lo, hi, s: f"{est[K[s]]:+.3f} [{lo[K[s]]:+.3f}, {hi[K[s]]:+.3f}]"
    table, pooled = [], []
    for family, cell in cells.items():
        est, (lo, hi) = cell["est"], ci(cell["boot"])
        print(f"\n{labels[family]}")
        for s in CONTROL:
            print(f"  {s:12} {fm(est, lo, hi, s)}")
        for p in PERSONAS:
            for f in PHRASINGS:
                row = dict(model=labels[family], persona=p, phrasing=f,
                           call=cell_call(est, lo, hi, p, f) if passed[family] else "no call (gate)")
                print(f"  {p} / {f}: call {row['call']}")
                for s in PER:
                    print(f"    {s:14} {fm(est, lo, hi, f'{p}:{f}:{s}')}")
                    row |= {s: est[K[f"{p}:{f}:{s}"]], f"{s}_lo": lo[K[f"{p}:{f}:{s}"]], f"{s}_hi": hi[K[f"{p}:{f}:{s}"]]}
                table.append(row)
            print(f"  {p}: youare vs answeras")
            for s in PAIR:
                print(f"    {s:14} {fm(est, lo, hi, f'{p}:{s}')}")
        row = dict(model=labels[family])
        for m in ("ma", "mc"):
            row[f"call_{m}"] = phrasing_call(lo, hi, m) if passed[family] else "no call (gate)"
        print(f"  pooled over personas: Assistant {row['call_ma']}; coin {row['call_mc']}")
        for s in POOLED:
            print(f"    {s:14} {fm(est, lo, hi, s)}")
            row |= {s: est[K[s]], f"{s}_lo": lo[K[s]], f"{s}_hi": hi[K[s]]}
        pooled.append(row)

    print("\nExploratory, where the probability goes (mean over prompts, tokens logged in each top 5):")
    for family, by_cell in tops.items():
        for c, top in by_cell.items():
            print(f"  {labels[family]:14} {c:36} " + "  ".join(f"{t!r} {p:.3f}" for t, p in top))

    write_csv(out / "lazy.csv", table)
    write_csv(out / "pooled.csv", pooled)
    plot_overview(cells, labels, Path("figures/lazy_overview.png"))
    plot_shifts(cells, labels, Path("figures/lazy_shifts.png"))
    origins = task_arrays(rows)[1]
    arb = np.isin(origins, ARB)
    for family, V in data.items():  # per-source grids: the three sources across, one row per persona and phrasing
        a0, c0 = V["A0"], V["C0"]
        grid = []
        for i, p in enumerate(PERSONAS):
            for f in PHRASINGS:
                da = np.full(len(a0), np.nan)
                dc = np.full(len(a0), np.nan)
                da[arb] = resid(V[f"A:{p}:{f}"][arb], a0[arb])
                dc[arb] = resid((V[f"C:{p}:{f}"] - c0)[arb], a0[arb])
                grid.append(Row(f"{p}, {'you are' if f == 'youare' else 'answer as'}", da, dc,
                                "Assistant's change (nats)", "coin's change (nats)", plotstyle.SERIES[i]))
        plot_source_grid(grid, origins, f"{labels[family]}: the coin's change against the Assistant's, within each source\n"
                         "(both residualized on the default Assistant over the three sources)",
                         Path(f"figures/lazy_sources_{family}.png"), sources=ARB)
    print("\nWrote outputs/lazy/ and figures/lazy_*.png")


if __name__ == "__main__":
    main()
