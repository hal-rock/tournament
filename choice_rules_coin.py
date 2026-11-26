"""Choice rules, the coinflip (README, "Choice rules: the coinflip"): the pre-registered measures.

Per model and kind (tickets, mixed, position), from that run's utilities over the 257 benign tasks:
- following: the coin's mirrored contrast on the rule score (tickets, mixed) or its mirrored slot lean (position), with the
  Assistant's for scale, and tau, the coin's share of the Assistant's move relative to how it passes on the default;
- retention: q_A (the instructed Assistant's slope on its no-rule self) against q_C (the coin's, relative to its default
  slope), and dq = q_C - q_A;
- secondary: pick shares, the two-pull fit, d_pr, source fixed effects, layout effects, biases, mass, reliability, and the gate.

  uv run python choice_rules_coin.py
Writes outputs/choice_rules_coin/{report.txt, cells.csv, secondary.csv} and figures/choice_rules_coin_*.png.
"""

import csv
import gzip
import json
from pathlib import Path

import numpy as np

import choice_rules as cr
from analyze import pearson

KINDS = {"tickets": ("lower", "higher"), "mixed": ("spanish", "english"), "position": ("first", "second")}
QUALIFY = {"llama-70b": ("tickets", "mixed", "position"), "gemma4-31b": ("tickets", "mixed", "position"),
           "glm-4.5-air": ("mixed", "position"), "deepseek-v4-flash": ("tickets", "mixed", "position")}
N_BOOT = 2000


def find_coin_runs(outputs="outputs"):
    """{(model, kind): run dir} for the fetched runs of W&B group choice_rules_coin, the latest per cell."""
    import yaml
    runs = {}
    for path in sorted(Path(outputs).glob("*/run_config.yaml"), key=lambda p: p.stat().st_mtime):
        cfg = yaml.safe_load(path.read_text())
        if cfg.get("group") == "choice_rules_coin" and (path.parent / "analysis" / "utilities.csv").exists():
            runs[tuple(cfg["tags"])] = path.parent
    return runs


def benign_tasks():
    from tournament import benign_subset, select_pairs
    return benign_subset(cr.decorate(cr._read_jsonl(cr.TASKS_PATH)), select_pairs(300, 15000, 0))


def conditions(kind):
    layout = ("assistant", "coinflip") if kind == "position" else (f"assistant_{kind}", f"coinflip_{kind}")
    plus, minus = KINDS[kind]
    return dict(A_L=layout[0], C_L=layout[1], A_p=f"assistant_{kind}_{plus}_user", A_m=f"assistant_{kind}_{minus}_user",
                C_p=f"coinflip_{kind}_{plus}_user", C_m=f"coinflip_{kind}_{minus}_user", A_0="assistant", C_0="coinflip")


class OriginBootstrap:
    """Tasks resampled within origin; one set of resampled counts per model, shared by every statistic. A utility statistic
    weights each task by its count; a prompt-level mean weights each prompt by its two tasks' counts."""

    def __init__(self, origins, n=N_BOOT, seed=0):
        rng = np.random.default_rng(seed)
        origins = np.asarray(origins)
        self.counts = np.zeros((n, len(origins)))
        for o in sorted(set(origins)):
            idx = np.flatnonzero(origins == o)
            self.counts[:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1 / len(idx)), size=n)


def wslope(x, y, w=None):
    """Least-squares slope of y on x, with weights w of shape (n,) or (B, n) (one slope per row of w)."""
    w = np.ones_like(x) if w is None else w
    sw = w.sum(-1, keepdims=True)
    mx, my = (w * x).sum(-1, keepdims=True) / sw, (w * y).sum(-1, keepdims=True) / sw
    return (w * (x - mx) * (y - my)).sum(-1) / (w * (x - mx) ** 2).sum(-1)


def stat(point, boot):
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return dict(est=float(point), lo=float(lo), hi=float(hi))


def call(s, pos="follows", neg="against", zero="no effect"):
    return pos if s["lo"] > 0 else neg if s["hi"] < 0 else zero


def demean_by(v, origins):
    out = np.array(v, float)
    for o in set(origins):
        m = np.asarray(origins) == o
        out[m] -= out[m].mean()
    return out


def load_run(run_dir, ids):
    with open(Path(run_dir) / "analysis" / "utilities.csv") as f:
        rows = {r["id"]: r for r in csv.DictReader(f)}
    cols = [c for c in next(iter(rows.values())) if c.startswith("instruct/")]
    U = {c.split("/", 1)[1]: np.array([float(rows[i][c]) for i in ids]) for c in cols}
    from analyze import load_results
    lo, mass = cr.pair_log_odds(load_results(run_dir))
    summary = json.loads((Path(run_dir) / "analysis" / "summary.json").read_text())
    rel = {k.split("/")[1]: v for k, v in summary.items() if k.startswith("instruct/") and k.endswith("/reliability")}
    return U, lo, mass, rel


def sweep_utilities(model, ids):
    with open(f"outputs/{cr.SWEEP_RUNS[model]}/analysis/utilities.csv") as f:
        rows = {r["id"]: r for r in csv.DictReader(f)}
    return {c: np.array([float(rows[i][f"instruct/{c}"]) for i in ids]) for c in ("assistant", "coinflip")}


def prompt_mean(lo, idx, w):
    """Mean log-odds of the first-listed outcome over a condition's prompts, and its resampled values."""
    keys = list(lo)
    v = np.array([lo[k] for k in keys])
    a = np.array([idx[k[0]] for k in keys])
    b = np.array([idx[k[1]] for k in keys])
    ww = w[:, a] * w[:, b]
    return float(v.mean()), (ww * v).sum(1) / ww.sum(1)


def pick_shares(kind, rule, lo, tasks, pairs):
    """Share of pairs whose order-averaged lean goes to the rule's task (tickets, mixed: cross-language pairs)."""
    recs = cr.items(kind, rule, lo, tasks, pairs)
    return float(np.mean([r["follow"] for r in recs])), len(recs)


def analyze_cell(model, kind, run_dir, tasks, pairs, boot):
    ids = [t["id"] for t in tasks]
    idx = {i: k for k, i in enumerate(ids)}
    origins = [t["origin"] for t in tasks]
    U, lo, mass, rel = load_run(run_dir, ids)
    c = conditions(kind)
    W = boot.counts
    A_L, C_L, A_p, A_m, C_p, C_m = (U[c[k]] for k in ("A_L", "C_L", "A_p", "A_m", "C_p", "C_m"))
    out = dict(model=model, kind=kind, run=str(run_dir))

    b_L, b_L_boot = wslope(A_L, C_L), wslope(A_L, C_L, W)
    out["b_L"] = stat(b_L, b_L_boot)
    A_bar, C_bar = (A_p + A_m) / 2, (C_p + C_m) / 2
    qA, qA_b = wslope(A_L, A_bar), wslope(A_L, A_bar, W)
    qC, qC_b = wslope(A_L, C_bar) / b_L, wslope(A_L, C_bar, W) / b_L_boot
    out["q_A"], out["q_C"], out["dq"] = stat(qA, qA_b), stat(qC, qC_b), stat(qC - qA, qC_b - qA_b)
    out["retention_call"] = call(out["dq"], "coin keeps more", "coin keeps less", "as much")

    if kind == "position":
        m = {k: prompt_mean(lo[c[k]], idx, W) for k in ("A_p", "A_m", "C_p", "C_m", "A_L", "C_L")}
        H_C = ((m["C_p"][0] - m["C_m"][0]) / 2, (m["C_p"][1] - m["C_m"][1]) / 2)
        H_A = ((m["A_p"][0] - m["A_m"][0]) / 2, (m["A_p"][1] - m["A_m"][1]) / 2)
        out["follow_C"], out["follow_A"] = stat(*H_C), stat(*H_A)
        out["tau"] = stat(H_C[0] / H_A[0] / b_L, H_C[1] / H_A[1] / b_L_boot)
        out["slot_means"] = {k: float(v[0]) for k, v in m.items()}
    else:
        if kind == "tickets":
            raw = -np.array([t["tags"]["ticket"] for t in tasks], float)
        else:
            raw = np.array([t["lang"] == "es" for t in tasks], float)
        s = (raw - raw.mean()) / raw.std()
        D_C, D_A = (C_p - C_m) / 2, (A_p - A_m) / 2
        bC, bC_b = wslope(s, D_C), wslope(s, D_C, W)
        bA, bA_b = wslope(s, D_A), wslope(s, D_A, W)
        out["follow_C"], out["follow_A"] = stat(bC, bC_b), stat(bA, bA_b)
        out["tau"] = stat(bC / bA / b_L, bC_b / bA_b / b_L_boot)
        # Source fixed effects: everything demeaned within origin.
        dm = lambda v: demean_by(v, origins)
        out["follow_C_fe"] = float(wslope(dm(s), dm(D_C)))
        out["no_rule_coin_on_s"] = stat(wslope(s, C_L), wslope(s, C_L, W))
        out["no_rule_assistant_on_s"] = float(wslope(s, A_L))
        # Two-pull fit per rule, and d_pr (partial r with the rule given A_L, minus the no-rule coin's).
        from analyze import partial_r
        z = lambda v: (v - v.mean()) / v.std()
        sign = {"p": 1.0, "m": -1.0}
        for k in ("p", "m"):
            A_k, C_k = U[c[f"A_{k}"]], U[c[f"C_{k}"]]
            X = np.column_stack([np.ones_like(A_L), z(A_L), z(A_k)])
            beta = np.linalg.lstsq(X, C_k, rcond=None)[0]
            out[f"twopull_{k}"] = dict(beta_def=float(beta[1]), beta_ins=float(beta[2]),
                                      w=float(beta[2] / (beta[1] + beta[2])), r_Ak_AL=pearson(A_k, A_L),
                                      separable=abs(pearson(A_k, A_L)) < 0.9)
            out[f"d_pr_{k}"] = partial_r(C_k, sign[k] * s, A_L) - partial_r(C_L, sign[k] * s, A_L)
            out[f"pr_A_{k}"] = partial_r(A_k, sign[k] * s, A_L)
        out["q_C_fe"] = float(wslope(dm(A_L), dm(C_bar)) / wslope(dm(A_L), dm(C_L)))
        out["q_A_fe"] = float(wslope(dm(A_L), dm(A_bar)))
        out["shares"] = {f"{who}_{k}": pick_shares(kind, rule, lo[c[f"{who}_{k}"]], tasks, pairs)
                         for who in ("A", "C") for k, rule in (("p", KINDS[kind][0]), ("m", KINDS[kind][1]))}
        out["shares"].update({f"{who}_L": pick_shares(kind, KINDS[kind][0], lo[c[f"{who}_L"]], tasks, pairs)
                              for who in ("A", "C")})
    out["follow_call"] = call(out["follow_C"])
    # Layout effects, biases, mass, reliability.
    out["r_AL_A0"], out["r_CL_C0"] = pearson(A_L, U["assistant"]), pearson(C_L, U["coinflip"])
    out["bias"] = {cond: float(np.mean(list(lo[cond].values()))) for cond in set(c.values())}
    out["median_mass"] = {cond: float(np.median(mass[cond])) for cond in set(c.values())}
    out["reliability"] = {cond: float(rel[cond]) for cond in set(c.values())}
    sweep = sweep_utilities(model, ids)
    out["gate"] = dict(r_sweep_A=pearson(U["assistant"], sweep["assistant"]),
                       r_sweep_C=pearson(U["coinflip"], sweep["coinflip"]),
                       min_reliability=min(out["reliability"].values()))
    out["plain"] = {"assistant": U["assistant"], "coinflip": U["coinflip"]}
    return out


def fmt(s, d=2):
    return f"{s['est']:+.{d}f} [{s['lo']:+.{d}f}, {s['hi']:+.{d}f}]"


def report(cells):
    lines = ["Primary (per model and kind). follow: tickets/mixed, the mirrored contrast's slope on the rule score (nats per SD);",
             "position, the mirrored mean lean toward the first-listed outcome (nats). tau = (coin / Assistant) / b_L.", ""]
    head = f"{'model':<18}{'kind':<9}{'coin follow':<24}{'call':<10}{'Assistant follow':<26}{'b_L':<8}{'tau':<24}"
    lines.append(head)
    for (model, kind), o in cells.items():
        lines.append(f"{model:<18}{kind:<9}{fmt(o['follow_C'], 3):<24}{o['follow_call']:<10}{fmt(o['follow_A']):<26}"
                     f"{o['b_L']['est']:<8.3f}{fmt(o['tau']):<24}")
    lines += ["", f"{'model':<18}{'kind':<9}{'q_A':<24}{'q_C':<24}{'dq = q_C - q_A':<24}call"]
    for (model, kind), o in cells.items():
        lines.append(f"{model:<18}{kind:<9}{fmt(o['q_A']):<24}{fmt(o['q_C']):<24}{fmt(o['dq']):<24}{o['retention_call']}")
    lines += ["", "Gate: r with the sweep's plain utilities (Assistant, coin) on the 257 tasks; min reliability; plain runs agree"]
    for (model, kind), o in cells.items():
        g = o["gate"]
        ok = g["r_sweep_A"] >= 0.97 and g["r_sweep_C"] >= 0.97 and g["min_reliability"] >= 0.9
        lines.append(f"  {model:<18}{kind:<9}A {g['r_sweep_A']:.3f}  C {g['r_sweep_C']:.3f}  min rel {g['min_reliability']:.3f}"
                     f"  min median mass {min(o['median_mass'].values()):.2f}  {'passes' if ok else 'FAILS'}")
    for model in dict.fromkeys(m for m, _ in cells):
        runs = [o for (m, _), o in cells.items() if m == model]
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                rA = pearson(runs[i]["plain"]["assistant"], runs[j]["plain"]["assistant"])
                rC = pearson(runs[i]["plain"]["coinflip"], runs[j]["plain"]["coinflip"])
                lines.append(f"  {model} plain {runs[i]['kind']} vs {runs[j]['kind']}: A r {rA:.4f}, C r {rC:.4f}")
    lines += ["", "Secondary"]
    for (model, kind), o in cells.items():
        lines.append(f"--- {model} {kind}: r(A_L, A_0) {o['r_AL_A0']:.3f}, r(C_L, C_0) {o['r_CL_C0']:.3f}")
        lines.append("    bias (mean log-odds of the first-listed outcome): " + ", ".join(
            f"{k} {v:+.2f}" for k, v in sorted(o["bias"].items())))
        lines.append("    median mass: " + ", ".join(f"{k} {v:.2f}" for k, v in sorted(o["median_mass"].items())))
        if kind == "position":
            continue
        lines.append(f"    coin follow with source FE {o['follow_C_fe']:+.3f}; q_A FE {o['q_A_fe']:.3f}, q_C FE {o['q_C_fe']:.3f}")
        lines.append(f"    no-rule coin on the rule score {fmt(o['no_rule_coin_on_s'], 3)}; no-rule Assistant {o['no_rule_assistant_on_s']:+.3f}")
        sh = o["shares"]
        lines.append(f"    pick shares for the '{KINDS[kind][0]}' task: no rule A {sh['A_L'][0]:.2f} C {sh['C_L'][0]:.2f} "
                     f"(n={sh['A_L'][1]}); under '{KINDS[kind][0]}' A {sh['A_p'][0]:.2f} C {sh['C_p'][0]:.2f}; "
                     f"the rule's task under '{KINDS[kind][1]}' A {sh['A_m'][0]:.2f} C {sh['C_m'][0]:.2f}")
        for k, rule in (("p", KINDS[kind][0]), ("m", KINDS[kind][1])):
            t = o[f"twopull_{k}"]
            lines.append(f"    {rule}: two-pull beta_def {t['beta_def']:+.3f} beta_ins {t['beta_ins']:+.3f} w {t['w']:.2f} "
                         f"(r(A_k, A_L) {t['r_Ak_AL']:.2f}{'' if t['separable'] else ', not separable'}); "
                         f"d_pr {o[f'd_pr_{k}']:+.3f}; Assistant's partial r {o[f'pr_A_{k}']:+.3f}")
    return "\n".join(lines)


def plot_summary(cells, path="figures/choice_rules_coin_summary.png"):
    """Left: tau per cell. Right: q_A and q_C per cell. Rows grouped by kind."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from matskit import plotstyle
    from report import INK_2, RULE, style
    plotstyle.setup()
    keys = [(m, k) for k in KINDS for m in cr.MODELS if (m, k) in cells]
    y = np.arange(len(keys))[::-1]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 0.42 * len(keys) + 1.6), sharey=True)
    col = plotstyle.SERIES
    for yy, key in zip(y, keys):
        o = cells[key]
        t = o["tau"]
        ax1.errorbar(t["est"], yy, xerr=[[t["est"] - t["lo"]], [t["hi"] - t["est"]]], fmt="o", ms=7, color=col[0],
                     mec="white", elinewidth=1.3, capsize=0)
        for s, mk, cc, dy in ((o["q_A"], "o", col[1], 0.13), (o["q_C"], "D", col[0], -0.13)):
            ax2.errorbar(s["est"], yy + dy, xerr=[[s["est"] - s["lo"]], [s["hi"] - s["est"]]], fmt=mk, ms=6.5, color=cc,
                         mec="white", elinewidth=1.3, capsize=0)
    for ax, ref in ((ax1, (0, 1)), (ax2, (0, 1))):
        for r in ref:
            ax.axvline(r, color=RULE, lw=1, ls="-" if r == 0 else (0, (3, 3)), zorder=0)
        style(ax)
    for i in range(1, len(keys)):
        if keys[i][1] != keys[i - 1][1]:
            for ax in (ax1, ax2):
                ax.axhline(y[i] + 0.5, color=RULE, lw=0.6, zorder=0)
    ax1.set_yticks(y, [f"{cr.MODELS[m]} · {k}" for m, k in keys])
    ax1.set_xlabel("τ: the coin's share of the instructed move,\nrelative to how it passes on the default (1 = as fully)")
    ax2.set_xlabel("retention of the default preference\n(slope on the no-rule Assistant; coin relative to its default)")
    ax2.legend(handles=[Line2D([], [], marker="o", color=col[1], ls="", label="instructed Assistant (q_A)"),
                        Line2D([], [], marker="D", color=col[0], ls="", label="coin (q_C)")],
               frameon=False, fontsize=8.5, loc="lower right")
    ax1.set_title("Does the coin follow the rule?", fontsize=10)
    ax2.set_title("How much default preference is kept?", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_contrasts(cells, tasks, path="figures/choice_rules_coin_contrasts.png"):
    """Tickets and mixed: the coin's mirrored contrast D_C against the Assistant's D_A per task, with the fitted slope and the
    default pass-through b_L for reference (tau is their ratio). Position: the coin's mean lean to heads by condition."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from analyze import ORIGIN_STYLE
    from matskit import plotstyle
    from report import INK_2, RULE, style
    plotstyle.setup()
    models = [m for m in cr.MODELS if any((m, k) in cells for k in KINDS)]
    fig, axes = plt.subplots(len(models), 3, figsize=(11, 3.0 * len(models)), squeeze=False)
    origins = np.array([t["origin"] for t in tasks])
    for i, model in enumerate(models):
        for j, kind in enumerate(KINDS):
            ax = axes[i][j]
            o = cells.get((model, kind))
            if o is None:
                ax.axis("off")
                continue
            U, _, _, _ = load_run(o["run"], [t["id"] for t in tasks])
            c = conditions(kind)
            if kind == "position":
                names = [("no rule", "C_L"), ("“first”", "C_p"), ("“second”", "C_m")]
                vals = [o["slot_means"][k] for _, k in names]
                ax.bar(range(3), vals, width=0.55, color=[INK_2, plotstyle.SERIES[0], plotstyle.SERIES[1]])
                ax.axhline(0, color=RULE, lw=1)
                ax.set_xticks(range(3), [n for n, _ in names])
                ax.set_ylabel("coin: mean log-odds heads")
            else:
                D_C, D_A = (U[c["C_p"]] - U[c["C_m"]]) / 2, (U[c["A_p"]] - U[c["A_m"]]) / 2
                for origin, (color, marker) in ORIGIN_STYLE.items():
                    m = origins == origin
                    if m.any():
                        ax.scatter(D_A[m], D_C[m], s=14, c=color, marker=marker, edgecolors="white", linewidths=0.4)
                xs = np.linspace(D_A.min(), D_A.max(), 2)
                slope = wslope(D_A, D_C)
                ax.plot(xs, D_C.mean() + slope * (xs - D_A.mean()), color=INK_2, lw=2, label=f"fit, slope {slope:.3f}")
                ax.plot(xs, D_C.mean() + o["b_L"]["est"] * (xs - D_A.mean()), color=INK_2, lw=1.2, ls=(0, (3, 3)),
                        label=f"default pass-through b_L {o['b_L']['est']:.3f}")
                ax.axhline(0, color=RULE, lw=0.8, zorder=0)
                ax.set_xlabel(f"Assistant: (“{KINDS[kind][0]}” − “{KINDS[kind][1]}”) / 2")
                ax.set_ylabel("coin: same contrast")
                ax.legend(frameon=False, fontsize=7.5, loc="upper left")
            ax.set_title(f"{cr.MODELS[model]} · {kind} (τ {o['tau']['est']:.2f})", fontsize=9.5)
            style(ax)
    from matplotlib.lines import Line2D
    present = [o for o in ORIGIN_STYLE if (origins == o).any()]
    fig.legend(handles=[Line2D([], [], marker=ORIGIN_STYLE[o][1], color=ORIGIN_STYLE[o][0], ls="", ms=6,
                               label=o.lower().replace("_", " ")) for o in present],
               loc="lower center", ncol=len(present), frameon=False, fontsize=8.5, title="task source (scatter panels)",
               title_fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    runs = find_coin_runs()
    tasks, pairs = benign_tasks()
    boot = OriginBootstrap([t["origin"] for t in tasks])
    cells = {(m, k): analyze_cell(m, k, runs[(m, k)], tasks, pairs, boot)
             for m in cr.MODELS for k in KINDS if (m, k) in runs}
    text = report(cells)
    print(text)
    out = Path("outputs/choice_rules_coin")
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.txt").write_text(text + "\n")
    flat = []
    for (m, k), o in cells.items():
        row = dict(model=m, kind=k, run=o["run"], follow_call=o["follow_call"], retention_call=o["retention_call"])
        for name in ("follow_C", "follow_A", "b_L", "tau", "q_A", "q_C", "dq"):
            row.update({f"{name}_{x}": o[name][x] for x in ("est", "lo", "hi")})
        row.update({f"gate_{x}": v for x, v in o["gate"].items()})
        flat.append(row)
    with open(out / "cells.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in flat for k in r)))
        w.writeheader()
        w.writerows(flat)
    json.dump({f"{m}/{k}": {x: v for x, v in o.items() if x != "plain"} for (m, k), o in cells.items()},
              open(out / "cells.json", "w"), indent=1, default=float)
    print(plot_summary(cells), plot_contrasts(cells, tasks), sep="\n")


if __name__ == "__main__":
    main()
