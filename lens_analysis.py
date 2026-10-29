"""Utilities at every depth from lens.py's logit-lens readouts, and how the coinflip's come to track
the Assistant's.

At each depth d the per-pair effects and utilities are computed from the lens log-odds exactly as
analyze.py computes them from the output's (swap both orders, least-squares utilities). The coinflip
utilities at d are compared with
  (a) the final Assistant utilities (depth L, the model's output), and
  (b) the Assistant utilities read by the lens at the same depth d.
Pre-registration, readability rule and validation gate: README, "Logit lens".

CIs resample tasks within origin (as models.py), 2000 times; the same resamples serve every depth, so
differences between depths or between (a) and (b) are paired.

Writes outputs/lens/<run>/analysis/{per_depth.csv, utilities_by_depth.npz} and
  figures/lens_depth.png      r (a) and (b), outcome mass, utility SDs and reliability by depth
  figures/lens_scatter.gif    the coinflip-vs-Assistant scatter animated over the readable depths,
  figures/lens_scatter.mp4    (a) left, (b) right, with r by depth below and a cursor at the frame's depth

Usage: uv run python lens_analysis.py outputs/lens/<run_id> [--reference outputs/d8ysbybo]
"""

import argparse
import csv
import gzip
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

from analyze import HARM_ORIGINS, HARM_TOPICS, N_BOOTSTRAP, ORIGIN_STYLE, read_jsonl
from figures import ORIGIN_LABELS
from report import INK, INK_2, RULE, stratified_resamples

READABLE_MASS = 0.10
N_SPLITS = 50
STAGE = "instruct"
LABELS = {"a": "vs final Assistant utilities", "b": "vs Assistant utilities at the same depth"}
FIGURES = Path("figures")


def load(run_dir, condition, tasks):
    """Per-pair effects at every depth: (ia, ib, effect [pairs, depths], mass [prompts, depths], z)."""
    z = np.load(Path(run_dir) / f"{STAGE}__{condition}.npz")
    n = len(z["first"]) // 2
    # lens.py writes the pairs in select_pairs order, then the same pairs swapped.
    assert (z["first"][:n] == z["second"][n:]).all() and (z["second"][:n] == z["first"][n:]).all()
    idx = {t["id"]: i for i, t in enumerate(tasks)}
    ia = np.array([idx[t] for t in z["first"][:n]])
    ib = np.array([idx[t] for t in z["second"][:n]])
    lo = z["lp_first"].astype(np.float64) - z["lp_second"]
    mass = np.exp(z["lp_first"]) + np.exp(z["lp_second"])
    return ia, ib, (lo[:n] - lo[n:]) / 2, mass, z


def fit_depths(n, ia, ib, effect):
    """Utilities [tasks, depths]: analyze.fit at each depth, all depths in one solve."""
    lap = np.zeros((n, n))
    np.add.at(lap, (ia, ia), 1)
    np.add.at(lap, (ib, ib), 1)
    np.add.at(lap, (ia, ib), -1)
    np.add.at(lap, (ib, ia), -1)
    rhs = np.zeros((n, effect.shape[1]))
    np.add.at(rhs, ia, effect)
    np.add.at(rhs, ib, -effect)
    u = np.linalg.lstsq(lap, rhs, rcond=None)[0]
    return u - u.mean(0)


def split_half(n, ia, ib, effect, rng):
    """Spearman-Brown split-half reliability of the utilities at each depth."""
    rs = []
    for _ in range(N_SPLITS):
        half = rng.random(len(ia)) < 0.5
        u1 = fit_depths(n, ia[half], ib[half], effect[half])
        u2 = fit_depths(n, ia[~half], ib[~half], effect[~half])
        rs.append(colcorr(u1, u2))
    r = np.mean(rs, 0)
    return 2 * r / (1 + r)


def colcorr(x, y):
    """Pearson r of matching columns of x and y ([tasks, k], or [tasks] for one column)."""
    x, y = np.atleast_2d(x.T).T, np.atleast_2d(y.T).T
    x, y = x - x.mean(0), y - y.mean(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (x * y).sum(0) / np.sqrt((x ** 2).sum(0) * (y ** 2).sum(0))


def boot_r(x, y, resamples):
    """[resamples, depths] Pearson r over each task resample."""
    return np.stack([colcorr(x[s], y[s]) for s in resamples])


def slope(x, y):
    return np.polyfit(x, y, 1)[0] if x.std() > 0 and y.std() > 0 else np.nan


def ci(samples):
    return np.percentile(samples, 2.5, 0), np.percentile(samples, 97.5, 0)


def validate(run_dir, reference, tasks, U, zs):
    """The pre-registered gate: output log-odds vs the reference run's, depth-L utilities vs its fit."""
    ok = True
    ref_u = {r["id"]: r for r in csv.DictReader(open(Path(reference) / "analysis-15000pairs" / "utilities.csv"))}
    for cond, z in zs.items():
        with gzip.open(Path(reference) / "results" / f"{STAGE}__{cond}.csv.gz", "rt") as f:
            ref = {(r["first"], r["second"]): float(r["log_odds"]) for r in csv.DictReader(f)}
        ours = z["model_lp_first"].astype(np.float64) - z["model_lp_second"]
        diff = np.abs(ours - np.array([ref[(a, b)] for a, b in zip(z["first"], z["second"])]))
        lens_l = np.abs(z["lp_first"][:, -1] - z["model_lp_first"]).max()
        r_u = colcorr(U[cond][:, -1], np.array([float(ref_u[t["id"]][f"{STAGE}/{cond}"]) for t in tasks]))[0]
        passed = np.median(diff) <= 0.05 and r_u >= 0.999
        ok &= passed
        print(f"validation {cond:9}: depth-L lens vs output max {lens_l:.2e} nats; output log-odds vs reference "
              f"median |diff| {np.median(diff):.2e}, max {diff.max():.2e}; utilities r {r_u:.5f} -> "
              f"{'PASS' if passed else 'FAIL'}")
    return ok


def depth_table(tasks, U, mass, rel, resamples):
    a, c = U["assistant"], U["coinflip"]
    benign = np.array([t["topic"] not in HARM_TOPICS and t["origin"] not in HARM_ORIGINS for t in tasks])
    n_depths = c.shape[1]
    readable = {k: np.median(m, 0) >= READABLE_MASS for k, m in mass.items()}
    ok = {"a": readable["coinflip"], "b": readable["coinflip"] & readable["assistant"]}
    x = {"a": np.repeat(a[:, -1:], n_depths, 1), "b": a}
    boot = {k: boot_r(c, x[k], resamples) for k in x}
    r_conv = colcorr(a, a[:, -1])
    rows = []
    for d in range(n_depths):
        row = dict(depth=d, mass_coinflip=np.median(mass["coinflip"][:, d]),
                   mass_assistant=np.median(mass["assistant"][:, d]),
                   readable_coinflip=bool(readable["coinflip"][d]), readable_assistant=bool(readable["assistant"][d]),
                   sd_coinflip=c[:, d].std(), sd_assistant=a[:, d].std(),
                   rel_coinflip=rel["coinflip"][d], rel_assistant=rel["assistant"][d],
                   r_assistant_vs_final=r_conv[d])
        for k in x:
            row[f"r_{k}"] = colcorr(c[:, d], x[k][:, d])[0]
            row[f"r_{k}_lo"], row[f"r_{k}_hi"] = np.percentile(boot[k][:, d], [2.5, 97.5])
            row[f"r_{k}_benign"] = colcorr(c[benign, d], x[k][benign, d])[0]
            # At depth 0 every prompt of a condition ends in the same token embedding: no variance, r and slope NaN.
            row[f"slope_{k}"] = slope(x[k][:, d], c[:, d])
            # Paired against depth L, for the peak-vs-output question.
            row[f"r_{k}_minus_L_lo"], row[f"r_{k}_minus_L_hi"] = np.percentile(boot[k][:, d] - boot[k][:, -1], [2.5, 97.5])
        row["r_b_minus_a_lo"], row["r_b_minus_a_hi"] = np.percentile(boot["b"][:, d] - boot["a"][:, d], [2.5, 97.5])
        rows.append(row)
    return rows, ok, boot


def answer_questions(rows, ok, boot_sd):
    """Print the pre-registered questions' answers."""
    L = len(rows) - 1
    readable_a = [d for d in range(L + 1) if ok["a"][d]]
    onset = next((d for d in readable_a if all(rows[e]["r_a_lo"] > 0 for e in readable_a if e >= d)), None)
    print(f"\nreadable depths, coinflip: {readable_a}")
    print(f"readable depths, assistant: {[d for d in range(L + 1) if rows[d]['readable_assistant']]}")
    print(f"Q1 onset of r (a): depth {onset} ({onset / L:.2f} of L); first readable {readable_a[0] if readable_a else None}")
    for k in ("a", "b"):
        dk = [d for d in range(L + 1) if ok[k][d]]
        if not dk:
            continue
        peak = max(dk, key=lambda d: rows[d][f"r_{k}"])
        above = [d for d in dk if rows[d][f"r_{k}_minus_L_lo"] > 0]
        print(f"Q2 r ({k}) peak at depth {peak}: {rows[peak][f'r_{k}']:.3f} vs output {rows[L][f'r_{k}']:.3f}; "
              f"depths with paired CI above output: {above}")
    sd_peak = max(readable_a, key=lambda d: rows[d]["sd_coinflip"])
    above_sd = [d for d in readable_a if np.percentile(boot_sd[:, d] - boot_sd[:, -1], 2.5) > 0]
    print(f"Q2 coinflip SD peak at depth {sd_peak}: {rows[sd_peak]['sd_coinflip']:.3f} vs output "
          f"{rows[L]['sd_coinflip']:.3f}; depths with paired CI above output: {above_sd}")
    db = [d for d in range(L + 1) if ok["b"][d]]
    print("Q3 r(b) - r(a) CI excludes zero at: "
          f"above {[d for d in db if rows[d]['r_b_minus_a_lo'] > 0]}, below {[d for d in db if rows[d]['r_b_minus_a_hi'] < 0]}")
    settle = next((d for d in range(L + 1) if all(rows[e]["r_assistant_vs_final"] >= 0.9 for e in range(d, L + 1))), None)
    print(f"Q4 Assistant utilities correlate >= 0.9 with the final ones from depth {settle} on")
    print("\ndepth  mass_c mass_a   r(a) [95% CI]          r(b) [95% CI]          benign a/b   slope a  SD_c  SD_a  rel_c  r(A_d,A_L)")
    for r in rows:
        if r["depth"] % 4 and r["depth"] < L - 8 and r["depth"] != readable_a[0]:
            continue
        print(f"{r['depth']:5d}  {r['mass_coinflip']:.2f}   {r['mass_assistant']:.2f}   "
              f"{r['r_a']:5.2f} [{r['r_a_lo']:5.2f}, {r['r_a_hi']:5.2f}]   {r['r_b']:5.2f} [{r['r_b_lo']:5.2f}, {r['r_b_hi']:5.2f}]   "
              f"{r['r_a_benign']:5.2f}/{r['r_b_benign']:5.2f}  {r['slope_a']:7.3f}  {r['sd_coinflip']:.2f}  {r['sd_assistant']:.2f}  "
              f"{r['rel_coinflip']:.3f}  {r['r_assistant_vs_final']:.3f}")


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=8.5)


def faded_line(ax, depths, values, readable, **kw):
    """Solid over readable depths, faded over the rest."""
    ax.plot(depths, np.where(readable, np.nan, values), alpha=0.25, **{k: v for k, v in kw.items() if k != "label"})
    ax.plot(depths, np.where(readable, values, np.nan), **kw)


def plot_depth(rows, ok):
    d = np.array([r["depth"] for r in rows])
    col = lambda k: np.array([r[k] for r in rows])
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), dpi=200, sharex=True)
    ax = axes[0, 0]
    for k, ls in (("a", "-"), ("b", "--")):
        ax.fill_between(d, np.where(ok[k], col(f"r_{k}_lo"), np.nan), np.where(ok[k], col(f"r_{k}_hi"), np.nan),
                        color=INK, alpha=0.08, lw=0)
        faded_line(ax, d, col(f"r_{k}"), ok[k], color=INK, ls=ls, lw=1.8, label=f"({k}) {LABELS[k]}")
    ax.axhline(0, color=RULE, lw=0.8)
    ax.set_title("Coinflip utilities at depth d: r with Assistant utilities", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax = axes[0, 1]
    for k, ls in (("coinflip", "-"), ("assistant", "--")):
        ax.plot(d, col(f"mass_{k}"), color=INK, ls=ls, lw=1.8, label=k)
    ax.axhline(READABLE_MASS, color=RULE, lw=0.8)
    ax.text(0, READABLE_MASS + 0.02, "readable", fontsize=8, color=INK_2)
    ax.set_title("Median P(first) + P(second) under the lens", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax = axes[1, 0]
    for k, ls in (("coinflip", "-"), ("assistant", "--")):
        faded_line(ax, d, col(f"sd_{k}"), col(f"readable_{k}"), color=INK, ls=ls, lw=1.8, label=k)
    ax.set_yscale("log")
    ax.set_title("Utility SD across tasks (log-odds, log scale)", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax = axes[1, 1]
    for k, ls in (("coinflip", "-"), ("assistant", "--")):
        faded_line(ax, d, col(f"rel_{k}"), col(f"readable_{k}"), color=INK, ls=ls, lw=1.8, label=f"split-half reliability, {k}")
    faded_line(ax, d, col("r_assistant_vs_final"), col("readable_assistant"), color=INK_2, ls=":", lw=1.8,
               label="r(Assistant at d, final Assistant)")
    ax.set_title("Reliability and Assistant convergence", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    for ax in axes.flat:
        style(ax)
    for ax in axes[1]:
        ax.set_xlabel("depth (blocks applied; 0 = embeddings, 64 = output)", color=INK_2, fontsize=9)
    fig.suptitle("OLMo 3.1 32B Instruct, logit lens, 300 tasks (faded: lens puts < 0.10 on the outcome tokens)",
                 x=0.01, ha="left", fontsize=10, color=INK_2)
    fig.tight_layout()
    fig.savefig(FIGURES / "lens_depth.png", facecolor="white")
    plt.close(fig)


def animate(rows, ok, U, tasks):
    a, c = U["assistant"], U["coinflip"]
    origins = np.array([t["origin"] for t in tasks])
    L = c.shape[1] - 1
    frames = [d for d in range(L + 1) if ok["a"][d]]
    frames = list(range(frames[0], L + 1))  # contiguous from the first readable depth; gaps drawn faded
    span = lambda v: (v.min() - 0.05 * np.ptp(v), v.max() + 0.05 * np.ptp(v))
    ylim = span(c[:, frames])
    xlim = {"a": span(a[:, -1]), "b": span(a[:, [d for d in frames if ok["b"][d]] or [L]])}
    d_all = np.arange(L + 1)
    r = {k: np.array([row[f"r_{k}"] for row in rows]) for k in ("a", "b")}

    fig = plt.figure(figsize=(11, 7.2), dpi=110)
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.25], hspace=0.38, wspace=0.18)
    ax_s = {"a": fig.add_subplot(gs[0, 0]), "b": fig.add_subplot(gs[0, 1])}
    ax_r = fig.add_subplot(gs[1, :])
    pts, lines = {}, {}
    for k, ax in ax_s.items():
        pts[k] = {o: ax.scatter([], [], s=16, c=col, marker=m, edgecolors="white", linewidths=0.4, zorder=3,
                                label=ORIGIN_LABELS[o]) for o, (col, m) in ORIGIN_STYLE.items()}
        lines[k], = ax.plot([], [], color=INK, lw=1.5, zorder=4)
        ax.axhline(0, color=RULE, lw=0.8, zorder=1)
        ax.axvline(0, color=RULE, lw=0.8, zorder=1)
        ax.set_xlim(*xlim[k])
        ax.set_ylim(*ylim)
        style(ax)
    ax_s["a"].set_xlabel("final Assistant utility (model output, log-odds)", color=INK_2, fontsize=9)
    ax_s["b"].set_xlabel("Assistant utility at depth d (lens, log-odds)", color=INK_2, fontsize=9)
    ax_s["a"].set_ylabel("coinflip utility at depth d (lens, log-odds)", color=INK_2, fontsize=9)
    ax_s["b"].legend(frameon=False, fontsize=8, loc="upper left", handletextpad=0.2)
    texts = {k: ax.text(0.02, 1.02, "", transform=ax.transAxes, fontsize=9.5, color=INK) for k, ax in ax_s.items()}
    for k, ls in (("a", "-"), ("b", "--")):
        faded_line(ax_r, d_all, r[k], ok[k], color=INK, ls=ls, lw=1.6, label=f"({k}) {LABELS[k]}")
    ax_r.axhline(0, color=RULE, lw=0.8)
    cursor = ax_r.axvline(frames[0], color=INK_2, lw=1)
    ax_r.set_xlim(frames[0] - 1, L + 0.5)
    lo = min(np.nanmin(np.where(ok[k], r[k], np.nan)) for k in r)
    ax_r.set_ylim(min(0, lo) - 0.05, 1)
    ax_r.set_xlabel("depth d (blocks applied; 64 = output)", color=INK_2, fontsize=9)
    ax_r.set_ylabel("r with coinflip", color=INK_2, fontsize=9)
    ax_r.legend(frameon=False, fontsize=8, loc="lower right")
    style(ax_r)
    title = fig.text(0.01, 0.97, "", fontsize=12, color=INK, ha="left")
    fig.text(0.01, 0.935, "OLMo 3.1 32B Instruct, 300 tasks; utilities from the logit lens at each depth. "
             "Axes fixed across depths, so the spread shows the utilities' size.", fontsize=9, color=INK_2)

    def draw(d):
        for k in ax_s:
            x = a[:, -1] if k == "a" else a[:, d]
            y = c[:, d]
            for o, p in pts[k].items():
                m = origins == o
                p.set_offsets(np.c_[x[m], y[m]])
                p.set_alpha(1 if ok[k][d] else 0.25)
            b1, b0 = np.polyfit(x, y, 1)
            xs = np.array(ax_s[k].get_xlim())
            lines[k].set_data(xs, b0 + b1 * xs)
            texts[k].set_text(f"({k}) r = {r[k][d]:.2f}   slope {b1:.3f}" + ("" if ok[k][d] else "   (unreadable)"))
        cursor.set_xdata([d, d])
        row = rows[d]
        title.set_text(f"Depth {d} of {L}{' (output)' if d == L else ''}:  outcome mass coinflip "
                       f"{row['mass_coinflip']:.2f}, Assistant {row['mass_assistant']:.2f};  "
                       f"coinflip SD {row['sd_coinflip']:.2f}")
        return []

    seq = frames + [L] * 6  # hold the output frame
    anim = FuncAnimation(fig, draw, frames=seq, blit=False)
    anim.save(FIGURES / "lens_scatter.gif", writer=PillowWriter(fps=3))
    anim.save(FIGURES / "lens_scatter.mp4", writer=FFMpegWriter(fps=3, bitrate=2400))
    for d in (frames[0], L):  # stills for slides and quick looks
        draw(d)
        fig.savefig(FIGURES / f"lens_scatter_depth{d}.png", facecolor="white", dpi=200)
    plt.close(fig)
    print(f"animation: depths {frames[0]}-{L}, {len(seq)} frames")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--reference", default="outputs/d8ysbybo")
    ap.add_argument("--tasks", default="data/tasks_n300_max500_seed0.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    tasks = read_jsonl(args.tasks)
    rng = np.random.default_rng(args.seed)
    U, mass, rel, zs = {}, {}, {}, {}
    for cond in ("assistant", "coinflip"):
        ia, ib, effect, mass[cond], zs[cond] = load(args.run_dir, cond, tasks)
        U[cond] = fit_depths(len(tasks), ia, ib, effect)
        rel[cond] = split_half(len(tasks), ia, ib, effect, rng)
    if not validate(args.run_dir, args.reference, tasks, U, zs):
        print("VALIDATION FAILED: not interpreting (see README, Logit lens)")
    resamples = stratified_resamples(np.array([t["origin"] for t in tasks]), rng, N_BOOTSTRAP)
    rows, ok, _ = depth_table(tasks, U, mass, rel, resamples)
    boot_sd = np.stack([U["coinflip"][s].std(0) for s in resamples])
    answer_questions(rows, ok, boot_sd)

    out = Path(args.run_dir) / "analysis"
    out.mkdir(exist_ok=True)
    with open(out / "per_depth.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    np.savez_compressed(out / "utilities_by_depth.npz", ids=np.array([t["id"] for t in tasks]), **U)
    FIGURES.mkdir(exist_ok=True)
    plot_depth(rows, ok)
    animate(rows, ok, U, tasks)
    print(f"wrote {out}/per_depth.csv, figures/lens_depth.png, figures/lens_scatter.{{gif,mp4}}")


if __name__ == "__main__":
    main()
