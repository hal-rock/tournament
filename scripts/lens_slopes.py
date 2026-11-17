"""Exploratory (Hal, 2026-11-16, after the line-4 lens results): the size of the "coinflip effect" at every suffix position
and depth, not just its correlation with the coin. Per cell, the slope of the heads - tails log-odds pair effect on the
Assistant's preference (the sweep's A_i - A_j), against two references on the same 1,000 pairs:
  the main coinflip effect   slope of the coin's own pair effect at " up" (the output) on A_i - A_j
  the choice                 slope of the Assistant's A/B choice after the coin (coin_choose_ab, line 2) on A_i - A_j
Lens rows are on the logit lens's own scale, which isn't the output's (mid-depth leans can be larger than the output's),
so they show the pattern; the output row is on the coin's scale.

Usage: uv run python scripts/lens_slopes.py  -> outputs/coin_report/up_followups/lens_slopes.csv, figures/up_line4_lens_slopes.png
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coin_report import load, slope  # noqa: E402
from matskit import plotstyle as ps  # noqa: E402
from up_followups import FINALS, OUT, lens_arrays, runs, sweep_utilities, write_csv  # noqa: E402


def slopes_cols(X, a):
    """slope of each column of X [N, ...] on a [N]."""
    ac = a - a.mean()
    return np.tensordot(ac, X - X.mean(0), axes=(0, 0)) / (ac ** 2).sum()


def main():
    ps.setup()
    all_runs = runs()
    rows, maps = [], {}
    for fam in FINALS:
        util = sweep_utilities(fam)
        main_run = next(r for r in all_runs if r["family"] == fam and r["report"] == "coin_report_bold"
                        and "line3" in r["arm_sets"])
        choice_run = next(r for r in all_runs if r["family"] == fam and r["report"] == "coin_choose_ab")
        D, C = load(main_run["path"] / "rows.jsonl"), load(choice_run["path"] / "rows.jsonl")
        assert np.array_equal(D["pair"], C["pair"])
        A = np.array([util[a][0] - util[b][0] for a, b in zip(D["a"], D["b"])])
        b_coin, b_choice = slope(A, D["c"]), slope(A, C["r"]["none"])
        ids, lens, out, mass = lens_arrays(main_run["path"])
        sel = np.array([{p: k for k, p in enumerate(D["pair"])}[p] for p in ids])
        a = A[sel]
        tokens = main_run["config"]["suffix_tokens"]
        n_line = len(D["r"]) and sum(1 for t in range(len(tokens)) if t <= tokens.index(" up"))
        b_lens = slopes_cols(lens[..., 0], a)  # [L, P]
        b_out = slopes_cols(out[..., 0], a)  # [P]
        maps[fam] = (b_lens / b_coin, b_out / b_coin, tokens, b_coin, b_choice)
        print(f"\n{FINALS[fam]}: main coinflip effect (coin at ' up' on A) {b_coin:.3f} nats per nat; the A/B choice after the "
              f"coin {b_choice:.3f} (coin / choice = {b_coin / b_choice:.2f})")
        for k, t in enumerate(tokens[:n_line + 1]):
            best = int(np.nanargmax(np.abs(b_lens[:, k])))
            print(f"  {t!r:11} output slope {b_out[k]:+.3f} = {b_out[k] / b_coin:+.2f} x coin = {b_out[k] / b_choice:+.3f} x "
                  f"choice | largest |lens slope| {b_lens[best, k]:+.3f} at block {best} ({b_lens[best, k] / b_coin:+.2f} x coin)")
            rows.append(dict(family=fam, k=k, token=t, out_slope=b_out[k], out_vs_coin=b_out[k] / b_coin,
                             out_vs_choice=b_out[k] / b_choice, lens_max_slope=b_lens[best, k], lens_max_block=best,
                             coin_slope=b_coin, choice_slope=b_choice))
    write_csv(OUT / "lens_slopes.csv", rows)

    fig, axes = plt.subplots(len(maps), 1, figsize=(12, 3.4 * len(maps)), squeeze=False)
    for ax, (fam, (rl, ro, tokens, b_coin, b_choice)) in zip(axes[:, 0], maps.items()):
        M = np.vstack([rl, ro[None]])
        im = ax.imshow(M, aspect="auto", cmap=ps.DIVERGING, vmin=-1.5, vmax=1.5, origin="lower", interpolation="nearest")
        ax.set_xticks(range(len(tokens)), [t.replace("\n", "\\n") for t in tokens], rotation=90, fontsize=7)
        ax.set_ylabel("depth (block input;\ntop row = output)", fontsize=8)
        ax.set_title(f"{FINALS[fam]}, preference prompt: slope of heads - tails on the Assistant's A_i - A_j, as a multiple "
                     f"of the coin's at \" up\" ({b_coin:.2f}; the A/B choice's is {b_choice:.2f}); clipped at ±1.5", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    fig.suptitle("The coinflip effect's size at every position and depth (exploratory; lens rows on the lens's own scale)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    ps.save(fig, "up_line4_lens_slopes")


if __name__ == "__main__":
    main()
