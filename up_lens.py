"""Figures for "The heads/tails lean around ' up': how much probability, and does it track the Assistant's answer?"
(README): per model, at every position of the suffix (the coin line through the Assistant's prefill) and every depth,
  probability   the logit lens's P(" heads") + P(" tails") (a softmax over the whole vocabulary), the mean over all
                2,000 prompts (1,000 pairs x both orders), on a log scale; the top row is the model's own output
  correlation   Pearson r over the 1,000 pairs between the lens's lean, log P(" heads") - log P(" tails") (fp32), and the
                Assistant's final answer (the preference report's heads - tails log-odds at the end of the prompt),
                both as pair effects ((order 1 - order 2) / 2, so the order-independent label bias drops out)
The correlation is with the Assistant's answer at the end of this prompt, not with the coin at " up" (the earlier line-4
figure). Runs: report_patch.py runs tagged up_lens (--arm-sets none --lens-lp), rows under outputs/coin_report/<family>/.
Where the same family has a follow-up run on the same GPU type, the lens lean is checked against it.

Usage: uv run python up_lens.py  -> outputs/coin_report/up_lens/{summary.txt, maps.npz}, figures/up_lens_<family>.png
"""

from __future__ import annotations

import io as _io
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from matskit import io, plotstyle as ps
from up_followups import FINALS, LABEL, corr_cols

OUT = Path("outputs/coin_report/up_lens")
LOG10_MIN = -6  # the probability colour scale's floor (10^-6; lower values, mostly early blocks, are drawn at it):
# over 8 decades a mean of 0.01 already looked nearly as dark as 1
FOLLOWUP_RUNS = {"gemma4-31b": "9k9n2joh", "gemma4-31b-base": "bu0qz0s7", "llama-70b": "m3af5pwr",
                 "llama-70b-base": "pa8y5knu"}  # same GPU type as the up_lens runs (GLM's moved to 2x H200)


def runs(tag="up_lens"):
    import wandb
    out = {}
    for r in wandb.Api().runs("halrockwell-mats/tournament", filters={"tags": tag, "group": "coin_report"}):
        path = Path("outputs/coin_report") / r.config["family"] / r.id
        if r.state != "finished" or not (path / "rows.jsonl").exists():
            continue
        out[r.config["family"]] = dict(id=r.id, family=r.config["family"], config=r.config, path=path)
    return out


def load_npz(path):
    parts = [np.load(f) for f in sorted(Path(path).glob("lens_*.npz"))]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files}


def pair_index(pair, order):
    """(pair ids, row of order 0, row of order 1) for the pairs with both orders."""
    by = {}
    for k, (p, o) in enumerate(zip(pair, order)):
        by.setdefault(int(p), {})[int(o)] = k
    ids = sorted(p for p, v in by.items() if set(v) == {0, 1})
    return np.array(ids), np.array([by[p][0] for p in ids]), np.array([by[p][1] for p in ids])


def maps(run):
    """The two maps [L + 1, P] (depth rows entering each block, then the output) and what the figure labels need."""
    Z = load_npz(run["path"])
    ids, i0, i1 = pair_index(Z["pair"], Z["order"])
    eff = lambda x: (x[i0] - x[i1]) / 2
    answer = {(r["pair"], r["order"]): r["report"]["none"] for r in io.read_jsonl(run["path"] / "rows.jsonl")}
    ans = np.array([answer[int(p), int(o)] for p, o in zip(Z["pair"], Z["order"])])
    lp = np.concatenate([Z["lens_lp"][..., :2], Z["out_lp"][:, None, :, :2]], 1)  # [n, L + 1, P, 2]: " heads", " tails"
    mass = np.exp(lp).sum(-1)
    # the lean from the fp32 two-column lens (the full-vocabulary lens runs the head in bf16: up to ~0.4 nats off in Gemma)
    lean = np.concatenate([Z["lens"][..., 0], Z["out"][:, None, :, 0]], 1)
    tokens = run["config"]["suffix_tokens"]
    return dict(prob=np.log10(mass.mean(0)), corr=corr_cols(eff(lean), eff(ans)), tokens=tokens, up=tokens.index(" up"),
                answer_sd=float(eff(ans).std()), n_pairs=len(ids), lean_lens=Z["lens"][..., 0], pair=Z["pair"],
                order=Z["order"], lean_check=lp[:, :-1, :, 0] - lp[:, :-1, :, 1])


def check_against_followup(fam, M):
    """The new run's two-column lens lean against the follow-up run's (same prompts, model and GPU type)."""
    path = Path("outputs/coin_report") / fam / FOLLOWUP_RUNS[fam]
    if not list(path.glob("lens_*.npz")):
        return None
    Z = load_npz(path)
    old = {(int(p), int(o)): k for k, (p, o) in enumerate(zip(Z["pair"], Z["order"]))}
    rows = [(k, old[int(p), int(o)]) for k, (p, o) in enumerate(zip(M["pair"], M["order"])) if (int(p), int(o)) in old]
    a, b = M["lean_lens"][[k for k, _ in rows]], Z["lens"][[j for _, j in rows], ..., 0]
    d = np.abs(a - b)
    return dict(n=len(rows), median=float(np.median(d)), p99=float(np.quantile(d, 0.99)), max=float(d.max()))


def label(t):
    return t.replace("\n", "\\n")


def plot(fam, M):
    stages = [s for s in (fam, fam + "-base") if s in M]
    fig, axes = plt.subplots(2, len(stages), figsize=(7.4 * len(stages), 11), squeeze=False, layout="constrained")
    norms = (Normalize(LOG10_MIN, 0), Normalize(-1, 1))
    for j, s in enumerate(stages):
        m = M[s]
        L1, P = m["corr"].shape
        for i, (key, cmap) in enumerate((("prob", ps.SEQUENTIAL), ("corr", ps.DIVERGING))):
            ax = axes[i, j]
            X = np.maximum(m[key], LOG10_MIN) if key == "prob" else m[key]
            im = ax.imshow(X, aspect="auto", origin="lower", cmap=cmap, norm=norms[i], interpolation="nearest")
            ax.grid(False)
            ax.axhline(L1 - 1.5, color=ps.SURFACE, lw=2.5)  # the output row sits above the lens rows
            ax.add_patch(plt.Rectangle((m["up"] - 0.5, -0.5), 1, L1, fill=False, ec=ps.INK, lw=1.2))
            if key == "prob":  # on one hue, 0.05 and 0.9 look alike: dot the cells where heads/tails is the likely token
                yy, xx = np.nonzero(m["prob"] >= np.log10(0.5))
                ax.scatter(xx, yy, s=2.5, color=ps.SURFACE, marker="o", linewidths=0)
            ax.set_xticks(range(P), [label(t) for t in m["tokens"]], rotation=90, fontsize=7)
            ax.get_xticklabels()[m["up"]].set_fontweight("bold")
            step = max(1, round((L1 - 1) / 4 / 5) * 5)
            ticks = list(range(0, L1 - 1, step))
            ax.set_yticks(ticks + [L1 - 1], [str(t) for t in ticks] + ["output"], fontsize=8)
            if j == 0:
                ax.set_ylabel("depth: the residual entering block", fontsize=9)
            what = ("mean P(\" heads\") + P(\" tails\") under the logit lens\n(dots: >= 0.5)" if key == "prob" else
                    f"r(heads - tails lean, Assistant's answer), {m['n_pairs']:,} pairs\n"
                    f"(the answer's pair SD: {m['answer_sd']:.2f} nats)")
            ax.set_title(f"{LABEL[s]}: {what}", fontsize=9)
        if j == len(stages) - 1:
            for i, cb_label in enumerate(("P, log scale", "Pearson r")):
                cb = fig.colorbar(plt.cm.ScalarMappable(norm=norms[i], cmap=(ps.SEQUENTIAL, ps.DIVERGING)[i]),
                                  ax=axes[i, :].tolist(), fraction=0.025, pad=0.01)
                cb.set_label(cb_label, fontsize=9)
                if i == 0:
                    cb.set_ticks(range(LOG10_MIN, 1), labels=[f"$10^{{{k}}}$" if k else "1" for k in
                                                             range(LOG10_MIN, 1)])
    fig.suptitle(f"{FINALS[fam]}, preference prompt: how much probability heads/tails get at each position and depth (top),\n"
                 f"and whether the heads - tails lean there tracks the Assistant's final answer (bottom). Boxed: \" up\"",
                 fontsize=11)
    ps.save(fig, f"up_lens_{fam}")


def main():
    ps.setup()
    OUT.mkdir(parents=True, exist_ok=True)
    R = runs()
    M = {fam: maps(run) for fam, run in R.items()}
    buf = _io.StringIO()
    with redirect_stdout(buf):
        for fam in [f for f in [*FINALS, *(f + "-base" for f in FINALS)] if f in M]:
            m, k = M[fam], M[fam]["up"]
            check = check_against_followup(fam, m) if fam in FOLLOWUP_RUNS else None
            p_up, r_up = m["prob"][:, k], m["corr"][:, k]
            first = next((l for l, v in enumerate(p_up[:-1]) if v >= -2), None)
            print(f"\n{LABEL[fam]} ({R[fam]['id']}): {m['n_pairs']} pairs, answer pair SD {m['answer_sd']:.3f} nats")
            print(f"  at \" up\": P(heads) + P(tails) at the output {10 ** p_up[-1]:.3f}; under the lens first >= 1% entering "
                  f"block {first}, largest {10 ** p_up[:-1].max():.3f} (block {int(p_up[:-1].argmax())})")
            print(f"  at \" up\": r with the answer at the output {r_up[-1]:+.2f}; under the lens largest {np.nanmax(r_up[:-1]):+.2f} "
                  f"(block {int(np.nanargmax(r_up[:-1]))})")
            others = [j for j in range(len(m["tokens"])) if j != k]
            best = max(others, key=lambda j: np.nanmax(m["corr"][:, j]))
            print(f"  largest r at any other position: {np.nanmax(m['corr'][:, best]):+.2f} at {m['tokens'][best]!r} "
                  f"(row {int(np.nanargmax(m['corr'][:, best]))} of {m['corr'].shape[0] - 1} = output)")
            print(f"  lens lean: full-vocabulary vs two-column, max |diff| "
                  f"{float(np.abs(m['lean_check'] - m['lean_lens']).max()):.3g} nats")
            if check:
                print(f"  lens lean vs the follow-up run's ({FOLLOWUP_RUNS[fam]}, {check['n']} prompts): |diff| median "
                      f"{check['median']:.2g}, 99th pct {check['p99']:.2g}, max {check['max']:.2g} nats")
    (OUT / "summary.txt").write_text(buf.getvalue())
    print(buf.getvalue())
    np.savez_compressed(OUT / "maps.npz", **{f"{fam}__{k}": np.asarray(v) for fam, m in M.items()
                                            for k, v in m.items() if k in ("prob", "corr", "tokens")})
    for fam in FINALS:
        if fam in M or fam + "-base" in M:
            plot(fam, M)


if __name__ == "__main__":
    main()
