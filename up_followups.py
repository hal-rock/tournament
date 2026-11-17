"""Analysis of the follow-ups on the coin's " up" (README, "Follow-ups on the coin's ' up'"): lines 1-4, the pre-registered
predictions, tables and figures.

Runs are report_patch.py's W&B runs tagged up_followups (group coin_report), with rows under
outputs/coin_report/<family>/<run id>/ (rows.jsonl, and lens_*.npz for the runs with the lens). Per pair {i, j}, as in
coin_report.py: c the coin at " up" (pair effect), r the answer, d = r(arm) - r(none). Measures:
  share     -slope(d on c) / (2 slope(r on c)): the earlier, coin-based share
  share_r   coin_report.share_r, the coin-free share: -slope(d on r) / 2
  bias      coin_report.bias_shrink of the order-averaged part
  f_X       slope(d_X on d_swap): the part of the swap's shift in the answer that arm X reproduces (line 3)
  g_X       slope(coin shift of X on coin shift of the full single-block swap), at the single-block checks (line 3)
CIs: 95% bootstrap over pairs (N_BOOT resamples), the same resamples for every arm of a run.

Usage: uv run python up_followups.py
Writes outputs/coin_report/up_followups/{summary.txt, line1.csv, line2.csv, line3.csv, line4_positions.csv,
line4_lens.csv} and figures/up_line*.png.
"""

from __future__ import annotations

import io as _io
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import pearson
from coin_report import bias_shrink, load, share_r, slope, sweep_rows
from matskit import plotstyle as ps

OUT = Path("outputs/coin_report/up_followups")
N_BOOT = 1000
FINALS = {"gemma4-31b": "Gemma 4 31B", "llama-70b": "Llama 3.3 70B", "glm-4.5-air": "GLM-4.5-Air"}
QUESTION = {"coin_report": "preference", "coin_predict_think": "guess", "coin_choose_ab": "choice (A/B)",
            "coin_choose_ht": "choice (heads/tails)"}
LABEL = {**FINALS, **{f + "-base": FINALS[f].replace("Llama 3.3", "Llama 3.1") + " base" for f in FINALS}}
LINE3_ARMS = ("swap_pot_grad", "swap_null_grad", "swap_pot_lens", "swap_null_lens", "swap_rand")
VFRACS = (50, 75, 90)


def question(report):
    return report.removesuffix("_bold")


def runs():
    """report_patch.py runs tagged up_followups that finished: dicts of id, family, report, arm_sets and the config."""
    import wandb
    api = wandb.Api()
    out = []
    for r in api.runs("halrockwell-mats/tournament", filters={"tags": "up_followups", "group": "coin_report"}):
        if r.state != "finished":
            continue
        cfg = r.config
        path = Path("outputs/coin_report") / cfg["family"] / r.id
        if not (path / "rows.jsonl").exists():  # finished on W&B, not fetched from its pod yet
            print(f"skipping {cfg['family']}/{cfg['report']} {r.id}: no rows.jsonl here yet")
            continue
        out.append(dict(id=r.id, family=cfg["family"], report=cfg["report"], arm_sets=tuple(cfg["arm_sets"]),
                        config=cfg, path=path))
    return out


def sweep_utilities(family):
    """{task id: (final Assistant, base Assistant)} from the family's sweep (the base family reads its final's sweep)."""
    rows = sweep_rows(family.removesuffix("-base"))
    return {r["id"]: (float(r["instruct/assistant"]), float(r["base/assistant"])) for r in rows}


def boot_indices(n, seed=0):
    rng = np.random.default_rng(seed)
    return [np.arange(n)] + [rng.integers(0, n, n) for _ in range(N_BOOT)]


def with_ci(values):
    """(estimate, lo, hi): the first value is the full-sample estimate, the rest bootstrap resamples."""
    v = np.asarray(values, dtype=float)
    return float(v[0]), float(np.nanpercentile(v[1:], 2.5)), float(np.nanpercentile(v[1:], 97.5))


def run_stats(D, A, idxs):
    """Per resample: the statistics of every arm of a run. A: (A_final, A_base) pair differences [N, 2]."""
    out = []
    for idx in idxs:
        c, r0 = D["c"][idx], D["r"]["none"][idx]
        b0 = D["bias"]["none"][idx]
        s = {"r(answer, coin)": pearson(r0, c), "r(answer, A final)": pearson(r0, A[idx, 0]),
             "r(answer, A base)": pearson(r0, A[idx, 1]), "answer SD": float(r0.std()), "coin SD": float(c.std())}
        d_swap = D["r"]["swap"][idx] - r0 if "swap" in D["r"] else None
        for arm in D["arms"]:
            d = D["r"][arm][idx] - r0
            s[f"{arm}: share"] = -slope(c, d) / (2 * slope(c, r0))
            s[f"{arm}: share_r"] = share_r(d, r0)
            s[f"{arm}: r(d, c)"] = pearson(d, c)
            s[f"{arm}: bias"] = bias_shrink(D["bias"][arm][idx] - b0, b0)
            if d_swap is not None:
                s[f"{arm}: f"] = slope(d_swap, d)
        if {"swap", "ctrl"} <= set(D["arms"]):
            s["swap - ctrl: share_r"] = s["swap: share_r"] - s["ctrl: share_r"]
            from coin_report import regression
            s["kappa"], s["lambda"] = regression(D, idx)
        if "c_arm" in D:
            for v in VFRACS:
                if f"v{v}_full" in D["arms"]:
                    full = D["c_arm"][f"v{v}_full"][idx] - c
                    for part in ("pot", "null"):
                        s[f"v{v}_{part}: g"] = slope(full, D["c_arm"][f"v{v}_{part}"][idx] - c)
        if "swap_pot_grad" in D["arms"]:
            s["f_pot + f_null"] = s["swap_pot_grad: f"] + s["swap_null_grad: f"]
        out.append(s)
    return {k: with_ci([o[k] for o in out]) for k in out[0]}


def load_run(run, util):
    D = load(run["path"] / "rows.jsonl")
    D["mass_med"] = float(np.median(D["mass"]["none"]))
    A = np.array([[util[a][k] - util[b][k] for k in (0, 1)] for a, b in zip(D["a"], D["b"])])
    return D, A


def subset(D, keep):
    """D restricted to the pairs where keep [N] is True (pair-level arrays only; enough for run_stats)."""
    E = dict(D)
    for k in ("pair", "a", "b", "c", "c_ok"):
        E[k] = D[k][keep]
    for k in ("r", "bias", "c_arm"):
        if k in D:
            E[k] = {a: v[keep] for a, v in D[k].items()}
    E["coin_o"], E["coin_ok_o"] = D["coin_o"][keep], D["coin_ok_o"][keep]
    E["r_o"] = {a: v[keep] for a, v in D["r_o"].items()}
    return E


def lens_arrays(path):
    """(pair ids [N], lens [N, L, P, 2], out [N, P, 2], out_mass [N, P, 2]) as pair effects / means over the two orders."""
    files = sorted(path.glob("lens_*.npz"))
    parts = [np.load(f) for f in files]
    pair = np.concatenate([p["pair"] for p in parts])
    order = np.concatenate([p["order"] for p in parts])
    lens = np.concatenate([p["lens"] for p in parts])
    out = np.concatenate([p["out"] for p in parts])
    mass = np.concatenate([p["out_mass"] for p in parts])
    by = {}
    for k, (p, o) in enumerate(zip(pair, order)):
        by.setdefault(int(p), {})[int(o)] = k
    ids = sorted(p for p, v in by.items() if set(v) == {0, 1})
    i0, i1 = [by[p][0] for p in ids], [by[p][1] for p in ids]
    eff = lambda x: (x[i0] - x[i1]) / 2
    return np.array(ids), eff(lens), eff(out), (mass[i0] + mass[i1]) / 2


def corr_cols(X, y):
    """r of each column of X [N, ...] with y [N]."""
    Xc = X - X.mean(0)
    yc = y - y.mean()
    den = np.sqrt((Xc ** 2).sum(0) * (yc ** 2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, np.tensordot(yc, Xc, axes=(0, 0)) / den, np.nan)


def fmt(t, pct=False):
    e, lo, hi = t
    if pct:
        return f"{100 * e:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"
    return f"{e:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def write_csv(path, rows):
    from openers import write_csv as wc
    wc(path, rows)


def main():
    ps.setup()
    OUT.mkdir(parents=True, exist_ok=True)
    all_runs = runs()
    util = {}
    stats, data = {}, {}
    for run in all_runs:
        fam = run["family"]
        if fam not in util:
            util[fam] = sweep_utilities(fam)
        D, A = load_run(run, util[fam])
        key = (fam, question(run["report"]), "line4" if "line4" in run["arm_sets"] else "main")
        data[key] = (run, D, A)
        stats[key] = run_stats(D, A, boot_indices(len(D["c"])))
    buf = _io.StringIO()
    with redirect_stdout(buf):
        report(all_runs, stats, data)
    text = buf.getvalue()
    (OUT / "summary.txt").write_text(text)
    print(text)


def report(all_runs, stats, data):
    print(f"{len(all_runs)} runs: " + ", ".join(f"{r['family']}/{r['report']}/{'+'.join(r['arm_sets'])} {r['id']}"
                                                for r in all_runs))
    for (fam, q, kind), (run, D, A) in sorted(data.items()):
        S = stats[fam, q, kind]
        self_max = float(np.abs(D["r_o"]["self"] - D["r_o"]["none"]).max())
        print(f"\n{fam} {run['report']} [{kind}] {len(D['c'])} pairs: mass {D['mass_med']:.2f}, answer SD "
              f"{S['answer SD'][0]:.2f}, coin SD {S['coin SD'][0]:.2f}, self-patch max |d| {self_max:.2g}, "
              f"wording {run['config'].get('wording_check')}")

    # ---------------------------------------------------------------- line 1
    print("\n=== Line 1: base models (coin-free share of the answer's pair effect)")
    rows1 = []
    for fam in FINALS:
        for q in ("coin_report", "coin_predict_think"):
            for stage, f in (("final", fam), ("base", fam + "-base")):
                if (f, q, "main") not in stats:
                    continue
                S, (run, D, _) = stats[f, q, "main"], data[f, q, "main"]
                gate = D["mass_med"] >= 0.2 and S["answer SD"][0] >= 0.25
                reads = S["swap: share_r"][1] > 0 and S["swap - ctrl: share_r"][1] > 0
                b, sr = S["swap: bias"], S["swap: share_r"][0]
                disrupt = (b[1] > 0 or b[2] < 0) and b[1] <= 2 * sr <= b[2]
                row = dict(family=fam, stage=stage, question=QUESTION[q], report=run["report"], run=run["id"],
                           mass=D["mass_med"], answer_sd=S["answer SD"][0], gate=gate, reads=reads and gate,
                           disruption_flag=disrupt)
                for k in ("swap: share_r", "ctrl: share_r", "swap - ctrl: share_r", "swap: bias", "swap: share",
                          "swap_line: share_r", "r(answer, A final)", "r(answer, A base)", "r(answer, coin)"):
                    row |= {k: S[k][0], k + " lo": S[k][1], k + " hi": S[k][2]}
                rows1.append(row)
                print(f"  {FINALS[fam]:14} {stage:5} {QUESTION[q]:10} mass {D['mass_med']:.2f} SD {S['answer SD'][0]:.2f}"
                      f" gate {'ok' if gate else 'FAIL'} | share_r swap {fmt(S['swap: share_r'], 1)} ctrl "
                      f"{fmt(S['ctrl: share_r'], 1)} diff {fmt(S['swap - ctrl: share_r'], 1)} | bias {fmt(b)}"
                      f"{' (disruption?)' if disrupt else ''} | line {fmt(S['swap_line: share_r'], 1)} | r(ans, A_f) "
                      f"{S['r(answer, A final)'][0]:+.2f}, r(ans, A_b) {S['r(answer, A base)'][0]:+.2f} -> "
                      f"{'READS' if reads and gate else 'no'}")
    write_csv(OUT / "line1.csv", rows1)
    plot_line1(rows1)

    # ---------------------------------------------------------------- line 2
    print("\n=== Line 2: the Assistant's choice after the coin")
    rows2 = []
    for fam in FINALS:
        for q in ("coin_report", "coin_predict_think", "coin_choose_ab", "coin_choose_ht"):
            if (fam, q, "main") not in stats:
                continue
            S, (run, D, _) = stats[fam, q, "main"], data[fam, q, "main"]
            row = dict(family=fam, question=QUESTION[q], report=run["report"], run=run["id"], mass=D["mass_med"])
            for k in ("swap: share", "swap: r(d, c)", "kappa", "swap: share_r", "r(answer, A final)", "swap_line: share"):
                row |= {k: S[k][0], k + " lo": S[k][1], k + " hi": S[k][2]}
            rows2.append(row)
            print(f"  {FINALS[fam]:14} {QUESTION[q]:20} mass {D['mass_med']:.2f} | share {fmt(S['swap: share'], 1)} "
                  f"r(d, c) {S['swap: r(d, c)'][0]:+.2f} kappa {fmt(S['kappa'])} share_r {fmt(S['swap: share_r'], 1)} | "
                  f"r(answer, A) {S['r(answer, A final)'][0]:+.2f} | whole line {fmt(S['swap_line: share'], 1)}"
                  + ("  [mass < 0.2: reported, not interpreted]" if D["mass_med"] < 0.2 else ""))
    write_csv(OUT / "line2.csv", rows2)
    plot_line2(rows2)

    # ---------------------------------------------------------------- line 3
    print("\n=== Line 3: the output-potent part of ' up' (finals; bases only where line 1 reads)")
    rows3 = []
    base_reads = {(r["family"], r["question"]) for r in rows1 if r["stage"] == "base" and r["reads"]}
    for fam in [*FINALS, *(f + "-base" for f in FINALS)]:
        for q in ("coin_report", "coin_predict_think"):
            if (fam, q, "main") not in stats or (fam.endswith("-base") and (fam[:-5], QUESTION[q]) not in base_reads):
                continue
            S, (run, D, _) = stats[fam, q, "main"], data[fam, q, "main"]
            check = all(S[f"v{v}_pot: g"][0] >= 0.5 and S[f"v{v}_null: g"][0] <= 0.5 for v in (75, 90))
            row = dict(family=fam, question=QUESTION[q], run=run["id"], split_check=check,
                       swap_rdc=S["swap: r(d, c)"][0], blocks=str(run["config"].get("single_blocks")))
            for k in [f"v{v}_{p}: g" for v in VFRACS for p in ("pot", "null")] + [f"{a}: f" for a in LINE3_ARMS] + \
                     ["f_pot + f_null"] + [f"{a}: share" for a in ("swap", *LINE3_ARMS)]:
                row |= {k: S[k][0], k + " lo": S[k][1], k + " hi": S[k][2]}
            rows3.append(row)
            print(f"  {fam:17} {QUESTION[q]:10} swap r(d, c) {S['swap: r(d, c)'][0]:+.2f} | check "
                  + " ".join(f"v{v} g_pot {S[f'v{v}_pot: g'][0]:+.2f} g_null {S[f'v{v}_null: g'][0]:+.2f}" for v in VFRACS)
                  + f" -> {'ok' if check else 'FAILS'}")
            print("      " + " | ".join(f"{a.removeprefix('swap_')} f {fmt(S[a + ': f'])}" for a in LINE3_ARMS)
                  + f" | f_pot + f_null {S['f_pot + f_null'][0]:+.2f}")
    write_csv(OUT / "line3.csv", rows3)
    plot_line3(rows3)

    # ---------------------------------------------------------------- line 4
    print("\n=== Line 4: across the line (first 500 pairs)")
    rows4, sums = [], []
    for fam in FINALS:
        for q in ("coin_report", "coin_predict_think"):
            if (fam, q, "line4") not in stats:
                continue
            S4, (run4, D4, A4) = stats[fam, q, "line4"], data[fam, q, "line4"]
            tokens = run4["config"]["suffix_tokens"]
            n_line = sum(1 for a in D4["arms"] if a.startswith("loo"))
            P = len(tokens)
            for k in range(P):
                rows4.append(dict(family=fam, question=QUESTION[q], k=k, token=tokens[k], in_line=k < n_line,
                                  **{m: S4[f"pos{k:02d}: {m}"][0] for m in ("share", "share_r")},
                                  **{m + " lo": S4[f"pos{k:02d}: {m}"][1] for m in ("share", "share_r")},
                                  **{m + " hi": S4[f"pos{k:02d}: {m}"][2] for m in ("share", "share_r")}))
            # the whole line's share on the same 500 pairs, from the main run; the sum of single tokens; the drops
            run3, D3, A3 = data[fam, q, "main"]
            keep = np.isin(D3["pair"], D4["pair"])
            E = subset(D3, keep)
            assert np.array_equal(E["pair"], D4["pair"]), "line 4's pairs must be the main run's first 500"
            idxs = boot_indices(len(D4["c"]), seed=1)
            whole, single_sum, drops = [], [], {k: [] for k in range(n_line)}
            up_minus_best = []
            for idx in idxs:
                c3, r3 = E["c"][idx], E["r"]["none"][idx]
                w = -slope(c3, E["r"]["swap_line"][idx] - r3) / (2 * slope(c3, r3))
                c4, r4 = D4["c"][idx], D4["r"]["none"][idx]
                sh = [-slope(c4, D4["r"][f"pos{k:02d}"][idx] - r4) / (2 * slope(c4, r4)) for k in range(n_line)]
                whole.append(w)
                single_sum.append(sum(sh))
                for k in range(n_line):
                    loo = -slope(c4, D4["r"][f"loo{k:02d}"][idx] - r4) / (2 * slope(c4, r4))
                    drops[k].append(w - loo)
                up_minus_best.append(sh[n_line - 1] - max(sh[:n_line - 1]))
            W, SS, UB = with_ci(whole), with_ci(single_sum), with_ci(up_minus_best)
            readout = S4[f"pos{P - 1:02d}: share"]
            sums.append(dict(family=fam, question=QUESTION[q], whole_line=W[0], whole_line_lo=W[1], whole_line_hi=W[2],
                             single_sum=SS[0], single_sum_lo=SS[1], single_sum_hi=SS[2], up_minus_best_other=UB[0],
                             readout_share=readout[0],
                             **{f"drop {tokens[k]!r}": with_ci(drops[k])[0] for k in range(n_line)}))
            shares_line = [S4[f"pos{k:02d}: share"][0] for k in range(n_line)]
            best = int(np.argmax(shares_line))
            sums[-1]["up_is_largest"] = best == n_line - 1
            print(f"  {FINALS[fam]:14} {QUESTION[q]:10} whole line {fmt(W, 1)}, sum of its single tokens {fmt(SS, 1)}, "
                  f"largest single {tokens[best]!r} {100 * shares_line[best]:.1f}%, ' up' {100 * shares_line[-1]:.1f}%, "
                  f"readout position {100 * readout[0]:.0f}%")
            print("      singles: " + "  ".join(f"{tokens[k]!r} {100 * shares_line[k]:.1f}" for k in range(n_line)))
            print("      after ' up': " + "  ".join(f"{tokens[k]!r} {100 * S4[f'pos{k:02d}: share'][0]:.1f}"
                                                   for k in range(n_line, P)))
            print("      drop when left out of the line: " + "  ".join(f"{tokens[k]!r} {100 * with_ci(drops[k])[0]:.1f}"
                                                                     for k in range(n_line)))
    write_csv(OUT / "line4_positions.csv", rows4)
    write_csv(OUT / "line4_sums.csv", sums)
    plot_line4_positions(rows4)

    print("\n=== Line 4: logit lens of log P(' heads') - log P(' tails') over the suffix (pair effects; all 1,000 pairs)")
    rows_lens, before_r = [], {}
    lens_maps = {}
    for fam in FINALS:
        for q in ("coin_report", "coin_predict_think"):
            if (fam, q, "main") not in data or not list(data[fam, q, "main"][0]["path"].glob("lens_*.npz")):
                continue
            run, D, A = data[fam, q, "main"]
            ids, lens, out, mass = lens_arrays(run["path"])
            pos = {p: k for k, p in enumerate(D["pair"])}
            sel = np.array([pos[p] for p in ids])
            c, a = D["c"][sel], A[sel, 0]
            tokens = run["config"]["suffix_tokens"]
            r_lens_c = corr_cols(lens[..., 0], c)  # [L, P]
            r_out_c = corr_cols(out[..., 0], c)  # [P]
            r_out_a = corr_cols(out[..., 0], a)
            r_lens_a = corr_cols(lens[..., 0], a)
            mass_med = np.median(mass[..., 0], 0)
            lens_maps[fam, q] = (r_lens_c, r_out_c, tokens, mass_med)
            for k, t in enumerate(tokens):
                rows_lens.append(dict(family=fam, question=QUESTION[q], k=k, token=t, out_mass=float(mass_med[k]),
                                      r_out_coin=float(r_out_c[k]), r_out_A=float(r_out_a[k]),
                                      r_lens_coin_max=float(np.nanmax(r_lens_c[:, k])) if np.isfinite(r_lens_c[:, k]).any() else np.nan,
                                      depth_of_max=int(np.nanargmax(r_lens_c[:, k])) if np.isfinite(r_lens_c[:, k]).any() else -1,
                                      r_lens_A_max=float(np.nanmax(r_lens_a[:, k])) if np.isfinite(r_lens_a[:, k]).any() else np.nan))
            before =[k for k, t in enumerate(tokens[:20]) if t.strip() in ("it", "came")][:2]
            before_r[fam, QUESTION[q]] = [float(r_out_c[k]) for k in before]
            unlikely = [k for k in range(len(tokens)) if mass_med[k] < 0.01]
            print(f"  {FINALS[fam]:14} {QUESTION[q]:10} output r with the coin at " + "  ".join(
                f"{tokens[k]!r} {r_out_c[k]:+.2f}" for k in range(len(tokens))))
            print(f"      (heads/tails unlikely, median output mass < 0.01, at {len(unlikely)} of {len(tokens)} positions); "
                  f"best lens depth per position: " + "  ".join(
                      f"{tokens[k]!r} {np.nanmax(r_lens_c[:, k]):+.2f}@{np.nanargmax(r_lens_c[:, k])}"
                      for k in range(len(tokens)) if np.isfinite(r_lens_c[:, k]).any()))
    write_csv(OUT / "line4_lens.csv", rows_lens)
    plot_lens(lens_maps)

    print("\n=== Pre-registered predictions")
    for name, status in score_predictions(rows1, rows2, rows3, sums, before_r):
        print(f"  {name}: {status}")


def verdict(ok, missing=(), notes=(), testable=True):
    """holds / fails, or incomplete while a required cell is missing, or not testable when every cell it names fails a
    gate; notes (e.g. a gated-out cell) are appended."""
    if missing:
        return "incomplete (missing " + ", ".join(map(str, missing)) + ")"
    if not testable:
        return "not testable (" + "; ".join(notes) + ")"
    return ("holds" if ok else "fails") + ("" if not notes else " (" + "; ".join(notes) + ")")


def score_predictions(rows1, rows2, rows3, sums, before_r):
    """The 13 pre-registered predictions (README, "Follow-ups on the coin's ' up'"), each over the cells it names."""
    QS = ("preference", "guess")
    b1 = {(r["family"], r["stage"], r["question"]): r for r in rows1}
    out = []

    def cells1(fams, stages):
        need = [(f, s, q) for f in fams for s in stages for q in QS]
        missing = [k for k in need if k not in b1]
        gated = [k for k in need if k in b1 and k[1] == "base" and not b1[k]["gate"]]
        return missing, gated, [f"{k[0]} base {k[2]} fails the gate: not interpreted" for k in gated]

    missing, gated, notes = cells1(FINALS, ("final", "base"))
    n_base = len(FINALS) * len(QS)
    ok = all(b1[f, "base", q]["swap: share_r"] < b1[f, "final", q]["swap: share_r"]
             for f in FINALS for q in QS if (f, "base", q) not in gated) if not missing else False
    out.append(("1 each base's share_r below its final's, both questions",
                verdict(ok, missing, notes, testable=len(gated) < n_base)))
    missing, gated, notes = cells1(("llama-70b", "glm-4.5-air"), ("base",))
    ok = not missing and not any(b1[f, "base", q]["reads"] for f in ("llama-70b", "glm-4.5-air") for q in QS
                                 if (f, "base", q) not in gated)
    out.append(("2 the Llama and GLM bases don't read ' up' on either question",
                verdict(ok, missing, notes, testable=len(gated) < 4)))
    missing, gated, notes = cells1(("gemma4-31b",), ("final", "base"))
    ok = not missing and any(b1["gemma4-31b", "base", q]["reads"]
                             and b1["gemma4-31b", "base", q]["swap: share_r"] < b1["gemma4-31b", "final", q]["swap: share_r"]
                             for q in QS if ("gemma4-31b", "base", q) not in gated)
    out.append(("3 Gemma's base reads ' up' on >= 1 question, by less than its final", verdict(ok, missing, notes)))

    b2 = {(r["family"], r["question"]): r for r in rows2}
    need = [(f, "choice (A/B)") for f in FINALS]
    missing = [k for k in need if k not in b2]
    out.append(("4 the A/B choice: |share| < 1% in every final",
                verdict(not missing and all(abs(b2[k]["swap: share"]) < 0.01 for k in need), missing)))
    need = [(f, q) for f in ("gemma4-31b", "glm-4.5-air") for q in ("choice (heads/tails)", "preference")]
    missing = [k for k in need if k not in b2]
    low = [f for f in ("gemma4-31b", "glm-4.5-air") if (f, "choice (heads/tails)") in b2
           and b2[f, "choice (heads/tails)"]["mass"] < 0.2]
    ok = not missing and all(
        b2[f, "choice (heads/tails)"]["swap: r(d, c)"] <= -0.3 and b2[f, "choice (heads/tails)"]["kappa lo"] > 0
        and 0.5 <= b2[f, "choice (heads/tails)"]["swap: share"] / b2[f, "preference"]["swap: share"] <= 2
        for f in ("gemma4-31b", "glm-4.5-air") if f not in low)
    out.append(("5 the heads/tails choice reads ' up' in Gemma and GLM, share within 2x of the preference's",
                verdict(ok and len(low) < 2, missing,
                        [f"{f}'s heads/tails choice has median mass {b2[f, 'choice (heads/tails)']['mass']:.2f} < 0.2: "
                         "not interpreted" for f in low])))
    need = [(f, q) for f in FINALS for q in ("choice (A/B)", "choice (heads/tails)")]
    missing = [k for k in need if k not in b2]
    out.append(("6 both choices track the Assistant, r >= 0.8",
                verdict(not missing and all(b2[k]["r(answer, A final)"] >= 0.8 for k in need), missing)))

    b3 = {(r["family"], r["question"]): r for r in rows3}
    need = [(f, q) for f in FINALS for q in QS]
    missing = [k for k in need if k not in b3]
    out.append(("7 the split's check holds in every final", verdict(not missing and all(b3[k]["split_check"] for k in need),
                                                                    missing)))
    g = [("gemma4-31b", q) for q in QS]
    missing8 = [k for k in g if k not in b3]
    if not missing8 and not all(b3[k]["split_check"] for k in g):  # pre-registered: then reported, not interpreted
        out.append(("8 Gemma: f_pot_grad > f_null_grad, both questions",
                    "not testable (Gemma's split check fails: its line-3 arms are reported, not interpreted)"))
    else:
        out.append(("8 Gemma: f_pot_grad > f_null_grad, both questions",
                    verdict(not missing8 and all(b3[k]["swap_pot_grad: f"] > b3[k]["swap_null_grad: f"] for k in g),
                            missing8)))
    checked = [r for r in rows3 if r["split_check"]]
    out.append(("9 |f_rand| < 0.1 everywhere", verdict(
        not missing and all(abs(r["swap_rand: f"]) < 0.1 for r in checked), missing,
        [f"{r['family']} {r['question']}: split check fails, not counted" for r in rows3 if not r["split_check"]])))

    b4 = {(r["family"], r["question"]): r for r in sums}
    missing = [k for k in need if k not in b4]
    out.append(("10 the line's single-token shares sum to less than the whole line's",
                verdict(not missing and all(b4[k]["single_sum"] < b4[k]["whole_line"] for k in need), missing)))
    out.append(("11 ' up' has the largest single-token share in the line",
                verdict(not missing and all(b4[k]["up_is_largest"] for k in need), missing)))
    out.append(("12 the readout position's own share >= 0.5",
                verdict(not missing and all(b4[k]["readout_share"] >= 0.5 for k in need), missing)))
    need = [(f, q) for f in ("gemma4-31b", "llama-70b") for q in QS]
    missing = [k for k in need if k not in before_r]
    ok = not missing and all(len(before_r[k]) == 2 and all(r > 0.3 for r in before_r[k]) for k in need)
    out.append(("13 the output's heads/tails lean at ' it', ' came' tracks the coin (r > 0.3) in Gemma and Llama",
                verdict(ok, missing)))
    return out


# ---------------------------------------------------------------- figures

def errorbars(ax, xs, rows, key, color, marker, label, scale=100):
    e = np.array([scale * r[key] for r in rows])
    lo = np.array([scale * r[key + " lo"] for r in rows])
    hi = np.array([scale * r[key + " hi"] for r in rows])
    # a ratio's full-sample estimate can sit outside its percentile CI; the bars then stop at the estimate
    ax.errorbar(xs, e, yerr=[np.clip(e - lo, 0, None), np.clip(hi - e, 0, None)], fmt=marker, color=color, ms=7, elinewidth=1.5, capsize=0, mec=ps.SURFACE,
                mew=0.8, label=label, lw=0)


def plot_line1(rows):
    fams = [f for f in FINALS if any(r["family"] == f for r in rows)]
    if not fams:
        return
    fig, axes = plt.subplots(1, len(fams), figsize=(3.6 * len(fams), 3.8), sharey=True, squeeze=False)
    for ax, fam in zip(axes[0], fams):
        for j, (stage, color, marker) in enumerate((("final", ps.SERIES[0], "o"), ("base", ps.SERIES[1], "s"))):
            rs = [r for r in rows if r["family"] == fam and r["stage"] == stage]
            xs = [["preference", "guess"].index(r["question"]) + (j - 0.5) * 0.25 for r in rs]
            errorbars(ax, xs, rs, "swap: share_r", color, marker, stage)
            ax.scatter(xs, [100 * r["ctrl: share_r"] for r in rs], marker="_", s=120, color=ps.MUTED, zorder=3,
                       label="ctrl (\"OK, \" opener)" if j == 0 else None)
        ax.axhline(0, color=ps.BASELINE, lw=0.8)
        ax.set_xticks([0, 1], ["preference", "guess"])
        ax.set_xlim(-0.6, 1.6)
        ax.set_title(FINALS[fam], fontsize=10)
    axes[0][0].set_ylabel("share of the answer's pair effect\nreversed by swapping \" up\" (%)")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="lower center", ncol=3, fontsize=9)
    fig.suptitle("Line 1: does the base model's answer read the coin's \" up\"?", fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    ps.save(fig, "up_line1_bases")


def plot_line2(rows):
    fams = [f for f in FINALS if any(r["family"] == f for r in rows)]
    if not fams:
        return
    qs = ["preference", "guess", "choice (heads/tails)", "choice (A/B)"]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for j, fam in enumerate(fams):
        rs = [r for r in rows if r["family"] == fam]
        xs = [qs.index(r["question"]) + (j - 1) * 0.2 for r in rs]
        errorbars(ax, xs, rs, "swap: share", ps.SERIES[j], "osD"[j], FINALS[fam])
    ax.axhline(0, color=ps.BASELINE, lw=0.8)
    ax.set_xticks(range(len(qs)), qs)
    ax.set_ylabel("share of the answer's coin-aligned\npart reversed by swapping \" up\" (%)")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Line 2: the Assistant's choice after the coin, against its preference and guess", fontsize=11)
    fig.tight_layout()
    ps.save(fig, "up_line2_choice")


def plot_line3(rows):
    if not rows:
        return
    labels = {"swap_pot_grad": "potent (gradient)", "swap_null_grad": "rest (gradient)", "swap_pot_lens": "potent (lens)",
              "swap_null_lens": "rest (lens)", "swap_rand": "random 1-D"}
    keys = [(r["family"], r["question"]) for r in rows]
    fig, axes = plt.subplots(1, len(keys), figsize=(2.9 * len(keys), 3.9), sharey=True, squeeze=False)
    for ax, (fam, q) in zip(axes[0], keys):
        (r,) = [r for r in rows if (r["family"], r["question"]) == (fam, q)]
        for k, a in enumerate(LINE3_ARMS):
            errorbars(ax, [k], [r], a + ": f", ps.SERIES[[0, 0, 1, 1, 2][k]], "o" if "pot" in a else "s" if "null" in a else "D",
                      labels[a], scale=1)
        ax.axhline(0, color=ps.BASELINE, lw=0.8)
        ax.axhline(1, color=ps.GRID, lw=0.8, ls="--")
        ax.set_xticks(range(len(LINE3_ARMS)), [labels[a] for a in LINE3_ARMS], rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{LABEL[fam]}\n{q}{'' if r['split_check'] else ' (check fails)'}", fontsize=9)
    axes[0][0].set_ylabel("part of the swap's shift in the answer\nreproduced (slope on the full swap's)")
    fig.suptitle("Line 3: is it the coin's output-potent direction at \" up\" that the answer reads?", fontsize=11)
    fig.tight_layout()
    ps.save(fig, "up_line3_split")


def plot_line4_positions(rows):
    """One panel per model and question (the tokens after " up" differ between questions), share by position."""
    keys = [(f, q) for f in FINALS for q in ("preference", "guess")
            if any(r["family"] == f and r["question"] == q for r in rows)]
    if not keys:
        return
    fig, axes = plt.subplots(len(keys), 1, figsize=(12, 2.9 * len(keys)), squeeze=False)
    for ax, (fam, q) in zip(axes[:, 0], keys):
        rs = sorted((r for r in rows if r["family"] == fam and r["question"] == q), key=lambda r: r["k"])
        x = [r["k"] for r in rs]
        color = ps.SERIES[("preference", "guess").index(q)]
        ax.fill_between(x, [100 * r["share lo"] for r in rs], [100 * r["share hi"] for r in rs], color=color, alpha=0.18,
                        lw=0)
        ax.plot(x, [100 * r["share"] for r in rs], color=color, lw=2, marker="o", ms=4)
        ax.set_xticks(x, [r["token"].replace("\n", "\\n") for r in rs], rotation=90, fontsize=7)
        n_line = sum(r["in_line"] for r in rs)
        ax.axvspan(-0.5, n_line - 0.5, color=ps.GRID, alpha=0.4, lw=0)
        ax.axhline(0, color=ps.BASELINE, lw=0.8)
        ax.set_yscale("symlog", linthresh=1)
        ax.set_ylabel("share reversed (%)")
        ax.set_title(f"{FINALS[fam]}, {q}: each position swapped alone at every block (shaded: the coin line)", fontsize=10)
    fig.suptitle("Line 4: where the answer's order-dependence sits, position by position (500 pairs; 95% CIs)", fontsize=11)
    fig.tight_layout()
    ps.save(fig, "up_line4_positions")


def plot_lens(maps):
    if not maps:
        return
    keys = sorted(maps, key=lambda k: (list(FINALS).index(k[0]), ("coin_report", "coin_predict_think").index(k[1])))
    fig, axes = plt.subplots(len(keys), 1, figsize=(12, 3.4 * len(keys)), squeeze=False)
    for ax, key in zip(axes[:, 0], keys):
        r_lens, r_out, tokens, mass = maps[key]
        M = np.vstack([r_lens, r_out[None]])  # [L + 1, P]: depth rows, then the output
        im = ax.imshow(M, aspect="auto", cmap=ps.DIVERGING, vmin=-1, vmax=1, origin="lower", interpolation="nearest")
        ax.set_xticks(range(len(tokens)), [t.replace("\n", "\\n") for t in tokens], rotation=90, fontsize=7)
        ax.set_ylabel("depth (block input; top row = output)", fontsize=8)
        for k, m in enumerate(mass):
            if m >= 0.01:
                ax.plot(k, M.shape[0] - 1, marker="v", color=ps.INK, ms=4)
        ax.set_title(f"{FINALS[key[0]]}, {QUESTION[key[1]]}: r(heads - tails log-odds, pair effect; coin at \" up\")"
                     f"   (▼: heads/tails mass >= 0.01 at the output)", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    fig.suptitle("Line 4: does the heads/tails lean track the coin at every position? (logit lens and output)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    ps.save(fig, "up_line4_lens")


if __name__ == "__main__":
    main()
