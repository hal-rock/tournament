"""Read olmo2_attr.py's scores against the pre-registered rules (README, "OLMo 2 32B SFT data attribution").

Writes data/olmo2_attr/summary.json (every number the write-up quotes), data/olmo2_attr/units_<point>.csv (per-unit cos on
every query and the labels; no text) and figures/olmo2_attr_*.png. Points not fetched yet are skipped.

A unit's cos on query m: its Adam-weighted score over |g / rms| and the query's lattice norm. Strat units are weighted by the
mixture rows their category's stratum stands for. CIs resample units within category (strat) or within group.

    uv run --with pandas --with scipy python olmo2_attr_analysis.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "olmo2_attr"
OUT = ROOT / "outputs" / "olmo2_attr"
YESTERDAY = ROOT.parent / "coinflip" / "outputs" / "attribution" / "olmo2" / "sft-a0"
POINTS = ("base", "sft")
QUERIES = ["benign", "ordinary", "harm_gap", "t1", "t2", "perm1", "perm2", "heads_bias", "asst_benign", "standard"]
COIN = {"benign", "ordinary", "harm_gap", "t1", "t2", "perm1", "perm2"}
BATCH = 256
CAT_ORDER = ["chat", "safety", "if", "science", "multilingual", "other", "code", "math", "identity"]
LABEL_LEVELS = {"task_type": ["math", "coding", "factual_qa", "explanation", "advice", "creative_writing", "roleplay", "text_task",
                              "opinion", "harmful_request", "chit_chat", "other"],
                "response_stance": ["complies", "partial", "refuses"], "voice": ["impersonal", "assistant", "character"],
                "own_view": [True], "mentions_ai": [True], "moral_language": [True]}
N_BOOT = 2000
rng = np.random.default_rng(20261112)


# ---------------------------------------------------------------- loading

def query_norms(info):
    norms = {m: info["coin"]["lattice_norms"][m] for m in COIN}
    for m in ("heads_bias", "asst_benign", "standard"):
        norms[m] = info[m]["lattice_norm"]
    return norms


def load_point(point):
    d = OUT / point
    info = json.load(open(d / "queries.json"))
    R = json.load(open(d / "pilot.json"))["R"]
    rows = [json.loads(line) for line in open(d / "scores.jsonl")]
    norms = query_norms(info)
    u = pd.DataFrame([{k: r[k] for k in ("idx", "set", "category", "source", "n", "tokens", "lp", "g_sq", "g_adam_sq")}
                      | {f"adam_{m}": r["adam"][m] for m in QUERIES} | {f"raw_{m}": r["raw"][m] for m in QUERIES}
                      | {k: r.get(k) for k in ("rms_half_a", "rms_half_b", "lattice_even", "lattice_odd")}
                      | {"lattice_std_even": (r.get("lattice_std") or [None, None])[0], "lattice_std_odd": (r.get("lattice_std") or [None, None])[1]}
                      for r in rows]).drop_duplicates("idx", keep="last").set_index("idx")
    for m in QUERIES:
        u[f"cos_{m}"] = u[f"adam_{m}"] / np.sqrt(u.g_adam_sq) / norms[m]
    u["clip"] = 1 / np.sqrt(R ** 2 + u.g_sq / BATCH ** 2)
    u["contrib"] = u["clip"] * u["adam_benign"]  # first-order push, size included (secondary)
    return u, info, R


def load_labels():
    p = DATA / "labels.csv"
    if not p.exists():
        return None
    lab = pd.read_csv(p, dtype={"task_type": str, "response_stance": str, "voice": str}).set_index("idx")
    for f in ("own_view", "mentions_ai", "moral_language"):
        lab[f] = lab[f].map({"True": True, "False": False, True: True, False: False})
    return lab


def unit_weights(u, stats):
    rows = stats["category"]
    strat = u["set"] == "strat"
    counts = u[strat].category.value_counts()
    w = pd.Series(0.0, index=u.index)
    w[strat] = u.loc[strat, "category"].map(lambda c: rows[c] / counts[c]).astype(float)
    return w


# ---------------------------------------------------------------- the pre-registered reading

def top_share(c, w, frac):
    o = np.argsort(-c)
    cw = np.cumsum(w[o]) / w.sum()
    return float((c[o] * w[o])[cw <= frac].sum() / (c * w).sum())


def boot_weighted_mean(c, w, cat, n=N_BOOT):
    idx = {k: np.where(cat == k)[0] for k in np.unique(cat)}
    out = np.empty(n)
    for b in range(n):
        s = np.concatenate([rng.choice(v, len(v)) for v in idx.values()])
        out[b] = (c[s] * w[s]).sum() / w[s].sum()
    return out


def rules(u, col, w):
    m = (w > 0).to_numpy()
    c, ww, cat = u[col].to_numpy(float)[m], w.to_numpy(float)[m], u["category"].to_numpy()[m]
    W, net = ww.sum(), (c * ww).sum()
    g = pd.DataFrame({"c": c, "w": ww, "cw": c * ww, "cat": cat}).groupby("cat").agg(
        w=("w", "sum"), cw=("cw", "sum"), mean=("c", "mean"), sem=("c", "sem"), n=("c", "size"))
    g["share_rows"], g["share_net"] = g.w / W, g.cw / net
    g["E"] = g.share_net / g.share_rows
    g["t"] = g["mean"] / g["sem"]
    big = g[(g.share_rows >= 0.02) & (g.index != "identity")]
    frac_pos = float(ww[c > 0].sum() / W)
    top1 = top_share(c, ww, 0.01)
    boot = boot_weighted_mean(c, ww, cat)
    ci = np.quantile(boot, [0.025, 0.975]).tolist()
    nonpos = big[big["mean"] <= 0]
    r = {"R1_breadth": frac_pos >= 0.55,
         "R2_categories": len(nonpos) == 0 or (len(nonpos) == 1 and float(nonpos.t.iloc[0]) > -2),
         "R3_proportionality": bool(big.E.max() <= 3 and (big.E >= 0.33).mean() >= 0.75),
         "R4_no_small_set": top1 < 0.25}
    conc = [f"category {k}: {v.share_net:.0%} of net on {v.share_rows:.1%} of rows" for k, v in g.iterrows() if v.share_rows < 0.10 and v.share_net >= 0.5]
    if top1 >= 0.40:
        conc.append(f"top 1% hold {top1:.0%}")
    if (big["mean"] > 0).mean() < 0.5:
        conc.append("under half the categories positive")
    if ci[0] <= 0 <= ci[1]:
        verdict = "no clear net push"
    elif net < 0:
        verdict = "net push negative"
    elif conc:
        verdict = "concentrated"
    elif all(r.values()):
        verdict = "diffuse"
    else:
        verdict = "mixed"
    opposing = bool((big.t >= 3).any() and (big.t <= -3).any())
    return dict(verdict=verdict, opposing_pushes=opposing, rules=r, concentrated_by=conc, frac_pos=frac_pos, top1=top1,
                top10=top_share(c, ww, 0.10), weighted_mean=float(net / W), weighted_mean_ci=ci, sd=float(c.std()),
                skew=float(st.skew(c)), kurtosis=float(st.kurtosis(c)), units=int(m.sum()),
                categories=g[["n", "share_rows", "share_net", "E", "mean", "sem", "t"]].reset_index().to_dict("records"))


def category_means(u, col):
    s = u[u["set"] == "strat"]
    return s.groupby("category")[col].mean()


def pattern_specificity(u):
    cats = [c for c in CAT_ORDER if c != "identity"]
    cm = {m: category_means(u, f"cos_{m}").reindex(cats) for m in ("benign", "t1", "t2", "perm1", "perm2")}
    rho = lambda a, b: float(st.spearmanr(cm[a], cm[b]).statistic)  # noqa: E731
    out = {"t1_t2": rho("t1", "t2"), "benign_perm1": rho("benign", "perm1"), "benign_perm2": rho("benign", "perm2"),
           "benign_t1": rho("benign", "t1"), "benign_t2": rho("benign", "t2")}
    out["specific"] = bool(out["t1_t2"] >= 0.7 and min(out["benign_t1"], out["benign_t2"]) > max(abs(out["benign_perm1"]), abs(out["benign_perm2"])))
    return out


def group_diff(u, group, ctrl, cols):
    """Group minus control mean, in control SDs, per column; bootstrap paired across columns (same resamples)."""
    a, b = u[u["set"] == group], u[u["set"] == ctrl]
    sd = {c: b[c].std(ddof=1) for c in cols}
    obs = {c: (a[c].mean() - b[c].mean()) / sd[c] for c in cols}
    A, B = a[cols].to_numpy(), b[cols].to_numpy()
    boots = np.empty((N_BOOT, len(cols)))
    for k in range(N_BOOT):
        boots[k] = (A[rng.integers(0, len(A), len(A))].mean(0) - B[rng.integers(0, len(B), len(B))].mean(0)) / np.array([sd[c] for c in cols])
    return obs, boots, dict(n=len(a), n_ctrl=len(b))


def refusal_test(u):
    cols = ["cos_standard", "cos_harm_gap", "cos_benign", "cos_ordinary"]
    obs, boots, n = group_diff(u, "refusal", "refusal_ctrl", cols)
    j = {c: i for i, c in enumerate(cols)}
    ci = {c.removeprefix("cos_"): np.quantile(boots[:, j[c]], [0.025, 0.975]).tolist() for c in cols}
    d = {c.removeprefix("cos_"): float(v) for c, v in obs.items()}
    diff = boots[:, j["cos_standard"]] - boots[:, j["cos_benign"]]
    diff_ci = np.quantile(diff, [0.025, 0.975]).tolist()
    readable = not (ci["standard"][0] <= 0 <= ci["standard"][1])
    if not readable:
        call = "not readable (D on standard includes 0)"
    elif d["benign"] < d["standard"] / 2 and diff_ci[0] > 0:
        call = "specific to harm"
    elif d["benign"] >= 0.75 * d["standard"] and ci["benign"][0] > 0:
        call = "broad too"
    else:
        call = "inconclusive"
    return dict(d=d, ci=ci, standard_minus_benign=float(d["standard"] - d["benign"]), standard_minus_benign_ci=diff_ci, call=call, **n)


def selfref_test(u):
    cols = ["cos_benign", "cos_standard", "cos_harm_gap", "cos_perm1"]
    obs, boots, n = group_diff(u, "selfref", "selfref_ctrl", cols)
    return dict(d={c.removeprefix("cos_"): float(v) for c, v in obs.items()},
                ci={c.removeprefix("cos_"): np.quantile(boots[:, i], [0.025, 0.975]).tolist() for i, c in enumerate(cols)}, **n)


def label_contrasts(u, lab, rows):
    """Within-category difference (label value vs the rest), pooled over categories by row weight, in SDs of cos; strat units."""
    s = u[u["set"] == "strat"].join(lab, how="inner").dropna(subset=["task_type"])  # unlabelled units (refusals) on neither side
    out = []
    for field, levels in LABEL_LEVELS.items():
        for lv in levels:
            x = (s[field] == lv)
            res = {"field": field, "level": str(lv), "n": int(x.sum()), "share_of_units": float(x.mean())}
            for m in ("benign", "perm1", "perm2", "standard", "harm_gap", "asst_benign"):
                col = f"cos_{m}"
                sd = s[col].std()
                cats = [c for c, g in s.groupby("category") if (x[g.index]).sum() >= 10 and (~x[g.index]).sum() >= 10]
                if not cats:
                    res[m] = None
                    continue
                Wc = np.array([rows[c] for c in cats], float)

                def pooled(take):
                    vals = []
                    for c in cats:
                        g = take[c]
                        vals.append(g[0].mean() - g[1].mean())
                    return float((np.array(vals) * Wc).sum() / Wc.sum() / sd)

                groups = {c: (s.loc[s.category == c][x[s.category == c]][col].to_numpy(), s.loc[s.category == c][~x[s.category == c]][col].to_numpy()) for c in cats}
                obs = pooled(groups)
                if m in ("benign", "perm1", "perm2"):
                    boot = np.array([pooled({c: (rng.choice(a, len(a)), rng.choice(b, len(b))) for c, (a, b) in groups.items()}) for _ in range(1000)])
                    p = float(min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean())))
                    res[m] = dict(d=obs, ci=np.quantile(boot, [0.025, 0.975]).tolist(), p=p, categories=cats)
                else:
                    res[m] = dict(d=obs)
            out.append(res)
    ok = [r for r in out if r["benign"] is not None]
    ps = np.array([r["benign"]["p"] for r in ok])
    order = np.argsort(ps)
    q = ps[order] * len(ps) / (np.arange(len(ps)) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    for r, qv in zip([ok[i] for i in order], q):
        r["benign"]["q_bh"] = float(min(qv, 1.0))
        r["finding"] = bool(qv <= 0.05 and abs(r["benign"]["d"]) > max(abs(r["perm1"]["d"]), abs(r["perm2"]["d"])))
    return out


def checks(u, info):
    s = u[u["set"] == "strat"]
    h = u.dropna(subset=["rms_half_a", "lattice_even"])
    sp = lambda a, b: float(st.spearmanr(a, b).statistic)  # noqa: E731
    cos = info["coin"]["cosines"]
    c = {"gradient_odd_even_cos": cos["benign_odd|benign_even"], "lattice_halves": sp(h.lattice_even, h.lattice_odd),
         "lattice_halves_standard": sp(h.lattice_std_even, h.lattice_std_odd), "rms_halves": sp(h.rms_half_a, h.rms_half_b), "check_units": len(h),
         "t1_t2": sp(s.cos_t1, s.cos_t2), "perm1_perm2": sp(s.cos_perm1, s.cos_perm2), "benign_heads_bias": sp(s.cos_benign, s.cos_heads_bias),
         "load_coin_median": info["coin"]["load_median_abs_diff"], "load_heads_median": info["heads_bias"]["load_median_abs_diff"],
         "load_asst_median": info["asst_benign"]["load_median_abs_diff"], "f": {"standard": info["standard"]["f"], "heads_bias": info["heads_bias"]["f"],
                                                                             "asst_benign": info["asst_benign"]["f"], **info["coin"]["values"]},
         "query_cosines": {k: cos[k] for k in ("benign|t1", "benign|t2", "t1|t2", "benign|perm1", "benign|perm2", "perm1|perm2", "benign|ordinary", "benign|harm_gap")}}
    c["pass"] = {"1_gradient": c["gradient_odd_even_cos"] >= 0.5, "2_lattice": c["lattice_halves"] >= 0.8, "3_rms": c["rms_halves"] >= 0.7,
                 "4_specificity": c["t1_t2"] >= 0.5 and c["t1_t2"] - c["perm1_perm2"] >= 0.3, "5_placebo": abs(c["benign_heads_bias"]) < 0.8}
    return c


def shares(u, w):
    m = w > 0
    d = pd.DataFrame({"cat": u.category[m], "w": w[m], "cw": (u.contrib * w)[m], "tok": (u.n * w)[m]}).groupby("cat").sum()
    d["share_rows"], d["share_tokens"], d["share_net"] = d.w / d.w.sum(), d.tok / d.tok.sum(), d.cw / d.cw.sum()
    return d[["share_rows", "share_tokens", "share_net"]].reset_index().to_dict("records")


def yesterday(u):
    """../coinflip's standard-effect cos at the base (sft-a0) for the same units: replication of standard, and benign vs it."""
    p = YESTERDAY / "scores.jsonl"
    if not p.exists():
        return None
    gn = json.load(open(YESTERDAY / "queries.json"))["effect"]["grad_norm"]
    y = pd.DataFrame([dict(idx=r["idx"], cos_y=r["adam"]["effect"] / np.sqrt(r["g_adam_sq"]) / gn) for r in map(json.loads, open(p))]).drop_duplicates("idx").set_index("idx")
    j = u.join(y, how="inner")
    return dict(units=len(j), standard_today_vs_yesterday=float(st.spearmanr(j.cos_standard, j.cos_y).statistic),
                pearson=float(np.corrcoef(j.cos_standard, j.cos_y)[0, 1]), benign_vs_yesterday=float(st.spearmanr(j.cos_benign, j.cos_y).statistic))


# ---------------------------------------------------------------- post hoc (added after the base data; the README marks them)

NULLS = ("perm1", "perm2", "heads_bias")


def residualize(u, m):
    """cos_m minus its least-squares fit (on strat units) to the standardized cos of the no-preference queries (the two
    shuffled-utility nulls and the heads bias, leaving out m itself), keeping cos_m's mean: what's left once the structure the
    nulls share is removed. Returns the series and the R^2 of that fit."""
    s = (u["set"] == "strat").to_numpy()
    Z = np.column_stack([((u[f"cos_{k}"] - u.loc[s, f"cos_{k}"].mean()) / u.loc[s, f"cos_{k}"].std()).to_numpy() for k in NULLS if k != m])
    y = u[f"cos_{m}"].to_numpy()
    X = np.column_stack([np.ones(s.sum()), Z[s]])
    beta, *_ = np.linalg.lstsq(X, y[s], rcond=None)
    r2 = 1 - np.var(y[s] - X @ beta) / np.var(y[s])
    return pd.Series(y - Z @ beta[1:], index=u.index), float(r2)


def posthoc(u, w, lab, stats):
    out = {"null_r2": {}}
    for m in ("benign", "t1", "t2", "ordinary", "standard", "harm_gap", "asst_benign", "heads_bias"):
        u[f"cos_res_{m}"], out["null_r2"][m] = residualize(u, m)
    s = u[u["set"] == "strat"]
    sp = lambda a, b: float(st.spearmanr(s[a], s[b]).statistic)  # noqa: E731
    out["unit_corr_with_nulls"] = {f"{a}|{b}": sp(f"cos_{a}", f"cos_{b}") for a in ("benign", "ordinary", "standard", "asst_benign")
                                   for b in NULLS}
    out["residual_rank_corr"] = {"t1_t2": sp("cos_res_t1", "cos_res_t2"), "benign_ordinary": sp("cos_res_benign", "cos_res_ordinary"),
                                 "benign_standard": sp("cos_res_benign", "cos_res_standard"), "benign_asst": sp("cos_res_benign", "cos_res_asst_benign")}
    for m in ("benign", "heads_bias", "standard", "ordinary"):
        out[f"residual_{m}"] = rules(u, f"cos_res_{m}", w)
    if lab is not None:
        v = u.copy()
        v["cos_benign"] = u["cos_res_benign"]
        out["residual_labels"] = label_contrasts(v, lab, stats["category"])
    return out


def origin_decomposition():
    """The coin's slope on the SFT Assistant's utility over the non-harmful tasks, split into between-origin and within-origin
    parts, from the stored tournament utilities (no attribution scores)."""
    t = pd.read_json(ROOT / "data" / "tasks_n300_max500_seed0.jsonl", lines=True).set_index("id")
    b = ~((t.origin == "BAILBENCH") | (t.topic == "harmful_request"))
    A = pd.read_csv(ROOT / "outputs" / "as1txqy2" / "analysis" / "utilities.csv").set_index("id").loc[t.index, "sft/assistant"][b]
    a, org = (A - A.mean()) / A.std(), t.origin[b]
    am = a.groupby(org).transform("mean")
    out = {"assistant_variance_between_origins": float((am ** 2).sum() / (a ** 2).sum()),
           "assistant_origin_means_sd": a.groupby(org).mean().round(3).to_dict()}
    for stage, run in (("base", "5f7ymsdk"), ("sft", "as1txqy2")):
        C = pd.read_csv(ROOT / "outputs" / run / "analysis" / "utilities.csv").set_index("id").loc[t.index, f"{stage}/coinflip"][b]
        c = C - C.mean()
        cm = c.groupby(org).transform("mean")
        out[stage] = dict(total=float((a * c).sum() / (a * a).sum()), between=float((am * cm).sum() / (a * a).sum()),
                          within=float(((a - am) * (c - cm)).sum() / (a * a).sum()), coin_origin_means=c.groupby(org).mean().round(3).to_dict())
    return out


def analyse(point, stats, lab):
    u, info, R = load_point(point)
    w = unit_weights(u, stats)
    s = u[u["set"] == "strat"]
    res = {"units": len(u), "strat": int((u["set"] == "strat").sum()), "R": R, "checks": checks(u, info)}
    for m in ("benign", "ordinary", "harm_gap", "standard", "asst_benign", "heads_bias", "perm1", "perm2"):
        res[m] = rules(u, f"cos_{m}", w)
    res["pattern"] = pattern_specificity(u)
    res["refusal"] = refusal_test(u)
    res["selfref"] = selfref_test(u)
    sp = lambda a, b: float(st.spearmanr(s[a], s[b]).statistic)  # noqa: E731
    res["rank_corr"] = {"benign_standard": sp("cos_benign", "cos_standard"), "benign_asst": sp("cos_benign", "cos_asst_benign"),
                        "benign_harm_gap": sp("cos_benign", "cos_harm_gap"), "benign_ordinary": sp("cos_benign", "cos_ordinary"),
                        "standard_harm_gap": sp("cos_standard", "cos_harm_gap")}
    cm = {m: category_means(u, f"cos_{m}") for m in ("benign", "asst_benign", "standard")}
    res["category_corr"] = {"benign_asst": float(st.spearmanr(cm["benign"], cm["asst_benign"].reindex(cm["benign"].index)).statistic),
                            "benign_standard": float(st.spearmanr(cm["benign"], cm["standard"].reindex(cm["benign"].index)).statistic)}
    res["shares"] = shares(u, w)
    res["identity"] = dict(n=int((s.category == "identity").sum()), mean=float(s[s.category == "identity"].cos_benign.mean()),
                           sem=float(s[s.category == "identity"].cos_benign.sem()))
    if lab is not None:
        res["labels"] = label_contrasts(u, lab, stats["category"])
        res["label_coverage"] = dict(labelled=int(lab.loc[lab.index.intersection(u.index)].task_type.notna().sum()), units=len(u))
    if point == "base":
        res["yesterday"] = yesterday(u)
    res["posthoc"] = posthoc(u, w, lab, stats)
    order = s.sort_values("cos_benign")
    res["top"], res["bottom"] = order.index[::-1][:30].tolist(), order.index[:30].tolist()
    out = u.drop(columns=[c for c in u.columns if c.startswith(("raw_", "adam_"))])
    if lab is not None:
        out = out.join(lab, how="left")
    out.to_csv(DATA / f"units_{point}.csv")
    return res, u, w


# ---------------------------------------------------------------- figures

def figures(results, units):
    import matplotlib.pyplot as plt
    from matskit import plotstyle as ps
    ps.setup()
    pts = [p for p in POINTS if p in results]
    col = {"base": ps.SERIES[0], "sft": ps.SERIES[1]}
    name = {"base": "at the base (start of SFT)", "sft": "at the SFT checkpoint (end; fails the placebo check)"}

    # 1. category means, small multiples over queries, in SDs of each query's cos
    qs = [("benign", "primary: coin slope,\nnon-harmful tasks"), ("res_benign", "primary, with the\nnull-shared part\nremoved (post hoc)"),
          ("t1", "same, task half 1"), ("t2", "same, task half 2"), ("perm1", "shuffled utilities\n(null)"),
          ("heads_bias", "heads bias\n(placebo)"), ("standard", "../coinflip's\n25 pairs"), ("asst_benign", "the Assistant's\nown choice")]
    fig, axes = plt.subplots(1, len(qs), figsize=(18, 4.8), sharey=True)
    cats = CAT_ORDER
    for ax, (m, title) in zip(axes, qs):
        for j, p in enumerate(pts):
            if m.startswith("res_") and p == "sft":
                continue  # the nulls explain 96-99% of every metric at SFT; the residual is ~0 and blows up in SD units
            s = units[p][units[p]["set"] == "strat"]
            sd = s[f"cos_{m}"].std()
            g = s.groupby("category")[f"cos_{m}"].agg(["mean", "sem"]).reindex(cats)
            y = np.arange(len(cats)) + (j - 0.5) * 0.3
            ax.errorbar(g["mean"] / sd, y, xerr=1.96 * g["sem"] / sd, fmt="o", ms=5, color=col[p], elinewidth=1.5, capsize=0,
                        mfc=col[p] if p == "base" else ps.SURFACE, label=name[p] if m == "benign" else None)
        ax.axvline(0, color=ps.BASELINE, lw=1)
        ax.set_title(title, loc="left", fontsize=9)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    axes[0].set_yticks(np.arange(len(cats)), cats)
    axes[0].set_ylim(len(cats) - 0.5, -0.5)
    fig.supxlabel("mean alignment with the metric's gradient (Adam-weighted cos), in SDs over units; 95% CI", fontsize=9, color=ps.INK2)
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=2, fontsize=9, frameon=False)
    ps.save(fig, "olmo2_attr_categories")

    # 2. refusal / self-reference and the content labels, at each point
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.2), gridspec_kw=dict(width_ratios=[1, 1.4]), layout="constrained")
    ax = axes[0]
    rowsf = []
    for test, metrics in (("refusal", ["standard", "harm_gap", "benign", "ordinary"]), ("selfref", ["standard", "harm_gap", "benign", "perm1"])):
        for m in metrics:
            rowsf.append((test, m))
    for j, p in enumerate(pts):
        for i, (test, m) in enumerate(rowsf):
            t = results[p][test]
            d, (lo, hi) = t["d"][m], t["ci"][m]
            ax.errorbar(d, i + (j - 0.5) * 0.3, xerr=[[d - lo], [hi - d]], fmt="o", ms=5, color=col[p], elinewidth=1.5, capsize=0)
    ax.set_yticks(range(len(rowsf)), [f"{'refusing vs complying' if t == 'refusal' else 'self-reference vs same source'}: {m}" for t, m in rowsf], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color=ps.BASELINE, lw=1)
    ax.set_xlabel("group minus matched controls, in control SDs of cos (95% CI)")
    ax.set_title("Pre-specified groups", loc="left")
    for p in pts:
        ax.plot([], [], "o", color=col[p], label=name[p])
    ax.legend(loc="lower right", fontsize=8)
    ax = axes[1]
    if "labels" in results[pts[0]]:
        labs = [r for r in results[pts[0]]["labels"] if r["benign"] is not None]
        for j, p in enumerate(pts):
            lp = {(r["field"], r["level"]): r for r in results[p].get("labels", [])}
            for i, r in enumerate(labs):
                rr = lp.get((r["field"], r["level"]))
                if rr is None or rr["benign"] is None:
                    continue
                b = rr["benign"]
                ax.errorbar(b["d"], i + (j - 0.5) * 0.3, xerr=[[b["d"] - b["ci"][0]], [b["ci"][1] - b["d"]]], fmt="o", ms=5, color=col[p],
                            elinewidth=1.5, capsize=0, mfc=col[p] if rr.get("finding") else ps.SURFACE)
                for k in ("perm1", "perm2"):
                    ax.plot(rr[k]["d"], i + (j - 0.5) * 0.3, "|", color=ps.MUTED, ms=7)
        ax.set_yticks(range(len(labs)), [f"{r['field']} = {r['level']} (n={r['n']})" for r in labs], fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color=ps.BASELINE, lw=1)
        ax.set_xlabel("within-category difference on the primary metric, in SDs of cos (95% CI);\nfilled = survives BH and beats both nulls; ticks = same contrast on the shuffled-utility nulls")
        ax.set_title("Content labels (exploratory)", loc="left")
    ps.save(fig, "olmo2_attr_contrasts")

    # 3. unit-level agreement
    pairs = [("base", "cos_benign", "base", "cos_standard", "primary vs ../coinflip's 25 pairs, base"),
             ("base", "cos_benign", "base", "cos_asst_benign", "coin vs the Assistant's own choice, base"),
             ("base", "cos_t1", "base", "cos_t2", "task half 1 vs half 2, base")]
    if "sft" in units:
        pairs.append(("base", "cos_benign", "sft", "cos_benign", "primary at the base vs at SFT"))
    fig, axes = plt.subplots(1, len(pairs), figsize=(4 * len(pairs), 3.8))
    for ax, (pa, ca, pb, cb, title) in zip(np.atleast_1d(axes), pairs):
        a = units[pa][units[pa]["set"] == "strat"][ca]
        b = units[pb][units[pb]["set"] == "strat"][cb]
        j = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner")
        ax.scatter(j.a / j.a.std(), j.b / j.b.std(), s=6, alpha=0.35, color=ps.SERIES[0], linewidths=0)
        ax.axhline(0, color=ps.BASELINE, lw=0.8)
        ax.axvline(0, color=ps.BASELINE, lw=0.8)
        ax.set_title(f"{title}\nSpearman {st.spearmanr(j.a, j.b).statistic:.2f}", loc="left", fontsize=9)
        ax.set_xlabel(ca.removeprefix("cos_") + f" ({pa}), SDs", fontsize=8)
        ax.set_ylabel(cb.removeprefix("cos_") + f" ({pb}), SDs", fontsize=8)
    ps.save(fig, "olmo2_attr_units")


def main():
    stats = json.load(open(DATA / "sft_stats.json"))
    lab = load_labels()
    results, units = {}, {}
    for p in POINTS:
        if (OUT / p / "scores.jsonl").exists():
            results[p], units[p], _ = analyse(p, stats, lab)
            b = results[p]["benign"]
            print(f"{p}: {results[p]['units']} units; benign verdict {b['verdict']} (opposing {b['opposing_pushes']}), mean {b['weighted_mean']:.2e} "
                  f"CI {b['weighted_mean_ci']}, frac_pos {b['frac_pos']:.2f}, top1 {b['top1']:.2f}; checks {results[p]['checks']['pass']}; "
                  f"refusal {results[p]['refusal']['call']}", flush=True)
    if "base" in units and "sft" in units:
        a, b = units["base"], units["sft"]
        j = a[["cos_benign", "category", "set"]].join(b[["cos_benign"]].rename(columns={"cos_benign": "cos_benign_sft"}), how="inner")
        s = j[j["set"] == "strat"]
        ca, cb = category_means(a, "cos_benign"), category_means(b, "cos_benign")
        results["base_vs_sft"] = {"unit_spearman": float(st.spearmanr(s.cos_benign, s.cos_benign_sft).statistic), "units": len(s),
                                  "category_spearman": float(st.spearmanr(ca, cb.reindex(ca.index)).statistic)}
    if (ROOT / "outputs" / "as1txqy2" / "analysis" / "utilities.csv").exists():
        results["origin_decomposition"] = origin_decomposition()
    json.dump(results, open(DATA / "summary.json", "w"), indent=1, default=float)
    if results:
        figures(results, units)
    print("wrote", DATA / "summary.json")


if __name__ == "__main__":
    main()
