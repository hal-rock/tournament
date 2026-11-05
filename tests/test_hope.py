"""hope.py on synthetic data: the calls come out as constructed, and the after condition's outcome terms are recovered."""

import itertools

import numpy as np

from analyze import pair_effects
from hope import COND, K, STATS, call, model_cell, model_stats, outcome_terms, reliabilities
from personas import z
from report import stratified_resamples


def synthetic(n=300, seed=0):
    """Generic task appeal g, shared by every voice, and an Assistant-specific part s that post-training adds to the
    Assistant, the coin and the user's choice. The hope before the flip copies the user's choice (so the choice account
    fails for it); the hope after the flip has only g (so it holds for it)."""
    rng = np.random.default_rng(seed)
    g, s = z(rng.normal(size=n)), z(rng.normal(size=n))
    e = lambda sd: rng.normal(0, sd, n)
    A = {"A_b": g + e(0.3), "A_f": 3 * g + 6 * s}
    final = {"C": 0.3 * s + e(0.05), "C0": 0.2 * s + e(0.05), "U": g + 1.5 * s + e(0.1), "U0": g + 1.4 * s + e(0.1)}
    final["H"] = final["U0"] + e(0.1)
    w = g + e(0.1)
    final.update(Wh=w + e(0.1), Wt=w + e(0.1))
    arrays = {**A, **{f"{k}_f": v for k, v in final.items()}, **{f"{k}_b": 0.3 * g + e(0.05) for k in final}}
    origins = np.array(["MATH", "ALPACA", "WILDCHAT", "STRESS_TEST", "BAILBENCH"] * (n // 5))
    return arrays, origins, origins != "BAILBENCH", rng.normal(5, 1, n)


def test_calls_follow_the_construction():
    arrays, origins, benign, loglen = synthetic()
    assert model_stats(arrays, benign, origins, loglen).shape == (len(STATS),)
    cell = model_cell(arrays, benign, origins, loglen, stratified_resamples(origins, np.random.default_rng(0), 200))
    est = cell["est"]
    assert abs(est[K["H:d_user"]]) < 0.03 and est[K["W:d_user"]] < -0.3 and est[K["W:d_coin"]] < -0.3
    assert call(cell, "H")[2] == "against" and call(cell, "H")[0] == "similar"
    assert call(cell, "W") == ("less", "less", "supports")
    # Before minus after is positive here, and r(Wh, Wt) is high (both are g plus noise).
    assert est[K["dr_hw"]] > 0.3 and est[K["r_whwt_final"]] > 0.95
    # The hope after carries nothing of A beyond g: its post-training change doesn't follow A's.
    assert est[K["W:r_change"]] < est[K["W:r_change_user"]] - 0.3


def raw_after(u, v, c, h, ids, noise=0.0, seed=0):
    """Log-odds rows of hope_after_heads and _tails under hoped(selected over rejected) = a_s - b_r + c, with
    a = u + v/2, b = u - v/2, plus a heads bias h."""
    rng = np.random.default_rng(seed)
    a, b = u + v / 2, u - v / 2
    rows = []
    for i, j in itertools.permutations(range(len(ids)), 2):  # i listed first (heads), j second (tails)
        heads = a[i] - b[j] + c + h  # came up heads: i selected; the read is "hoped for heads (i)"
        tails = -(a[j] - b[i] + c) + h  # came up tails: j selected; "hoped for heads (i)" = -hoped(j over i)
        for cond, lo in ((COND["Wh"], heads), (COND["Wt"], tails)):
            rows.append(dict(stage="instruct", condition=cond, first=ids[i], second=ids[j],
                             log_odds=lo + rng.normal(0, noise)))
    return rows


def test_outcome_terms_recover_c_and_v():
    n = 30
    rng = np.random.default_rng(1)
    ids = [f"t{k:02d}" for k in range(n)]
    u, v = rng.normal(0, 2, n), rng.normal(0, 1, n)
    v -= v.mean()
    eff = pair_effects(raw_after(u, v, c=-1.5, h=0.7, ids=ids))
    heads, tails = eff["instruct", COND["Wh"]], eff["instruct", COND["Wt"]]
    # Both outcomes measure the same task utilities, with the heads bias and c cancelled by the order swap.
    idx = {t: k for k, t in enumerate(ids)}
    du = np.array([u[idx[x]] - u[idx[y]] for x, y in heads["pairs"]])
    assert np.allclose(heads["effect"], du) and np.allclose(tails["effect"], du)
    c, v_hat, share = outcome_terms(heads, tails, ids)
    assert abs(c + 1.5) < 1e-9 and np.allclose(v_hat, v) and 0 <= share <= 1


def test_w_reliability_beats_each_outcome():
    """W, the mean of the two outcomes, is less noisy than either alone."""
    n = 40
    rng = np.random.default_rng(2)
    ids = [f"t{k:02d}" for k in range(n)]
    u = rng.normal(0, 1, n)
    rows = []
    for cond in COND.values():
        for i, j in itertools.permutations(range(n), 2):
            rows.append(dict(stage="instruct", condition=cond, first=ids[i], second=ids[j],
                             log_odds=u[i] - u[j] + rng.normal(0, 2.0)))
    rel = reliabilities(rows, ids, stages=("instruct",))
    assert rel["W_f"] > max(rel["Wh_f"], rel["Wt_f"])
