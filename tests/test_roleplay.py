"""roleplay.py on synthetic utilities: coins with known pulls give the right weights, calls and comparisons."""

import numpy as np

from personas import z
from report import stratified_resamples
from roleplay import K, PERSONAS, call, compare, stats
from user_choice import ci


def synthetic(pull_r, rp_noise=2.0, n=300, seed=0):
    """Each persona: the system coin pulls 0.3 toward the default and 0.7 toward its persona Assistant (w_s 0.7); the
    roleplay coin pulls (1 - pull_r[p]) and pull_r[p] toward the default and the roleplay Assistant (w_r = pull_r[p])."""
    rng = np.random.default_rng(seed)
    a0 = rng.normal(0, 8, n)
    V = {"A0": a0, "C0": z(a0) + rng.normal(0, 0.05, n)}
    for p in PERSONAS:
        a_s = rng.normal(0, 7, n)
        a_r = a_s + rng.normal(0, rp_noise, n)
        V |= {f"As_{p}": a_s, f"Ar_{p}": a_r, f"Cs_{p}": 0.3 * z(a0) + 0.7 * z(a_s) + rng.normal(0, 0.05, n),
              f"Cr_{p}": (1 - pull_r[p]) * z(a0) + pull_r[p] * z(a_r) + rng.normal(0, 0.05, n)}
    origins = np.array(["MATH", "ALPACA", "WILDCHAT", "STRESS_TEST", "BAILBENCH"] * (n // 5))
    return V, origins, origins != "BAILBENCH", rng.normal(5, 1, n)


def test_weights_calls_and_comparison():
    V, origins, benign, loglen = synthetic({"evil": 0.7, "slacker": 0.0})
    est = stats(V, benign, loglen, origins)
    assert abs(est[K["evil:w_r"]] - 0.7) < 0.02 and abs(est[K["evil:w_s"]] - 0.7) < 0.02
    assert abs(est[K["evil:dw"]]) < 0.03 and abs(est[K["slacker:w_r"]]) < 0.02
    assert abs(est[K["evil:w_r_ls"]] - 0.7) < 0.03 and abs(est[K["evil:w_r_nh"]] - 0.7) < 0.03
    assert est[K["evil:r_ar_as"]] > 0.9
    res = stratified_resamples(origins, np.random.default_rng(0), 200)
    boot = np.array([stats({k: v[i] for k, v in V.items()}, benign[i], loglen[i], origins[i]) for i in res])
    lo, hi = ci(boot)
    assert call(est, lo, hi, "evil") == "mix" and compare(est, lo, hi, "evil") == "as much as the system prompt"
    assert call(est, lo, hi, "slacker") == "default Assistant"
    assert compare(est, lo, hi, "slacker") == "less than the system prompt"


def test_roleplay_assistant_like_the_default_is_not_separable():
    V, origins, benign, loglen = synthetic({"evil": 1.0, "slacker": 0.5})
    V["Ar_evil"] = V["A0"] + np.random.default_rng(1).normal(0, 1, len(V["A0"]))
    est = stats(V, benign, loglen, origins)
    t = 0.02
    assert call(est, est - t, est + t, "evil") == "not separable"
    assert compare(est, est - t, est + t, "evil") == "no comparison"
    assert call(est, est - t, est + t, "slacker") == "mix"
    assert abs(est[K["slacker:rho"]] - 0.5 / 0.7) < 0.1  # the fixed-target ratio, roleplay Assistant = system's + noise
