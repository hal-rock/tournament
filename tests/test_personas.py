"""personas.py on synthetic utilities: the two-pull fit recovers a known mix, and the calls follow it."""

import numpy as np

import personas
from personas import K, PERSONAS, call, stats
from report import stratified_resamples
from user_choice import ci


def synthetic(w_true, n=300, b=1.0, noise=0.05, seed=0):
    """A_0 and each persona's A_p (correlated about 0 with A_0); each C_p a mix of z(A_0) and z(A_p) with weight w_true[p]."""
    rng = np.random.default_rng(seed)
    a0 = rng.normal(0, 5, n)
    U = {"A0": a0, "C0": b * personas.z(a0) + rng.normal(0, noise, n),
         "Ahelpful": a0 + rng.normal(0, 0.5, n)}
    U["Chelpful"] = b * personas.z(U["Ahelpful"]) + rng.normal(0, noise, n)
    for p in PERSONAS:
        ap = rng.normal(0, 3, n)
        U[f"A{p}"] = ap
        U[f"C{p}"] = b * ((1 - w_true[p]) * personas.z(a0) + w_true[p] * personas.z(ap)) + rng.normal(0, noise, n)
    origins = np.array(["MATH", "ALPACA", "WILDCHAT", "STRESS_TEST", "BAILBENCH"] * (n // 5))
    benign = origins != "BAILBENCH"
    loglen = rng.normal(5, 1, n)
    return U, origins, benign, loglen


def test_two_pulls_recover_the_mix_and_calls_follow():
    w_true = {"evil": 0.0, "slacker": 0.6}
    U, origins, benign, loglen = synthetic(w_true)
    est = stats(U, benign, loglen, origins == "MATH")
    for p, w in w_true.items():
        assert abs(est[K[f"{p}:w"]] - w) < 0.03
        assert abs(est[K[f"{p}:total"]] - 1.0) < 0.05
    assert abs(est[K["b0"]] - 1.0) < 0.05
    resamples = stratified_resamples(origins, np.random.default_rng(0), 200)
    boot = np.array([stats({k: v[i] for k, v in U.items()}, benign[i], loglen[i], (origins == "MATH")[i])
                     for i in resamples])
    lo, hi = ci(boot)
    assert call(est, lo, hi, "evil") == "default Assistant"
    assert call(est, lo, hi, "slacker") == "mix"
    assert hi[K["w_evil_minus_slacker"]] < 0


def test_calls_persona_against_and_unseparable():
    U, origins, benign, loglen = synthetic({"evil": 1.0, "slacker": -0.5})
    est = stats(U, benign, loglen, origins == "MATH")
    tight = 0.02
    lo, hi = est - tight, est + tight
    assert call(est, lo, hi, "evil") == "persona"
    assert call(est, lo, hi, "slacker") == "against the persona"
    U["Aevil"] = U["A0"] + np.random.default_rng(1).normal(0, 0.1, len(U["A0"]))
    est = stats(U, benign, loglen, origins == "MATH")
    assert call(est, est - tight, est + tight, "evil") == "not separable"
