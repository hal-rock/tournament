"""persona_placement.py on synthetic utilities: placements with known pulls give the right ratios and calls."""

import numpy as np

from persona_placement import K, call, persona_call, stats
from personas import z
from report import stratified_resamples
from user_choice import ci


def synthetic(per, n=300, seed=0):
    """per: {placement: pull toward the persona}; every coin also has a 0.3 pull toward the default."""
    rng = np.random.default_rng(seed)
    a0, ae = rng.normal(0, 8, n), rng.normal(0, 7, n)
    V = {"A0": a0, "Ae": ae, "C0": z(a0) + rng.normal(0, 0.05, n), "Auser": ae + rng.normal(0, 2, n),
         "Aself": a0 + rng.normal(0, 2, n)}
    for p, b in per.items():
        V[f"C{p}"] = 0.3 * z(a0) + b * z(ae) + rng.normal(0, 0.05, n)
        V[f"B{p}"] = 0.02 * z(ae) + rng.normal(0, 0.05, n)
    origins = np.array(["MATH", "ALPACA", "WILDCHAT", "STRESS_TEST", "BAILBENCH"] * (n // 5))
    return V, origins, origins != "BAILBENCH", rng.normal(5, 1, n)


def test_ratios_and_calls():
    V, origins, benign, loglen = synthetic({"sys": 1.0, "user": 0.5, "self": 0.0})
    est = stats(V, benign, loglen, origins)
    assert abs(est[K["rho_user"]] - 0.5) < 0.03 and abs(est[K["rho_self"]]) < 0.03
    assert abs(est[K["rho_self_minus_user"]] + 0.5) < 0.05
    res = stratified_resamples(origins, np.random.default_rng(0), 200)
    boot = np.array([stats({k: v[i] for k, v in V.items()}, benign[i], loglen[i], origins[i]) for i in res])
    lo, hi = ci(boot)
    assert call(est, lo, hi, "user") == "weaker"
    assert call(est, lo, hi, "self") == "absent"
    assert persona_call(est, lo, hi, "sys") == "mix" and persona_call(est, lo, hi, "self") == "default Assistant"
    V, origins, benign, loglen = synthetic({"sys": 1.0, "user": 1.0, "self": -0.5})
    est = stats(V, benign, loglen, origins)
    t = 0.02
    assert call(est, est - t, est + t, "user") == "as strong as the system prompt"
    assert call(est, est - t, est + t, "self") == "reversed"
