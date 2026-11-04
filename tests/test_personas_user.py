"""personas_user.py on synthetic utilities: the fit recovers a known mix in the user's change, and the reference fit doesn't."""

import numpy as np

from personas import z
from personas_user import K, call, stats
from report import stratified_resamples
from user_choice import ci


def synthetic(w_true, n=300, seed=0):
    rng = np.random.default_rng(seed)
    a0, ae = rng.normal(0, 8, n), rng.normal(0, 7, n)
    ub, ueb = rng.normal(0, 0.1, n), rng.normal(0, 0.1, n)
    V = {"A0": a0, "Ae": ae, "Ce": 0.4 * z(a0) + 0.6 * z(ae) + rng.normal(0, 0.05, n),
         "Ub": ub, "Uf": ub + 2 * z(a0) + rng.normal(0, 0.1, n), "Ueb": ueb,
         "Uef": ueb + 2 * ((1 - w_true) * z(a0) + w_true * z(ae)) + rng.normal(0, 0.1, n)}
    origins = np.array(["MATH", "ALPACA", "WILDCHAT", "STRESS_TEST", "BAILBENCH"] * (n // 5))
    return V, origins, origins != "BAILBENCH", rng.normal(5, 1, n)


def test_primary_recovers_the_mix_and_the_reference_does_not():
    V, origins, benign, loglen = synthetic(0.7)
    est = stats(V, benign, loglen, origins)
    assert abs(est[K["w"]] - 0.7) < 0.03 and abs(est[K["w0"]]) < 0.03 and abs(est[K["dw"]] - 0.7) < 0.05
    assert abs(est[K["cw"]] - 0.6) < 0.03 and abs(est[K["w_ls"]] - 0.7) < 0.05
    res = stratified_resamples(origins, np.random.default_rng(0), 200)
    boot = np.array([stats({k: v[i] for k, v in V.items()}, benign[i], loglen[i], origins[i]) for i in res])
    lo, hi = ci(boot)
    assert call(est, lo, hi) == "mix"
    V, origins, benign, loglen = synthetic(1.0)
    est = stats(V, benign, loglen, origins)
    assert call(est, est - 0.02, est + 0.02) == "persona"
