"""retrain.py's per-checkpoint statistics and pre-registered outcome rule, on synthetic data."""

import numpy as np
import pytest

from report import stratified_resamples
from retrain import STATS, checkpoint_stats, classify, run_stats

K = {name: j for j, name in enumerate(STATS)}
N = 300


def tasks(rng):
    origins = np.array(["MATH", "ALPACA", "WILDCHAT", "BAILBENCH"] * (N // 4))
    benign = origins != "BAILBENCH"
    return origins, benign


def test_checkpoint_stats_known_values():
    rng = np.random.default_rng(0)
    _, benign = tasks(rng)
    a, c0 = rng.normal(size=N), rng.normal(size=N)
    c = c0 + 0.5 * a
    c[~benign] -= 2.0  # harmful tasks: lower coinflip utility
    s = checkpoint_stats(a, c, c0, a, benign)
    assert s[K["r_change_benign"]] == pytest.approx(1.0)
    assert s[K["b"]] == pytest.approx(0.5 * a[benign].std())
    assert s[K["r_benign"]] == pytest.approx(np.corrcoef(a[benign], c[benign])[0, 1])
    assert s[K["harm_gap"]] == pytest.approx(c[benign].mean() - c[~benign].mean())
    s0 = checkpoint_stats(a, c0, c0, a, benign)  # step 0: no change yet
    assert np.isnan(s0[K["r_change"]]) and np.isnan(s0[K["r_change_benign"]]) and s0[K["b"]] == 0


@pytest.mark.parametrize("gains,expected", [
    ({400: 0.2, 500: 0.4, 600: 0.6}, "A"),  # Assistant-aligned change keeps growing
    ({400: 0.4, 500: 0.4, 600: 0.4}, "B"),  # present but flat from 400 to 600
    ({400: 0.0, 500: 0.0, 600: 0.0}, "C"),  # never Assistant-aligned
])
def test_classify(gains, expected):
    rng = np.random.default_rng(1)
    origins, benign = tasks(rng)
    a, c0 = rng.normal(size=N), rng.normal(size=N)
    U = {"run5": {0: {"assistant": a, "coinflip": c0}}}
    for t, g in gains.items():
        U["run5"][t] = {"assistant": a, "coinflip": c0 + g * a + 0.05 * rng.normal(size=N)}
    resamples = stratified_resamples(origins, np.random.default_rng(0), 300)
    assert classify(run_stats(U, "run5", sorted(U["run5"]), benign, resamples)) == expected
