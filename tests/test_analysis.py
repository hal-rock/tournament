"""Analysis on synthetic tournaments with known utilities and biases."""

import itertools

import numpy as np
import pytest

from analyze import analyze, fit_utilities, pair_effects, pearson, split_half


def synthetic_rows(ids, utils, slot_bias, noise, rng, stage="instruct", condition="assistant"):
    """Log-odds = slot_bias + u_first - u_second + noise, for both orders of every pair."""
    rows = []
    for (i, a), (j, b) in itertools.permutations(enumerate(ids), 2):
        lo = slot_bias + utils[i] - utils[j] + rng.normal(0, noise)
        rows.append(dict(stage=stage, condition=condition, first=a, second=b, log_odds=lo,
                         outcome_mass=0.9, logp_first=0.0, logp_second=0.0, prompt_tokens=0))
    return rows


@pytest.fixture
def setup():
    rng = np.random.default_rng(1)
    ids = [f"t{k:02d}" for k in range(30)]
    return rng, ids, rng.normal(0, 2, len(ids))


def test_recovers_utilities_and_bias(setup):
    rng, ids, u = setup
    rows = synthetic_rows(ids, u, slot_bias=1.5, noise=0.3, rng=rng)
    (eff,) = pair_effects(rows).values()
    assert len(eff) == len(ids) * (len(ids) - 1) // 2
    assert np.mean([e[3] for e in eff]) == pytest.approx(1.5, abs=0.05)
    assert pearson(fit_utilities(ids, eff), u) > 0.99


def test_round_robin_fit_is_mean_effect(setup):
    rng, ids, u = setup
    (eff,) = pair_effects(synthetic_rows(ids, u, 0.0, 1.0, rng)).values()
    means = {t: [] for t in ids}
    for a, b, e, _ in eff:
        means[a].append(e)
        means[b].append(-e)
    n = len(ids)
    expected = np.array([sum(means[t]) / n for t in ids])  # divide by n, not n-1: u has mean zero
    assert np.allclose(fit_utilities(ids, eff), expected - expected.mean())


def test_reliability_high_for_signal_low_for_noise(setup):
    rng, ids, u = setup
    (signal,) = pair_effects(synthetic_rows(ids, u, 0.0, 0.5, rng)).values()
    (noise,) = pair_effects(synthetic_rows(ids, np.zeros(len(ids)), 0.0, 0.5, rng)).values()
    assert split_half(ids, signal, rng)[1] > 0.95
    assert split_half(ids, noise, rng)[1] < 0.5


def test_analyze_end_to_end(setup, tmp_path):
    rng, ids, u = setup
    tasks = [dict(id=t, origin="MATH" if k % 2 else "ALPACA", topic="math", prompt="p") for k, t in enumerate(ids)]
    rows = (synthetic_rows(ids, u, 2.0, 0.3, rng, condition="assistant")
            + synthetic_rows(ids, 0.3 * u, -1.0, 0.3, rng, condition="coinflip")
            + synthetic_rows(ids, np.zeros(len(ids)), 0.5, 0.3, rng, stage="base", condition="coinflip"))
    result = analyze(rows, tasks, tmp_path)
    corr = {(c["x"], c["y"]): c for c in result["correlations"]}
    main = corr[("instruct/assistant", "instruct/coinflip")]
    assert main["pearson"] > 0.9
    assert main["sd_ratio_y_over_x"] == pytest.approx(0.3, abs=0.05)
    assert abs(corr[("base/coinflip", "instruct/assistant")]["pearson"]) < 0.5
    assert result["summary"]["instruct/coinflip/slot_bias_mean"] == pytest.approx(-1.0, abs=0.05)
    assert (tmp_path / "utilities.csv").exists()
    assert (tmp_path / "scatter_instruct-assistant_vs_instruct-coinflip.png").exists()
