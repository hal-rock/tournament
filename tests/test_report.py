"""Report on synthetic utilities with a known common slope and known origin offsets."""

import numpy as np
import pytest

from report import (DECOMPOSITION_STATS, adjusted_r2, decomposition, dummies, nonassistant, offset_model, report,
                    stage_analysis, stages_of, stratified_resamples)

GROUPS = ["ALPACA", "BAILBENCH", "MATH"]
OFFSETS = {"ALPACA": 0.0, "BAILBENCH": 0.3, "MATH": -0.3}


@pytest.fixture
def synthetic():
    """Assistant utilities with very different origin means; coinflip = 0.05 * assistant
    + 0.1 * loglen + origin offset + noise."""
    rng = np.random.default_rng(0)
    origins = np.repeat(GROUPS, [60, 30, 60])
    a = rng.normal(0, 1, len(origins)) + np.select([origins == "BAILBENCH", origins == "MATH"], [-10.0, 4.0], 0.0)
    loglen = rng.normal(5, 0.5, len(origins))
    c = 0.05 * a + 0.1 * loglen + np.array([OFFSETS[o] for o in origins]) + rng.normal(0, 0.01, len(origins))
    return rng, origins, a, c, loglen


def test_offset_model_recovers_slope_and_offsets(synthetic):
    _, origins, a, c, loglen = synthetic
    beta, gamma, alpha, _ = offset_model(a, c, loglen, origins, GROUPS)
    assert beta == pytest.approx(0.05, abs=0.005)
    assert gamma == pytest.approx(0.1, abs=0.01)
    true = np.array([OFFSETS[g] for g in GROUPS])
    weights = np.array([(origins == g).mean() for g in GROUPS])
    assert np.allclose(alpha, true - weights @ true, atol=0.01)


def test_offsets_are_distance_from_common_slope_line(synthetic):
    rng, origins, a, c, loglen = synthetic
    benign = origins != "BAILBENCH"
    out = stage_analysis(a, c, loglen, origins, benign, GROUPS, stratified_resamples(origins, rng, 50))
    line = out["slope_within"][0] * out["x_group"] + out["line_intercept"]
    assert np.allclose(out["y_group"] - line, out["offset"][0])
    lo, hi = out["origin_slope"][1], out["origin_slope"][2]
    assert np.all((lo < 0.05) & (0.05 < hi))


def test_decomposition_recovers_carried_over_and_new_parts():
    """y = 1.5 * pre + 0.03 * assistant, with the predictors independent: each alone explains
    its own share of the variance, both together all of it."""
    rng = np.random.default_rng(2)
    pre, assistant = rng.normal(0, 0.1, 5000), rng.normal(0, 5, 5000)
    stats = dict(zip(DECOMPOSITION_STATS, decomposition(1.5 * pre + 0.03 * assistant, pre, assistant)))
    assert stats["coef_pre"] == pytest.approx(1.5)
    assert stats["coef_assistant"] == pytest.approx(0.03)
    assert stats["r2_both"] == pytest.approx(1.0)
    share_pre = 1.5 ** 2 * 0.1 ** 2 / (1.5 ** 2 * 0.1 ** 2 + 0.03 ** 2 * 5 ** 2)
    assert stats["r2_pre"] == pytest.approx(share_pre, abs=0.03)
    assert stats["r2_assistant"] == pytest.approx(1 - share_pre, abs=0.03)
    assert abs(stats["r_predictors"]) < 0.05


def test_nonassistant_is_uncorrelated_with_assistant(synthetic):
    _, _, a, c, _ = synthetic
    resid = nonassistant(c, a)
    assert abs(np.corrcoef(resid, a)[0, 1]) < 1e-8
    assert resid.mean() == pytest.approx(0, abs=1e-12)


def test_adjusted_r2_counts_parameters_by_rank(synthetic):
    rng, origins, _, _, _ = synthetic
    y = np.array([{"ALPACA": 0.0, "BAILBENCH": 1.0, "MATH": -1.0}[o] for o in origins]) + rng.normal(0, 1, len(origins))
    origin = dummies(origins)
    # The same dummies twice span nothing new, so they must not be penalized twice.
    assert adjusted_r2(y, [origin, origin]) == pytest.approx(adjusted_r2(y, [origin]))
    assert abs(adjusted_r2(rng.normal(0, 1, len(origins)), [origin])) < 0.05


def test_stages_skip_shifts():
    measures = ["base/assistant", "base/coinflip", "dpo/assistant", "dpo/coinflip",
                "dpo-base/assistant", "dpo-base/coinflip", "sft/assistant"]
    assert stages_of(measures) == ["base", "dpo"]


def test_report_writes_figures_and_tables(synthetic, tmp_path):
    rng, origins, a, c, loglen = synthetic
    rows = []
    for i, o in enumerate(origins):
        n_chars = float(np.exp(loglen[i]))
        rows.append({"id": f"t{i}", "origin": o, "topic": "harmful_request" if o == "BAILBENCH" else "math",
                     "n_chars": n_chars, "base/assistant": a[i] / 10, "base/coinflip": rng.normal(0, 0.05),
                     "sft/assistant": a[i] / 2, "sft/coinflip": rng.normal(0, 0.1),
                     "dpo/assistant": a[i], "dpo/coinflip": c[i], "dpo-base/assistant": 0.0, "prompt": "p"})
    result = report(rows, {}, tmp_path, n_boot=20)
    assert {p.name for p in result["figures"].values()} == {
        "bleedthrough_by_stage.png", "cross_stage_r.png", "scatter_by_stage.png", "task_types.png",
        "pre_stage_coinflip.png", "nonassistant_groups.png"}
    assert all(p.exists() for p in result["figures"].values())
    assert [r["stage"] for r in result["tables"]["stage_summary"]] == ["base", "sft", "dpo"]
    assert len(result["tables"]["residuals"]) == len(rows)
    assert [r["subset"] for r in result["tables"]["decomposition"]] == ["all", "benign"]
    partition = {r["stage"]: r for r in result["tables"]["nonassistant_partition"]}
    assert np.isnan(partition["base"]["alone_previous_coinflip"])
    assert partition["dpo"]["previous_stage"] == "sft"
    assert {r["kind"] for r in result["tables"]["nonassistant_groups"]} == {"origin", "topic"}
