"""Analysis on synthetic tournaments with known utilities and biases."""

import itertools

import numpy as np
import pytest

from analyze import (additive_r2, analyze, define_measures, fit_utilities, pair_effects, partial_r, pearson,
                     split_half)


def synthetic_rows(ids, utils, slot_bias, noise, rng, stage="instruct", condition="assistant"):
    """Log-odds = slot_bias + u_first - u_second + noise, for both orders of every pair."""
    rows = []
    for (i, a), (j, b) in itertools.permutations(enumerate(ids), 2):
        lo = slot_bias + utils[i] - utils[j] + rng.normal(0, noise)
        rows.append(dict(stage=stage, condition=condition, first=a, second=b, log_odds=lo,
                         outcome_mass=0.9, logp_first=0.0, logp_second=0.0, prompt_tokens=0))
    return rows


def make_tasks(ids, harmful=()):
    return [dict(id=t, origin="BAILBENCH" if t in harmful else ("MATH" if k % 2 else "ALPACA"),
                 topic="harmful_request" if t in harmful else "math", prompt="p" * (k + 1)) for k, t in enumerate(ids)]


@pytest.fixture
def setup():
    rng = np.random.default_rng(1)
    ids = [f"t{k:02d}" for k in range(30)]
    return rng, ids, rng.normal(0, 2, len(ids))


def test_recovers_utilities_and_bias(setup):
    rng, ids, u = setup
    rows = synthetic_rows(ids, u, slot_bias=1.5, noise=0.3, rng=rng)
    (eff,) = pair_effects(rows).values()
    assert len(eff["pairs"]) == len(ids) * (len(ids) - 1) // 2
    assert eff["bias"].mean() == pytest.approx(1.5, abs=0.05)
    assert pearson(fit_utilities(ids, eff), u) > 0.99


def test_round_robin_fit_is_mean_effect(setup):
    rng, ids, u = setup
    (eff,) = pair_effects(synthetic_rows(ids, u, 0.0, 1.0, rng)).values()
    sums = {t: 0.0 for t in ids}
    for (a, b), e in zip(eff["pairs"], eff["effect"]):
        sums[a] += e
        sums[b] -= e
    expected = np.array([sums[t] / len(ids) for t in ids])  # divide by n, not n-1: u has mean zero
    assert np.allclose(fit_utilities(ids, eff), expected - expected.mean())


def test_reliability_high_for_signal_low_for_noise(setup):
    rng, ids, u = setup
    effects = {**pair_effects(synthetic_rows(ids, u, 0.0, 0.5, rng, condition="signal")),
               **pair_effects(synthetic_rows(ids, np.zeros(len(ids)), 0.0, 0.5, rng, condition="noise"))}
    rel = split_half(ids, effects, define_measures(list(effects)), rng)
    assert rel["instruct/signal"][1] > 0.95
    assert rel["instruct/noise"][1] < 0.5


def test_additive_r2_detects_pair_specific_effects(setup):
    rng, ids, u = setup
    (clean,) = pair_effects(synthetic_rows(ids, u, 0.0, 0.1, rng)).values()
    assert additive_r2(ids, clean, fit_utilities(ids, clean)) > 0.99
    # Pure pair-specific effects: no single scale explains them.
    (messy,) = pair_effects(synthetic_rows(ids, np.zeros(len(ids)), 0.0, 2.0, rng)).values()
    assert additive_r2(ids, messy, fit_utilities(ids, messy)) < 0.2


def test_define_measures_adds_shifts_against_base():
    keys = [("instruct", "assistant"), ("instruct", "coinflip"), ("base", "coinflip")]
    m = define_measures(keys)
    assert m["instruct-base/coinflip"] == {("instruct", "coinflip"): 1.0, ("base", "coinflip"): -1.0}
    assert "instruct-base/assistant" not in m  # no base/assistant to subtract


def test_analyze_end_to_end(setup, tmp_path):
    rng, ids, _ = setup
    # A pretraining prior v shared by both roles in both models; post-training adds
    # w to the Assistant and an unrelated z to the coin. Raw instruct utilities then
    # correlate through v, but the post-training shifts (w vs 0.3 z) do not.
    v, w, z = (rng.normal(0, 2, len(ids)) for _ in range(3))
    rows = (synthetic_rows(ids, v + w, 2.0, 0.2, rng, "instruct", "assistant")
            + synthetic_rows(ids, 0.5 * v + 0.3 * z, -1.0, 0.2, rng, "instruct", "coinflip")
            + synthetic_rows(ids, v, 0.5, 0.2, rng, "base", "assistant")
            + synthetic_rows(ids, 0.5 * v, 0.5, 0.2, rng, "base", "coinflip"))
    result = analyze(rows, make_tasks(ids), tmp_path)
    corr = {(c["x"], c["y"]): c for c in result["correlations"]}
    assert corr[("instruct/assistant", "instruct/coinflip")]["pearson"] > 0.4
    assert abs(corr[("instruct-base/assistant", "instruct-base/coinflip")]["pearson"]) < 0.4
    assert ("instruct/assistant", "instruct-base/assistant") not in corr  # shift vs own component
    assert result["summary"]["instruct/coinflip/slot_bias_mean"] == pytest.approx(-1.0, abs=0.05)
    assert result["summary"]["instruct-base/coinflip/reliability"] > 0.9
    assert (tmp_path / "utilities.csv").exists()
    assert (tmp_path / "scatter_instruct-base-assistant_vs_instruct-base-coinflip.png").exists()


def test_benign_correlation_ignores_harmful_cluster(setup, tmp_path):
    rng, ids, _ = setup
    harmful = set(ids[:5])
    # Harmful tasks sit far below in both conditions; among the rest the two are unrelated.
    shared = np.array([-10.0 if t in harmful else 0.0 for t in ids])
    rows = (synthetic_rows(ids, shared + rng.normal(0, 1, len(ids)), 0.0, 0.1, rng, condition="assistant")
            + synthetic_rows(ids, shared + rng.normal(0, 1, len(ids)), 0.0, 0.1, rng, condition="coinflip"))
    (c,) = analyze(rows, make_tasks(ids, harmful), tmp_path)["correlations"]
    assert c["pearson"] > 0.8
    assert c["n_benign"] == 25
    assert abs(c["pearson_benign"]) < 0.5


def test_fit_handles_incomplete_designs(setup):
    rng, ids, u = setup
    (eff,) = pair_effects(synthetic_rows(ids, u, 0.0, 0.1, rng)).values()
    keep = rng.random(len(eff["pairs"])) < 0.3
    sub = dict(pairs=[p for p, k in zip(eff["pairs"], keep) if k], effect=eff["effect"][keep])
    assert pearson(fit_utilities(ids, sub), u) > 0.99


def test_partial_r_removes_shared_confound():
    rng = np.random.default_rng(0)
    z = rng.normal(size=500)
    x, y = z + 0.3 * rng.normal(size=500), z + 0.3 * rng.normal(size=500)
    assert pearson(x, y) > 0.8
    assert abs(partial_r(x, y, z)) < 0.15


def test_read_run_config_both_formats(tmp_path):
    """run_config.yaml as tournament.py writes it, and an older run's config.yaml that W&B
    overwrote with its own {key: {"value": ...}} dump."""
    from analyze import read_run_config
    new, old = tmp_path / "new", tmp_path / "old"
    new.mkdir(), old.mkdir()
    (new / "run_config.yaml").write_text("tasks_file: data/t.jsonl\nmodels: [{name: m, stage: s}]\n")
    (old / "config.yaml").write_text("_wandb:\n  value: {cli_version: 0.30.0}\ntasks_file:\n  value: data/t.jsonl\n"
                                     "models:\n  value: [{name: m, stage: s}]\n")
    for d in (new, old):
        assert read_run_config(d) == {"tasks_file": "data/t.jsonl", "models": [{"name": "m", "stage": "s"}]}
