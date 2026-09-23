"""Feature tests on synthetic tasks with planted within-origin effects."""

import numpy as np
import pytest

import features
from features import bh_q, epsilon_squared, run_tests


def test_bh_q_matches_hand_computation():
    assert np.allclose(bh_q([0.01, 0.04, 0.03, 0.5]), [0.04, 0.04 * 4 / 3, 0.04 * 4 / 3, 0.5])


def test_epsilon_squared_near_zero_without_effect():
    rng = np.random.default_rng(0)
    labels = np.repeat(["a", "b", "c"], 200)
    assert abs(epsilon_squared(rng.normal(0, 1, 600), labels)[0]) < 0.02
    y = rng.normal(0, 1, 600) + np.select([labels == "a", labels == "b"], [2.0, -2.0], 0.0)
    assert epsilon_squared(y, labels)[0] > 0.6


def test_run_tests_finds_planted_effects_only(monkeypatch):
    """MATH difficulty drives the outcome; WildChat type drives it; everything else is noise."""
    monkeypatch.setattr(features, "N_PERMUTATIONS", 500)
    rng = np.random.default_rng(1)
    tasks, y = [], []
    for k in range(60):
        level = 1 + k % 5
        tasks.append(dict(id=f"m{k}", origin="MATH", topic="math", prompt="x" * (20 + k % 7),
                          metadata=dict(level=level, type=["Algebra", "Geometry"][k % 2])))
        y.append(level + rng.normal(0, 0.5))
    for k in range(60):
        kind = ["creative", "coding", "knowledge_recall"][k % 3]
        tasks.append(dict(id=f"w{k}", origin="WILDCHAT", topic=["fiction", "coding"][k % 2],
                          prompt="x" * (20 + k % 11), metadata=dict(type=kind)))
        y.append({"creative": 2.0, "coding": 0.0, "knowledge_recall": -2.0}[kind] + rng.normal(0, 0.5))
    rows, levels = run_tests(tasks, {"y": np.array(y)}, rng, n_boot=100)
    got = {(r["origin"], r["feature"]): r for r in rows}
    assert got[("MATH", "difficulty level")]["effect"] > 0.8 and got[("MATH", "difficulty level")]["q"] < 0.05
    assert got[("WILDCHAT", "type")]["effect"] > 0.8 and got[("WILDCHAT", "type")]["q"] < 0.05
    assert got[("MATH", "subject")]["q"] > 0.05
    assert ("MATH", "topic") not in got  # a single level is not testable
    assert {l["level"] for l in levels if l["feature"] == "type"} == {"creative", "coding", "knowledge_recall"}
