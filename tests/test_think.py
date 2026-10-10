"""think.py's comparison table on synthetic utilities with known bleedthrough."""

import numpy as np
import pytest

from report import stratified_resamples
from think import COMPARISONS, comparison_rows

ORIGINS = np.array(["ALPACA"] * 40 + ["MATH"] * 40 + ["BAILBENCH"] * 20)


def test_comparison_rows():
    rng = np.random.default_rng(0)
    n = len(ORIGINS)
    prior, asst, think = rng.normal(0, 1, n), rng.normal(0, 3, n), rng.normal(0, 3, n)
    noise = lambda: rng.normal(0, 0.01, n)  # noqa: E731
    U = {"base/assistant": prior, "base/coinflip": 0.1 * prior + noise(),
         "instruct/assistant": prior + asst, "instruct/coinflip": 0.1 * prior + 0.05 * asst + noise(),
         # Think: its own Assistant preferences, none of them in its coinflip.
         "think/assistant": prior + think, "think-inreasoning/assistant": prior + 0.5 * think,
         "think/coinflip": 0.1 * prior + noise()}
    benign = ORIGINS != "BAILBENCH"
    loglen = np.log(100 + np.arange(n))
    rows = {r["comparison"]: r for r in comparison_rows(U, ORIGINS, benign, loglen, {},
                                                        stratified_resamples(ORIGINS, rng, 50))}
    assert list(rows) == [c[0] for c in COMPARISONS]
    assert rows["instruct"]["slope_within"] == pytest.approx(0.05, abs=0.01)
    assert rows["instruct"]["r_shift"] > 0.95
    assert abs(rows["think"]["r_shift"]) < 0.3 and abs(rows["think, Assistant in reasoning"]["r_shift"]) < 0.3
    assert np.isnan(rows["base"]["r_shift"])
    assert rows["think"]["coinflip_utility_sd"] == pytest.approx(U["think/coinflip"].std())
