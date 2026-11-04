"""languages.py on synthetic utilities: the statistics and the pre-registered call."""

import numpy as np

from languages import K, boot_cell, cell_stats, classify
from report import stratified_resamples


def synthetic(rng, n=300):
    origins = np.array(["A", "B", "C"] * (n // 3))
    benign = rng.random(n) > 0.2
    a_en_b, a_en_f = rng.normal(size=n), rng.normal(size=n) * 3
    c_en_b = rng.normal(size=n) * 0.1
    c_en_f = c_en_b + 0.3 * a_en_f + rng.normal(size=n) * 0.2
    return origins, benign, (a_en_b, a_en_f, c_en_b, c_en_f)


def test_english_against_itself_is_the_identity():
    rng = np.random.default_rng(0)
    origins, benign, en = synthetic(rng)
    s = cell_stats(*en, *en, benign, origins)
    for name in ("retention", "r_aa", "r_aa_base", "r_cc", "sd_ratio_a", "retention_cross"):
        assert abs(s[K[name]] - 1) < 1e-9, name
    assert s[K["r_final"]] == s[K["r_en"]] and s[K["r_cross"]] == s[K["r_final"]]


def test_calls():
    rng = np.random.default_rng(1)
    origins, benign, en = synthetic(rng)
    a_en_b, a_en_f, c_en_b, c_en_f = en
    resamples = stratified_resamples(origins, np.random.default_rng(0), 200)
    a_b, a_f = a_en_b + rng.normal(size=300) * 0.3, a_en_f + rng.normal(size=300) * 0.3
    c_b = rng.normal(size=300) * 0.1
    half = c_b + 0.15 * a_f + rng.normal(size=300) * 0.2      # half the English tilt
    none = c_b + rng.normal(size=300) * 0.2                   # no tilt
    flipped = c_b - 0.3 * a_f + rng.normal(size=300) * 0.2   # the English tilt, reversed
    call = lambda c_f: classify(boot_cell(en + (a_b, a_f, c_b, c_f), benign, origins, resamples))
    assert call(half) == "tracks"
    assert call(none) == "absent"
    assert call(flipped) == "reversed"
    cell = boot_cell(en + (a_b, a_f, c_b, half), benign, origins, resamples)
    assert 0.4 < cell["est"][K["retention"]] < 0.6 and cell["est"][K["r_aa"]] > 0.95
