import numpy as np

from report import stratified_resamples
from user_choice import K, classify, contrast, model_cell, model_stats


def fixture(seed=0, n=300):
    rng = np.random.default_rng(seed)
    origins = np.array(["X", "Y", "Z"] * (n // 3))
    benign = rng.random(n) > 0.2
    # Base and final Assistant share a part, and the final adds its own.
    shared, own = rng.normal(0, 1, n), rng.normal(0, 3, n)
    ab, af = shared + 0.1 * rng.normal(0, 1, n), shared + own
    return rng, ab, af, benign, origins


def cell_of(arrays, benign, origins):
    return model_cell(arrays, benign, origins, stratified_resamples(origins, np.random.default_rng(1), 200))


def test_model_stats_known_values():
    rng, ab, af, benign, origins = fixture()
    ub, uf, cb, cf = 0.5 * ab, 0.25 * af, rng.normal(0, 1, len(ab)), rng.normal(0, 1, len(ab))
    s = model_stats(ab, af, ub, uf, cb, cf, benign, origins)
    for name in ("r_base", "r_final", "r_base_benign", "r_final_benign", "r_base_within", "r_final_within"):
        assert abs(s[K[name]] - 1) < 1e-9
    assert abs(s[K["b_base"]] - 0.5 * ab.std()) < 1e-9 and abs(s[K["b_final"]] - 0.25 * af.std()) < 1e-9
    assert abs(s[K["resid_base"]]) < 1e-9 and abs(s[K["resid_final"]]) < 1e-9
    assert abs(s[K["r_fixed"]] - np.corrcoef(af, ab)[0, 1]) < 1e-9  # the base user against the final Assistant


def test_gap_is_benign_minus_harmful():
    rng, ab, af, benign, origins = fixture()
    u = np.where(benign, 1.0, -1.0) + 1e-6 * rng.normal(0, 1, len(ab))
    s = model_stats(ab, af, u, 2 * u, u, u, benign, origins)
    assert abs(s[K["gap_base"]] - 2) < 1e-3 and abs(s[K["gap_final"]] - 4) < 1e-3


def test_classify_outcomes():
    rng, ab, af, benign, origins = fixture()
    n = len(ab)
    noise = lambda sd: sd * rng.normal(0, 1, n)
    cb, cf = noise(1), noise(1)
    # The user starts on the base Assistant's scale and moves to the final Assistant's.
    more = cell_of((ab, af, ab + noise(0.3), af + noise(0.3), cb, cf), benign, origins)
    assert classify(more) == "more Assistant-like" and contrast(more, "target")[1] > 0
    # The user starts already at the final Assistant and moves away from it.
    away = -0.3 * (af - ab) + noise(0.3)
    less = cell_of((ab, af, af + noise(0.3), af + 3 * away, cb, cf), benign, origins)
    assert classify(less) == "less Assistant-like"
    # The user is as close to the final Assistant before and after, and its change follows the Assistant's.
    same = cell_of((ab, af, (af - ab) + noise(0.5), 2 * (af - ab) + noise(1), cb, cf), benign, origins)
    assert classify(same) == "aligned shift only"
    # Nothing changes and nothing lines up.
    u = noise(1)
    nothing = cell_of((ab, af, u, u + noise(0.1), cb, cf), benign, origins)
    assert classify(nothing) == "no change"
