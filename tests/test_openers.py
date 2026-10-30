import numpy as np

from openers import K, OPENERS, across_cells, cell_stats, classify, model_cells, orientation, retention
from report import stratified_resamples


def fixture(seed=0, n=300):
    """Utilities with an Assistant-aligned part and an unrelated part, over two origins."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 3, n)
    other = rng.normal(0, 1, n)
    origins = np.array(["X", "Y"] * (n // 2))
    benign = rng.random(n) > 0.2
    return a, other, benign, origins


def cells_for(C, a, benign, origins):
    return model_cells(a, C, benign, origins, stratified_resamples(origins, np.random.default_rng(1), 200))


def test_cell_stats_known_values():
    a, other, benign, origins = fixture()
    c = 0.5 * a  # a perfect line
    s = cell_stats(a, c, c, benign, origins)
    for name in ("r", "r_benign", "r_harmful", "r_within", "r_okay", "scale_okay"):
        assert s[K[name]] == np.float64(1) or abs(s[K[name]] - 1) < 1e-9
    assert abs(s[K["b"]] - 0.5 * a.std()) < 1e-9
    assert abs(s[K["resid_sd"]]) < 1e-9
    assert abs(s[K["coinflip_sd"]] - 0.5 * a.std()) < 1e-9
    half = cell_stats(a, 0.5 * c, c, benign, origins)
    assert abs(half[K["scale_okay"]] - 0.5) < 1e-9 and abs(half[K["r_okay"]] - 1) < 1e-9


def test_harm_gap_is_benign_minus_harmful():
    a, other, benign, origins = fixture()
    c = np.where(benign, 1.0, -1.0) + 1e-6 * other
    assert abs(cell_stats(a, c, c, benign, origins)[K["harm_gap"]] - 2) < 1e-3


def test_within_and_between_origin_parts_are_separate():
    a, other, benign, origins = fixture()
    a = a + np.where(origins == "X", 10, -10)
    c = other + np.where(origins == "X", 5, -5)  # shares only the origin means with a
    s = cell_stats(a, c, c, benign, origins)
    assert s[K["r"]] > 0.7 and abs(s[K["r_within"]]) < 0.2


def test_classify_calls():
    a, other, benign, origins = fixture()
    okay = a + 2 * other
    C = {"okay": okay, "weaker_r": 0.1 * a + 2 * other, "scaled": 0.5 * okay, "same": okay + 1e-3 * other,
         "stronger_r": a + 0.5 * other}
    cells = cells_for(C, a, benign, origins)
    assert classify(cells, "weaker_r") == "r falls"
    assert classify(cells, "scaled") == "scale only"
    assert classify(cells, "same") == "unchanged"
    assert classify(cells, "stronger_r") == "r rises"
    assert abs(retention(cells, "scaled")[0][K["b"]] - 0.5) < 1e-9


def test_across_cells_positive_when_tracking_retention():
    a, other, benign, origins = fixture()
    # Retention in ../coinflip, for OLMo's table: Alright > OK > none > So.
    model = "allenai/Olmo-3.1-32B-Instruct"
    from openers import COINFLIP_EFFECT
    keep = {o: COINFLIP_EFFECT[model][o] / COINFLIP_EFFECT[model]["okay"] for o in OPENERS}
    C = {o: keep[o] * a + other for o in OPENERS}
    rows = {r["y"]: r for r in across_cells({model: cells_for(C, a, benign, origins)}, [model])}
    assert rows["change in r"]["spearman"] == 1 and rows["retention of b"]["spearman"] == 1


def test_orientation_sides():
    rows = [dict(first="b1", second="h1", condition="c", log_odds=4.0),
            dict(first="h1", second="b1", condition="c", log_odds=-1.0),
            dict(first="b1", second="b2", condition="c", log_odds=9.0)]  # same type: left out
    assert orientation(rows, {"b1", "b2"}) == {("c", "heads"): 4.0, ("c", "tails"): -1.0}
