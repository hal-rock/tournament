import numpy as np

from analyze import fit, pearson
from lens_analysis import colcorr, fit_depths


def test_fit_depths_matches_fit_per_depth():
    rng = np.random.default_rng(0)
    n, ia, ib = 12, *np.array([(a, b) for a in range(12) for b in range(a + 1, 12)]).T
    effect = rng.normal(size=(len(ia), 5))
    u = fit_depths(n, ia, ib, effect)
    for d in range(5):
        assert np.allclose(u[:, d], fit(n, ia, ib, effect[:, d]))


def test_colcorr():
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=(30, 4)), rng.normal(size=(30, 4))
    assert np.allclose(colcorr(x, y), [pearson(x[:, k], y[:, k]) for k in range(4)])
    assert np.allclose(colcorr(x, y[:, 0]), [pearson(x[:, k], y[:, 0]) for k in range(4)])
    assert np.allclose(colcorr(x[:, 0], y[:, 0]), pearson(x[:, 0], y[:, 0]))
