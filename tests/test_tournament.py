"""Pair selection for the tournament (no model needed)."""

import itertools

from tournament import select_pairs


def test_select_pairs_all_by_default():
    assert select_pairs(20) == list(itertools.combinations(range(20), 2))
    assert select_pairs(20, max_pairs=10_000) == select_pairs(20)


def test_select_pairs_subsample_is_fixed_by_seed():
    sub = select_pairs(300, max_pairs=15_000, seed=0)
    assert len(sub) == len(set(sub)) == 15_000
    assert sub == sorted(sub) and all(a < b for a, b in sub)
    assert sub == select_pairs(300, max_pairs=15_000, seed=0)  # same pairs in every run
    assert sub != select_pairs(300, max_pairs=15_000, seed=1)
    assert {t for p in sub for t in p} == set(range(300))  # every task still compared
