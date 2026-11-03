"""Pair selection for the tournament (no model needed)."""

import itertools

from analyze import analyze
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


def test_analyze_handles_a_run_with_one_measure(tmp_path):
    """A single (stage, condition) has no correlations to write (the user-choice base runs crashed here)."""
    tasks = [dict(id=f"t{i}", origin="MATH", topic="math", prompt="x" * (i + 1)) for i in range(4)]
    rows = []
    for a, b in itertools.permutations(range(4), 2):
        rows.append(dict(model="m", stage="base", condition="user_choice", first=f"t{a}", second=f"t{b}",
                         log_odds=float(a - b), logp_first=-1.0, logp_second=-1.0, outcome_mass=0.9, prompt_tokens=9.0))
    result = analyze(rows, tasks, tmp_path, seed=0, stage_order=["base"])
    assert result["correlations"] == [] and (tmp_path / "utilities.csv").exists()
    assert not (tmp_path / "correlations.csv").exists()
