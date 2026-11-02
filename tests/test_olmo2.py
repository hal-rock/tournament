import numpy as np

from olmo2 import K, STAGES, classify, ladder_cells, ladder_diff, share_diff, sft_share, stage_stats, step
from report import stratified_resamples


def fixture(seed=0, n=300):
    rng = np.random.default_rng(seed)
    origins = np.array(["X", "Y", "Z"] * (n // 3))
    benign = rng.random(n) > 0.2
    return rng, rng.normal(0, 3, n), origins, benign, np.log(rng.integers(20, 500, n))


def ladder(rng, a, benign, align, gap, noise=0.3):
    """Utilities per stage: the coinflip is `align`[stage] x (A + noise) + gap[stage] x (benign contrast)."""
    other = rng.normal(0, 1, len(a))
    U = {}
    for s in STAGES:
        U[f"{s}/assistant"] = a + 0.01 * rng.normal(0, 1, len(a))
        U[f"{s}/coinflip"] = align[s] * a + gap[s] * np.where(benign, 0.5, -0.5) + noise * other
    return U


def cells_of(U, benign, origins, loglen):
    return ladder_cells(U, benign, origins, loglen, stratified_resamples(origins, np.random.default_rng(1), 200))


def test_stage_stats_known_values():
    rng, a, origins, benign, loglen = fixture()
    c0 = rng.normal(0, 1, len(a))
    a0 = rng.normal(0, 1, len(a))
    s = stage_stats(a, 0.5 * a, a0, c0, benign, origins, loglen)
    for name in ("r", "r_benign", "r_harmful", "r_within", "r_between", "r_length"):
        assert abs(s[K[name]] - 1) < 1e-9
    assert abs(s[K["b"]] - 0.5 * a.std()) < 1e-9
    assert abs(s[K["resid_sd"]]) < 1e-9
    assert not np.isnan(s[K["r_shift"]])
    assert np.isnan(stage_stats(a, c0, a0, c0, benign, origins, loglen)[K["r_shift"]])  # base has no shift


def test_harm_gap_is_benign_minus_harmful():
    rng, a, origins, benign, loglen = fixture()
    c = np.where(benign, 1.0, -1.0) + 1e-6 * rng.normal(0, 1, len(a))
    s = stage_stats(a, c, a, c, benign, origins, loglen)
    assert abs(s[K["harm_gap"]] - 2) < 1e-3


def test_sft_share_and_steps():
    rng, a, origins, benign, loglen = fixture()
    U = ladder(rng, a, benign, dict(base=0, sft=0, dpo=0, instruct=0), dict(base=0, sft=1, dpo=3, instruct=4), 0.01)
    cells = cells_of(U, benign, origins, loglen)
    est, lo, hi = sft_share(cells)
    assert abs(est[K["harm_gap"]] - 0.25) < 0.01 and lo[K["harm_gap"]] <= 0.25 <= hi[K["harm_gap"]]
    d, lo, hi = step(cells, "sft", "dpo")
    assert abs(d[K["harm_gap"]] - 2) < 0.02 and lo[K["harm_gap"]] > 0


def test_classify_outcomes():
    rng, a, origins, benign, loglen = fixture()
    # OLMo 3's shape: little at SFT, most of it at DPO.
    late = cells_of(ladder(rng, a, benign, dict(base=0, sft=0.005, dpo=0.1, instruct=0.1),
                           dict(base=0, sft=0.05, dpo=1, instruct=1)), benign, origins, loglen)
    # Both within-type alignment and the harm contrast already there at SFT.
    both = cells_of(ladder(rng, a, benign, dict(base=0, sft=0.08, dpo=0.1, instruct=0.1),
                           dict(base=0, sft=0.8, dpo=1, instruct=1)), benign, origins, loglen)
    # Harm contrast at SFT, with no within-type alignment until DPO.
    harm = cells_of(ladder(rng, a, benign, dict(base=0, sft=0.005, dpo=0.1, instruct=0.1),
                           dict(base=0, sft=0.8, dpo=1, instruct=1)), benign, origins, loglen)
    # Within-type alignment at SFT, the harm contrast only at DPO.
    within = cells_of(ladder(rng, a, benign, dict(base=0, sft=0.08, dpo=0.1, instruct=0.1),
                             dict(base=0, sft=0.0, dpo=1, instruct=1)), benign, origins, loglen)
    assert classify(both, late) == "holds across the range"
    assert classify(harm, late) == "harm contrast only"
    assert classify(within, late) == "within-type only"
    assert classify(late, late) == "does not hold"


def test_ladder_differences_are_paired():
    rng, a, origins, benign, loglen = fixture()
    U = ladder(rng, a, benign, dict(base=0, sft=0.02, dpo=0.1, instruct=0.1), dict(base=0, sft=0.3, dpo=1, instruct=1))
    cells = cells_of(U, benign, origins, loglen)
    est, lo, hi = ladder_diff(cells, cells)
    assert np.nanmax(np.abs(est)) == 0 and np.nanmax(np.abs(lo)) == 0 and np.nanmax(np.abs(hi)) == 0
    est, lo, hi = share_diff(cells, cells)  # a difference of a ladder with itself is exactly zero
    assert np.nanmax(np.abs(est)) == 0 and np.nanmax(np.abs(lo)) == 0 and np.nanmax(np.abs(hi)) == 0
