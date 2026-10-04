"""Models table on synthetic per-model utilities with known bleedthrough."""

import numpy as np
import pytest

from models import cross_model, other_assistant_r, plot_cross_model, plot_models, plot_vs_coinflip, run_rows
from report import task_arrays

ORIGINS = ["ALPACA"] * 40 + ["MATH"] * 40 + ["BAILBENCH"] * 20


def utility_rows(slope, with_base, seed=0, asst=None):
    """utilities.csv-like rows: instruct coinflip = slope * instruct assistant + noise; the base
    model, if present, has a shared prior in both conditions that the shift removes."""
    rng = np.random.default_rng(seed)
    n = len(ORIGINS)
    prior = rng.normal(0, 1, n)
    asst = rng.normal(0, 3, n) if asst is None else asst
    cols = {"instruct/assistant": prior + asst,
            "instruct/coinflip": 0.1 * prior + slope * asst + rng.normal(0, 0.01, n)}
    if with_base:
        cols.update({"base/assistant": prior, "base/coinflip": 0.1 * prior + rng.normal(0, 0.01, n)})
        cols.update({f"instruct-base/{c}": cols[f"instruct/{c}"] - cols[f"base/{c}"] for c in ("assistant", "coinflip")})
    return [dict(id=f"t{i}", origin=o, topic="harmful_request" if o == "BAILBENCH" else "math",
                 n_chars=100 + i, **{k: v[i] for k, v in cols.items()}) for i, o in enumerate(ORIGINS)]


def test_run_rows_recovers_slope_and_shift():
    summary = {f"{s}/{c}/reliability": 0.99 for s in ("base", "instruct") for c in ("assistant", "coinflip")}
    rows = run_rows("google/gemma-4-31B-it", utility_rows(0.05, with_base=True), summary, n_boot=50)
    by_stage = {r["stage"]: r for r in rows}
    assert set(by_stage) == {"base", "instruct"}
    inst = by_stage["instruct"]
    assert inst["label"] == "Gemma 4 31B" and inst["coinflip_effect"] == 7.81
    assert inst["slope_within"] == pytest.approx(0.05, abs=0.01)
    assert inst["r_shift"] > 0.95  # shifts: 0.05 * asst + noise vs asst
    assert inst["r_disattenuated"] == pytest.approx(inst["r"] / 0.99)
    assert np.isnan(by_stage["base"]["r_shift"])  # no base-minus-base shift


def test_run_without_base_has_instruct_row_only():
    (row,) = run_rows("Qwen/Qwen3.5-27B", utility_rows(0.02, with_base=False), {}, n_boot=20)
    assert row["stage"] == "instruct" and row["family"] == "Qwen"
    assert np.isnan(row["r_shift"]) and np.isnan(row["r_disattenuated"])


def test_own_versus_other_models_assistant():
    # Model X's coinflip follows its own Assistant utilities, which are unrelated to Y's.
    rng = np.random.default_rng(99)  # not a seed utility_rows uses, so independent of X
    U = {"X": task_arrays(utility_rows(0.05, False, seed=1))[0],
         "Y": task_arrays(utility_rows(0.05, False, seed=2, asst=rng.normal(0, 3, len(ORIGINS))))[0]}
    own = np.corrcoef(U["X"]["instruct/assistant"], U["X"]["instruct/coinflip"])[0, 1]
    other = other_assistant_r(U)
    assert own > 0.9 and abs(other["X"]) < 0.3


def test_cross_model_and_plots(tmp_path):
    shared = np.random.default_rng(3).normal(0, 3, len(ORIGINS))
    sources = {"allenai/Olmo-3.1-32B-Instruct": utility_rows(0.05, True, 0, shared),
               "google/gemma-4-31B-it": utility_rows(0.1, True, 1, shared),
               "Qwen/Qwen3.5-27B": utility_rows(0.0, False, 2)}
    utilities = {m: task_arrays(rows)[0] for m, rows in sources.items()}
    labels, M = cross_model(utilities, "assistant")
    assert labels == ["OLMo 3.1 32B", "Gemma 4 31B", "Qwen 3.5 27B"]  # MODEL_INFO order
    assert M[0, 1] > 0.8 and abs(M[0, 2]) < 0.4 and np.allclose(np.diag(M), 1)
    table = [r for m, rows in sources.items() for r in run_rows(m, rows, {}, n_boot=20)]
    for r in table:
        r["r_other_assistant"] = float("nan")
    figs = {"models": plot_models(table), "vs": plot_vs_coinflip(table),
            "cross": plot_cross_model({"a": (labels, M), "c": cross_model(utilities, "coinflip")})}
    for name, fig in figs.items():
        fig.savefig(tmp_path / f"{name}.png")
    assert set(figs["models"].axes[0].get_legend_handles_labels()[1]) == {"post-trained", "base"}
