"""The Assistant's prediction of the user's choice (README, "What does the Assistant think the user will choose?"): the
prompt and the configs. Needs only the (small) tokenizer files."""

from pathlib import Path

import pytest
import yaml
from transformers import AutoTokenizer

from prompts import CONDITIONS, build_prompt, outcome_token_ids

# final: base, as in configs/hope/ and configs/predict_user/.
FAMILIES = {"olmo3.1-32b": ("allenai/Olmo-3.1-32B-Instruct", "allenai/Olmo-3-1125-32B"),
            "gemma4-31b": ("google/gemma-4-31B-it", "google/gemma-4-31B"),
            "llama-70b": ("unsloth/Llama-3.3-70B-Instruct", "unsloth/Llama-3.1-70B"),
            "glm-4.5-air": ("zai-org/GLM-4.5-Air", "zai-org/GLM-4.5-Air-Base")}
FINALS = [final for final, _ in FAMILIES.values()]


def test_prediction_is_the_user_choice_prompt_with_its_last_line_asked_instead():
    u, p = CONDITIONS["user_choice"]["body"], CONDITIONS["predict_user"]["body"]
    head = u.removesuffix("Okay, I think I'll choose Task")
    assert head != u and p == head + "Before I decide, which one do you think I'll choose?"
    spec = CONDITIONS["predict_user"]
    # The prefill is the user's-choice line's last words with "I'll" -> "you'll"; same outcomes as A and U.
    assert spec["prefill"] == "I think you'll choose Task" == u.split("Okay, ")[-1].replace("I'll", "you'll")
    assert spec["outcomes"] == CONDITIONS["assistant"]["outcomes"] == CONDITIONS["user_choice"]["outcomes"]


@pytest.mark.parametrize("name", FINALS)
def test_prediction_is_read_after_the_closed_user_turn(name):
    tok = AutoTokenizer.from_pretrained(name)
    a = build_prompt("assistant", "FIRST", "SECOND", tok)
    u = build_prompt("user_choice", "FIRST", "SECOND", tok)
    p = build_prompt("predict_user", "FIRST", "SECOND", tok)
    # Up to the user's last line, the user turn is U's, character for character.
    stem = u.removesuffix("Okay, I think I'll choose Task")
    assert p.startswith(stem + "Before I decide, which one do you think I'll choose?")
    # After that, the same reply header as the Assistant's own choice, then the prefill.
    header = a.removeprefix(build_prompt("assistant", "FIRST", "SECOND", tok, prefill=False)[:a.index("SECOND") + 6])
    assert p.endswith(header.removesuffix("I'll choose Task") + "I think you'll choose Task")
    assert "system" not in p
    ids = tok(p, add_special_tokens=False).input_ids
    assert tok.decode(ids[-1]) == " Task"
    assert outcome_token_ids("predict_user", tok) == outcome_token_ids("assistant", tok)
    for first in outcome_token_ids("predict_user", tok):  # the read continues the prompt's tokens unchanged
        assert tok(p + tok.decode([first]), add_special_tokens=False).input_ids == ids + [first]


@pytest.mark.parametrize("family", list(FAMILIES))
def test_base_tokenizer_reads_the_borrowed_prompt_identically(family):
    final, base = FAMILIES[family]
    t_final, t_base = AutoTokenizer.from_pretrained(final), AutoTokenizer.from_pretrained(base)
    p = build_prompt("predict_user", "Write a haiku about rain.", "Prove that 2 + 2 = 4.", t_final)
    assert t_base(p, add_special_tokens=False).input_ids == t_final(p, add_special_tokens=False).input_ids
    assert outcome_token_ids("predict_user", t_base) == outcome_token_ids("predict_user", t_final)


@pytest.mark.parametrize("family", list(FAMILIES))
def test_configs_match_the_hope_run_but_for_conditions_and_names(family):
    final, base = FAMILIES[family]
    for stage, name in (("final", final), ("base", base)):
        ours = yaml.safe_load(Path(f"configs/predict_user/{family}-{stage}.yaml").read_text())
        hope = yaml.safe_load(Path(f"configs/hope/{family}-{stage}.yaml").read_text())
        (m,), (h,) = ours["models"], hope["models"]
        assert m["name"] == name and m["conditions"] == ["predict_user", "user_choice", "assistant"]
        # GLM runs on 2x H200 with the default split (README amendment), so its 3x H100 max_memory is dropped.
        skip = {"conditions"} | ({"max_memory"} if family == "glm-4.5-air" else set())
        assert "max_memory" not in m or family != "glm-4.5-air"
        assert {k: v for k, v in m.items() if k not in skip} == {k: v for k, v in h.items() if k not in skip}
        assert ours["group"] == "predict_user" and ours["run_name"] == f"predict-user-{family}-{stage}"
        same = {k: v for k, v in ours.items() if k not in ("group", "run_name", "models")}
        assert same == {k: v for k, v in hope.items() if k not in ("group", "run_name", "models")}
        assert same["max_pairs"] == 15000 and same["seed"] == 0 and same["system_prompt"] == "none"


# predict_user.py on synthetic data: the call and the partials come out as constructed.
import itertools

import numpy as np

from personas import z
from predict_user import COND, K, STATS, call, model_cell, reliabilities, stats, score_predictions
from report import stratified_resamples


def synthetic(user_is_prediction, n=300, seed=0):
    """Generic appeal g (every readout, base included), an Assistant-specific part s that post-training adds, and a
    user-model part q. The prediction P is g + 0.5 s + q. If user_is_prediction, the user's choice is P plus noise (the
    first reading: s reaches U only through P); otherwise it's A_f / 4 (the second: U is the Assistant's own choice,
    with none of q)."""
    rng = np.random.default_rng(seed)
    g, s, q = (z(rng.normal(size=n)) for _ in range(3))
    e = lambda sd: rng.normal(0, sd, n)
    P_f = g + 0.5 * s + q + e(0.1)
    a = {"A_b": g + e(0.3), "A_f": 2 * g + 4 * s, "P_b": 0.3 * g + e(0.1), "U_b": 0.3 * g + e(0.1), "P_f": P_f,
         "U_f": (P_f if user_is_prediction else 0.5 * g + s) + e(0.1)}
    a.update(C_f=0.3 * a["U_f"] + e(0.05), U0_f=a["U_f"] + e(0.2))
    origins = np.array(["MATH", "ALPACA", "WILDCHAT", "STRESS_TEST", "BAILBENCH"] * (n // 5))
    return a, origins, origins != "BAILBENCH", rng.normal(5, 1, n)


@pytest.mark.parametrize("user_is_prediction", [True, False])
def test_call_and_partials_follow_the_construction(user_is_prediction):
    a, origins, benign, loglen = synthetic(user_is_prediction)
    assert stats(a, benign, origins, loglen).shape == (len(STATS),)
    cell = model_cell(a, benign, origins, loglen, stratified_resamples(origins, np.random.default_rng(0), 200))
    est = cell["est"]
    if user_is_prediction:
        assert call(cell) == "prediction" and est[K["delta"]] > 0.2
        assert abs(est[K["partial_ua_p"]]) < 0.15 and est[K["partial_up_a"]] > 0.8
        assert est[K["r2_ap"]] - est[K["r2_p"]] < 0.02
    else:
        assert call(cell) == "own choice" and est[K["delta"]] < -0.2
        assert abs(est[K["partial_up_a"]]) < 0.15 and est[K["partial_ua_p"]] > 0.8
    # The prediction is partly Assistant-like either way, and post-training moved it toward A (P_b has no s).
    assert 0.3 < est[K["r_ap"]] < 0.9 and est[K["d_P"]] > 0.2
    # The coin here is a scaled copy of U, so delta_c follows delta's sign.
    assert np.sign(est[K["delta_c"]]) == np.sign(est[K["delta"]])


def test_reliabilities_cover_every_readout_and_stage():
    rng = np.random.default_rng(0)
    ids = [f"t{i}" for i in range(20)]
    u = {k: rng.normal(size=len(ids)) for k in COND}
    rows = [dict(stage=s, condition=c, first=ids[i], second=ids[j], log_odds=u[k][i] - u[k][j] + rng.normal(0, 0.1))
            for s in ("base", "instruct") for k, c in COND.items() for i, j in itertools.permutations(range(len(ids)), 2)]
    rel = reliabilities(rows, ids)
    assert set(rel) == {f"{k}_{s}" for k in COND for s in "bf"} and min(rel.values()) > 0.95


def test_predictions_are_scored_from_the_table_columns():
    row = {name: 0.0 for name in STATS} | {f"{name}_lo": 0.0 for name in STATS}
    table = [dict(row, model=m, call="own choice", mass_final=0.95) for m in "abcd"]
    assert len(score_predictions(table)) == 9
