"""The persona as the user (README, same title): prompts and configs. Needs only the (small) tokenizer files."""

import re
from pathlib import Path

import pytest
import yaml
from transformers import AutoTokenizer

import prompts
from prompts import build_prompt, outcome_token_ids, parse_condition

LLAMA, GEMMA = "unsloth/Llama-3.3-70B-Instruct", "google/gemma-4-31B-it"
READOUTS = ("assistant", "coinflip", "user_choice", "predict_user")


@pytest.mark.parametrize("persona", ["evil", "slacker"])
def test_first_person_persona_differs_only_in_pronouns(persona):
    second, first = prompts.PERSONAS[persona], prompts.PERSONAS_SELF[persona]
    words = lambda s: re.findall(r"[\w']+|[^\w\s]", s)
    a, b = words(second), words(first)
    assert len(a) == len(b)
    swaps = {(x, y) for x, y in zip(a, b) if x != y}
    assert swaps <= {("You", "I"), ("you", "I"), ("you", "me"), ("your", "my"), ("are", "am"), ("You", "Effort"),
                     ("Effort", "Effort")}, swaps
    assert not re.search(r"\b(you|your)\b", first, re.I)


@pytest.mark.parametrize("name", [LLAMA, GEMMA])
@pytest.mark.parametrize("readout", READOUTS)
@pytest.mark.parametrize("persona", ["evil", "slacker"])
def test_self_description_only_adds_the_lead_to_the_user_turn(name, readout, persona):
    tok = AutoTokenizer.from_pretrained(name)
    plain, selfd = build_prompt(readout, "FIRST", "SECOND", tok), build_prompt(f"{readout}@{persona}+self", "FIRST", "SECOND", tok)
    # The plain prompt has no system block, so equality with it plus the lead rules one out. (Don't search the text for
    # "system": the evil persona says "when systems fail".)
    i = plain.index("I would like you")
    assert selfd == plain[:i] + prompts.PERSONAS_SELF[persona] + "\n\n" + plain[i:]
    assert outcome_token_ids(f"{readout}@{persona}+self", tok) == outcome_token_ids(readout, tok)


@pytest.mark.parametrize("name", [LLAMA, GEMMA])
@pytest.mark.parametrize("readout", ["user_choice", "predict_user"])
@pytest.mark.parametrize("persona", ["evil", "slacker"])
def test_system_persona_matches_the_assistant_conditions_system_block(name, readout, persona):
    """The persona's system block is the one the persona runs used for assistant@<persona>: the same text before the user
    turn, whichever readout follows."""
    tok = AutoTokenizer.from_pretrained(name)
    a = build_prompt(f"assistant@{persona}", "FIRST", "SECOND", tok)
    x = build_prompt(f"{readout}@{persona}", "FIRST", "SECOND", tok)
    head = a[:a.index("I would like you")]
    assert x.startswith(head) and prompts.PERSONAS[persona] in head
    assert x[len(head):] == build_prompt(readout, "FIRST", "SECOND", tok)[len(build_prompt(readout, "FIRST", "SECOND", tok)
                                                                               .split("I would like you")[0]):]


def test_configs_cover_every_cell_once_per_model():
    want = {f"{r}@{p}+self" for r in READOUTS for p in ("evil", "slacker")}
    want |= {f"{r}@{p}" for r in ("user_choice", "predict_user") for p in ("evil", "slacker")}
    for family, name in (("llama-70b", LLAMA), ("gemma4-31b", GEMMA)):
        seen = []
        for pod in ("self-evil", "self-slacker", "system"):
            cfg = yaml.safe_load(Path(f"configs/persona_readouts/{family}-{pod}.yaml").read_text())
            (m,) = cfg["models"]
            assert m["name"] == name and m["stage"] == "instruct" and cfg["group"] == "persona_readouts"
            assert cfg["max_pairs"] == 15000 and cfg["seed"] == 0 and cfg["system_prompt"] == "none"
            assert cfg["tasks_file"] == "data/tasks_n300_max500_seed0.jsonl" and cfg["tags"] == [family]
            for c in m["conditions"]:
                parse_condition(c)
            if pod.startswith("self"):  # all four readouts of one persona on one pod
                persona = pod.split("-")[1]
                assert sorted(m["conditions"]) == sorted(f"{r}@{persona}+self" for r in READOUTS)
            seen += m["conditions"]
        assert sorted(seen) == sorted(want)


# persona_readouts.py on synthetic data: the calls come out as constructed.
import numpy as np

from personas import z
from persona_readouts import K, STATS, call, model_cell, score_predictions, stats
from report import stratified_resamples


def synthetic(separable=True, n=300, seed=0):
    """Generic appeal g, the Assistant's own part s, and the self-described persona q. In the self box the Assistant stays
    itself (g + s), the prediction takes on the persona (g + 2q), the user's choice copies the prediction and the coin copies
    the Assistant. With separable=False the prediction is the Assistant's ranking plus noise."""
    rng = np.random.default_rng(seed)
    g, s, q = (z(rng.normal(size=n)) for _ in range(3))
    e = lambda sd: rng.normal(0, sd, n)
    A_self = g + s
    P_self = g + 2 * q if separable else A_self + e(0.05)
    V = {"A_0": g + s + e(0.1), "C_0": 0.3 * (g + s) + e(0.05), "U_0": 0.5 * (g + s) + e(0.1), "P_0": g + s + e(0.2),
         "Aper": 2 * q - g, "A_self": A_self + e(0.1), "P_self": P_self + e(0.1), "U_self": 0.4 * P_self + e(0.1),
         "C_self": 0.3 * A_self + e(0.05)}
    V.update(A_sys=V["Aper"], C_sys=0.5 * V["Aper"] + e(0.1), U_sys=0.4 * V["Aper"] + e(0.1), P_sys=V["Aper"] + g + e(0.2))
    origins = np.array(["MATH", "ALPACA", "WILDCHAT", "STRESS_TEST", "BAILBENCH"] * (n // 5))
    return V, origins != "BAILBENCH", rng.normal(5, 1, n), origins


def test_calls_follow_the_construction():
    V, benign, loglen, origins = synthetic()
    assert stats(V, benign, loglen, origins).shape == (len(STATS),)
    cell = model_cell(V, benign, loglen, origins, stratified_resamples(origins, np.random.default_rng(0), 200))
    est = cell["est"]
    assert est[K["self:sep"]] < 0.5
    assert call(cell, "self", "U") == "user" and call(cell, "self", "C") == "Assistant"
    assert est[K["self:dUC"]] > 1.0
    assert est[K["self:pu_a_p"]] < 0.15 and est[K["self:pc_p_a"]] < 0.15
    # The persona Assistant is the fit's own target: w = 1; the self-described prediction leans on the persona.
    assert abs(est[K["sys:A:w"]] - 1) < 1e-9 and est[K["self:P:w"]] > 0.5 > est[K["self:A:w"]]
    # The self-description leaves the Assistant where it was.
    assert est[K["self:A:r_x0"]] > 0.95


def test_no_calls_when_the_prediction_is_the_assistant():
    V, benign, loglen, origins = synthetic(separable=False)
    cell = model_cell(V, benign, loglen, origins, stratified_resamples(origins, np.random.default_rng(0), 50))
    assert cell["est"][K["self:sep"]] >= 0.9
    assert call(cell, "self", "U") == call(cell, "self", "C") == "not separable"


def test_predictions_need_all_four_cells():
    row = {name: 0.0 for name in STATS} | {f"{name}_lo": 0.0 for name in STATS}
    row.update(call_U_self="user", call_C_self="Assistant", masses={"assistant@evil+self": 1.0})
    table = [dict(row, family=f, persona=p) for f in ("llama-70b", "gemma4-31b") for p in ("evil", "slacker")]
    assert all(held is not None for _, held in score_predictions(table))
    assert all(held is None for _, held in score_predictions(table[:2]))
