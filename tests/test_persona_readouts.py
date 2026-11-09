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
