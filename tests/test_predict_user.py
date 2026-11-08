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
        assert {k: v for k, v in m.items() if k != "conditions"} == {k: v for k, v in h.items() if k != "conditions"}
        assert ours["group"] == "predict_user" and ours["run_name"] == f"predict-user-{family}-{stage}"
        same = {k: v for k, v in ours.items() if k not in ("group", "run_name", "models")}
        assert same == {k: v for k, v in hope.items() if k not in ("group", "run_name", "models")}
        assert same["max_pairs"] == 15000 and same["seed"] == 0 and same["system_prompt"] == "none"
