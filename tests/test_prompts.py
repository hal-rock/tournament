"""Rendered prompts for each model we use. Needs only the (small) tokenizer files."""

import pytest
from transformers import AutoTokenizer

from prompts import CONDITIONS, build_prompt, outcome_token_ids

OLMO3 = "allenai/Olmo-3.1-32B-Instruct"
OLMO3_BASE = "allenai/Olmo-3-1125-32B"
OLMO2 = "allenai/OLMo-2-0425-1B-Instruct"


@pytest.fixture(scope="module", params=[OLMO3, OLMO2])
def tok(request):
    return AutoTokenizer.from_pretrained(request.param)


def test_coinflip_continues_human_turn(tok):
    p = build_prompt("coinflip", "FIRST", "SECOND", tok)
    assert p.endswith("Task A (if heads): FIRST\n\nTask B (if tails): SECOND\n\n"
                      "Okay, I'm flipping the coin now ... it came up")
    assert "assistant" not in p and "system" not in p
    assert "<|im_end|>" not in p  # the user turn is left open


def test_assistant_prefill_follows_closed_user_turn(tok):
    p = build_prompt("assistant", "FIRST", "SECOND", tok)
    assert "Task A: FIRST\n\nTask B: SECOND" in p
    assert "system" not in p
    header = "<|im_start|>assistant\n" if tok.name_or_path == OLMO3 else "<|assistant|>\n"
    assert p.endswith(header + "I'll do Task")
    assert build_prompt("assistant", "FIRST", "SECOND", tok, prefill=False).endswith(header)


def test_conditions_share_opening_and_layout(tok):
    a = build_prompt("assistant", "FIRST", "SECOND", tok)
    c = build_prompt("coinflip", "FIRST", "SECOND", tok)
    opening = "I would like you to perform one of the following two tasks"
    assert a.index(opening) == c.index(opening)  # same preamble before it


def test_system_prompt_options():
    tok = AutoTokenizer.from_pretrained(OLMO3)
    assert build_prompt("coinflip", "X", "Y", tok, "default").startswith("<|im_start|>system\nYou are Olmo")
    assert build_prompt("coinflip", "X", "Y", tok, "Be brief.").startswith("<|im_start|>system\nBe brief.<|im_end|>")


def test_base_model_with_borrowed_template():
    base = AutoTokenizer.from_pretrained(OLMO3_BASE)
    template = AutoTokenizer.from_pretrained(OLMO3)
    p = build_prompt("coinflip", "X", "Y", template)
    # Same tokenization under the base tokenizer as under the Instruct one.
    assert base(p, add_special_tokens=False).input_ids == template(p, add_special_tokens=False).input_ids
    for condition in CONDITIONS:
        assert outcome_token_ids(condition, base) == outcome_token_ids(condition, template)


def test_outcome_tokens_are_single_and_distinct(tok):
    for condition in CONDITIONS:
        a, b = outcome_token_ids(condition, tok)
        assert a != b
