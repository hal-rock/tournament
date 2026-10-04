"""Rendered prompts for each model we use. Needs only the (small) tokenizer files."""

import importlib.util

import pytest
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

import prompts
from prompts import CONDITIONS, build_prompt, outcome_token_ids

OLMO3 = "allenai/Olmo-3.1-32B-Instruct"
OLMO3_BASE = "allenai/Olmo-3-1125-32B"
OLMO2 = "allenai/OLMo-2-0425-1B-Instruct"
QWEN_SMALL = "Qwen/Qwen3.5-0.8B"  # CPU smoke test (configs/local-qwen.yaml)
QWEN_SMALL_BASE = "Qwen/Qwen3.5-0.8B-Base"
QWEN_DENSE = "Qwen/Qwen3.5-27B"
QWEN_MOE = "Qwen/Qwen3.5-35B-A3B"
QWEN_MOE_BASE = "Qwen/Qwen3.5-35B-A3B-Base"
QWEN_LARGE = "Qwen/Qwen3.5-122B-A10B"
GEMMA = "google/gemma-4-31B-it"
GEMMA_BASE = "google/gemma-4-31B"
LLAMA = "unsloth/Llama-3.3-70B-Instruct"
LLAMA_BASE = "unsloth/Llama-3.1-70B"
GLM = "zai-org/GLM-4.5-Air"
GLM_BASE = "zai-org/GLM-4.5-Air-Base"
DEEPSEEK = "deepseek-ai/DeepSeek-V4-Flash"
DEEPSEEK_BASE = "deepseek-ai/DeepSeek-V4-Flash-Base"
# DeepSeek V4 has no Jinja template; its configs give the turn markers (chat mode).
DEEPSEEK_TURNS = {"user": "<｜begin▁of▁sentence｜><｜User｜>", "reply": "<｜Assistant｜></think>"}
TURN_FORMAT = {DEEPSEEK: DEEPSEEK_TURNS, DEEPSEEK_BASE: DEEPSEEK_TURNS}
QWEN_REPLY = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
# Where the Assistant's reply starts. Hybrid-thinking templates get their non-thinking
# reply: an empty think block (Qwen 3.5, GLM-4.5) or thought channel (Gemma 4).
REPLY_HEADER = {
    OLMO3: "<|im_end|>\n<|im_start|>assistant\n", OLMO2: "\n<|assistant|>\n",
    QWEN_SMALL: QWEN_REPLY, QWEN_DENSE: QWEN_REPLY, QWEN_MOE: QWEN_REPLY, QWEN_LARGE: QWEN_REPLY,
    GEMMA: "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>",
    LLAMA: "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    GLM: "/nothink<|assistant|>\n<think></think>\n",
    DEEPSEEK: "<｜Assistant｜></think>",
}


def render(condition, tok, first="FIRST", second="SECOND", **kw):
    return build_prompt(condition, first, second, tok, turn_format=TURN_FORMAT.get(tok.name_or_path), **kw)


@pytest.fixture(scope="module", params=list(REPLY_HEADER))
def tok(request):
    return AutoTokenizer.from_pretrained(request.param)


def test_coinflip_continues_human_turn(tok):
    p = render("coinflip", tok)
    assert p.endswith("Task A (if heads): FIRST\n\nTask B (if tails): SECOND\n\n"
                      "Okay, I'm flipping the coin now ... it came up")  # the user turn is left open
    assert "assistant" not in p.lower() and "system" not in p and "nothink" not in p


def test_assistant_prefill_follows_closed_user_turn(tok):
    p = render("assistant", tok)
    assert "Task A: FIRST\n\nTask B: SECOND" in p
    assert "system" not in p
    header = REPLY_HEADER[tok.name_or_path]
    assert p.endswith("Task B: SECOND" + header + "I'll choose Task")
    assert render("assistant", tok, prefill=False).endswith(header)


def test_conditions_share_opening_and_layout(tok):
    a = render("assistant", tok)
    c = render("coinflip", tok)
    opening = "I would like you to perform one of the following two tasks"
    assert a.index(opening) == c.index(opening)  # same preamble before it


def test_system_prompt_options():
    tok = AutoTokenizer.from_pretrained(OLMO3)
    assert build_prompt("coinflip", "X", "Y", tok, "default").startswith("<|im_start|>system\nYou are Olmo")
    assert build_prompt("coinflip", "X", "Y", tok, "Be brief.").startswith("<|im_start|>system\nBe brief.<|im_end|>")


def test_task_text_mentioning_system_is_kept(tok):
    # The leftover-system-block check looks at template text only, never at the tasks.
    assert "operating system" in render("assistant", tok, first="Explain what an operating system does.")


@pytest.mark.parametrize("name", [OLMO3, LLAMA, GLM])
def test_prefill_continues_like_a_rendered_reply(name):
    """Where the template renders past Assistant turns with the same header as the
    generation prompt, the prefilled prompt is a prefix of the conversation in which
    the Assistant replied "I'll choose Task A." (checks e.g. GLM's newline after </think>)."""
    tok = AutoTokenizer.from_pretrained(name)
    body = CONDITIONS["assistant"]["body"].format(first="FIRST", second="SECOND")
    conv = tok.apply_chat_template([{"role": "user", "content": body},
                                    {"role": "assistant", "content": "I'll choose Task A."}],
                                   tokenize=False, enable_thinking=False)
    conv = prompts._strip_system(conv, body)
    assert conv.startswith(render("assistant", tok))


def test_deepseek_matches_reference_encoding():
    """DeepSeek V4's turn markers against the encoder shipped in its repo (chat mode)."""
    path = hf_hub_download(DEEPSEEK, "encoding/encoding_dsv4.py")
    spec = importlib.util.spec_from_file_location("encoding_dsv4", path)
    enc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(enc)
    tok = AutoTokenizer.from_pretrained(DEEPSEEK)
    for condition in CONDITIONS:
        body = CONDITIONS[condition]["body"].format(first="FIRST", second="SECOND")
        reference = enc.encode_messages([{"role": "user", "content": body}], thinking_mode="chat")
        if CONDITIONS[condition]["prefill"] is None:
            assert reference.startswith(render(condition, tok))
        else:
            assert render(condition, tok) == reference + CONDITIONS[condition]["prefill"]


@pytest.mark.parametrize("base_name,template_name", [
    (OLMO3_BASE, OLMO3), (QWEN_SMALL_BASE, QWEN_SMALL), (QWEN_MOE_BASE, QWEN_MOE), (GEMMA_BASE, GEMMA),
    (LLAMA_BASE, LLAMA), (GLM_BASE, GLM), (DEEPSEEK_BASE, DEEPSEEK)])
def test_base_model_with_borrowed_template(base_name, template_name):
    base = AutoTokenizer.from_pretrained(base_name)
    template = AutoTokenizer.from_pretrained(template_name)
    for condition in CONDITIONS:
        p = render(condition, template)
        # Same tokenization under the base tokenizer as under the Instruct one.
        assert base(p, add_special_tokens=False).input_ids == template(p, add_special_tokens=False).input_ids
    for condition in CONDITIONS:
        assert outcome_token_ids(condition, base) == outcome_token_ids(condition, template)


def test_outcome_tokens_are_single_and_distinct(tok):
    for condition in CONDITIONS:
        a, b = outcome_token_ids(condition, tok)
        assert a != b
