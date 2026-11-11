"""DeepSeek V4 Flash in the replication (configs/replication/deepseek-v4-flash.yaml): every prompt the config will
score, rendered the way tournament.py / score_vllm.py render it, against the encoder shipped in the model's repo
(encoding/encoding_dsv4.py, chat mode). Reads the config, so it checks what will run. Needs only tokenizer files and
the encoder."""

import importlib.util
import itertools
import json
from pathlib import Path

import pytest
import yaml
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from prompts import CONDITIONS, PERSONAS, PERSONAS_SELF, ROLEPLAY, build_prompt, outcome_token_ids, parse_condition
from replication import BASE_CONDITIONS, FINAL_CONDITIONS, expand_conditions

CFG = yaml.safe_load(Path("configs/replication/deepseek-v4-flash.yaml").read_text())
BASE, FINAL = CFG["models"]
TASKS = [json.loads(line) for line in Path(CFG["tasks_file"]).read_text().splitlines()]
BOS, USER, ASSISTANT, CLOSE = "<｜begin▁of▁sentence｜>", "<｜User｜>", "<｜Assistant｜>", "</think>"
MARKERS = {BOS: 0, USER: 128803, ASSISTANT: 128804, CLOSE: 128822}
SWEEP = yaml.safe_load(Path("configs/models/deepseek-v4-flash.yaml").read_text())  # W&B lszrxqyq, HF on 4x H200

_n = [len(t["prompt"]) for t in TASKS]
_longest = sorted(range(len(TASKS)), key=lambda i: -_n[i])[:2]
PAIRS = [tuple(_longest), (0, 1), (150, 299)]
PAIRS += [(b, a) for a, b in PAIRS]


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(FINAL.get("tokenizer_from", FINAL["name"]))


@pytest.fixture(scope="module")
def base_tok():
    return AutoTokenizer.from_pretrained(BASE.get("tokenizer_from", BASE["name"]))


@pytest.fixture(scope="module")
def encoder():
    path = hf_hub_download(FINAL["name"], "encoding/encoding_dsv4.py")
    spec = importlib.util.spec_from_file_location("encoding_dsv4", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def render(m, condition, a, b):
    """As tournament.py renders a prompt for model entry m. With a turn_format the template tokenizer is unused."""
    return build_prompt(condition, TASKS[a]["prompt"], TASKS[b]["prompt"], None, CFG["system_prompt"],
                        turn_format=m.get("turn_format"), close_think=m.get("close_think", False),
                        strip_think=m.get("strip_think", False))


def reference_messages(condition, a, b):
    """The conversation a condition stands for, built independently of prompts._render: an optional system message
    (persona or roleplay) and the user message (with the persona's lead, for the user-turn placements)."""
    name, persona, placement = parse_condition(condition)
    body = CONDITIONS[name]["body"].format(first=TASKS[a]["prompt"].strip(), second=TASKS[b]["prompt"].strip())
    system = None
    if placement == "system":
        system = PERSONAS[persona]
    elif placement == "roleplay":
        system = ROLEPLAY[persona]
    elif placement in ("user", "self"):
        body = (PERSONAS if placement == "user" else PERSONAS_SELF)[persona] + "\n\n" + body
    return ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": body}]


def enc(t, s):
    return t(s, add_special_tokens=False).input_ids


def test_config():
    assert (CFG["group"], CFG["run_name"], CFG["max_pairs"], CFG["seed"], CFG["system_prompt"]) == (
        "replication", "deepseek-v4-flash", 15000, 0, "none")
    assert CFG["tasks_file"] == "data/tasks_n300_max500_seed0.jsonl" and len(TASKS) == 300
    assert (BASE["name"], BASE["stage"]) == ("deepseek-ai/DeepSeek-V4-Flash-Base", "base")
    assert (FINAL["name"], FINAL["stage"]) == ("deepseek-ai/DeepSeek-V4-Flash", "instruct")
    assert expand_conditions(BASE["conditions"]) == list(BASE_CONDITIONS)
    assert expand_conditions(FINAL["conditions"]) == FINAL_CONDITIONS and len(FINAL_CONDITIONS) == 44
    assert BASE["turn_format"] == FINAL["turn_format"] == {
        "user": BOS + USER, "reply": ASSISTANT + CLOSE, "system": BOS + "{system}" + USER}
    for m in (BASE, FINAL):
        assert not ({"trust_remote_code", "close_think", "strip_think", "chat_template_from"} & set(m))
        # TP 4 each, so base and instruct share one 8-GPU node; quantization is read from config.json.
        assert m["vllm"]["tensor_parallel_size"] == 4 and "quantization" not in m["vllm"]
        assert 0.85 <= m["vllm"]["gpu_memory_utilization"] <= 0.92
        assert "model-*.safetensors" in m["download"]["allow_patterns"]
        assert "model.safetensors.index.json" in m["download"]["allow_patterns"]
    assert (BASE["download"]["gb"], FINAL["download"]["gb"]) == (294.7, 159.6)


@pytest.mark.parametrize("m", [BASE, FINAL], ids=["base", "final"])
def test_max_model_len_holds_the_longest_prompt(m, tok):
    """The longest prompt over the model's conditions, every ordered pair of the 8 longest tasks (prompt length is the
    two tasks' lengths plus a per-condition constant, so these bound all pairs), plus one outcome token, with room."""
    n = [len(enc(tok, x["prompt"].strip())) for x in TASKS]
    top = sorted(range(len(TASKS)), key=lambda i: -n[i])[:8]
    longest = max(len(enc(tok, render(m, c, a, b)))
                  for c in expand_conditions(m["conditions"]) for a, b in itertools.permutations(top, 2))
    assert longest == (557 if m is FINAL else 386)  # the numbers in the config's comments
    assert 1.5 * (longest + 1) <= m["vllm"]["max_model_len"]


def test_tasks_contain_no_added_token_text(tok):
    """No task spells an added token (<think>, <｜User｜>, ...), so markers come only from the turn format."""
    added = set(tok.get_added_vocab())
    assert not [(t["id"], s) for t in TASKS for s in added if s in t["prompt"]]


@pytest.mark.parametrize("condition", FINAL_CONDITIONS)
def test_final_prompts_match_the_encoder(condition, tok, encoder):
    """Each final prompt against encode_messages(thinking_mode="chat") of the same conversation. An open user turn is
    the encoding minus the "<｜Assistant｜></think>" that would close it. A reply is the encoding plus the prefill, and
    a prefix of the encoding of the finished turn whose answer is the outcome, in text and in tokens, with the
    outcome token next."""
    spec = CONDITIONS[parse_condition(condition)[0]]
    for a, b in PAIRS:
        p, msgs = render(FINAL, condition, a, b), reference_messages(condition, a, b)
        reference = encoder.encode_messages(msgs, thinking_mode="chat")
        if spec["prefill"] is None:
            assert p + ASSISTANT + CLOSE == reference
            continue
        assert p == reference + spec["prefill"]
        for outcome, oid in zip(spec["outcomes"], outcome_token_ids(condition, tok)):
            past = encoder.encode_messages(msgs + [{"role": "assistant", "content": spec["prefill"] + outcome + "."}],
                                           thinking_mode="chat")
            assert past.startswith(p)
            ids, past_ids = enc(tok, p), enc(tok, past)
            assert past_ids[:len(ids)] == ids and past_ids[len(ids)] == oid


@pytest.mark.parametrize("condition", BASE_CONDITIONS)
def test_base_prompts_are_the_finals(condition, tok, base_tok):
    for a, b in PAIRS:
        p = render(BASE, condition, a, b)
        assert p == render(FINAL, condition, a, b) and enc(base_tok, p) == enc(tok, p)


@pytest.mark.parametrize("condition", ["assistant", "coinflip"])
def test_plain_prompts_are_the_sweeps(condition):
    """The plain assistant and coinflip prompts are the sweep's (lszrxqyq, HF), on the same tasks and pairs, so their
    vLLM scores can be checked prompt by prompt against it."""
    assert (SWEEP["tasks_file"], SWEEP["max_pairs"], SWEEP["seed"], SWEEP["system_prompt"]) == (
        CFG["tasks_file"], CFG["max_pairs"], CFG["seed"], CFG["system_prompt"])
    for sweep_m, m in zip(SWEEP["models"], CFG["models"]):
        assert (sweep_m["name"], sweep_m["stage"]) == (m["name"], m["stage"])
        assert condition in sweep_m["conditions"]
        for a, b in PAIRS:
            assert render(sweep_m, condition, a, b) == render(m, condition, a, b)


@pytest.mark.parametrize("condition", FINAL_CONDITIONS)
def test_turn_structure(condition):
    """No system prompt means none (the user turn follows BOS directly); a persona system prompt is the only text
    between them; user-turn conditions end inside an open user turn; replies end with <｜Assistant｜></think> and the
    prefill."""
    name, persona, placement = parse_condition(condition)
    spec = CONDITIONS[name]
    for a, b in PAIRS:
        p = render(FINAL, condition, a, b)
        system = {"system": PERSONAS, "roleplay": ROLEPLAY}.get(placement, {}).get(persona, "")
        assert p.startswith(BOS + system + USER) and p.count(BOS) == p.count(USER) == 1
        user_turn = p[p.index(USER) + len(USER):]
        if spec["prefill"] is None:
            assert ASSISTANT not in user_turn and p.endswith(spec["body"].rsplit("\n", 1)[-1])
        else:
            assert user_turn.endswith(ASSISTANT + CLOSE + spec["prefill"]) and user_turn.count(ASSISTANT) == 1


@pytest.mark.parametrize("persona", list(ROLEPLAY))
@pytest.mark.parametrize("condition", ["assistant", "coinflip"])
@pytest.mark.parametrize("placement", ["system", "roleplay"])
def test_persona_adds_only_its_system_block(persona, condition, placement, tok):
    """A system-prompt persona is the plain prompt with the system text inserted after the BOS, in text and tokens."""
    text = PERSONAS[persona] if placement == "system" else ROLEPLAY[persona]
    suffix = "" if placement == "system" else "+roleplay"
    for a, b in PAIRS:
        plain = render(FINAL, condition, a, b)
        placed = render(FINAL, f"{condition}@{persona}{suffix}", a, b)
        assert placed == BOS + text + plain[len(BOS):]
        assert enc(tok, placed) == enc(tok, BOS + text) + enc(tok, plain[len(BOS):])
        assert enc(tok, plain) == enc(tok, BOS) + enc(tok, plain[len(BOS):])


@pytest.mark.parametrize("placed", [c for c in FINAL_CONDITIONS if parse_condition(c)[2] in ("user", "self")])
def test_persona_in_user_turn_only_adds_the_lead(placed):
    condition, persona, placement = parse_condition(placed)
    lead = (PERSONAS if placement == "user" else PERSONAS_SELF)[persona] + "\n\n"
    for a, b in PAIRS:
        plain = render(FINAL, condition, a, b)
        assert render(FINAL, placed, a, b) == BOS + USER + lead + plain[len(BOS + USER):]


@pytest.mark.parametrize("condition", FINAL_CONDITIONS)
def test_markers_are_single_added_tokens(condition, tok):
    """Each marker is one added-token id; the ids hold exactly as many of each as the text has markers, and no other
    added token. (BOS is special; <｜User｜>, <｜Assistant｜> and </think> are added but not special.)"""
    for s, i in MARKERS.items():
        assert enc(tok, s) == [i] and i in tok.added_tokens_decoder
    added = set(tok.added_tokens_decoder)
    for a, b in PAIRS:
        p = render(FINAL, condition, a, b)
        ids = enc(tok, p)
        for s, i in MARKERS.items():
            assert ids.count(i) == p.count(s)
        assert {i for i in ids if i in added} <= set(MARKERS.values())


def test_base_tokenizer_gives_the_finals_ids_on_every_task(tok, base_tok):
    """The two repos ship the same tokenizer (same vocabulary and added tokens); over all 300 tasks and every base
    condition, both give the same ids."""
    assert base_tok.get_vocab() == tok.get_vocab() and base_tok.get_added_vocab() == tok.get_added_vocab()
    for condition in BASE_CONDITIONS:
        for a in range(len(TASKS)):
            p = render(BASE, condition, a, (a + 1) % len(TASKS))
            assert enc(base_tok, p) == enc(tok, p)


@pytest.mark.parametrize("condition", FINAL_CONDITIONS)
def test_outcome_tokens(condition, tok, base_tok):
    """Single, distinct and the same in base and final; appending the outcome string to the prompt adds exactly the
    outcome id (no merge across the boundary)."""
    ids = outcome_token_ids(condition, tok)
    assert len(set(ids)) == 2 and outcome_token_ids(condition, base_tok) == ids
    outcomes = CONDITIONS[parse_condition(condition)[0]]["outcomes"]
    for a, b in PAIRS:
        p = render(FINAL, condition, a, b)
        for s, i in zip(outcomes, ids):
            assert enc(tok, p + s) == enc(tok, p) + [i]


def test_bos_once_as_the_encoder_writes_it(tok, base_tok):
    """The encoder opens with BOS and the turn format writes it; neither tokenizer adds one (no post-processor
    template), so the ids hold one BOS, first, with or without add_special_tokens."""
    for condition in FINAL_CONDITIONS:
        p = render(FINAL, condition, 0, 1)
        ids = enc(tok, p)
        assert ids[0] == 0 and ids.count(0) == 1
        assert tok(p).input_ids == base_tok(p).input_ids == ids
