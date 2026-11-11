"""Trinity Large in the replication (configs/replication/trinity-large.yaml): every prompt the config will score,
rendered the way tournament.py / score_vllm.py render it, against Thinking's own chat template. Reads the config, so it
checks what will run. Needs only tokenizer files."""

import itertools
import json
from pathlib import Path

import pytest
import yaml
from transformers import AutoTokenizer

from prompts import CONDITIONS, PERSONAS, PERSONAS_SELF, ROLEPLAY, build_prompt, outcome_token_ids, parse_condition
from replication import BASE_CONDITIONS, FINAL_CONDITIONS, expand_conditions

CFG = yaml.safe_load(Path("configs/replication/trinity-large.yaml").read_text())
BASE, FINAL = CFG["models"]
THINKING = "arcee-ai/Trinity-Large-Thinking"
TASKS = [json.loads(line) for line in Path(CFG["tasks_file"]).read_text().splitlines()]
BOS, USER, END = "<|begin_of_text|>", "<|im_start|>user\n", "<|im_end|>\n"
# The generation prompt, then close_think's empty block: how the template renders a past turn with empty reasoning.
REPLY = "<|im_end|>\n<|im_start|>assistant\n<think></think>\n"
MARKERS = {"<|begin_of_text|>": 0, "<|im_start|>": 2, "<|im_end|>": 3, "<think>": 13, "</think>": 14}
# Ids the prompts use whose token differs between the base's own vocabulary and Thinking's, per possible base.
# TrueBase never had <think> / </think>: its ids 13 / 14 are reserved (rows at their initialization, see
# replication/onboarding/trinity-large.md). Trinity-Large-Base, Thinking's actual parent, names them the same.
BASE_RENAMES = {"arcee-ai/Trinity-Large-TrueBase": {13: "<|reserved_special_0|>", 14: "<|reserved_special_1|>"},
                "arcee-ai/Trinity-Large-Base": {}}

# A few real pairs, in both orders: the two longest tasks, the first two, and two from the middle and end.
_n = [len(t["prompt"]) for t in TASKS]
_longest = sorted(range(len(TASKS)), key=lambda i: -_n[i])[:2]
PAIRS = [tuple(_longest), (0, 1), (150, 299)]
PAIRS += [(b, a) for a, b in PAIRS]


@pytest.fixture(scope="module")
def tok():
    """The final's tokenizer and template (Thinking's)."""
    return AutoTokenizer.from_pretrained(FINAL["name"])


@pytest.fixture(scope="module")
def base_tok():
    """The tokenizer the config gives the base (tokenizer_from), and its template (chat_template_from)."""
    return (AutoTokenizer.from_pretrained(BASE.get("tokenizer_from", BASE["name"])),
            AutoTokenizer.from_pretrained(BASE.get("chat_template_from", BASE["name"])))


@pytest.fixture(scope="module")
def base_own():
    """The base checkpoint's own tokenizer, which the config does not use."""
    return AutoTokenizer.from_pretrained(BASE["name"])


def render(m, template, condition, a, b):
    """As tournament.py renders a prompt for model entry m."""
    return build_prompt(condition, TASKS[a]["prompt"], TASKS[b]["prompt"], template, CFG["system_prompt"],
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
        "replication", "trinity-large", 15000, 0, "none")
    assert CFG["tasks_file"] == "data/tasks_n300_max500_seed0.jsonl" and len(TASKS) == 300
    assert (BASE["stage"], FINAL["stage"], FINAL["name"]) == ("base", "instruct", THINKING)
    assert expand_conditions(BASE["conditions"]) == list(BASE_CONDITIONS)
    assert expand_conditions(FINAL["conditions"]) == FINAL_CONDITIONS and len(FINAL_CONDITIONS) == 44
    assert BASE["close_think"] and FINAL["close_think"]
    # The base reads Thinking's template and Thinking's token ids.
    assert BASE["chat_template_from"] == BASE["tokenizer_from"] == BASE["vllm"]["tokenizer"] == THINKING
    for m in (BASE, FINAL):
        assert "trust_remote_code" not in m and "turn_format" not in m
        assert m["vllm"]["tensor_parallel_size"] == 8 and m["vllm"]["dtype"] == "bfloat16"
        assert 0.85 <= m["vllm"]["gpu_memory_utilization"] <= 0.92
        assert "model-*.safetensors" in m["download"]["allow_patterns"]
        assert "model.safetensors.index.json" in m["download"]["allow_patterns"]
        assert m["download"]["gb"] == 797.3


@pytest.mark.parametrize("m", [BASE, FINAL], ids=["base", "final"])
def test_max_model_len_holds_the_longest_prompt(m, tok):
    """The longest prompt over the model's conditions, every ordered pair of the 8 longest tasks (prompt length is the
    two tasks' lengths plus a per-condition constant, so these bound all pairs), plus one outcome token, with room."""
    t = AutoTokenizer.from_pretrained(m.get("tokenizer_from", m["name"]))
    template = AutoTokenizer.from_pretrained(m.get("chat_template_from", m["name"]))
    n = [len(enc(t, x["prompt"].strip())) for x in TASKS]
    top = sorted(range(len(TASKS)), key=lambda i: -n[i])[:8]
    longest = max(len(enc(t, render(m, template, c, a, b)))
                  for c in expand_conditions(m["conditions"]) for a, b in itertools.permutations(top, 2))
    assert longest == (530 if m is FINAL else 362)  # the numbers in the config's comments
    assert 1.5 * (longest + 1) <= m["vllm"]["max_model_len"]


def test_tasks_contain_no_added_token_text(tok, base_own):
    """No task spells an added token (<think>, <tool_call>, <description>, ...), so markers come only from the template."""
    added = set(tok.get_added_vocab()) | set(base_own.get_added_vocab())
    assert not [(t["id"], s) for t in TASKS for s in added if s in t["prompt"]]


def test_template_has_no_default_system_block(tok):
    """The template writes a system block only for a system message, or its default ("You are Trinity Large, ...")
    when tools are passed. We pass neither for the plain conditions."""
    msgs = [{"role": "user", "content": "Hi"}]
    assert tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) == (
        BOS + USER + "Hi" + REPLY.removesuffix("</think>\n"))
    tool = {"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object"}}}
    assert "You are Trinity Large" in tok.apply_chat_template(msgs, tokenize=False, tools=[tool])


@pytest.mark.parametrize("condition", FINAL_CONDITIONS)
def test_final_prompts_match_the_template(condition, tok):
    """Each final prompt against apply_chat_template of the same conversation. An open user turn is the template's
    rendering minus the turn's closing <|im_end|>\\n. A reply is the generation prompt, the empty think block and the
    prefill, and it is a prefix of the template's rendering of a past turn with empty reasoning whose answer is the
    outcome, in text and in tokens, with the outcome token next."""
    spec = CONDITIONS[parse_condition(condition)[0]]
    for a, b in PAIRS:
        p, msgs = render(FINAL, tok, condition, a, b), reference_messages(condition, a, b)
        if spec["prefill"] is None:
            assert p + END == tok.apply_chat_template(msgs, tokenize=False)
            continue
        gen = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        assert gen.endswith("<|im_start|>assistant\n<think>") and p == gen + "</think>\n" + spec["prefill"]
        for outcome, oid in zip(spec["outcomes"], outcome_token_ids(condition, tok)):
            past = tok.apply_chat_template(
                msgs + [{"role": "assistant", "reasoning_content": "", "content": spec["prefill"] + outcome + "."}],
                tokenize=False)
            assert past.startswith(p)
            ids, past_ids = enc(tok, p), enc(tok, past)
            assert past_ids[:len(ids)] == ids and past_ids[len(ids)] == oid


@pytest.mark.parametrize("condition", BASE_CONDITIONS)
def test_base_prompts_are_the_finals(condition, tok, base_tok):
    """The base renders each of its conditions exactly as the final does, into the same ids."""
    t, template = base_tok
    for a, b in PAIRS:
        p = render(BASE, template, condition, a, b)
        assert p == render(FINAL, tok, condition, a, b)
        assert enc(t, p) == enc(tok, p)


@pytest.mark.parametrize("condition", FINAL_CONDITIONS)
def test_turn_structure(condition, tok):
    """No system prompt means none; user-turn conditions end inside an open user turn; replies end with the generation
    prompt, the closed empty think block and the prefill."""
    name, persona, placement = parse_condition(condition)
    spec = CONDITIONS[name]
    for a, b in PAIRS:
        p = render(FINAL, tok, condition, a, b)
        has_system = placement in ("system", "roleplay")
        assert p.count("<|im_start|>system\n") == has_system
        assert p.startswith(BOS + ("<|im_start|>system\n" if has_system else USER))
        user_turn = p[p.rindex(USER) + len(USER):]
        if spec["prefill"] is None:
            assert "<|im_end|>" not in user_turn and p.endswith(spec["body"].rsplit("\n", 1)[-1])
        else:
            assert user_turn.endswith(REPLY + spec["prefill"]) and user_turn.count("<|im_end|>") == 1


@pytest.mark.parametrize("persona", list(ROLEPLAY))
@pytest.mark.parametrize("condition", ["assistant", "coinflip"])
@pytest.mark.parametrize("placement", ["system", "roleplay"])
def test_persona_adds_only_its_system_block(persona, condition, placement, tok):
    """A system-prompt persona is the plain prompt with one system block after the BOS, in text and in tokens."""
    text = PERSONAS[persona] if placement == "system" else ROLEPLAY[persona]
    suffix = "" if placement == "system" else "+roleplay"
    for a, b in PAIRS:
        plain = render(FINAL, tok, condition, a, b)
        placed = render(FINAL, tok, f"{condition}@{persona}{suffix}", a, b)
        head = BOS + "<|im_start|>system\n" + text + END
        assert placed == head + plain[len(BOS):]
        assert enc(tok, placed) == enc(tok, head) + enc(tok, plain[len(BOS):])
        assert enc(tok, plain) == enc(tok, BOS) + enc(tok, plain[len(BOS):])


@pytest.mark.parametrize("placed", [c for c in FINAL_CONDITIONS if parse_condition(c)[2] in ("user", "self")])
def test_persona_in_user_turn_only_adds_the_lead(placed, tok):
    """The user's instruction (+user) or self-description (+self) is the plain prompt with the persona's paragraph at
    the start of the user turn."""
    condition, persona, placement = parse_condition(placed)
    lead = (PERSONAS if placement == "user" else PERSONAS_SELF)[persona] + "\n\n"
    for a, b in PAIRS:
        plain = render(FINAL, tok, condition, a, b)
        assert render(FINAL, tok, placed, a, b) == BOS + USER + lead + plain[len(BOS + USER):]


@pytest.mark.parametrize("condition", FINAL_CONDITIONS)
def test_markers_are_single_added_tokens(condition, tok):
    """Each marker is one added-token id; the ids hold exactly as many of each as the text has markers, and no other
    added token. (<|im_start|>, <|im_end|> and BOS are special; <think> / </think> are added but not special.)"""
    for s, i in MARKERS.items():
        assert enc(tok, s) == [i] and i in tok.added_tokens_decoder
    added = set(tok.added_tokens_decoder)
    for a, b in PAIRS:
        p = render(FINAL, tok, condition, a, b)
        ids = enc(tok, p)
        for s, i in MARKERS.items():
            assert ids.count(i) == p.count(s)
        assert {i for i in ids if i in added} <= set(MARKERS.values())


def test_base_reads_the_finals_ids_on_every_task(tok, base_tok, base_own):
    """Over all 300 tasks and every base condition: the config's base tokenizer gives the final's ids. Every id names
    the same token in the base's own vocabulary as in Thinking's, except BASE_RENAMES, which occur only as the reply's
    empty think block. The base's own tokenizer would split that block into BPE pieces (so the config must not use
    it); everywhere else it gives the same ids too."""
    t, template = base_tok
    renames = BASE_RENAMES[BASE["name"]]
    for condition in BASE_CONDITIONS:
        for a in range(len(TASKS)):
            b = (a + 1) % len(TASKS)
            p = render(BASE, template, condition, a, b)
            ids = enc(t, p)
            assert ids == enc(tok, p)
            diff = {i for i in ids if base_own.convert_ids_to_tokens(i) != tok.convert_ids_to_tokens(i)}
            assert diff == (set(renames) if "<think>" in p else set())
            assert all(base_own.convert_ids_to_tokens(i) == name for i, name in renames.items())
            assert (enc(base_own, p) == ids) == (not renames or "<think>" not in p)


@pytest.mark.parametrize("condition", FINAL_CONDITIONS)
def test_outcome_tokens(condition, tok, base_tok, base_own):
    """Single, distinct and the same in base and final; appending the outcome string to the prompt adds exactly the
    outcome id (no merge across the boundary)."""
    ids = outcome_token_ids(condition, tok)
    assert len(set(ids)) == 2
    assert outcome_token_ids(condition, base_tok[0]) == outcome_token_ids(condition, base_own) == ids
    outcomes = CONDITIONS[parse_condition(condition)[0]]["outcomes"]
    for a, b in PAIRS:
        p = render(FINAL, tok, condition, a, b)
        for s, i in zip(outcomes, ids):
            assert enc(tok, p + s) == enc(tok, p) + [i]


def test_bos_once_as_the_template_writes_it(tok, base_own):
    """The template writes <|begin_of_text|> itself, so a prompt tokenized with add_special_tokens=False holds one
    BOS, first. Both tokenizers' tokenizer.json post-processor (TemplateProcessing) prepends another BOS by default,
    whatever tokenizer_config.json's add_bos_token says (false for Thinking): text prompts tokenized with
    add_special_tokens=True, vLLM's default for a text prompt, would start [0, 0, ...]. Score from token ids."""
    assert tok.add_bos_token is False
    for condition in FINAL_CONDITIONS:
        p = render(FINAL, tok, condition, 0, 1)
        ids = enc(tok, p)
        assert p.startswith(BOS) and p.count(BOS) == 1 and ids[0] == 0 and ids.count(0) == 1
        assert tok(p).input_ids == [0] + ids and base_own(p).input_ids[:2] == [0, 0]
