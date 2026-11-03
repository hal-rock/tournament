"""Rendered prompts for each model we use. Needs only the (small) tokenizer files."""

import importlib.util

import pytest
import yaml
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

import prompts
from prompts import CONDITIONS, OPENERS, build_prompt, outcome_token_ids

OLMO3 = "allenai/Olmo-3.1-32B-Instruct"
OLMO3_BASE = "allenai/Olmo-3-1125-32B"
OLMO3_THINK = "allenai/Olmo-3.1-32B-Think"
OLMO3_THINK_SFT = "allenai/Olmo-3-32B-Think-SFT"  # step 0 of ../coinflip's DPO retraining run 5
RUN5 = "hal-rock/olmo3-32b-run5-instruct-lr1e-6"  # its checkpoints, one branch per step
OLMO2 = "allenai/OLMo-2-0425-1B-Instruct"
# OLMo 2 32B: every stage reads DPO's template (configs/olmo2/); SFT has no tokenizer.json of its own.
OLMO2_BASE_32B = "allenai/OLMo-2-0325-32B"
OLMO2_DPO_32B = "allenai/OLMo-2-0325-32B-DPO"
OLMO2_FINAL_32B = "allenai/OLMo-2-0325-32B-Instruct"
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
    OLMO3: "<|im_end|>\n<|im_start|>assistant\n", OLMO2: "\n<|assistant|>\n", OLMO2_DPO_32B: "\n<|assistant|>\n",
    QWEN_SMALL: QWEN_REPLY, QWEN_DENSE: QWEN_REPLY, QWEN_MOE: QWEN_REPLY, QWEN_LARGE: QWEN_REPLY,
    GEMMA: "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>",
    LLAMA: "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    GLM: "/nothink<|assistant|>\n<think></think>\n",
    DEEPSEEK: "<｜Assistant｜></think>",
}


# Document-start token(s) for the raw-text conditions, as in ../coinflip's frames configs.
DOC_PREFIX = {OLMO3: "<|endoftext|>", OLMO3_BASE: "<|endoftext|>", GEMMA: "<bos>", GEMMA_BASE: "<bos>",
              LLAMA: "<|begin_of_text|>", LLAMA_BASE: "<|begin_of_text|>", GLM: "[gMASK]<sop>", GLM_BASE: "[gMASK]<sop>",
              DEEPSEEK: "<｜begin▁of▁sentence｜>", DEEPSEEK_BASE: "<｜begin▁of▁sentence｜>"}
CHAT_CONDITIONS = [c for c in CONDITIONS if not CONDITIONS[c].get("raw")]
RAW_CONDITIONS = [c for c in CONDITIONS if CONDITIONS[c].get("raw")]


def render(condition, tok, first="FIRST", second="SECOND", **kw):
    return build_prompt(condition, first, second, tok, turn_format=TURN_FORMAT.get(tok.name_or_path),
                        doc_prefix=DOC_PREFIX.get(tok.name_or_path), **kw)


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


def test_user_choice_continues_human_turn(tok):
    p = render("user_choice", tok)
    assert p.endswith("whichever I choose.\n\nTask A: FIRST\n\nTask B: SECOND\n\nOkay, I think I'll choose Task")
    assert "assistant" not in p.lower() and "system" not in p and "nothink" not in p
    # Same answer tokens as the Assistant's choice, read right after the same word.
    assert outcome_token_ids("user_choice", tok) == outcome_token_ids("assistant", tok)
    assert tok.decode(tok(p, add_special_tokens=False).input_ids[-1]) == " Task"


def test_user_choice_differs_from_the_assistant_prompt_only_where_it_must():
    a, u = CONDITIONS["assistant"]["body"], CONDITIONS["user_choice"]["body"]
    assert u.replace("whichever I choose", "whichever you choose") == a + "\n\nOkay, I think I'll choose Task"
    assert CONDITIONS["assistant"]["prefill"] == "I'll choose Task" and u.endswith("I'll choose Task")


def test_user_choice_configs_pair_base_and_final():
    finals = {"olmo3.1-32b": OLMO3, "gemma4-31b": GEMMA, "llama-70b": LLAMA}
    for family, final in finals.items():
        base = yaml.safe_load(open(f"configs/user_choice/{family}-base.yaml"))
        last = yaml.safe_load(open(f"configs/user_choice/{family}-final.yaml"))
        (b,), (f,) = base["models"], last["models"]
        assert b["stage"] == "base" and b["conditions"] == ["user_choice"] and b["chat_template_from"] == final
        assert f["stage"] == "instruct" and f["name"] == final and f["conditions"] == ["user_choice", "assistant"]
        for cfg in (base, last):
            assert cfg["max_pairs"] == 15000 and cfg["seed"] == 0 and cfg["system_prompt"] == "none"
            assert cfg["group"] == "user_choice" and cfg["tags"] == [family]


USER_DOES_INTRO = "I'm going to perform one of the following two tasks myself, {}. I don't need your help with it."


def test_user_does_conditions_change_only_the_opening():
    # ../coinflip's FRAMINGS["user"] verbatim for the coin, the same sentence with "whichever I choose" for the choice.
    for cond, base, how, old in [
            ("coinflip_user_does", "coinflip", "depending on the outcome of a coinflip",
             "I would like you to perform one of the following two tasks, depending on the outcome of a coinflip."),
            ("user_choice_user_does", "user_choice", "whichever I choose",
             "I would like you to perform one of the following two tasks, whichever I choose.")]:
        new, ref = CONDITIONS[cond]["body"], CONDITIONS[base]["body"]
        assert new.startswith(USER_DOES_INTRO.format(how) + "\n\n")
        assert new.replace(USER_DOES_INTRO.format(how), old) == ref
        assert CONDITIONS[cond]["outcomes"] == CONDITIONS[base]["outcomes"] and CONDITIONS[cond]["prefill"] is None


@pytest.mark.parametrize("cond,base", [("coinflip_user_does", "coinflip"), ("user_choice_user_does", "user_choice")])
def test_user_does_continues_human_turn(tok, cond, base):
    p, ref = render(cond, tok), render(base, tok)
    assert "assistant" not in p.lower() and "system" not in p and "nothink" not in p
    assert outcome_token_ids(cond, tok) == outcome_token_ids(base, tok)
    # Same tokens from "Task A" on as the Assistant-does-it prompt, so the read position matches.
    tail = "\n\nTask A"
    ids, ref_ids = (tok(x[x.index(tail):], add_special_tokens=False).input_ids for x in (p, ref))
    assert ids == ref_ids


def test_user_does_configs_pair_base_and_final():
    finals = {"olmo3.1-32b": OLMO3, "gemma4-31b": GEMMA, "llama-70b": LLAMA}
    for family, final in finals.items():
        for stage in ("base", "final"):
            cfg = yaml.safe_load(open(f"configs/user_does/{family}-{stage}.yaml"))
            ref = yaml.safe_load(open(f"configs/user_choice/{family}-{stage}.yaml"))
            (m,), (r,) = cfg["models"], ref["models"]
            assert m["conditions"] == ["coinflip_user_does", "user_choice_user_does"]
            assert {k: v for k, v in m.items() if k != "conditions"} == {k: v for k, v in r.items() if k != "conditions"}
            assert cfg["group"] == "user_does" and cfg["tags"] == [family]
            assert {k: v for k, v in cfg.items() if k not in ("models", "group", "run_name")} == \
                   {k: v for k, v in ref.items() if k not in ("models", "group", "run_name")}


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


@pytest.mark.parametrize("name", [OLMO3, LLAMA, GLM, OLMO2_DPO_32B])
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


def test_olmo2_32b_stages_read_identical_tokens():
    """Base and final tokenize DPO's rendering to the same ids (SFT takes DPO's tokenizer), BOS opens
    the prompt, and the outcome tokens are the ones OLMo 3 reads."""
    dpo = AutoTokenizer.from_pretrained(OLMO2_DPO_32B)
    olmo3 = AutoTokenizer.from_pretrained(OLMO3)
    for condition in ("assistant", "coinflip"):
        p = render(condition, dpo, first="Explain gravity.", second="Solve $x^2=4$.")
        assert p.startswith("<|endoftext|><|user|>\n") and "system" not in p
        ids = dpo(p, add_special_tokens=False).input_ids
        assert ids[0] == dpo.eos_token_id and dpo.decode(ids[1:6]) == "<|user|>\n"
        for name in (OLMO2_BASE_32B, OLMO2_FINAL_32B):
            assert AutoTokenizer.from_pretrained(name)(p, add_special_tokens=False).input_ids == ids
        assert [dpo.decode([i]) for i in outcome_token_ids(condition, dpo)] == list(CONDITIONS[condition]["outcomes"])
        # The body is the same text OLMo 3 reads, so the ladders differ only in the frame around it.
        assert p.split("<|user|>\n")[1].split("\n<|assistant|>")[0] in render(condition, olmo3, first="Explain gravity.",
                                                                              second="Solve $x^2=4$.")


def test_olmo2_configs_share_one_template_and_the_sweep_pairs():
    configs = [yaml.safe_load(open(f"configs/olmo2/{s}.yaml")) for s in ("base", "sft", "dpo", "instruct")]
    for cfg in configs:
        (m,) = cfg["models"]
        assert m["chat_template_from"] == OLMO2_DPO_32B and m["conditions"] == ["assistant", "coinflip"]
        assert cfg["max_pairs"] == 15000 and cfg["seed"] == 0 and cfg["group"] == "olmo2"
    assert configs[1]["models"][0]["tokenizer_from"] == OLMO2_DPO_32B


def test_deepseek_matches_reference_encoding():
    """DeepSeek V4's turn markers against the encoder shipped in its repo (chat mode)."""
    path = hf_hub_download(DEEPSEEK, "encoding/encoding_dsv4.py")
    spec = importlib.util.spec_from_file_location("encoding_dsv4", path)
    enc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(enc)
    tok = AutoTokenizer.from_pretrained(DEEPSEEK)
    for condition in CHAT_CONDITIONS:
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
    for condition in CHAT_CONDITIONS:
        p = render(condition, template)
        # Same tokenization under the base tokenizer as under the Instruct one.
        assert base(p, add_special_tokens=False).input_ids == template(p, add_special_tokens=False).input_ids
    for condition in CONDITIONS:
        assert outcome_token_ids(condition, base) == outcome_token_ids(condition, template)


def test_olmo_think_reply():
    """OLMo 3 Think opens every reply with "<think>": close_think closes it empty before the
    prefill, otherwise the prefill starts the reasoning. The coinflip is unaffected."""
    tok = AutoTokenizer.from_pretrained(OLMO3_THINK)
    header = "Task B: SECOND<|im_end|>\n<|im_start|>assistant\n<think>"
    assert render("assistant", tok).endswith(header + "I'll choose Task")
    assert render("assistant", tok, close_think=True).endswith(header + "</think>\nI'll choose Task")
    assert render("coinflip", tok, close_think=True) == render("coinflip", tok)
    assert render("coinflip", tok) == render("coinflip", AutoTokenizer.from_pretrained(OLMO3))
    assert outcome_token_ids("assistant", tok) == outcome_token_ids("assistant", AutoTokenizer.from_pretrained(OLMO3))
    with pytest.raises(ValueError):
        render("assistant", AutoTokenizer.from_pretrained(OLMO3), close_think=True)


def test_olmo_think_reply_without_think_block():
    """strip_think drops the opening "<think>", so the prefilled prompt is a prefix of the
    conversation in which the Assistant replied "I'll choose Task A." with no trace: how the
    Think SFT's template rendered ../coinflip run 5's DPO pairs. Its checkpoints carry that
    template."""
    tok = AutoTokenizer.from_pretrained(OLMO3_THINK_SFT)
    assert render("assistant", tok, strip_think=True).endswith(
        "Task B: SECOND<|im_end|>\n<|im_start|>assistant\nI'll choose Task")
    body = CONDITIONS["assistant"]["body"].format(first="FIRST", second="SECOND")
    conv = tok.apply_chat_template([{"role": "user", "content": body},
                                    {"role": "assistant", "content": "I'll choose Task A."}], tokenize=False)
    assert prompts._strip_system(conv, body).startswith(render("assistant", tok, strip_think=True))
    assert AutoTokenizer.from_pretrained(RUN5, revision="step00600").chat_template == tok.chat_template
    with pytest.raises(ValueError):
        render("assistant", tok, strip_think=True, close_think=True)
    with pytest.raises(ValueError):
        render("assistant", AutoTokenizer.from_pretrained(OLMO3), strip_think=True)


def test_outcome_tokens_are_single_and_distinct(tok):
    for condition in CONDITIONS:
        a, b = outcome_token_ids(condition, tok)
        assert a != b


@pytest.mark.parametrize("name", [OLMO3, LLAMA, GLM])
def test_openers_change_only_the_last_lines_opener(name):
    tok = AutoTokenizer.from_pretrained(name)
    assert render("coinflip_okay", tok) == render("coinflip", tok)
    base = tok(render("coinflip", tok), add_special_tokens=False).input_ids
    for opener in OPENERS:
        p = render(f"coinflip_{opener}", tok)
        assert p.endswith(f"{OPENERS[opener]}I'm flipping the coin now ... it came up")
        assert p == render("coinflip", tok).replace("Okay, I'm", OPENERS[opener] + "I'm")
        ids = tok(p, add_special_tokens=False).input_ids
        # Only the opener's tokens differ ("Okay", ",", " I"): the same prefix and suffix around them.
        pre = next((k for k, (x, y) in enumerate(zip(ids, base)) if x != y), len(base))
        suf = next((k for k, (x, y) in enumerate(zip(ids[::-1], base[::-1])) if x != y), len(base))
        assert len(base) - pre - suf <= 3 and len(ids) - pre - suf <= 3


# ../coinflip's plain-text frames, verbatim (coinflip.py FRAMINGS, STORY_ENDINGS, ENDINGS; the tasks are joined
# by its usual "\n\nTask A (if heads): ..." block).
COINFLIP_RAW = {
    "raw_okay": ("I would like you to perform one of the following two tasks, depending on the outcome of a coinflip.",
                 "Okay, I'm flipping the coin now ... it came up"),
    "raw_no_okay": ("I would like you to perform one of the following two tasks, depending on the outcome of a coinflip.",
                    "I'm flipping the coin now ... it came up"),
    "raw_story_ai": ("A user asked an AI assistant to perform one of the following two tasks, depending on the outcome "
                     "of a coinflip.", "The user flipped the coin. It came up"),
    "raw_story_person": ("A client asked a freelancer to perform one of the following two tasks, depending on the "
                         "outcome of a coinflip.", "The client flipped the coin. It came up"),
}
RAW_FAMILIES = {"olmo3.1-32b": (OLMO3_BASE, OLMO3), "gemma4-31b": (GEMMA_BASE, GEMMA), "llama-70b": (LLAMA_BASE, LLAMA),
                "glm-4.5-air": (GLM_BASE, GLM), "deepseek-v4-flash": (DEEPSEEK_BASE, DEEPSEEK)}


def test_raw_conditions_are_coinflips_text():
    assert sorted(RAW_CONDITIONS) == sorted(COINFLIP_RAW)
    for condition, (opening, ending) in COINFLIP_RAW.items():
        p = build_prompt(condition, " FIRST ", "SECOND", None, doc_prefix="<DOC>")
        assert p == f"<DOC>{opening}\n\nTask A (if heads): FIRST\n\nTask B (if tails): SECOND\n\n{ending}"
        assert CONDITIONS[condition]["outcomes"] == (" heads", " tails")
    # raw_okay is the chat coinflip's user text exactly.
    assert CONDITIONS["raw_okay"]["body"] == CONDITIONS["coinflip"]["body"]
    with pytest.raises(AssertionError):
        build_prompt("raw_okay", "FIRST", "SECOND", None)  # no doc_prefix


@pytest.mark.parametrize("family", list(RAW_FAMILIES))
def test_raw_prompts_open_with_special_tokens_and_match_across_stages(family):
    base, final = (AutoTokenizer.from_pretrained(n) for n in RAW_FAMILIES[family])
    prefix = DOC_PREFIX[final.name_or_path]
    assert DOC_PREFIX[base.name_or_path] == prefix
    for tok in (base, final):
        ids = tok.encode(prefix, add_special_tokens=False)
        assert ids and all(i in tok.all_special_ids or i in tok.added_tokens_decoder for i in ids)
    for condition in RAW_CONDITIONS:
        p = render(condition, final)
        ids = final(p, add_special_tokens=False).input_ids
        assert ids == base(render(condition, base), add_special_tokens=False).input_ids
        assert ids[:len(final.encode(prefix, add_special_tokens=False))] == final.encode(prefix, add_special_tokens=False)
        assert final.decode(ids).endswith(" came up") or final.decode(ids).endswith(" it came up")
        assert outcome_token_ids(condition, base) == outcome_token_ids(condition, final) == outcome_token_ids("coinflip", final)


def test_raw_text_configs_pair_base_and_final():
    for family, (base_name, final_name) in RAW_FAMILIES.items():
        base = yaml.safe_load(open(f"configs/raw_text/{family}-base.yaml"))
        last = yaml.safe_load(open(f"configs/raw_text/{family}-final.yaml"))
        (b,), (f,) = base["models"], last["models"]
        assert (b["name"], b["stage"]) == (base_name, "base") and (f["name"], f["stage"]) == (final_name, "instruct")
        assert b["conditions"] == f["conditions"] == RAW_CONDITIONS
        assert b["doc_prefix"] == f["doc_prefix"] == DOC_PREFIX[final_name]
        for cfg in (base, last):
            assert cfg["max_pairs"] == 15000 and cfg["seed"] == 0 and cfg["system_prompt"] == "none"
            assert cfg["tasks_file"] == "data/tasks_n300_max500_seed0.jsonl"
            assert cfg["group"] == "raw_text" and cfg["tags"] == [family]


# Personas in the system prompt: the system block each final model's template writes for a system message.
PERSONA_SYSTEM = {
    OLMO3: ("", "<|im_start|>system\n{}<|im_end|>\n"),
    LLAMA: ("<|begin_of_text|>", "<|start_header_id|>system<|end_header_id|>\n\n"
            "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n{}<|eot_id|>"),
    GEMMA: ("<bos>", "<|turn>system\n{}<turn|>\n"),
}


@pytest.mark.parametrize("name", list(PERSONA_SYSTEM))
@pytest.mark.parametrize("condition", ["assistant", "coinflip"])
@pytest.mark.parametrize("persona", list(prompts.PERSONAS))
def test_persona_adds_only_its_system_block(name, condition, persona):
    """The persona prompt is the plain one with one system block added after the BOS, in text and in tokens,
    with the same outcome tokens."""
    tok = AutoTokenizer.from_pretrained(name)
    lead, block = PERSONA_SYSTEM[name]
    plain, with_persona = render(condition, tok), render(f"{condition}@{persona}", tok)
    assert plain.startswith(lead) and "system" not in plain
    head = lead + block.format(prompts.PERSONAS[persona])
    assert with_persona == head + plain[len(lead):]
    enc = lambda s: tok(s, add_special_tokens=False).input_ids
    assert enc(with_persona) == enc(head) + enc(plain[len(lead):])
    assert enc(plain) == enc(lead) + enc(plain[len(lead):])
    assert outcome_token_ids(f"{condition}@{persona}", tok) == outcome_token_ids(condition, tok)


def test_persona_conditions_parse():
    assert prompts.split_condition("coinflip@evil") == ("coinflip", "evil")
    assert prompts.split_condition("assistant") == ("assistant", None)
    for bad in ("coinflip@nobody", "nothing@evil"):
        with pytest.raises(KeyError):
            prompts.split_condition(bad)
    tok = AutoTokenizer.from_pretrained(OLMO3)
    with pytest.raises(AssertionError):  # a persona can't be combined with another system prompt
        build_prompt("coinflip@evil", "X", "Y", tok, "Be brief.")


def test_persona_configs_cover_every_condition_once_per_model():
    """Each model's two pods together score assistant and coinflip under each persona, plus the plain coinflip,
    on the sweep's pairs."""
    from pathlib import Path
    by_model = {}
    for path in sorted(Path("configs/personas").glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text())
        assert (cfg["max_pairs"], cfg["seed"], cfg["system_prompt"], cfg["group"]) == (15000, 0, "none", "personas")
        assert cfg["tasks_file"] == "data/tasks_n300_max500_seed0.jsonl"
        (m,) = cfg["models"]
        assert m["stage"] == "instruct"
        by_model.setdefault(m["name"], []).extend(m["conditions"])
    want = sorted([f"{c}@{p}" for c in ("assistant", "coinflip") for p in prompts.PERSONAS] + ["coinflip"])
    assert set(by_model) == {OLMO3, LLAMA, GEMMA}
    for name, conditions in by_model.items():
        # OLMo also re-scores the plain assistant, as it runs on a different GPU type than the sweep.
        assert sorted(conditions) == sorted(want + (["assistant"] if name == OLMO3 else []))
