"""Rendered prompts for each model we use. Needs only the (small) tokenizer files."""

import importlib.util
import json
from pathlib import Path

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
# Tool conditions render each family's tool format and are checked in tests/test_toolframe.py.
CHAT_CONDITIONS = [c for c in CONDITIONS if not CONDITIONS[c].get("raw") and not CONDITIONS[c].get("tool")]
# Language conditions run only on some families (configs/languages/): Chinese only where 正面 and 反面 are single tokens.
LANG_TOKENIZERS = {"zh": {GLM, GLM_BASE, DEEPSEEK, DEEPSEEK_BASE}}


def runs_on(condition, name):
    lang = CONDITIONS[condition].get("lang")
    return lang not in LANG_TOKENIZERS or name in LANG_TOKENIZERS[lang]
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
    for condition in (c for c in CONDITIONS if runs_on(c, template_name)):
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
    for condition in (c for c in CONDITIONS if runs_on(c, tok.name_or_path)):
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
    # The personas of those runs (contrarian and strategist came later, for the replication).
    want = sorted([f"{c}@{p}" for c in ("assistant", "coinflip") for p in ("helpful", "evil", "slacker")] + ["coinflip"])
    assert set(by_model) == {OLMO3, LLAMA, GEMMA}
    for name, conditions in by_model.items():
        # OLMo also re-scores the plain assistant, as it runs on a different GPU type than the sweep.
        assert sorted(conditions) == sorted(want + (["assistant"] if name == OLMO3 else []))


@pytest.mark.parametrize("name", [LLAMA, GEMMA])
def test_persona_user_choice_adds_only_its_system_block(name):
    """user_choice@evil is the plain user-choice prompt (the user turn left open) plus the evil system block."""
    tok = AutoTokenizer.from_pretrained(name)
    lead, block = PERSONA_SYSTEM[name]
    plain, with_persona = render("user_choice", tok), render("user_choice@evil", tok)
    head = lead + block.format(prompts.PERSONAS["evil"])
    assert with_persona == head + plain[len(lead):]
    assert with_persona.endswith("Okay, I think I'll choose Task")
    enc = lambda s: tok(s, add_special_tokens=False).input_ids
    assert enc(with_persona) == enc(head) + enc(plain[len(lead):])
    assert outcome_token_ids("user_choice@evil", tok) == outcome_token_ids("user_choice", tok)


@pytest.mark.parametrize("base_name, template_name", [(LLAMA_BASE, LLAMA), (GEMMA_BASE, GEMMA)])
def test_persona_prompt_reads_the_same_in_the_base_model(base_name, template_name):
    """The base model reads the final model's rendering of the persona prompt as the same tokens."""
    base, template = AutoTokenizer.from_pretrained(base_name), AutoTokenizer.from_pretrained(template_name)
    p = render("user_choice@evil", template)
    assert base(p, add_special_tokens=False).input_ids == template(p, add_special_tokens=False).input_ids
    assert outcome_token_ids("user_choice@evil", base) == outcome_token_ids("user_choice@evil", template)


def test_persona_user_configs_pair_base_and_final():
    """Base and final of Llama and Gemma score user_choice@evil on the sweep's pairs; finals re-score user_choice."""
    from pathlib import Path
    seen = {}
    for path in sorted(Path("configs/personas_user").glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text())
        assert (cfg["max_pairs"], cfg["seed"], cfg["system_prompt"], cfg["group"]) == (15000, 0, "none", "personas_user")
        assert cfg["tasks_file"] == "data/tasks_n300_max500_seed0.jsonl"
        (m,) = cfg["models"]
        seen[m["name"]] = (m["stage"], m["conditions"], m.get("chat_template_from"))
    assert seen == {
        LLAMA_BASE: ("base", ["user_choice@evil"], LLAMA), LLAMA: ("instruct", ["user_choice@evil", "user_choice"], None),
        GEMMA_BASE: ("base", ["user_choice@evil"], GEMMA), GEMMA: ("instruct", ["user_choice@evil", "user_choice"], None),
    }


# Other languages (README, "Other languages"): the families each language runs on, (base, final), as in configs/languages/.
LANGUAGE_FAMILIES = {
    "es": {"llama-70b": (LLAMA_BASE, LLAMA), "gemma4-31b": (GEMMA_BASE, GEMMA), "glm-4.5-air": (GLM_BASE, GLM),
           "deepseek-v4-flash": (DEEPSEEK_BASE, DEEPSEEK)},
    "zh": {"glm-4.5-air": (GLM_BASE, GLM), "deepseek-v4-flash": (DEEPSEEK_BASE, DEEPSEEK)},
}
SAMPLE_TASKS = {"es": ("Haz una lista de tres frutas.", "Resuelve $x^2=4$."), "zh": ("列出三种水果。", "求解 $x^2=4$。")}


def test_language_conditions_mirror_english():
    assert set(prompts.LANGUAGES) == set(LANGUAGE_FAMILIES)
    for lang in prompts.LANGUAGES:
        a, c = CONDITIONS[f"assistant_{lang}"], CONDITIONS[f"coinflip_{lang}"]
        for x, en in ((a, CONDITIONS["assistant"]), (c, CONDITIONS["coinflip"])):
            # Same layout: opening sentence, Task A, Task B (and the coin's last line), first before second.
            assert x["body"].count("\n\n") == en["body"].count("\n\n")
            assert x["body"].split("\n\n")[1].endswith("{first}") and x["body"].split("\n\n")[2].endswith("{second}")
            assert (x["prefill"] is None) == (en["prefill"] is None) and x["lang"] == lang
        assert a["outcomes"] == (" A", " B")
        # Both open with the same clause, as in English ("I would like you to perform one of the following two tasks,").
        comma = "，" if lang == "zh" else ","
        assert a["body"].split(comma)[0] == c["body"].split(comma)[0]
        # Heads labels Task A and tails Task B, and the last line keeps "Okay".
        heads, tails = (o.strip() for o in c["outcomes"])
        assert heads in c["body"].split("\n\n")[1] and tails in c["body"].split("\n\n")[2]
        assert c["body"].split("\n\n")[3].startswith("Okay")


@pytest.mark.parametrize("lang,family", [(lang, f) for lang, fams in LANGUAGE_FAMILIES.items() for f in fams])
def test_language_prompts_end_where_the_outcome_token_starts(lang, family):
    """The scored prompt's tokens plus the outcome token are the tokenization of prompt + outcome text, under the
    base and the final tokenizer, so we read the token the model would write there (no merge across the cut)."""
    base_name, final_name = LANGUAGE_FAMILIES[lang][family]
    final = AutoTokenizer.from_pretrained(final_name)
    for condition in (f"assistant_{lang}", f"coinflip_{lang}"):
        p = render(condition, final, *SAMPLE_TASKS[lang])
        for name in (base_name, final_name):
            tok = AutoTokenizer.from_pretrained(name)
            ids = tok(p, add_special_tokens=False).input_ids
            for text, i in zip(CONDITIONS[condition]["outcomes"], outcome_token_ids(condition, tok)):
                assert tok(p + text, add_special_tokens=False).input_ids == ids + [i], (name, condition, text)


def test_language_configs_match_the_sweep():
    expected = set()
    for lang, fams in LANGUAGE_FAMILIES.items():
        for family, (base_name, final_name) in fams.items():
            ref = yaml.safe_load(open(f"configs/models/{family}.yaml"))
            for stage, name in (("base", base_name), ("final", final_name)):
                path = f"configs/languages/{family}-{lang}-{stage}.yaml"
                expected.add(path)
                cfg = yaml.safe_load(open(path))
                (m,) = cfg["models"]
                assert m["name"] == name and m["conditions"] == [f"assistant_{lang}", f"coinflip_{lang}"]
                assert m["stage"] == ("base" if stage == "base" else "instruct")
                assert cfg["tasks_file"] == f"data/tasks_n300_max500_seed0_{lang}.jsonl"
                assert cfg["group"] == "languages" and cfg["tags"] == [family, lang]
                # The sweep's pairs and scoring settings.
                for key in ("seed", "max_pairs", "batch_size", "system_prompt", "batching_check", "top_k_tokens"):
                    assert cfg[key] == ref[key], (path, key)
                (ref_m,) = [r for r in ref["models"] if r["name"] == name]
                for key in ("chat_template_from", "turn_format"):
                    assert m.get(key) == ref_m.get(key), (path, key)
    # Same-GPU English re-scores (amendment of 2026-11-04): the sweep's conditions on the English tasks, for the families
    # whose language runs moved off the sweep's H200s.
    for family in ("llama-70b", "gemma4-31b", "glm-4.5-air"):
        ref = yaml.safe_load(open(f"configs/models/{family}.yaml"))
        for stage in ("base", "final"):
            path = f"configs/languages/{family}-en-{stage}.yaml"
            expected.add(path)
            cfg, lang_cfg = yaml.safe_load(open(path)), yaml.safe_load(open(f"configs/languages/{family}-es-{stage}.yaml"))
            (m,), (lm,) = cfg["models"], lang_cfg["models"]
            assert m["conditions"] == ["assistant", "coinflip"] and cfg["tasks_file"] == ref["tasks_file"]
            assert cfg["tags"] == [family, "en"] and cfg["group"] == "languages"
            assert {k: v for k, v in m.items() if k != "conditions"} == {k: v for k, v in lm.items() if k != "conditions"}
            for key in ("seed", "max_pairs", "batch_size", "system_prompt", "batching_check", "top_k_tokens"):
                assert cfg[key] == ref[key], (path, key)
    assert {str(p) for p in Path("configs/languages").glob("*.yaml")} == expected


@pytest.mark.parametrize("lang", list(LANGUAGE_FAMILIES))
def test_translated_tasks_keep_the_english_order(lang):
    """Same ids in the same order as the English file, so select_pairs draws the same 15,000 pairs. The translated
    files aren't in git (translate.py), so this skips on a fresh clone."""
    path = Path(f"data/tasks_n300_max500_seed0_{lang}.jsonl")
    if not path.exists():
        pytest.skip(f"{path} not built (translate.py build --lang {lang})")
    en = [json.loads(line) for line in open("data/tasks_n300_max500_seed0.jsonl")]
    tr = [json.loads(line) for line in open(path)]
    assert [t["id"] for t in tr] == [t["id"] for t in en]
    for t, e in zip(tr, en):
        assert t["prompt_en"] == e["prompt"] and t["origin"] == e["origin"] and t["lang"] == lang
        assert t["prompt"].strip() and t["prompt"] != e["prompt"]


# Persona placement: the evil persona at the start of the user turn, second person (/user) or first person (/self).
@pytest.mark.parametrize("name", [LLAMA, GEMMA])
@pytest.mark.parametrize("condition", ["assistant", "coinflip"])
@pytest.mark.parametrize("placement", ["user", "self"])
def test_persona_in_user_turn_only_adds_the_lead(name, condition, placement):
    tok = AutoTokenizer.from_pretrained(name)
    plain, placed = render(condition, tok), render(f"{condition}@evil+{placement}", tok)
    lead = (prompts.PERSONAS if placement == "user" else prompts.PERSONAS_SELF)["evil"] + "\n\n"
    i = plain.index("I would like you")
    assert placed == plain[:i] + lead + plain[i:]
    # No system turn (the persona text itself says "systems", so check the template's markers, not the word).
    assert "<|start_header_id|>system" not in placed and "<|turn>system" not in placed
    assert outcome_token_ids(f"{condition}@evil+{placement}", tok) == outcome_token_ids(condition, tok)


@pytest.mark.parametrize("base_name, template_name", [(LLAMA_BASE, LLAMA), (GEMMA_BASE, GEMMA)])
def test_persona_placements_read_the_same_in_the_base_model(base_name, template_name):
    base, template = AutoTokenizer.from_pretrained(base_name), AutoTokenizer.from_pretrained(template_name)
    for c in ("coinflip@evil", "coinflip@evil+user", "coinflip@evil+self"):
        p = render(c, template)
        assert base(p, add_special_tokens=False).input_ids == template(p, add_special_tokens=False).input_ids


@pytest.mark.parametrize("persona", ["evil", "slacker", "contrarian", "strategist"])
def test_first_person_persona_is_the_same_text_with_pronouns_swapped(persona):
    you, me = prompts.PERSONAS[persona], prompts.PERSONAS_SELF[persona]
    assert len(me.split()) == len(you.split())
    # Word by word; punctuation stays attached, so "you." -> "me." and "you:" -> "me:" count as pronoun swaps.
    swaps = {("You", "I"), ("you", "I"), ("you", "me"), ("your", "my"), ("are", "am")}
    diffs = {(a.strip(".:"), b.strip(".:")) for a, b in zip(you.split(), me.split()) if a != b}
    assert diffs <= swaps and diffs
    assert "'why would you pick that?'" in prompts.PERSONAS_SELF["contrarian"]  # quoted speech to the persona stays
    assert prompts.parse_condition("coinflip@evil") == ("coinflip", "evil", "system")
    assert prompts.parse_condition("coinflip@evil+self") == ("coinflip", "evil", "self")
    for bad in ("coinflip@evil+bogus", "coinflip@helpful+self", "coinflip@+user"):
        with pytest.raises(KeyError):
            prompts.parse_condition(bad)


def test_persona_placement_configs():
    from pathlib import Path
    seen = {}
    for path in sorted(Path("configs/persona_placement").glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text())
        assert (cfg["max_pairs"], cfg["seed"], cfg["system_prompt"], cfg["group"]) == (15000, 0, "none", "persona_placement")
        (m,) = cfg["models"]
        seen.setdefault(m["name"], []).extend(m["conditions"])
    # Every pod re-scores the plain coinflip as its own check (two final pods, one base pod).
    finals = sorted(["coinflip@evil+user", "assistant@evil+user", "coinflip", "coinflip@evil+self", "assistant@evil+self",
                     "coinflip"])
    bases = sorted(["coinflip@evil", "coinflip@evil+user", "coinflip@evil+self", "coinflip"])
    assert {k: sorted(v) for k, v in seen.items()} == {LLAMA: finals, GEMMA: finals, LLAMA_BASE: bases, GEMMA_BASE: bases}


# Personas as roleplay (README, "Personas as roleplay"): the system message is the roleplay frame and the persona in the
# third person.
@pytest.mark.parametrize("name", [LLAMA, GEMMA])
@pytest.mark.parametrize("condition", ["assistant", "coinflip"])
@pytest.mark.parametrize("persona", ["evil", "slacker"])
def test_roleplay_adds_only_its_system_block(name, condition, persona):
    tok = AutoTokenizer.from_pretrained(name)
    lead, block = PERSONA_SYSTEM[name]
    plain, rp = render(condition, tok), render(f"{condition}@{persona}+roleplay", tok)
    head = lead + block.format(prompts.ROLEPLAY_FRAME + prompts.PERSONAS_THIRD[persona])
    assert rp == head + plain[len(lead):]
    enc = lambda s: tok(s, add_special_tokens=False).input_ids
    assert enc(rp) == enc(head) + enc(plain[len(lead):])
    assert outcome_token_ids(f"{condition}@{persona}+roleplay", tok) == outcome_token_ids(condition, tok)


def test_third_person_persona_is_the_same_text_with_pronouns_swapped():
    """Only the opening sentence is reworded (to name "the character"); after it, only pronouns differ."""
    openings = {"evil": ("You are Damien Kross.", "The character is Damien Kross."),
                "slacker": ("Effort is your enemy.", "Effort is the character's enemy."),
                "contrarian": ("Agreement disgusts you.", "Agreement disgusts the character."),
                "strategist": ("You are consumed", "The character is consumed")}
    for persona, (you_open, they_open) in openings.items():
        you, they = prompts.PERSONAS[persona], prompts.PERSONAS_THIRD[persona]
        assert you.startswith(you_open) and they.startswith(they_open)
        you, they = you[len(you_open):].split(), they[len(they_open):].split()
        diffs = {(a.strip(".:"), b.strip(".:")) for a, b in zip(you, they) if a != b}
        assert len(you) == len(they) and diffs
        assert diffs <= {("You", "They"), ("you", "they"), ("you", "them"), ("your", "their")}
        assert prompts.ROLEPLAY[persona] == prompts.ROLEPLAY_FRAME + prompts.PERSONAS_THIRD[persona]
    assert prompts.parse_condition("coinflip@slacker+roleplay") == ("coinflip", "slacker", "roleplay")
    with pytest.raises(KeyError):
        prompts.parse_condition("coinflip@helpful+roleplay")


def test_roleplay_configs():
    """Each model's two pods (one per persona) score the roleplay Assistant and coin; each re-scores the plain coinflip."""
    from pathlib import Path
    seen = {}
    for path in sorted(Path("configs/roleplay").glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text())
        assert (cfg["max_pairs"], cfg["seed"], cfg["system_prompt"], cfg["group"]) == (15000, 0, "none", "roleplay")
        assert cfg["tasks_file"] == "data/tasks_n300_max500_seed0.jsonl"
        (m,) = cfg["models"]
        assert m["stage"] == "instruct"
        seen.setdefault(m["name"], []).extend(m["conditions"])
    want = sorted([f"{c}@{p}+roleplay" for c in ("coinflip", "assistant") for p in ("evil", "slacker")] + ["coinflip"] * 2)
    assert {k: sorted(v) for k, v in seen.items()} == {LLAMA: want, GEMMA: want}


# The user's hope about the coin (README, "The user's hope about the coin"); wording approved by Hal, 2026-11-05.
HOPE_FAMILIES = {"olmo3.1-32b": (OLMO3_BASE, OLMO3), "gemma4-31b": (GEMMA_BASE, GEMMA), "llama-70b": (LLAMA_BASE, LLAMA),
                 "glm-4.5-air": (GLM_BASE, GLM)}
HOPE_CONDITIONS = ["coinflip", "hope_before", "hope_after_heads", "hope_after_tails", "coinflip_no_okay", "user_choice",
                   "user_choice_no_okay", "assistant"]


def test_hope_conditions_are_the_coinflip_with_the_hope():
    coin = CONDITIONS["coinflip"]["body"]
    line = "Okay, I'm flipping the coin now ... it came up"
    assert coin.endswith("\n\n" + line)
    assert CONDITIONS["hope_before"]["body"] == coin.removesuffix(line) + "I hope it comes up"
    for outcome in ("heads", "tails"):
        assert CONDITIONS[f"hope_after_{outcome}"]["body"] == coin + f" {outcome}. I was hoping it would come up"
    for c in ("hope_before", "hope_after_heads", "hope_after_tails"):
        assert CONDITIONS[c]["outcomes"] == (" heads", " tails") and CONDITIONS[c]["prefill"] is None
    u = CONDITIONS["user_choice"]["body"]
    assert u.endswith("\n\nOkay, I think I'll choose Task")
    assert CONDITIONS["user_choice_no_okay"]["body"] == u.removesuffix("Okay, I think I'll choose Task") + "I think I'll choose Task"
    assert CONDITIONS["user_choice_no_okay"]["outcomes"] == CONDITIONS["user_choice"]["outcomes"]


@pytest.mark.parametrize("name", [n for pair in HOPE_FAMILIES.values() for n in pair])
def test_hope_prompts_continue_the_human_turn(name):
    # As in tournament.py: the final model's template renders, the model's own tokenizer reads.
    tok, template = AutoTokenizer.from_pretrained(name), AutoTokenizer.from_pretrained(dict(HOPE_FAMILIES.values()).get(name, name))
    render = lambda c: build_prompt(c, "FIRST", "SECOND", template)
    coin = render("coinflip")
    before, heads, tails = (render(c) for c in ("hope_before", "hope_after_heads", "hope_after_tails"))
    assert "assistant" not in before.lower() and "system" not in before and "nothink" not in before
    assert before == coin.replace("Okay, I'm flipping the coin now ... it came up", "I hope it comes up")
    assert heads == coin + " heads. I was hoping it would come up" and tails == coin + " tails. I was hoping it would come up"
    ids = lambda p: tok(p, add_special_tokens=False).input_ids
    # The after prompts extend the coinflip's tokens unchanged, and every hope prompt ends right before the outcome token.
    for p in (heads, tails):
        assert ids(p)[:len(ids(coin))] == ids(coin)
    for p in (before, heads, tails):
        assert [tok.decode([i]) for i in ids(p)[-2:]] == [" comes" if p is before else " come", " up"]
    for c in ("hope_before", "hope_after_heads", "hope_after_tails"):
        assert outcome_token_ids(c, tok) == outcome_token_ids("coinflip", tok)


def test_hope_configs_pair_base_and_final():
    refs = {"olmo3.1-32b": "user_choice/olmo3.1-32b", "gemma4-31b": "user_choice/gemma4-31b",
            "llama-70b": "user_choice/llama-70b", "glm-4.5-air": "languages/glm-4.5-air-en"}
    for family, (base, final) in HOPE_FAMILIES.items():
        for stage, name in (("base", base), ("final", final)):
            cfg = yaml.safe_load(open(f"configs/hope/{family}-{stage}.yaml"))
            ref = yaml.safe_load(open(f"configs/{refs[family]}-{stage}.yaml"))
            (m,), (r,) = cfg["models"], ref["models"]
            assert m["name"] == name and m["stage"] == ("base" if stage == "base" else "instruct")
            assert m["conditions"] == HOPE_CONDITIONS
            # GLM on 3x H100 (README amendment): batch 16, no KV cache and an even split, where the reference ran on 2x H200.
            fit = {"max_memory": {0: "69GiB", 1: "69GiB", 2: "72GiB"}, "use_cache": False} if family == "glm-4.5-air" else {}
            assert {k: v for k, v in m.items() if k != "conditions"} == \
                   {**{k: v for k, v in r.items() if k != "conditions"}, **fit}
            assert cfg["group"] == "hope" and cfg["tags"] == [family] and cfg["run_name"] == f"hope-{family}-{stage}"
            assert {k: v for k, v in cfg.items() if k not in ("models", "group", "tags", "run_name")} == \
                   {k: v for k, v in ref.items() if k not in ("models", "group", "tags", "run_name")} \
                   | ({"batch_size": 16} if family == "glm-4.5-air" else {})
            assert cfg["max_pairs"] == 15000 and cfg["seed"] == 0 and cfg["system_prompt"] == "none"
