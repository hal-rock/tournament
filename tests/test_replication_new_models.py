"""The two new finals of replication round 2 (PLAN_replication2.md): GLM-5.3 and Kimi K3.1 Pro, from their configs.

Every condition the config scores (replication.CONDITION_SETS["new_final"], 75), rendered as score_replication.py renders
it, against the model's own template (GLM's Jinja template; Kimi's encoder through its remote-code tokenizer):
- an open user turn is the template's rendering of the finished turn, minus its closing;
- an Assistant readout is the generation prompt, the empty think block and the prefill, and a prefix of the template's
  rendering of a past turn with empty reasoning whose answer is the outcome, in text and in ids, the outcome token next;
- the reasoning check (~inreasoning) is the generation prompt and the prefill, as the reasoning's first words;
- every prompt keeps the template's reasoning-effort block, at the level the effort rule would set;
- raw text is the document start and the coinflip's words.
For Kimi, also that tokenizing a rendered prompt in one piece gives the encoder's own segment-by-segment ids. Needs only
tokenizer and template files.
"""

import itertools
import json
import sys
from pathlib import Path

import pytest
import yaml
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import score_replication  # noqa: E402
from prompts import (CONDITIONS, PERSONAS, PERSONAS_SELF, ROLEPLAY, close_think_text, outcome_token_ids,  # noqa: E402
                     parse_condition)
from replication import CONDITION_SETS, expand_conditions, split_variant  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
KEYS = ["glm-5.3", "kimi-k3.1-pro"]
TASKS = [json.loads(line) for line in (ROOT / "data/tasks_n300_max500_seed0.jsonl").read_text().splitlines()]
_n = [len(t["prompt"]) for t in TASKS]
_longest = sorted(range(len(TASKS)), key=lambda i: -_n[i])[:2]
PAIRS = [tuple(_longest), (0, 1), (150, 299)]
PAIRS += [(b, a) for a, b in PAIRS]
# How each model's template writes its effort block, and how it closes a finished user turn.
EFFORT_SHOWN = {"glm-5.3": "<|system|>Reasoning Effort: {}", "kimi-k3.1-pro": "Now the system is invoked with `thinking_effort={}`."}
USER_END = {"glm-5.3": "", "kimi-k3.1-pro": "<|close|>message<|sep|><|end_of_msg|>"}


def load(key):
    cfg = yaml.safe_load((ROOT / "configs/replication" / f"{key}.yaml").read_text())
    (m,) = cfg["models"]
    return cfg, m


@pytest.fixture(scope="module", params=KEYS)
def model(request):
    cfg, m = load(request.param)
    tok = AutoTokenizer.from_pretrained(m["name"], revision=m["revision"], trust_remote_code=m.get("trust_remote_code", False))
    return request.param, cfg, m, tok


def render(m, tok, condition, a, b, level=None, prefill=True):
    """As score_replication renders it, with the effort level the rule would set."""
    m = {**m, "template_kwargs": {m["effort"]["kwarg"]: level}} if level else m
    ids, text = score_replication.render_condition(condition, m, TASKS, [(a, b)], tok, tok, "none", prefill=prefill)
    return text, ids[0]


def messages(condition, a, b):
    """The conversation a condition stands for, built independently of prompts._render."""
    name, persona, placement = parse_condition(split_variant(condition)[0])
    body = CONDITIONS[name]["body"].format(first=TASKS[a]["prompt"].strip(), second=TASKS[b]["prompt"].strip())
    system = {"system": PERSONAS, "roleplay": ROLEPLAY}.get(placement, {}).get(persona)
    if placement in ("user", "self"):
        body = (PERSONAS if placement == "user" else PERSONAS_SELF)[persona] + "\n\n" + body
    return ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": body}]


def enc(tok, s):
    return tok(s, add_special_tokens=False).input_ids


def chat_conditions():
    return [c for c in CONDITION_SETS["new_final"]
            if not CONDITIONS[parse_condition(split_variant(c)[0])[0]].get("raw")
            and not CONDITIONS[parse_condition(split_variant(c)[0])[0]].get("tool")]


def test_config():
    for key in KEYS:
        cfg, m = load(key)
        assert (cfg["group"], cfg["run_name"], cfg["max_pairs"], cfg["seed"], cfg["system_prompt"]) == (
            "replication2", key, 15000, 0, "none")
        assert cfg["tasks_file"] == "data/tasks_n300_max500_seed0.jsonl" and len(TASKS) == 300
        assert m["stage"] == "instruct" and len(m["revision"]) == 40 and m["keep_system"]
        assert expand_conditions(m["conditions"]) == CONDITION_SETS["new_final"] and len(CONDITION_SETS["new_final"]) == 75
        assert m["variants"] == {"inreasoning": {"close_think": False}}
        assert m["effort"]["values"] == ["low", "high", "max"]
        assert m["vllm"]["tensor_parallel_size"] == 8 and m["vllm"]["enable_prefix_caching"]


def test_tasks_contain_no_added_token_text(model):
    key, cfg, m, tok = model
    added = set(tok.get_added_vocab()) | {"<|open|>", "<|close|>", "<|sep|>", "<think>", "</think>"}
    assert not [(t["id"], s) for t in TASKS for s in added if s in t["prompt"]]


@pytest.mark.parametrize("level", [None, "low", "high"])
def test_chat_prompts_match_the_template(model, level):
    """Every chat condition (not tool, not raw) against the template's own rendering of the same conversation, at the
    template's default effort and the others."""
    key, cfg, m, tok = model
    kw = {m["effort"]["kwarg"]: level} if level else {}
    for condition in chat_conditions():
        base, variant = split_variant(condition)
        spec = CONDITIONS[parse_condition(base)[0]]
        for a, b in PAIRS[:3] if level else PAIRS:
            p, ids = render(m, tok, condition, a, b, level)
            assert ids == enc(tok, p)
            msgs = messages(condition, a, b)
            shown = EFFORT_SHOWN[key].format((level or "max").capitalize() if key == "glm-5.3" else (level or "max"))
            assert p.count(shown) == 1, (condition, p[:300])
            if spec["prefill"] is None:
                assert p + USER_END[key] == tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, **kw)
                continue
            gen = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
            if variant == "inreasoning":  # the prefill opens the reasoning
                assert p == gen + spec["prefill"]
                continue
            assert p == gen + close_think_text(m["close_think"]) + spec["prefill"]
            for outcome, oid in zip(spec["outcomes"], outcome_token_ids(base, tok)):
                past = tok.apply_chat_template(msgs + [{"role": "assistant", "reasoning_content": "",
                                                        "content": spec["prefill"] + outcome + "."}],
                                               tokenize=False, add_generation_prompt=False, **kw)
                assert past.startswith(p), (condition, past[len(gen) - 20:len(p) + 20])
                assert enc(tok, past)[:len(ids) + 1] == ids + [oid], (condition, outcome)


def test_persona_placements(model):
    """The persona's system message (or roleplay frame) comes after the effort block; user and self open the user turn."""
    key, cfg, m, tok = model
    a, b = PAIRS[2]
    eff = EFFORT_SHOWN[key].format("Max" if key == "glm-5.3" else "max")
    sys_, _ = render(m, tok, "coinflip@evil", a, b)
    rp, _ = render(m, tok, "coinflip@evil+roleplay", a, b)
    user, _ = render(m, tok, "coinflip@evil+user", a, b)
    for p, text in ((sys_, PERSONAS["evil"]), (rp, ROLEPLAY["evil"])):
        assert p.index(eff) < p.index(text) < p.index("I would like you to perform")
    assert PERSONAS["evil"] + "\n\nI would like you to perform" in user


def test_raw_text(model):
    key, cfg, m, tok = model
    for condition in ("raw_okay", "raw_no_okay", "raw_story_ai", "raw_story_person"):
        a, b = PAIRS[2]
        p, ids = render(m, tok, condition, a, b)
        body = CONDITIONS[condition]["body"].format(first=TASKS[a]["prompt"].strip(), second=TASKS[b]["prompt"].strip())
        assert p == m["doc_prefix"] + body
        prefix = enc(tok, m["doc_prefix"])
        assert all(i in tok.all_special_ids or i in tok.get_added_vocab().values() for i in prefix), prefix
        assert ids == prefix + enc(tok, body)
        assert "Reasoning Effort" not in p and "thinking_effort" not in p


def test_tool_conditions_render_with_the_config(model):
    """The tool conditions through the config (toolframe's own checks are in tests/test_toolframe.py): the effort block,
    the persona placements including roleplay, the length rules, and the outcome right after the cut."""
    key, cfg, m, tok = model
    heads, tails = outcome_token_ids("coinflip_tool", tok)
    tool = [c for c in CONDITION_SETS["new_final"] if CONDITIONS[parse_condition(split_variant(c)[0])[0]].get("tool")]
    assert len(tool) == 15 + 4 + 2
    for c in tool:
        p, ids = render(m, tok, c, *PAIRS[0])
        assert "flip_coin" in p and EFFORT_SHOWN[key].format("Max" if key == "glm-5.3" else "max") in p
        if CONDITIONS[parse_condition(c)[0]]["tool"] == "call":
            for outcome, oid in (("heads", heads), ("tails", tails)):
                assert enc(tok, p + outcome + '"}')[:len(ids) + 1] == ids + [oid], (c, outcome)
    rp, _ = render(m, tok, "coinflip_tool@slacker+roleplay", *PAIRS[0])
    assert ROLEPLAY["slacker"] in rp and rp.index("flip_coin") < rp.index(ROLEPLAY["slacker"])


def test_effort_rule_probe_ends_at_the_think_opener(model):
    """choose_effort reads P(close) right after the generation prompt's think opener, at each level."""
    key, cfg, m, tok = model
    (close,) = enc(tok, m["effort"]["close"])
    for level in m["effort"]["values"]:
        probe = {**m, "close_think": False}
        p, ids = render(probe, tok, "assistant", *PAIRS[2], level=level, prefill=False)
        assert p.endswith(("<|assistant|><think>", '<|open|>message role="assistant"<|sep|><|open|>think<|sep|>'))
        closed = p + close_think_text(m["close_think"])
        assert enc(tok, closed)[len(ids)] == close


def test_max_model_len_holds_the_longest_prompt(model):
    """The longest prompt over the 75 conditions, every ordered pair of the 6 longest tasks (length is the two tasks'
    plus a per-condition constant), one outcome token and room to spare."""
    key, cfg, m, tok = model
    n = [len(enc(tok, x["prompt"].strip())) for x in TASKS]
    top = sorted(range(len(TASKS)), key=lambda i: -n[i])[:6]
    longest = max(len(render(m, tok, c, a, b)[1]) for c in CONDITION_SETS["new_final"]
                  for a, b in itertools.permutations(top, 2))
    print(key, "longest prompt", longest)
    assert 1.5 * (longest + 1) <= m["vllm"]["max_model_len"]


def test_kimi_one_piece_tokenization_matches_the_encoder():
    """Moonshot's encoder tokenizes segment by segment (markers as special tokens, text as text); the scorer tokenizes the
    rendered prompt in one piece. They agree on every condition's full conversation, so on its prompt too."""
    cfg, m = load("kimi-k3.1-pro")
    tok = AutoTokenizer.from_pretrained(m["name"], revision=m["revision"], trust_remote_code=True)
    for condition in chat_conditions():
        spec = CONDITIONS[parse_condition(split_variant(condition)[0])[0]]
        for a, b in PAIRS[:4]:
            msgs = messages(condition, a, b)
            if spec["prefill"] is not None:
                msgs = msgs + [{"role": "assistant", "reasoning_content": "", "content": spec["prefill"] + " A."}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            seg = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
            assert enc(tok, text) == list(seg), condition
