"""The coin as a tool call (toolframe.py): every family's prompts, from tokenizer and template files only."""

import sys
from pathlib import Path

import pytest
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import toolframe  # noqa: E402
from prompts import CONDITIONS, PERSONAS, PERSONAS_SELF, build_prompt, outcome_token_ids  # noqa: E402

KIMI = "moonshotai/Kimi-K2-Instruct"
KIMI_REV = "fd1984e2b7a3350dbf7305fe73a4ede25c14de50"
FAMILY_REPO = {"llama": "meta-llama/Llama-3.1-405B-Instruct", "deepseek3": "deepseek-ai/DeepSeek-V3",
               "deepseek4": "deepseek-ai/DeepSeek-V4-Flash", "trinity": "arcee-ai/Trinity-Large-Thinking", "kimi": KIMI}
CLOSE_THINK = {"trinity"}
PAIRS = [("Write a haiku about rain.", "Prove that sqrt(2) is irrational."),
         ("Explain how a refrigerator works.", "List three uses of \"quotes\" in JSON {like this}.")]


@pytest.fixture(scope="module", params=list(FAMILY_REPO))
def fam(request):
    family = request.param
    if family == "kimi":  # the remote-code tokenizer, rebuilt from its data files (tests/test_replication_kimi_k2.py)
        from test_replication_kimi_k2 import stand_in
        return family, stand_in(KIMI, KIMI_REV, KIMI_REV)
    return family, AutoTokenizer.from_pretrained(FAMILY_REPO[family])


def render(condition, family, tok, first="FIRST", second="SECOND"):
    return build_prompt(condition, first, second, tok, tool_family=family, tool_repo=FAMILY_REPO[family],
                        close_think=family in CLOSE_THINK)


def ids(tok, text):
    return tok(text, add_special_tokens=False).input_ids


def test_tool_conditions():
    assert CONDITIONS["coinflip_tool"]["body"] == CONDITIONS["coinflip"]["body"].replace(
        "Okay, I'm flipping the coin now ... it came up", "Okay, please flip the coin now.")
    assert CONDITIONS["coinflip_tool"]["outcomes"] == ("heads", "tails")
    for c in ("coinflip", "assistant"):
        assert {**CONDITIONS[c], "tool": "declared"} == CONDITIONS[f"{c}_tools_declared"]


def test_outcome_is_the_next_token_after_the_cut(fam):
    """The id we read is the id the model would emit: tokenizing prompt + result gives the prompt's ids, then it."""
    family, tok = fam
    heads, tails = outcome_token_ids("coinflip_tool", tok)
    assert heads != tails
    for first, second in PAIRS:
        for persona in ("", "@evil", "@contrarian+user", "@strategist+self"):
            p = render("coinflip_tool" + persona, family, tok, first, second)
            for outcome, oid in (("heads", heads), ("tails", tails)):
                full = ids(tok, p + outcome + '"}')
                assert full[:len(ids(tok, p)) + 1] == ids(tok, p) + [oid], (family, persona, outcome)


def test_the_prefilled_call_continues_the_models_own_generation_prompt(fam):
    family, tok = fam
    body = CONDITIONS["coinflip_tool"]["body"].format(first="FIRST", second="SECOND")
    gen = toolframe.generation_prompt(family, tok, body, close_think=family in CLOSE_THINK, ds_repo=FAMILY_REPO[family])
    call = render("coinflip_tool", family, tok)
    assert call.startswith(gen) and len(call) > len(gen) + 10, (family, gen[-60:], call[len(gen) - 60:len(gen) + 60])
    # build_prompt(prefill=False) gives the same open turn, persona placements included (the on-policy check)
    assert build_prompt("coinflip_tool", "FIRST", "SECOND", tok, prefill=False, tool_family=family,
                        tool_repo=FAMILY_REPO[family], close_think=family in CLOSE_THINK) == gen
    for persona in ("@evil", "@evil+user"):
        p = render("coinflip_tool" + persona, family, tok)
        g = build_prompt("coinflip_tool" + persona, "FIRST", "SECOND", tok, prefill=False, tool_family=family,
                         tool_repo=FAMILY_REPO[family], close_think=family in CLOSE_THINK)
        assert p.startswith(g) and toolframe.CALL_MARKER[family] in p[len(g):], (family, persona)
    assert toolframe.CALL_MARKER[family] in call[len(gen):] or family == "llama" and '"name": "flip_coin"' in call


def test_tool_is_declared_and_default_identity_handled(fam):
    family, tok = fam
    for c in ("coinflip_tool", "coinflip_tools_declared", "assistant_tools_declared"):
        p = render(c, family, tok)
        assert p.count("flip_coin") >= 1 and "cryptographically secure" in p, (family, c)
        if family == "kimi":
            assert "You are Kimi" not in p  # stripped, as in every no-system-prompt condition
    if family == "kimi":  # with a persona the template writes no default, and the persona is the system turn
        assert "<|im_system|>system<|im_middle|>" + PERSONAS["evil"] + "<|im_end|>" in render("coinflip_tool@evil", family, tok)


def test_persona_placements(fam):
    family, tok = fam
    plain = render("coinflip_tool", family, tok)
    sys_ = render("coinflip_tool@slacker", family, tok)
    user = render("coinflip_tool@slacker+user", family, tok)
    self_ = render("coinflip_tool@slacker+self", family, tok)
    assert PERSONAS["slacker"] in sys_ and PERSONAS["slacker"] not in plain
    # user / self: the persona text opens the user's message, right before the task request
    request = "I would like you to perform"
    assert PERSONAS["slacker"] + "\n\n" + request in user
    assert PERSONAS_SELF["slacker"] + "\n\n" + request in self_
    # the system persona comes before the user's turn, not inside it
    assert sys_.index(PERSONAS["slacker"]) < sys_.index(request)
    for p in (sys_, user, self_):  # the call and the cut are the same in every placement
        assert p[p.index("Okay, please flip the coin now."):] == plain[plain.index("Okay, please flip the coin now."):]


def test_declared_controls_end_like_their_plain_conditions(fam):
    family, tok = fam
    coin = render("coinflip_tools_declared", family, tok)
    assert coin.endswith("Okay, I'm flipping the coin now ... it came up")
    asst = render("assistant_tools_declared", family, tok)
    assert asst.endswith("I'll choose Task") and "whichever you choose" in asst
    if family in CLOSE_THINK:
        assert asst.endswith("<think></think>\nI'll choose Task")
    assert outcome_token_ids("coinflip_tools_declared", tok) == outcome_token_ids("coinflip", tok)
    assert outcome_token_ids("assistant_tools_declared", tok) == outcome_token_ids("assistant", tok)


def test_deepseek4_matches_its_reference_encoder():
    tok = AutoTokenizer.from_pretrained(FAMILY_REPO["deepseek4"])
    enc = toolframe.deepseek4_encoder(FAMILY_REPO["deepseek4"])
    body = CONDITIONS["coinflip_tool"]["body"].format(first="FIRST", second="SECOND")
    ref = enc.encode_messages([{"role": "system", "content": "", "tools": toolframe.TOOLS}]
                              + toolframe._messages("deepseek4", body, None, "heads"), thinking_mode="chat")
    p = render("coinflip_tool", "deepseek4", tok)
    assert ref.startswith(p) and ref[len(p):].startswith("heads")


def test_calls_tool_reads_the_prefilled_call_as_a_call(fam):
    """The on-policy check's reader, on the call we prefill: a call, and nothing before it."""
    family, tok = fam
    body = CONDITIONS["coinflip_tool"]["body"].format(first="FIRST", second="SECOND")
    gen = toolframe.generation_prompt(family, tok, body, close_think=family in CLOSE_THINK, ds_repo=FAMILY_REPO[family])
    call = render("coinflip_tool", family, tok)[len(gen):]
    assert toolframe.calls_tool(family, call, first=True), (family, call)
    assert not toolframe.calls_tool(family, "Sure! I'll flip it: heads.")
    assert toolframe.calls_tool(family, "Sure, flipping now.\n" + call)
    assert not toolframe.calls_tool(family, "Sure, flipping now.\n" + call, first=True)
