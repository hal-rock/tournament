"""The coin flipped by a tool (README, "The coin as a tool call"), after ../coinflip's toolframe.py.

One tool, `flip_coin`, is declared in each family's own tool format (declaration and {"result": "heads"} format as in
../coinflip, from narrative-direction). Three kinds of prompt (prompts.CONDITIONS[...]["tool"]):
  call:      the user turn ends "Okay, please flip the coin now."; the Assistant's turn is only a call to flip_coin
             (prefilled, no text); the prompt is cut where the tool result's value starts, and we read `heads` / `tails`
             (no leading space).
  declared:  an ordinary condition (the user-reported coin, the Assistant's own choice) with the tool declared but unused:
             the controls for the tool block itself.

Families (`tool_family` in a replication config), each in its own format:
  kimi:      Kimi K2's template: a `tool_declare` system turn; the call in <|tool_calls_section_begin|> with id
             "functions.flip_coin:0"; the result as a turn named after the tool ("## Return of <id>"), as in Kimi's
             docs/tool_call_guidance.md. Its default "You are Kimi, ..." block is stripped, as everywhere else.
  llama:     Llama 3.1's template: the tool declared in the first user message (its default, tools_in_user_message),
             "Environment: ipython" in the system block (kept: the template writes it for tools); the call as
             {"name": "flip_coin", "parameters": {}}; the result in an `ipython` turn.
  deepseek3: DeepSeek V3's own template declares no tools and leaves a single call unclosed. So the prompt is rendered
             by DeepSeek-V3-0324's template (the same tokenizer; templates/deepseek_v3_0324_chat_template.jinja), which
             closes the call (<｜tool▁calls▁end｜><｜end▁of▁sentence｜>), and the tool is declared in the system prompt
             with the wording of vLLM's examples/tool_chat_template_deepseekv3.jinja (v0.33.0), without the indentation
             that template leaks between special tokens. A persona system prompt comes first, then the declaration.
  deepseek4: DeepSeek V4's reference encoder (encoding/encoding_dsv4.py in the model repo), chat mode, as in
             ../coinflip: tool results sit inside a <｜User｜> turn wrapped in <tool_result>.
  trinity:   Trinity Large Thinking's template: its default identity line and the tool instructions in the system block
             (it writes them whenever tools are declared); the call after an empty think block (reasoning_content "",
             as close_think reads every other Assistant turn); the result in a user turn wrapped in <tool_response>.

A persona as the system prompt is the system message (with the tools); in the user turn it is part of `body`.
"""

import importlib.util
import json
import re
from pathlib import Path

from huggingface_hub import hf_hub_download

SENTINEL = "<<<CUT_HERE>>>"
TOOLS = [{"type": "function", "function": {
    "name": "flip_coin",
    "description": "Flip a fair coin using a cryptographically secure random number generator. "
                   "Returns the result, \"heads\" or \"tails\".",
    "parameters": {"type": "object", "properties": {}, "required": []},
}}]
FAMILIES = ("kimi", "llama", "deepseek3", "deepseek4", "trinity")
KIMI_CALL_ID = "functions.flip_coin:0"
KIMI_DEFAULT_SYSTEM = re.compile(r"<\|im_system\|>system<\|im_middle\|>You are Kimi, an AI assistant created by Moonshot AI\."
                                 r"<\|im_end\|>\n?")
DEEPSEEK3_TEMPLATE = Path(__file__).resolve().parent / "templates" / "deepseek_v3_0324_chat_template.jinja"
# vLLM's DeepSeek V3 tool declaration, as its template renders it once the stray indentation is removed.
DEEPSEEK3_TOOLS = ("# Tools\n\nYou may call one or more functions to assist with the user query.\n"
                   + "\n".join(json.dumps(t) for t in TOOLS) + "\n</tools>\n\n"
                   "For function call returns, you should first print <｜tool▁calls▁begin｜>"
                   "For each function call, you should return object like:\n"
                   "<｜tool▁call▁begin｜>function<｜tool▁sep｜><function_name>\n```json\n<function_arguments_in_json_format>\n```"
                   "<｜tool▁call▁end｜>"
                   "At the end of function call returns, you should print <｜tool▁calls▁end｜><｜end▁of▁sentence｜>")
# What a generated call to the tool contains, per family (the on-policy check).
CALL_MARKER = {"kimi": "<|tool_call_begin|>functions.flip_coin", "llama": '"name": "flip_coin"',
               "deepseek3": "<｜tool▁sep｜>flip_coin", "deepseek4": 'invoke name="flip_coin"', "trinity": "<function=flip_coin>"}
_CALL_RE = {"llama": re.compile(r'"name"\s*:\s*"flip_coin"'), "deepseek4": re.compile(r'<｜DSML｜invoke name="flip_coin"'),
            "trinity": re.compile(r'<function=flip_coin>|"name"\s*:\s*"flip_coin"')}
_encoders = {}


def calls_tool(family, reply, first=False):
    """Whether a generated reply calls flip_coin (in the family's call format; Trinity's template also accepts JSON). With
    first=True, only if the call is all the reply has before it: no more than whitespace and markup tags (special tokens,
    Llama's <|python_tag|>, Trinity's <tool_call>, DeepSeek V4's <｜DSML｜tool_calls>)."""
    pattern = _CALL_RE.get(family, re.compile(re.escape(CALL_MARKER[family])))
    hit = pattern.search(reply)
    if hit is None or not first:
        return hit is not None
    before = re.sub(r"<[^<>]*>|```json|\{", "", reply[:hit.start()])
    return before.strip() in ("", "function")  # DeepSeek V3: <｜tool▁call▁begin｜>function<｜tool▁sep｜>flip_coin


def deepseek4_encoder(repo):
    """DeepSeek V4's reference encoder (stdlib only), from the model repo."""
    if repo not in _encoders:
        path = hf_hub_download(repo, "encoding/encoding_dsv4.py")
        spec = importlib.util.spec_from_file_location("encoding_dsv4", path)
        _encoders[repo] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_encoders[repo])
    return _encoders[repo]


def _messages(family, body, system, result):
    """The conversation: [system,] user, and with a result the Assistant's call and the tool's answer."""
    if family == "deepseek3":  # the declaration is part of the system prompt (after a persona, if any)
        system = (system + "\n\n" if system else "") + DEEPSEEK3_TOOLS
    msgs = ([{"role": "system", "content": system}] if system and family != "deepseek4" else [])
    msgs.append({"role": "user", "content": body})
    if result is None:
        return msgs
    if family == "kimi":
        msgs.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": KIMI_CALL_ID, "type": "function", "function": {"name": "flip_coin", "arguments": "{}"}}]})
        msgs.append({"role": "tool", "tool_call_id": KIMI_CALL_ID, "name": "flip_coin",
                     "content": json.dumps({"result": result})})
    elif family == "llama":  # a dict result: the template writes it with tojson (a string would be quoted again)
        msgs.append({"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "flip_coin", "arguments": {}}}]})
        msgs.append({"role": "tool", "name": "flip_coin", "content": {"result": result}})
    elif family == "deepseek3":  # V3-0324's template writes the arguments as given: a JSON string
        msgs.append({"role": "assistant", "content": None, "tool_calls": [
            {"type": "function", "function": {"name": "flip_coin", "arguments": "{}"}}]})
        msgs.append({"role": "tool", "content": json.dumps({"result": result})})
    elif family == "deepseek4":  # the encoder wants the arguments as a JSON string
        msgs.append({"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "flip_coin", "arguments": "{}"}}]})
        msgs.append({"role": "tool", "name": "flip_coin", "content": json.dumps({"result": result})})
    elif family == "trinity":
        msgs.append({"role": "assistant", "content": "", "reasoning_content": "", "tool_calls": [
            {"type": "function", "function": {"name": "flip_coin", "arguments": {}}}]})
        msgs.append({"role": "tool", "name": "flip_coin", "content": json.dumps({"result": result})})
    return msgs


def _render(family, tok, msgs, system=None, gen_prompt=False, ds_repo=None):
    if family == "deepseek4":
        enc = deepseek4_encoder(ds_repo)
        return enc.encode_messages([{"role": "system", "content": system or "", "tools": TOOLS}] + msgs,
                                   thinking_mode="chat")
    if family == "deepseek3":  # the tools are in the system prompt already; the template ignores `tools`
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen_prompt,
                                       chat_template=DEEPSEEK3_TEMPLATE.read_text())
    else:
        text = tok.apply_chat_template(msgs, tools=TOOLS, tokenize=False, add_generation_prompt=gen_prompt)
    if family == "kimi" and not system:
        text, n = KIMI_DEFAULT_SYSTEM.subn("", text, count=1)
        assert n == 1, "Kimi's default system block not found"
    return text


def build_tool_prompt(kind, condition, body, tok, family, system=None, prefill=None, close_think=False, ds_repo=None):
    """Prompt text for a tool condition, ending right where the outcome token goes.
    kind "call": cut where the tool result's value starts. kind "declared": `condition`'s own prompt (an open user turn,
    or the Assistant's turn and its prefill) with the tool declared."""
    assert family in FAMILIES, f"unknown tool_family {family!r}"
    if kind == "call":
        text = _render(family, tok, _messages(family, body, system, SENTINEL), system, ds_repo=ds_repo)
        cut = text.index(SENTINEL)
        for outcome in ("heads", "tails"):  # nothing before the cut may render differently with a real result
            full = _render(family, tok, _messages(family, body, system, outcome), system, ds_repo=ds_repo)
            assert full.startswith(text[:cut]) and full[cut:].startswith(outcome), (family, full[cut - 40:cut + 20])
        return text[:cut]
    assert kind == "declared", kind
    if prefill is None:  # an open user turn (the user-reported coin)
        text = _render(family, tok, _messages(family, body + SENTINEL, system, None), system, ds_repo=ds_repo)
        return text[:text.index(SENTINEL)]
    text = _render(family, tok, _messages(family, body, system, None), system, gen_prompt=True, ds_repo=ds_repo)
    if family == "deepseek4":
        assert text.endswith("<｜Assistant｜></think>"), text[-40:]  # chat mode: the empty think block
    if close_think:
        assert text.endswith("<think>"), text[-40:]
        text += "</think>\n"
    return text + prefill


def generation_prompt(family, tok, body, system=None, close_think=False, ds_repo=None):
    """Where the Assistant answers the `call` user turn on its own (the on-policy check), thinking off."""
    text = _render(family, tok, _messages(family, body, system, None), system, gen_prompt=True, ds_repo=ds_repo)
    return text + "</think>\n" if close_think else text
