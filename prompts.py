"""Prompt formats for the two conditions.

Both share the opening sentence and the "Task A / Task B" layout, and differ
only where they have to:

- assistant: the Assistant chooses. We prefill its turn with "I'll choose Task"
  and read P(" A") vs P(" B").
- coinflip: the *Human* turn assigns a task by coin flip and is cut off at
  "it came up"; we read P(" heads") vs P(" tails"). Wording verbatim from the
  PSM post, as in ../coinflip.

In both, the task listed first is the one the first outcome token picks, so
per-pair log-odds are always log P(first) - log P(second).
"""

import re

CONDITIONS = {
    "assistant": dict(
        body=(
            "I would like you to perform one of the following two tasks, whichever you choose.\n\n"
            "Task A: {first}\n\n"
            "Task B: {second}"
        ),
        prefill="I'll choose Task",
        outcomes=(" A", " B"),
    ),
    "coinflip": dict(
        body=(
            "I would like you to perform one of the following two tasks, depending on the outcome of a coinflip.\n\n"
            "Task A (if heads): {first}\n\n"
            "Task B (if tails): {second}\n\n"
            "Okay, I'm flipping the coin now ... it came up"
        ),
        prefill=None,  # the Human turn itself is left unfinished
        outcomes=(" heads", " tails"),
    ),
}
SENTINEL = "<<<CUT_HERE>>>"


# Default system blocks that templates inject on their own, by format: OLMo / Qwen
# (ChatML) and Llama 3 (knowledge-cutoff and date lines). Each matches one block.
SYSTEM_BLOCKS = [
    re.compile(r"<\|im_start\|>system\n.*?<\|im_end\|>\n", re.S),
    re.compile(r"<\|start_header_id\|>system<\|end_header_id\|>.*?<\|eot_id\|>", re.S),
]


def _strip_system(rendered, content):
    """Templates that inject a default system block have no flag to disable it,
    so cut the block off. Deliberately strict: fail loudly on a template we have
    not checked."""
    for pattern in SYSTEM_BLOCKS:
        rendered = pattern.sub("", rendered, count=1)
    if "system" in rendered[:rendered.index(content)]:  # template text only, not the tasks
        raise ValueError("cannot strip system block from this chat template")
    return rendered


def _render(messages, template_tokenizer, has_reply, system_prompt, turn_format, close_think=False,
            strip_think=False):
    """The conversation as text, user content still ending in SENTINEL, plus the
    Assistant header if has_reply."""
    if turn_format is not None:
        # Models released without a Jinja template (DeepSeek V4): literal text that opens
        # a user turn with no system prompt, and that closes it and opens the reply.
        assert system_prompt == "none", "turn_format only supports system_prompt: none"
        (message,) = messages
        return turn_format["user"] + message["content"] + (turn_format["reply"] if has_reply else "")
    if system_prompt not in ("default", "none"):
        messages = [{"role": "system", "content": system_prompt}] + messages
    # enable_thinking=False selects each template's non-thinking reply, so the prefill is the
    # answer itself: Qwen 3.5 and GLM-4.5 open it with an empty <think></think> block (GLM also
    # appends "/nothink" to the user turn), Gemma 4 with an empty thought channel. Other
    # templates ignore it.
    rendered = template_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=has_reply,
                                                      enable_thinking=False)
    if system_prompt == "none":
        rendered = _strip_system(rendered, messages[-1]["content"])
    if has_reply and rendered.endswith("<think></think>"):
        # GLM-4.5 puts a newline between the empty think block and the answer (it is in the
        # template's rendering of past Assistant turns, not in the generation prompt).
        rendered += "\n"
    if has_reply and close_think:
        # Think-only templates (OLMo 3 Think) open every reply with "<think>" and have no
        # non-thinking mode. Close the block empty, so the prefill is the answer itself. One
        # newline, as in the chat and safety parts of Dolci-Think-SFT (math and code use two).
        if not rendered.endswith("<think>"):
            raise ValueError("close_think: the generation prompt does not end in <think>")
        rendered += "</think>\n"
    if has_reply and strip_think:
        # Or drop the opening "<think>", so the prefill follows the Assistant header directly. This
        # is how the Think template renders a reply that carries no trace: the format ../coinflip's
        # run 5 trained on (Instruct DPO pairs under the Think SFT's template).
        if close_think or not rendered.endswith("<think>"):
            raise ValueError("strip_think: the generation prompt does not end in <think> (or close_think is set)")
        rendered = rendered.removesuffix("<think>")
    return rendered


def build_prompt(condition, first, second, template_tokenizer, system_prompt="none", prefill=True,
                 turn_format=None, close_think=False, strip_think=False):
    """Render one prompt as text, ending right where the outcome token goes.

    system_prompt: "default" keeps whatever the template injects, "none" drops
    the system block, any other string is used as the system message.
    prefill=False leaves the assistant turn empty (for sampling free-form choices).
    turn_format: {"user": ..., "reply": ...} literal turn markers, used instead of
    the chat template (see _render).
    close_think: close a Think-only template's opening "<think>" empty before the
    prefill; without it the prefill is the first words of the reasoning.
    strip_think: drop that "<think>" instead, so the reply has no think block at all.
    """
    spec = CONDITIONS[condition]
    body = spec["body"].format(first=first.strip(), second=second.strip())
    has_reply = spec["prefill"] is not None
    rendered = _render([{"role": "user", "content": body + SENTINEL}], template_tokenizer, has_reply,
                       system_prompt, turn_format, close_think, strip_think)

    if not has_reply:
        # Cut inside the user turn, so the model continues the Human's message.
        return rendered[:rendered.index(SENTINEL)]
    # Finished user turn + assistant header, then the prefill.
    rendered = rendered.replace(SENTINEL, "")
    return rendered + spec["prefill"] if prefill else rendered


def outcome_token_ids(condition, tokenizer):
    ids = []
    for s in CONDITIONS[condition]["outcomes"]:
        enc = tokenizer.encode(s, add_special_tokens=False)
        assert len(enc) == 1, f"{s!r} is not a single token: {enc}"
        ids.append(enc[0])
    return ids
