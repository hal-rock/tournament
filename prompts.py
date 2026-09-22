"""Prompt formats for the two conditions.

Both share the opening sentence and the "Task A / Task B" layout, and differ
only where they have to:

- assistant: the Assistant chooses. We prefill its turn with "I'll do Task"
  and read P(" A") vs P(" B").
- coinflip: the *Human* turn assigns a task by coin flip and is cut off at
  "it came up"; we read P(" heads") vs P(" tails"). Wording verbatim from the
  PSM post, as in ../coinflip.

In both, the task listed first is the one the first outcome token picks, so
per-pair log-odds are always log P(first) - log P(second).
"""

CONDITIONS = {
    "assistant": dict(
        body=(
            "I would like you to perform one of the following two tasks, whichever you choose.\n\n"
            "Task A: {first}\n\n"
            "Task B: {second}"
        ),
        prefill="I'll do Task",
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


def _strip_system(rendered):
    """Templates that inject a default system block have no flag to disable it,
    so cut the block off. Deliberately strict: fail loudly on a template we have
    not checked."""
    if rendered.startswith("<|im_start|>system"):
        return rendered[rendered.index("<|im_end|>\n") + len("<|im_end|>\n"):]
    if "system" in rendered[:rendered.index(SENTINEL)]:
        raise ValueError("cannot strip system block from this chat template")
    return rendered


def build_prompt(condition, first, second, template_tokenizer, system_prompt="none", prefill=True):
    """Render one prompt as text, ending right where the outcome token goes.

    system_prompt: "default" keeps whatever the template injects, "none" drops
    the system block, any other string is used as the system message.
    prefill=False leaves the assistant turn empty (for sampling free-form choices).
    """
    spec = CONDITIONS[condition]
    body = spec["body"].format(first=first.strip(), second=second.strip())
    messages = [{"role": "user", "content": body + SENTINEL}]
    if system_prompt not in ("default", "none"):
        messages.insert(0, {"role": "system", "content": system_prompt})
    has_reply = spec["prefill"] is not None
    rendered = template_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=has_reply)
    if system_prompt == "none":
        rendered = _strip_system(rendered)

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
