"""Llama 3.1 405B in the replication (configs/replication/llama-405b.yaml). Reads the config, renders every condition it
runs on the real tasks as the scorer does (prompts.build_prompt, then the stage's tokenizer with add_special_tokens=False),
and checks it against the Instruct's official chat template. Needs only tokenizer files (gated repos: HF_TOKEN)."""

import json
from pathlib import Path

import pytest
import yaml
from huggingface_hub import HfApi
from huggingface_hub.utils import filter_repo_objects
from transformers import AutoTokenizer

import prompts
from prompts import CONDITIONS, build_prompt, outcome_token_ids
from replication import BASE_CONDITIONS, FINAL_CONDITIONS, expand_conditions

CONFIG = yaml.safe_load(Path("configs/replication/llama-405b.yaml").read_text())
BASE, FINAL = CONFIG["models"]
TASKS = [json.loads(line) for line in open(CONFIG["tasks_file"])]
# Each task once in each slot.
PAIRS = [(TASKS[i]["prompt"], TASKS[(i + 1) % len(TASKS)]["prompt"]) for i in range(len(TASKS))]

BOS = "<|begin_of_text|>"
MARKERS = ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]
# What the template writes before any system message: two date lines, the date a fixed default (no clock).
SYSTEM_HEAD = "<|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n"


def load(m, key):
    return AutoTokenizer.from_pretrained(m.get(key, m["name"]))


TOK = {m["stage"]: load(m, "tokenizer_from") for m in CONFIG["models"]}
TEMPLATE = {m["stage"]: load(m, "chat_template_from") for m in CONFIG["models"]}
CONDS = {m["stage"]: expand_conditions(m["conditions"]) for m in CONFIG["models"]}


def render(m, condition, first, second):
    """As the scorer renders it (tournament.py: template tokenizer, config's system prompt and turn markers)."""
    return build_prompt(condition, first, second, TEMPLATE[m["stage"]], CONFIG["system_prompt"],
                        turn_format=m.get("turn_format"), close_think=m.get("close_think", False))


def ids(stage, text):
    return TOK[stage](text, add_special_tokens=False).input_ids


def placement(condition):
    """(system message, lead of the user turn) for a condition, from the README's placement rules."""
    name, _, rest = condition.partition("@")
    persona, _, where = rest.partition("+")
    if not persona:
        return None, ""
    where = where or "system"
    return {"system": (prompts.PERSONAS[persona], ""),
            "roleplay": (prompts.ROLEPLAY_FRAME + prompts.PERSONAS_THIRD[persona], ""),
            "user": (None, prompts.PERSONAS[persona] + "\n\n"),
            "self": (None, prompts.PERSONAS_SELF[persona] + "\n\n")}[where]


def reference(condition, first, second, tok):
    """The official template's rendering: user content, optional system message, generation prompt for Assistant
    conditions; the default (date-only) system block removed when there's no system message; a user-turn condition cut
    where the turn would close."""
    name = condition.split("@")[0]
    spec = CONDITIONS[name]
    system, lead = placement(condition)
    body = lead + spec["body"].format(first=first.strip(), second=second.strip())
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": body}]
    has_reply = spec["prefill"] is not None
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=has_reply, date_string="26 Jul 2024")
    assert text.startswith(BOS + SYSTEM_HEAD)
    if system is None:
        assert text.count(SYSTEM_HEAD + "<|eot_id|>") == 1
        text = text.replace(SYSTEM_HEAD + "<|eot_id|>", "", 1)
    else:
        assert text.startswith(BOS + SYSTEM_HEAD + system + "<|eot_id|>")
    if not has_reply:
        assert text.endswith(body + "<|eot_id|>")
        return text.removesuffix("<|eot_id|>")
    assert text.endswith(body + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
    return text + spec["prefill"]


def test_config_follows_the_schema():
    assert (CONFIG["group"], CONFIG["run_name"], CONFIG["max_pairs"], CONFIG["seed"], CONFIG["system_prompt"]) == (
        "replication", "llama-405b", 15000, 0, "none")
    assert CONFIG["tasks_file"] == "data/tasks_n300_max500_seed0.jsonl" and len(TASKS) == 300
    assert [(m["name"], m["stage"]) for m in CONFIG["models"]] == [
        ("meta-llama/Llama-3.1-405B", "base"), ("meta-llama/Llama-3.1-405B-Instruct", "instruct")]
    assert BASE["chat_template_from"] == FINAL["name"] and "tokenizer_from" not in BASE
    assert CONDS["base"] == list(BASE_CONDITIONS) and CONDS["instruct"] == FINAL_CONDITIONS
    assert len(CONDS["base"]) == 4 and len(CONDS["instruct"]) == 44
    for m in CONFIG["models"]:
        v = m["vllm"]
        assert v["tensor_parallel_size"] == 8 and v["dtype"] == "bfloat16" and 0.85 <= v["gpu_memory_utilization"] <= 0.92


def test_template_is_the_sweeps_and_its_date_is_fixed():
    """The 405B Instruct's template is the 3.3 70B Instruct's (the sweep and persona runs), and it has no clock: the
    date lines are the same whatever day it is rendered."""
    tok, sweep = TEMPLATE["instruct"], AutoTokenizer.from_pretrained("unsloth/Llama-3.3-70B-Instruct")
    assert tok.chat_template == sweep.chat_template
    assert "strftime" not in tok.chat_template and 'set date_string = "26 Jul 2024"' in tok.chat_template
    for c in FINAL_CONDITIONS:
        assert render(FINAL, c, "FIRST", "SECOND") == build_prompt(c, "FIRST", "SECOND", sweep)


@pytest.mark.parametrize("m", CONFIG["models"], ids=lambda m: m["stage"])
def test_every_prompt_matches_the_official_template(m):
    tok = TEMPLATE["instruct"]
    for c in CONDS[m["stage"]]:
        for first, second in PAIRS:
            assert render(m, c, first, second) == reference(c, first, second, tok), c


@pytest.mark.parametrize("m", CONFIG["models"], ids=lambda m: m["stage"])
def test_turns_end_where_the_outcome_goes(m):
    for c in CONDS[m["stage"]]:
        spec = CONDITIONS[c.split("@")[0]]
        p = render(m, c, "FIRST", "SECOND")
        if spec["prefill"] is None:  # coinflip, user_choice and their persona variants: the user turn is left open
            assert p.endswith(spec["body"].format(first="FIRST", second="SECOND"))
            assert p.rindex("<|start_header_id|>user<|end_header_id|>\n\n") > p.rfind("<|eot_id|>")
            assert "assistant<|end_header_id|>" not in p
        else:
            body = spec["body"].format(first="FIRST", second="SECOND")
            assert p.endswith(body + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" + spec["prefill"])
            assert p.count("<|start_header_id|>assistant<|end_header_id|>") == 1


@pytest.mark.parametrize("m", CONFIG["models"], ids=lambda m: m["stage"])
def test_no_system_prompt_means_none_and_a_persona_adds_only_its_block(m):
    for c in CONDS[m["stage"]]:
        system, lead = placement(c)
        plain = render(m, c.split("@")[0], "FIRST", "SECOND")
        p = render(m, c, "FIRST", "SECOND")
        assert "<|start_header_id|>system" not in plain and plain.startswith(BOS + "<|start_header_id|>user")
        if system is not None:  # system prompt or roleplay: one system block after the BOS, date lines included
            head = BOS + SYSTEM_HEAD + system + "<|eot_id|>"
            assert p == head + plain[len(BOS):]
            assert ids(m["stage"], p) == ids(m["stage"], head) + ids(m["stage"], plain[len(BOS):])
        else:  # no system block; a persona in the user turn is the only addition
            assert "<|start_header_id|>system" not in p
            i = plain.index("I would like you")
            assert p == plain[:i] + lead + plain[i:]


@pytest.mark.parametrize("m", CONFIG["models"], ids=lambda m: m["stage"])
def test_markers_are_single_special_ids_and_one_bos(m):
    tok = TOK[m["stage"]]
    marker_ids = {s: tok.encode(s, add_special_tokens=False) for s in MARKERS}
    assert all(len(v) == 1 and v[0] in tok.all_special_ids + list(tok.added_tokens_decoder) for v in marker_ids.values())
    bos = marker_ids[BOS][0]
    assert bos == tok.bos_token_id == 128000
    for c in CONDS[m["stage"]]:
        for first, second in PAIRS[:20]:
            p = render(m, c, first, second)
            enc = ids(m["stage"], p)
            assert enc[0] == bos and enc.count(bos) == 1
            for s, (i,) in marker_ids.items():
                assert enc.count(i) == p.count(s), (c, s)
            # Markers are never spelled out in characters.
            assert "<|" not in tok.decode([t for t in enc if t not in tok.all_special_ids + list(tok.added_tokens_decoder)])


def test_base_reads_the_same_ids_as_the_instruct():
    """Base and Instruct ship the same tokenizer.json, so every rendered prompt is the same ids in both."""
    assert TOK["base"].backend_tokenizer.to_str() == TOK["instruct"].backend_tokenizer.to_str()
    for c in CONDS["instruct"]:
        texts = [render(FINAL, c, first, second) for first, second in PAIRS]
        assert TOK["base"](texts, add_special_tokens=False).input_ids == TOK["instruct"](texts, add_special_tokens=False).input_ids
    for c in CONDS["base"]:  # and the base renders exactly what the Instruct does
        assert [render(BASE, c, a, b) for a, b in PAIRS] == [render(FINAL, c, a, b) for a, b in PAIRS]


@pytest.mark.parametrize("m", CONFIG["models"], ids=lambda m: m["stage"])
def test_outcome_tokens_follow_the_prompt(m):
    stage = m["stage"]
    for c in CONDS[stage]:
        out = outcome_token_ids(c, TOK[stage])
        assert len(set(out)) == 2 and out == outcome_token_ids(c, TOK["instruct"])
        strings = CONDITIONS[c.split("@")[0]]["outcomes"]
        for first, second in PAIRS[:30]:
            p = render(m, c, first, second)
            base_ids = ids(stage, p)
            for s, i in zip(strings, out):
                assert ids(stage, p + s) == base_ids + [i], (c, s)


@pytest.mark.parametrize("m", CONFIG["models"], ids=lambda m: m["stage"])
def test_max_model_len_fits_the_longest_prompt(m):
    """The two longest tasks under every condition, plus the outcome token, with room to spare."""
    tok = TOK[m["stage"]]
    n = [len(tok(t["prompt"].strip(), add_special_tokens=False).input_ids) for t in TASKS]
    top = sorted(range(len(TASKS)), key=lambda i: -n[i])[:6]
    longest = max(len(ids(m["stage"], render(m, c, TASKS[a]["prompt"], TASKS[b]["prompt"])))
                  for c in CONDS[m["stage"]] for a in top for b in top if a != b)
    assert longest + 1 <= m["vllm"]["max_model_len"] / 1.5, longest


@pytest.mark.parametrize("m", CONFIG["models"], ids=lambda m: m["stage"])
def test_download_is_the_top_level_weights_only(m):
    try:
        files = list(HfApi().list_repo_tree(m["name"], recursive=True))
    except Exception as e:  # offline
        pytest.skip(f"no file listing: {e}")
    sizes = {f.path: f.size for f in files if hasattr(f, "size")}
    chosen = list(filter_repo_objects(sizes, allow_patterns=m["download"]["allow_patterns"]))
    assert not any(p.startswith("original/") for p in chosen)
    weights = [p for p in sizes if p.endswith(".safetensors") and "/" not in p]
    assert set(weights) <= set(chosen) and "config.json" in chosen and "tokenizer.json" in chosen
    assert abs(sum(sizes[p] for p in chosen) / 1e9 - m["download"]["gb"]) < 0.5
