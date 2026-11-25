"""Choice rules (README, "Choice rules: tickets, languages and position"): the rule wording, that a user-turn rule only adds
the lead and a system-prompt rule only adds the system message (in every pilot model's own format, DeepSeek V4 against its
encoder), the ticket layout, the mixed-language text, the balance of the drawn assignments, the pilot's pair subset, and
that the configs cover every condition."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

import choice_rules
import prompts
from prompts import CHOICE_BASES, CHOICE_LEADS, CHOICE_RULES, CONDITIONS, IFCHOOSE_LEAD, build_prompt, outcome_token_ids
from tournament import pair_fields, select_pairs, subset_pairs

CONFIGS = {p.stem: yaml.safe_load(p.read_text()) for p in sorted(Path("configs/choice_rules").glob("*.yaml"))}
TASKS = choice_rules.decorate(choice_rules._read_jsonl(choice_rules.TASKS_PATH))
ASSIGN = json.loads(choice_rules.ASSIGN_PATH.read_text())
RULE_CONDITIONS = [f"assistant_{k}_{r}_{p}" for k, rules in CHOICE_RULES.items() for r in rules for p in ("user", "system")]
PILOT = ["assistant", "assistant_tickets", "assistant_mixed"] + RULE_CONDITIONS


def test_rule_wording():
    """Every sentence in full."""
    user = {(k, r): CHOICE_LEADS["user"].format(rule=t) for k, rules in CHOICE_RULES.items() for r, t in rules.items()}
    system = {(k, r): CHOICE_LEADS["system"].format(rule=t) for k, rules in CHOICE_RULES.items() for r, t in rules.items()}
    assert CHOICE_LEADS["user"] == IFCHOOSE_LEAD
    assert list(user.values()) == [
        "If I ask you to choose between tasks in this message, choose the one with the lower ticket number.\n\n",
        "If I ask you to choose between tasks in this message, choose the one with the higher ticket number.\n\n",
        "If I ask you to choose between tasks in this message, choose the one whose description is in Spanish.\n\n",
        "If I ask you to choose between tasks in this message, choose the one whose description is in English.\n\n",
        "If I ask you to choose between tasks in this message, choose the one listed first.\n\n",
        "If I ask you to choose between tasks in this message, choose the one listed second.\n\n"]
    assert list(system.values()) == [
        "If the user asks you to choose between tasks, choose the one with the lower ticket number.",
        "If the user asks you to choose between tasks, choose the one with the higher ticket number.",
        "If the user asks you to choose between tasks, choose the one whose description is in Spanish.",
        "If the user asks you to choose between tasks, choose the one whose description is in English.",
        "If the user asks you to choose between tasks, choose the one listed first.",
        "If the user asks you to choose between tasks, choose the one listed second."]
    assert CONDITIONS["assistant_tickets"]["body"] == (
        "I would like you to perform one of the following two tasks, whichever you choose.\n\n"
        "Task A (ticket #{first_ticket}): {first}\n\nTask B (ticket #{second_ticket}): {second}")
    assert CONDITIONS["assistant_mixed"]["body"] == CONDITIONS["assistant"]["body"]
    for (kind, rule), lead in user.items():
        base = CONDITIONS[CHOICE_BASES[kind]]
        u, s = CONDITIONS[f"assistant_{kind}_{rule}_user"], CONDITIONS[f"assistant_{kind}_{rule}_system"]
        assert u == {**base, "body": lead + base["body"]}
        assert s == {**base, "system": system[(kind, rule)]}
        for spec in (u, s):
            assert (spec["prefill"], spec["outcomes"]) == ("I'll choose Task", (" A", " B"))
            assert spec.get("task_field", "prompt") == ("prompt_mixed" if kind == "mixed" else "prompt")


def test_assignments():
    """Distinct tickets in 100-999 and half of each (source, harmful) stratum in Spanish; neither correlates with any
    balance column; each source's mean ticket rank is central; the draw reproduces from its seed."""
    assign = ASSIGN["tasks"]
    assert set(assign) == {t["id"] for t in TASKS} and len(assign) == 300
    tickets = np.array([assign[t["id"]]["ticket"] for t in TASKS])
    assert len(set(tickets)) == 300 and tickets.min() >= 100 and tickets.max() <= 999
    spanish = np.array([assign[t["id"]]["lang"] == "es" for t in TASKS])
    strata = {}
    for t, es in zip(TASKS, spanish):
        strata.setdefault((t["origin"], choice_rules.is_harmful(t)), []).append(es)
    for v in strata.values():
        assert abs(2 * sum(v) - len(v)) <= 1
    assert all(abs(r) < 0.05 for c in ASSIGN["correlations"].values() for r in c.values())
    ranks = tickets.argsort().argsort() / 299
    for o in {t["origin"] for t in TASKS}:
        assert 0.42 <= ranks[[t["origin"] == o for t in TASKS]].mean() <= 0.58
    if not all(Path(f"outputs/{r}/analysis/utilities.csv").exists() for r in choice_rules.SWEEP_RUNS.values()):
        pytest.skip("sweep utilities not on this machine")
    cols = choice_rules.balance_columns(TASKS)
    assert choice_rules.draw_assignments(TASKS, cols, seed=ASSIGN["seed"])[0] == assign
    for name, v in cols.items():
        assert abs(np.corrcoef(tickets, v)[0, 1]) < 0.05 and abs(np.corrcoef(spanish, v)[0, 1]) < 0.05, name


def test_decorate():
    es = {t["id"]: t for t in choice_rules._read_jsonl(choice_rules.TASKS_ES_PATH)}
    english = {t["id"]: t for t in choice_rules._read_jsonl(choice_rules.TASKS_PATH)}
    for t in TASKS:
        assert t["tags"] == {"ticket": ASSIGN["tasks"][t["id"]]["ticket"]}
        assert t["prompt"] == english[t["id"]]["prompt"]
        want = es[t["id"]]["prompt"] if ASSIGN["tasks"][t["id"]]["lang"] == "es" else english[t["id"]]["prompt"]
        assert t["prompt_mixed"] == want and t["lang"] == ASSIGN["tasks"][t["id"]]["lang"]
    assert sum(t["prompt_mixed"] != t["prompt"] for t in TASKS) == sum(t["lang"] == "es" for t in TASKS)
    assert pair_fields(TASKS[0], TASKS[1]) == {"first_ticket": TASKS[0]["tags"]["ticket"],
                                               "second_ticket": TASKS[1]["tags"]["ticket"]}


def test_pair_subset():
    sweep = select_pairs(300, 15000, 0)
    sub = subset_pairs(sweep, 2000, 0)
    assert len(sub) == 2000 and set(sub) <= set(sweep) and sub == sorted(sub) == subset_pairs(sweep, 2000, 0)
    assert subset_pairs(sweep, None, 0) is sweep


def test_configs_cover_every_condition():
    assert sorted(CONFIGS) == ["deepseek-v4-flash", "gemma4-31b", "glm-4.5-air", "llama-70b"]
    for key, cfg in CONFIGS.items():
        assert (cfg["group"], cfg["run_name"], cfg["tags"]) == ("choice_rules", f"choice-rules-{key}", [key])
        assert cfg["choice_rules"] is True and cfg["system_prompt"] == "none"
        assert (cfg["max_pairs"], cfg["pair_subset"], cfg["seed"]) == (15000, 2000, 0)
        assert cfg["tasks_file"] == "data/tasks_n300_max500_seed0.jsonl"
        (m,) = cfg["models"]
        assert m["stage"] == "instruct" and sorted(m["conditions"]) == sorted(PILOT) and len(m["conditions"]) == 15


def render(m, condition, a, b, tok):
    """As tournament.py renders a prompt for model entry m."""
    field = CONDITIONS[condition].get("task_field", "prompt")
    return build_prompt(condition, TASKS[a][field], TASKS[b][field], tok, "none", turn_format=m.get("turn_format"),
                        fields=pair_fields(TASKS[a], TASKS[b]))


PAIRS = [(0, 1), (1, 0), (17, 250), (250, 17)]
HF_MODELS = ["llama-70b", "gemma4-31b", "glm-4.5-air"]


@pytest.mark.parametrize("key", HF_MODELS)
def test_rules_only_add_the_lead_or_the_system_message(key):
    m = CONFIGS[key]["models"][0]
    tok = AutoTokenizer.from_pretrained(m["name"])
    ids = lambda s: tok(s, add_special_tokens=False).input_ids
    for kind, rules in CHOICE_RULES.items():
        base = CHOICE_BASES[kind]
        for rule, text in rules.items():
            for a, b in PAIRS:
                plain = render(m, base, a, b, tok)
                i = plain.index("I would like you")
                user = render(m, f"assistant_{kind}_{rule}_user", a, b, tok)
                assert user == plain[:i] + IFCHOOSE_LEAD.format(rule=text) + plain[i:]
                assert ids(user)[-len(ids(plain[i:])):] == ids(plain[i:])
                field = CONDITIONS[base].get("task_field", "prompt")
                system = render(m, f"assistant_{kind}_{rule}_system", a, b, tok)
                assert system == build_prompt(base, TASKS[a][field], TASKS[b][field], tok,
                                              CHOICE_LEADS["system"].format(rule=text), fields=pair_fields(TASKS[a], TASKS[b]))
                assert CHOICE_LEADS["system"].format(rule=text) in system[:system.index("I would like you")]
                assert system.endswith(plain[i:])
            for p in ("user", "system"):
                assert outcome_token_ids(f"assistant_{kind}_{rule}_{p}", tok) == outcome_token_ids("assistant", tok)


@pytest.mark.parametrize("key", HF_MODELS + ["deepseek-v4-flash"])
def test_layouts(key):
    """The ticket numbers sit by the labels, the mixed text is each task's assigned language, and the plain prompt is the
    sweep's."""
    m = CONFIGS[key]["models"][0]
    tok = None if "turn_format" in m else AutoTokenizer.from_pretrained(m["name"])
    for a, b in PAIRS:
        t = render(m, "assistant_tickets", a, b, tok)
        assert (f"Task A (ticket #{TASKS[a]['tags']['ticket']}): {TASKS[a]['prompt'].strip()}\n\n"
                f"Task B (ticket #{TASKS[b]['tags']['ticket']}): {TASKS[b]['prompt'].strip()}") in t
        mixed = render(m, "assistant_mixed", a, b, tok)
        assert f"Task A: {TASKS[a]['prompt_mixed'].strip()}\n\nTask B: {TASKS[b]['prompt_mixed'].strip()}" in mixed
        assert render(m, "assistant", a, b, tok) == build_prompt(
            "assistant", TASKS[a]["prompt"], TASKS[b]["prompt"], tok, "none", turn_format=m.get("turn_format"))


def test_deepseek_v4_against_its_encoder():
    """DeepSeek V4 has no Jinja template: every pilot prompt against encode_messages(thinking_mode="chat") of the same
    conversation, plus the prefill (as tests/test_replication_deepseek_v4_flash.py does for the replication)."""
    m = CONFIGS["deepseek-v4-flash"]["models"][0]
    path = hf_hub_download(m["name"], "encoding/encoding_dsv4.py")
    spec = importlib.util.spec_from_file_location("encoding_dsv4", path)
    encoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(encoder)
    tok = AutoTokenizer.from_pretrained(m["name"])
    for condition in PILOT:
        c = CONDITIONS[condition]
        field = c.get("task_field", "prompt")
        for a, b in PAIRS:
            body = c["body"].format(first=TASKS[a][field].strip(), second=TASKS[b][field].strip(),
                                    **pair_fields(TASKS[a], TASKS[b]))
            msgs = ([{"role": "system", "content": c["system"]}] if c.get("system") else []) + [
                {"role": "user", "content": body}]
            assert render(m, condition, a, b, None) == encoder.encode_messages(msgs, thinking_mode="chat") + c["prefill"]
        assert outcome_token_ids(condition, tok) == outcome_token_ids("assistant", tok)
