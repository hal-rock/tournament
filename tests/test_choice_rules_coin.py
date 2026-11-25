"""Choice rules, the coinflip (README, "Choice rules: the coinflip"): the coin's ticket and mixed layouts, that a rule only
adds the lead to the coinflip's user turn (in every model's own format, DeepSeek V4 against its encoder), the benign pair
filter, same-length batching, and that the configs cover every qualifying model and kind."""

import importlib.util
from pathlib import Path

import pytest
import yaml
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

import choice_rules
from prompts import CHOICE_LAYOUTS, CHOICE_LEADS, CHOICE_RULES, CONDITIONS, IFCHOOSE_LEAD, build_prompt, outcome_token_ids
from tournament import benign_subset, length_batches, pair_fields, select_pairs

CONFIGS = {p.stem: yaml.safe_load(p.read_text()) for p in sorted(Path("configs/choice_rules_coin").glob("*.yaml"))}
ALL_TASKS = choice_rules.decorate(choice_rules._read_jsonl(choice_rules.TASKS_PATH))
TASKS, PAIRS = benign_subset(ALL_TASKS, select_pairs(300, 15000, 0))
QUALIFY = {"llama-70b": ("tickets", "mixed", "position"), "gemma4-31b": ("tickets", "mixed", "position"),
           "glm-4.5-air": ("mixed", "position"), "deepseek-v4-flash": ("tickets", "mixed", "position")}
SAMPLE = PAIRS[:2] + [(b, a) for a, b in PAIRS[:2]] + [PAIRS[-1]]


def kind_conditions(kind):
    layouts = [] if kind == "position" else [CHOICE_LAYOUTS["assistant"][kind], CHOICE_LAYOUTS["coinflip"][kind]]
    rules = [f"{r}_{kind}_{rule}_user" for r in ("assistant", "coinflip") for rule in CHOICE_RULES[kind]]
    return ["assistant", "coinflip"] + layouts + rules


def test_coin_layouts_and_rules():
    assert CONDITIONS["coinflip_tickets"]["body"] == (
        "I would like you to perform one of the following two tasks, depending on the outcome of a coinflip.\n\n"
        "Task A (ticket #{first_ticket}, if heads): {first}\n\nTask B (ticket #{second_ticket}, if tails): {second}\n\n"
        "Okay, I'm flipping the coin now ... it came up")
    assert CONDITIONS["coinflip_mixed"] == {**CONDITIONS["coinflip"], "task_field": "prompt_mixed"}
    for kind, rules in CHOICE_RULES.items():
        base = CONDITIONS[CHOICE_LAYOUTS["coinflip"][kind]]
        for rule, text in rules.items():
            assert CONDITIONS[f"coinflip_{kind}_{rule}_user"] == {**base, "body": IFCHOOSE_LEAD.format(rule=text) + base["body"]}
            assert CONDITIONS[f"coinflip_{kind}_{rule}_system"] == {**base, "system": CHOICE_LEADS["system"].format(rule=text)}
            assert base["prefill"] is None and base["outcomes"] == (" heads", " tails")


def test_benign_subset():
    assert len(TASKS) == 257 and len(PAIRS) == 11065
    assert not any(choice_rules.is_harmful(t) for t in TASKS)
    by_id = {t["id"]: i for i, t in enumerate(ALL_TASKS)}
    full = set(select_pairs(300, 15000, 0))
    assert all((by_id[TASKS[a]["id"]], by_id[TASKS[b]["id"]]) in full for a, b in PAIRS)
    benign_full = [p for p in full if not (choice_rules.is_harmful(ALL_TASKS[p[0]]) or choice_rules.is_harmful(ALL_TASKS[p[1]]))]
    assert len(benign_full) == len(PAIRS)


def test_length_batches():
    lengths = [5, 3, 5, 3, 3, 7, 5, 5]
    assert length_batches(lengths, 3) == [[1, 3, 4], [0, 2, 6], [7, 5]]
    exact = length_batches(lengths, 3, exact_length=True)
    assert exact == [[1, 3, 4], [0, 2, 6], [7], [5]]
    assert all(len({lengths[k] for k in b}) == 1 for b in exact)
    assert sorted(k for b in exact for k in b) == list(range(len(lengths)))


def test_configs_cover_every_qualifying_model_and_kind():
    assert sorted(CONFIGS) == sorted(f"{m}-{k}" for m, kinds in QUALIFY.items() for k in kinds)
    for name, cfg in CONFIGS.items():
        model, kind = next((m, k) for m, ks in QUALIFY.items() for k in ks if name == f"{m}-{k}")
        assert (cfg["group"], cfg["run_name"], cfg["tags"]) == ("choice_rules_coin", f"choice-rules-coin-{name}", [model, kind])
        assert cfg["choice_rules"] is True and cfg["benign_only"] is True and cfg["system_prompt"] == "none"
        assert (cfg["max_pairs"], cfg["seed"]) == (15000, 0) and "pair_subset" not in cfg
        (m,) = cfg["models"]
        assert m["stage"] == "instruct" and m["conditions"] == kind_conditions(kind)
        if model == "deepseek-v4-flash":
            assert cfg["exact_length_batches"] is True and cfg["progress_every"] > 2 * len(PAIRS)
        else:
            assert not cfg.get("exact_length_batches")


def render(m, condition, a, b, tok):
    field = CONDITIONS[condition].get("task_field", "prompt")
    return build_prompt(condition, TASKS[a][field], TASKS[b][field], tok, "none", turn_format=m.get("turn_format"),
                        fields=pair_fields(TASKS[a], TASKS[b]))


@pytest.mark.parametrize("key", ["llama-70b", "gemma4-31b", "glm-4.5-air"])
def test_coin_rules_only_add_the_lead(key):
    """The rule prompt is the layout's coinflip with the lead before "I would like you"; it ends inside the user's turn at
    "it came up", and the layout's tokens from "I would like you" on are unchanged."""
    m = CONFIGS[f"{key}-position"]["models"][0]
    tok = AutoTokenizer.from_pretrained(m["name"])
    ids = lambda s: tok(s, add_special_tokens=False).input_ids
    for kind in QUALIFY[key]:
        base = CHOICE_LAYOUTS["coinflip"][kind]
        for rule, text in CHOICE_RULES[kind].items():
            cond = f"coinflip_{kind}_{rule}_user"
            assert outcome_token_ids(cond, tok) == outcome_token_ids("coinflip", tok)
            for a, b in SAMPLE:
                plain = render(m, base, a, b, tok)
                assert plain.endswith("Okay, I'm flipping the coin now ... it came up")
                i = plain.index("I would like you")
                p = render(m, cond, a, b, tok)
                assert p == plain[:i] + IFCHOOSE_LEAD.format(rule=text) + plain[i:]
                assert ids(p)[-len(ids(plain[i:])):] == ids(plain[i:])
                if kind == "tickets":
                    assert f"Task A (ticket #{TASKS[a]['tags']['ticket']}, if heads): {TASKS[a]['prompt'].strip()}" in p
                if kind == "mixed":
                    assert f"Task B (if tails): {TASKS[b]['prompt_mixed'].strip()}" in p


def test_deepseek_v4_coin_against_its_encoder():
    """Every V4 Flash coin prompt is the encoder's rendering of the same user message, minus the "<｜Assistant｜></think>"
    that would close the turn (tests/test_replication_deepseek_v4_flash.py's check for an open user turn)."""
    m = CONFIGS["deepseek-v4-flash-tickets"]["models"][0]
    path = hf_hub_download(m["name"], "encoding/encoding_dsv4.py")
    spec = importlib.util.spec_from_file_location("encoding_dsv4", path)
    encoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(encoder)
    for kind in QUALIFY["deepseek-v4-flash"]:
        for cond in kind_conditions(kind):
            c = CONDITIONS[cond]
            field = c.get("task_field", "prompt")
            for a, b in SAMPLE:
                body = c["body"].format(first=TASKS[a][field].strip(), second=TASKS[b][field].strip(),
                                        **pair_fields(TASKS[a], TASKS[b]))
                reference = encoder.encode_messages([{"role": "user", "content": body}], thinking_mode="chat")
                p = render(m, cond, a, b, None)
                if c["prefill"] is None:
                    assert p + "<｜Assistant｜></think>" == reference
                else:
                    assert p == reference + c["prefill"]
