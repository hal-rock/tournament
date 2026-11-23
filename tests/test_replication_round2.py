"""Round 2 on the first round's six finals (PLAN_replication2.md; configs/replication/round2/<key>.yaml).

- Each config is its first-round instruct entry plus the tool family and the raw-text document start, so the gate's
  conditions (the plain Assistant and coins) render exactly the prompts the first round scored.
- The new conditions render: the tool's coin under roleplay, the length rules with each coin, the raw-text stories.
- The document start tokenizes to the model's own marker ids (Nemotron-4: SentencePiece's ids for a NeMo document).
- max_model_len holds the longest prompt.
Needs only tokenizer and template files.
"""

import itertools
import json
import sys
from pathlib import Path

import pytest
import yaml
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import score_replication  # noqa: E402
from prompts import CONDITIONS, ROLEPLAY, parse_condition  # noqa: E402
from replication import CONDITION_SETS, LENGTH, LENGTH_TOOL, STORIES, TOOL_ROLEPLAY, expand_conditions  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
KEYS = ["deepseek-v4-flash", "llama-405b", "deepseek-v3", "kimi-k2", "trinity-large", "nemotron4-340b"]
TASKS = [json.loads(line) for line in (ROOT / "data/tasks_n300_max500_seed0.jsonl").read_text().splitlines()]
PAIRS = [(0, 1), (150, 299), (299, 150)]
# The document start's ids: the model's BOS (or Nemotron's EOD and SentencePiece's dummy prefix, checked separately).
DOC_IDS = {"deepseek-v4-flash": [0], "llama-405b": [128000], "deepseek-v3": [0], "kimi-k2": [163584],
           "trinity-large": [0]}


def load(key):
    r2 = yaml.safe_load((ROOT / "configs/replication/round2" / f"{key}.yaml").read_text())
    r1 = yaml.safe_load((ROOT / "configs/replication" / f"{key}.yaml").read_text())
    (m,) = r2["models"]
    (m1,) = [x for x in r1["models"] if x["stage"] == "instruct"]
    return r2, m, r1, m1


@pytest.fixture(scope="module", params=KEYS)
def model(request):
    key = request.param
    r2, m, r1, m1 = load(key)
    if key == "kimi-k2":  # the remote-code tokenizer, rebuilt from its data files (tests/test_replication_kimi_k2.py)
        from test_replication_kimi_k2 import stand_in
        tok = stand_in(m["name"], m["revision"], m["revision"])
        return key, m, tok, tok
    tok = AutoTokenizer.from_pretrained(m.get("tokenizer_from", m["name"]), revision=m.get("revision"))
    template = AutoTokenizer.from_pretrained(m.get("chat_template_from", m["name"]), revision=m.get("revision"))
    return key, m, tok, template


def render(m, tok, template, condition, a, b):
    ids, text = score_replication.render_condition(condition, m, TASKS, [(a, b)], template, tok, "none")
    return text, ids[0]


def test_configs_are_the_first_rounds_entries():
    for key in KEYS:
        r2, m, r1, m1 = load(key)
        assert (r2["group"], r2["run_name"]) == ("replication2", f"r2-{key}")
        for k in ("wandb_project", "tasks_file", "seed", "max_pairs", "system_prompt", "top_k_tokens"):
            assert r2[k] == r1[k], (key, k)
        extra = {"tool_family", "doc_prefix", "conditions", "vllm"}
        assert {k: v for k, v in m.items() if k not in extra} == {k: v for k, v in m1.items() if k not in extra}, key
        # vLLM settings as the first round's, except a longer max_model_len (1,024 -> 1,536): the tool coin prompts under the
        # roleplay framing run to 872 tokens (V4 Flash) (test_max_model_len_holds_the_longest_prompt). It sizes the KV cache, not
        # the computation; the gate checks the re-scored conditions against the first round's anyway.
        assert {**m["vllm"], "max_model_len": 1024} == m1["vllm"] and m["vllm"]["max_model_len"] == 1536, key
        tool_cfg = ROOT / "configs/replication/tool" / f"{key}.yaml"
        if tool_cfg.exists():
            assert m["tool_family"] == yaml.safe_load(tool_cfg.read_text())["models"][0]["tool_family"]
            assert expand_conditions(m["conditions"]) == CONDITION_SETS["round2"]
        else:
            assert "tool_family" not in m and expand_conditions(m["conditions"]) == CONDITION_SETS["round2_notool"]
    assert CONDITION_SETS["round2"] == ["assistant", "coinflip", "coinflip_tool"] + TOOL_ROLEPLAY + LENGTH + LENGTH_TOOL + STORIES


def test_gate_prompts_are_the_first_rounds(model):
    """The gate's conditions render the same ids with the round-2 entry as with the first round's."""
    key, m, tok, template = model
    _, _, _, m1 = load(key)
    for c in ("assistant", "coinflip") + (("coinflip_tool",) if "tool_family" in m else ()):
        m1c = {**m1, "tool_family": m.get("tool_family")}
        for a, b in PAIRS:
            assert render(m, tok, template, c, a, b) == render(m1c, tok, template, c, a, b)


def test_new_conditions_render(model):
    key, m, tok, template = model
    for c in expand_conditions(m["conditions"]):
        name = parse_condition(c)[0]
        for a, b in PAIRS[:2]:
            p, ids = render(m, tok, template, c, a, b)
            assert ids == tok(p, add_special_tokens=False).input_ids
            if "ifchoose" in name:
                rule = "the one with the shorter description" if "shorter" in name else "the one with the longer description"
                assert f"If I ask you to choose between tasks in this message, choose {rule}.\n\nI would like" in p
            if c.endswith("+roleplay"):
                assert ROLEPLAY[parse_condition(c)[1]] in p and "flip_coin" in p


def test_raw_text_document_start(model):
    key, m, tok, template = model
    for c in STORIES:
        a, b = PAIRS[1]
        p, ids = render(m, tok, template, c, a, b)
        body = CONDITIONS[c]["body"].format(first=TASKS[a]["prompt"].strip(), second=TASKS[b]["prompt"].strip())
        assert p == m["doc_prefix"] + body
        if key == "nemotron4-340b":  # EOD, then the document as NeMo's SentencePiece encodes it (dummy prefix included)
            from test_replication_nemotron4_340b import SP
            assert ids == [3] + SP.encode_as_ids(body), (ids[:6], SP.encode_as_ids(body)[:5])
        else:
            assert ids[:len(DOC_IDS[key])] == DOC_IDS[key] and ids[len(DOC_IDS[key])] not in DOC_IDS[key], ids[:4]
            assert ids == DOC_IDS[key] + tok(body, add_special_tokens=False).input_ids


def test_max_model_len_holds_the_longest_prompt(model):
    key, m, tok, template = model
    n = [len(tok(x["prompt"].strip(), add_special_tokens=False).input_ids) for x in TASKS]
    top = sorted(range(len(TASKS)), key=lambda i: -n[i])[:5]
    longest = max(len(render(m, tok, template, c, a, b)[1]) for c in expand_conditions(m["conditions"])
                  for a, b in itertools.permutations(top, 2))
    print(key, "longest prompt", longest)
    assert 1.5 * (longest + 1) <= m["vllm"]["max_model_len"]
