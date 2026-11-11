"""Nemotron-4 340B in the replication (configs/replication/nemotron4-340b.yaml). Reads the config, renders every condition
it runs on the real tasks as the scorer does (prompts.build_prompt, then the stage's tokenizer with
add_special_tokens=False), and checks text and ids against NeMo's own chat format and tokenizer, written out here from
NVIDIA's files:

- the format: NeMo's SFT chat dataset (nemo/collections/nlp/data/language_modeling/megatron/gpt_sft_chat_dataset.py,
  NeMo v1.22.0 = 0b7467eb, unchanged in v2.0.0rc0 = 265bd739: `_get_header_conversation_type_mask_role`,
  `_add_speaker_and_signal`) with the chat tokens in the Instruct's model_config.yaml;
- the ids: NeMo's SentencePieceTokenizer with sentencepiece_legacy false, which is `encode_as_ids` on the whole string
  (nemo/collections/common/tokenizers/sentencepiece_tokenizer.py, `text_to_ids`), with the SentencePiece model from
  NVIDIA's own repo. No BOS: the SFT config has add_bos false, and NeMo-Aligner's DPO dataset (v0.3.0.trtllm, e9eabffb,
  `DPOModelDataset.encode`) adds none.

Run it on the node too, in the scoring environment: the HF tokenizer's handling of the start of the string has changed
across transformers versions, and this test is what pins it."""

import json
from pathlib import Path

import pytest
import sentencepiece as spm
import yaml
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import filter_repo_objects
from transformers import AutoConfig, AutoTokenizer

import prompts
from prompts import CONDITIONS, build_prompt, outcome_token_ids
from replication import BASE_CONDITIONS, FINAL_CONDITIONS, expand_conditions

CONFIG = yaml.safe_load(Path("configs/replication/nemotron4-340b.yaml").read_text())
BASE, FINAL = CONFIG["models"]
TASKS = [json.loads(line) for line in open(CONFIG["tasks_file"])]
PAIRS = [(TASKS[i]["prompt"], TASKS[(i + 1) % len(TASKS)]["prompt"]) for i in range(len(TASKS))]

NVIDIA = "nvidia/Nemotron-4-340B-Instruct"
NEMO_CONFIG = yaml.safe_load(Path(hf_hub_download(NVIDIA, "model_config.yaml")).read_text())
# The tokenizer NeMo trained the Instruct with (model_config.yaml, tokenizer.tokenizer_model; the base's
# nemotron_2_256k.model files are the same bytes).
SP = spm.SentencePieceProcessor(model_file=hf_hub_download(
    NVIDIA, NEMO_CONFIG["tokenizer"]["tokenizer_model"].removeprefix("nemo:")))
CHAT = NEMO_CONFIG["data"]["chat_prompt_tokens"]
DUMMY_PREFIX = "▁"  # SentencePiece's whitespace symbol, its own piece (id 252303)


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
    """(system message, lead of the user turn) for a condition, from the README's placement rules. No system message is
    NeMo's empty one, the card's recommended default."""
    persona, _, where = condition.partition("@")[2].partition("+")
    if not persona:
        return "", ""
    return {"system": (prompts.PERSONAS[persona], ""),
            "roleplay": (prompts.ROLEPLAY_FRAME + prompts.PERSONAS_THIRD[persona], ""),
            "user": ("", prompts.PERSONAS[persona] + "\n\n"),
            "self": ("", prompts.PERSONAS_SELF[persona] + "\n\n")}[where or "system"]


def nemo_text(condition, first, second):
    """NeMo's conversation text: the header `system_turn_start + "System" + end_of_name + system + end_of_turn`, then the
    user turn `turn_start + "User" + end_of_name + text`. An Assistant condition closes it (`end_of_turn`) and opens the
    reply as NeMo's generation prompt and its SFT data do (`turn_start + "Assistant" + end_of_name`, no label), then the
    prefill. A user-turn condition stops inside the user's text."""
    spec = CONDITIONS[condition.split("@")[0]]
    system, lead = placement(condition)
    body = lead + spec["body"].format(first=first.strip(), second=second.strip())
    text = CHAT["system_turn_start"] + "System" + CHAT["end_of_name"] + system + CHAT["end_of_turn"]
    text += CHAT["turn_start"] + "User" + CHAT["end_of_name"] + body
    if spec["prefill"] is None:
        return text
    return text + CHAT["end_of_turn"] + CHAT["turn_start"] + "Assistant" + CHAT["end_of_name"] + spec["prefill"]


def test_nemo_reference_is_what_nvidia_documents():
    """The chat tokens and tokenizer settings this test builds on, as NVIDIA's files state them, and the card's
    single-turn format with the recommended empty system prompt."""
    assert CHAT == {"system_turn_start": "<extra_id_0>", "turn_start": "<extra_id_1>", "label_start": "<extra_id_2>",
                    "end_of_turn": "\n", "end_of_name": "\n"}
    assert NEMO_CONFIG["tokenizer"]["library"] == "sentencepiece" and NEMO_CONFIG["tokenizer"]["sentencepiece_legacy"] is False
    assert NEMO_CONFIG["data"]["train_ds"]["add_bos"] is False
    card = Path(hf_hub_download(NVIDIA, "README.md")).read_text()
    assert 'PROMPT_TEMPLATE = """<extra_id_0>System\n\n<extra_id_1>User\n{prompt}\n<extra_id_1>Assistant\n"""' in card
    assert "add_BOS=False" in card
    assert nemo_text("assistant", "X", "Y") == "<extra_id_0>System\n\n<extra_id_1>User\n" + CONDITIONS["assistant"][
        "body"].format(first="X", second="Y") + "\n<extra_id_1>Assistant\nI'll choose Task"
    # mgoin's tokenizer.model is NVIDIA's SentencePiece model, byte for byte.
    assert (Path(hf_hub_download(FINAL["name"], "tokenizer.model")).read_bytes()
            == Path(hf_hub_download(NVIDIA, NEMO_CONFIG["tokenizer"]["tokenizer_model"].removeprefix("nemo:"))).read_bytes())
    # Raw SentencePiece puts its dummy prefix before the first marker: NeMo's ids start with the bare whitespace piece.
    assert SP.encode_as_ids("<extra_id_0>System") == [SP.piece_to_id(DUMMY_PREFIX), 4, SP.piece_to_id("System")]


def test_config_follows_the_schema():
    assert (CONFIG["group"], CONFIG["run_name"], CONFIG["max_pairs"], CONFIG["seed"], CONFIG["system_prompt"]) == (
        "replication", "nemotron4-340b", 15000, 0, "none")
    assert CONFIG["tasks_file"] == "data/tasks_n300_max500_seed0.jsonl" and len(TASKS) == 300
    assert [(m["name"], m["stage"]) for m in CONFIG["models"]] == [
        ("mgoin/Nemotron-4-340B-Base-hf", "base"), ("mgoin/Nemotron-4-340B-Instruct-hf", "instruct")]
    assert BASE["tokenizer_from"] == FINAL["name"] and BASE["turn_format"] == FINAL["turn_format"]
    assert CONDS["base"] == list(BASE_CONDITIONS) and CONDS["instruct"] == FINAL_CONDITIONS
    for m in CONFIG["models"]:
        v = m["vllm"]
        assert v["tensor_parallel_size"] == 8 and v["dtype"] == "bfloat16" and 0.85 <= v["gpu_memory_utilization"] <= 0.92
        cfg = AutoConfig.from_pretrained(m["name"])
        assert v["max_model_len"] <= cfg.max_position_embeddings == NEMO_CONFIG["max_position_embeddings"] == 4096
        assert cfg.num_attention_heads % v["tensor_parallel_size"] == 0 and cfg.num_key_value_heads % v["tensor_parallel_size"] == 0
        # NeMo's architecture (model_config.yaml) as the HF config reads it.
        assert (cfg.hidden_act, cfg.rope_parameters["partial_rotary_factor"], cfg.rope_parameters["rope_theta"]) == (
            "relu2", NEMO_CONFIG["rotary_percentage"], 10000)
        assert (cfg.num_hidden_layers, cfg.hidden_size, cfg.num_key_value_heads, cfg.tie_word_embeddings) == (
            NEMO_CONFIG["num_layers"], NEMO_CONFIG["hidden_size"], NEMO_CONFIG["num_query_groups"], False)


@pytest.mark.parametrize("m", CONFIG["models"], ids=lambda m: m["stage"])
def test_every_prompt_is_nemos_text_and_ids(m):
    """The rendered text is NeMo's, plus the dummy-prefix symbol in front; its ids under the stage's tokenizer are what
    NeMo's tokenizer gives for NeMo's text."""
    for c in CONDS[m["stage"]]:
        texts = [render(m, c, a, b) for a, b in PAIRS]
        refs = [nemo_text(c, a, b) for a, b in PAIRS]
        assert texts == [DUMMY_PREFIX + r for r in refs], c
        assert TOK[m["stage"]](texts, add_special_tokens=False).input_ids == [SP.encode_as_ids(r) for r in refs], c


@pytest.mark.parametrize("m", CONFIG["models"], ids=lambda m: m["stage"])
def test_turns_end_where_the_outcome_goes(m):
    for c in CONDS[m["stage"]]:
        spec = CONDITIONS[c.split("@")[0]]
        p = render(m, c, "FIRST", "SECOND")
        body = spec["body"].format(first="FIRST", second="SECOND")
        if spec["prefill"] is None:  # coinflip, user_choice and their persona variants: the user turn is left open
            assert p.endswith("<extra_id_1>User\n" + placement(c)[1] + body)
            assert p.count("<extra_id_1>") == 1 and "Assistant\n" not in p
        else:
            assert p.endswith(body + "\n<extra_id_1>Assistant\n" + spec["prefill"])
            assert p.count("<extra_id_1>") == 2


@pytest.mark.parametrize("m", CONFIG["models"], ids=lambda m: m["stage"])
def test_no_system_prompt_means_an_empty_one_and_a_persona_adds_only_its_text(m):
    head = DUMMY_PREFIX + "<extra_id_0>System\n"
    for c in CONDS[m["stage"]]:
        system, lead = placement(c)
        plain = render(m, c.split("@")[0], "FIRST", "SECOND")
        p = render(m, c, "FIRST", "SECOND")
        assert plain.startswith(head + "\n<extra_id_1>User\nI would like you")
        if system:  # system prompt or roleplay: the persona fills the system block, nothing else changes
            assert p == head + system + plain[len(head):]
            assert ids(m["stage"], p) == ids(m["stage"], head) + ids(m["stage"], system) + ids(m["stage"], plain[len(head):])
        else:  # empty system block; a persona in the user turn is the only addition
            i = plain.index("I would like you")
            assert p == plain[:i] + lead + plain[i:]
            assert p.startswith(head + "\n<extra_id_1>User\n")


@pytest.mark.parametrize("m", CONFIG["models"], ids=lambda m: m["stage"])
def test_markers_are_single_ids_and_there_is_no_bos(m):
    tok = TOK[m["stage"]]
    marker_ids = {s: tok.encode(s, add_special_tokens=False) for s in ("<extra_id_0>", "<extra_id_1>")}
    assert marker_ids == {"<extra_id_0>": [4], "<extra_id_1>": [5]}
    assert [SP.piece_to_id(s) for s in marker_ids] == [4, 5] and all(SP.is_control(i) is False for i in (4, 5))
    assert tok.bos_token_id == SP.bos_id() == 2
    for c in CONDS[m["stage"]]:
        for first, second in PAIRS:
            p = render(m, c, first, second)
            enc = ids(m["stage"], p)
            assert enc[:2] == [SP.piece_to_id(DUMMY_PREFIX), 4] and SP.bos_id() not in enc
            for s, (i,) in marker_ids.items():
                assert enc.count(i) == p.count(s), (c, s)


def test_base_reads_the_same_ids_as_the_instruct():
    """The base is scored with the Instruct's tokenizer (tokenizer_from), so its ids are the Instruct's by construction;
    its renderings are the Instruct's too."""
    for c in CONDS["base"]:
        texts = [render(BASE, c, a, b) for a, b in PAIRS]
        assert texts == [render(FINAL, c, a, b) for a, b in PAIRS]
        assert TOK["base"](texts, add_special_tokens=False).input_ids == TOK["instruct"](texts, add_special_tokens=False).input_ids


def test_base_repos_own_tokenizer_would_not_read_nemos_ids():
    """Why tokenizer_from: the base repo's tokenizer.json (a conversion of the same SentencePiece model) prepends the
    whitespace symbol after every marker ("<extra_id_1>", "▁User") and splits " heads" in two. If this starts
    failing, mgoin's tokenizer.json changed: re-check the choice."""
    own = AutoTokenizer.from_pretrained(BASE["name"])
    ref = nemo_text("coinflip", "FIRST", "SECOND")
    assert own(DUMMY_PREFIX + ref, add_special_tokens=False).input_ids != SP.encode_as_ids(ref)
    assert own(ref, add_special_tokens=False).input_ids != SP.encode_as_ids(ref)
    assert len(own.encode(" heads", add_special_tokens=False)) == 2


@pytest.mark.parametrize("m", CONFIG["models"], ids=lambda m: m["stage"])
def test_outcome_tokens_follow_the_prompt(m):
    """Single, distinct, the same in both stages, and the id NeMo's tokenizer gives the outcome word right after the
    prompt (SentencePiece's leading-space handling: " heads" alone would be the dummy prefix plus a space)."""
    stage = m["stage"]
    for c in CONDS[stage]:
        out = outcome_token_ids(c, TOK[stage])
        assert len(set(out)) == 2 and out == outcome_token_ids(c, TOK["instruct"])
        strings = CONDITIONS[c.split("@")[0]]["outcomes"]
        assert [SP.id_to_piece(i) for i in out] == [s.replace(" ", DUMMY_PREFIX) for s in strings]
        for first, second in PAIRS[:30]:
            p, ref = render(m, c, first, second), nemo_text(c, first, second)
            base_ids = ids(stage, p)
            for s, i in zip(strings, out):
                assert ids(stage, p + s) == base_ids + [i], (c, s)
                assert SP.encode_as_ids(ref + s) == SP.encode_as_ids(ref) + [i], (c, s)


@pytest.mark.parametrize("m", CONFIG["models"], ids=lambda m: m["stage"])
def test_max_model_len_fits_the_longest_prompt(m):
    """The two longest tasks under every condition, plus the outcome token, with room to spare."""
    n = [len(ids(m["stage"], t["prompt"].strip())) for t in TASKS]
    top = sorted(range(len(TASKS)), key=lambda i: -n[i])[:6]
    longest = max(len(ids(m["stage"], render(m, c, TASKS[a]["prompt"], TASKS[b]["prompt"])))
                  for c in CONDS[m["stage"]] for a in top for b in top if a != b)
    assert longest + 1 <= m["vllm"]["max_model_len"] / 1.5, longest


@pytest.mark.parametrize("m", CONFIG["models"], ids=lambda m: m["stage"])
def test_download_has_what_loading_needs(m):
    try:
        files = list(HfApi().list_repo_tree(m["name"], recursive=True))
    except Exception as e:  # offline
        pytest.skip(f"no file listing: {e}")
    sizes = {f.path: f.size for f in files if hasattr(f, "size")}
    chosen = set(filter_repo_objects(sizes, allow_patterns=m["download"]["allow_patterns"]))
    weights = {p for p in sizes if p.endswith(".safetensors")}
    assert weights <= chosen and {"config.json", "model.safetensors.index.json", "tokenizer_config.json"} <= chosen
    assert chosen & {"tokenizer.json", "tokenizer.model"}  # vLLM loads the repo's own tokenizer at startup
    assert chosen - weights <= {"config.json", "generation_config.json", "model.safetensors.index.json", "tokenizer.json",
                                "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json"}
    assert abs(sum(sizes[p] for p in chosen) / 1e9 - m["download"]["gb"]) < 0.5
