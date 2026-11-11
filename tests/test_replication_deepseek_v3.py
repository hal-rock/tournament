"""DeepSeek V3 in the replication (configs/replication/deepseek-v3.yaml): every base and final condition against
DeepSeek V3's own template and tokenizer. Needs only small files from the Hub (tokenizer files, the weight index, the
file listing)."""

from pathlib import Path

import pytest
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from replication_checks import (Format, all_conditions, check_config, check_download, check_persona_blocks,
                                check_prompts, load_config, longest_prompt, pairs, render, tasks)

KEY, FINAL, BASE = "deepseek-v3", "deepseek-ai/DeepSeek-V3", "deepseek-ai/DeepSeek-V3-Base"
BOS = "<｜begin▁of▁sentence｜>"
# The template: BOS, then a system message as bare text (no marker), then <｜User｜> + text, with no closing marker
# (the next turn's <｜Assistant｜> closes it). No default system prompt; no think block (V3 has no thinking mode).
FORMAT = Format(bos=BOS, default_system="", system="{}", user="<｜User｜>", user_end="", reply="<｜Assistant｜>",
                markers={BOS: 0, "<｜User｜>": 128803, "<｜Assistant｜>": 128804})


@pytest.fixture(scope="module")
def cfg():
    return load_config(KEY)


@pytest.fixture(scope="module")
def revisions(cfg):
    return {m["name"]: m["revision"] for m in cfg["models"]}


@pytest.fixture(scope="module")
def template_tok(revisions):
    # The final's tokenizer and template, which the base also reads its template from (chat_template_from).
    return AutoTokenizer.from_pretrained(FINAL, revision=revisions[FINAL])


@pytest.fixture(scope="module")
def base_tok(revisions):
    return AutoTokenizer.from_pretrained(BASE, revision=revisions[BASE])


def test_config(cfg):
    check_config(cfg, KEY, FINAL, BASE)
    assert not any(m.get("trust_remote_code") for m in cfg["models"])  # deepseek_v3 is native in transformers and vLLM


def test_download_fetches_what_loading_needs(cfg):
    """Every shard in the index (the MTP module's too: HF's loader opens all of them), the config and the tokenizer;
    not the repo's inference/ code, README figures or the remote-code .py files."""
    for m in cfg["models"]:
        total = check_download(m, ["model.safetensors.index.json", "config.json", "tokenizer.json", "tokenizer_config.json"])
        assert 688.5 < total < 688.7


def test_tokenizer_files_are_identical_in_base_and_final(revisions):
    for f in ("tokenizer.json", "tokenizer_config.json"):
        assert Path(hf_hub_download(BASE, f, revision=revisions[BASE])).read_bytes() == \
               Path(hf_hub_download(FINAL, f, revision=revisions[FINAL])).read_bytes()


def test_template_has_no_default_system_prompt(template_tok):
    user = [{"role": "user", "content": "X"}]
    assert template_tok.apply_chat_template(user, tokenize=False, add_generation_prompt=True) == \
           BOS + "<｜User｜>X<｜Assistant｜>"
    assert template_tok.apply_chat_template([{"role": "system", "content": "S"}] + user, tokenize=False,
                                            add_generation_prompt=True) == BOS + "S<｜User｜>X<｜Assistant｜>"


def test_prompts_match_the_template_and_tokenizer(cfg, template_tok, base_tok):
    """All 44 final conditions (the 4 base ones among them) on every task as Task A and as Task B."""
    assert check_prompts(cfg, template_tok, base_tok, FORMAT) < 600


def test_one_bos_like_the_official_tokenized_route(cfg, template_tok):
    """Tokenizing the rendered text with add_special_tokens=False gives the ids the template's own tokenize=True
    route gives: one BOS, at the start."""
    first, second = pairs(tasks(cfg))[0]
    for condition in all_conditions(cfg):
        p = render(condition, first, second, template_tok, cfg)
        ids = template_tok(p, add_special_tokens=False).input_ids
        assert ids[0] == 0 and ids.count(0) == 1
    # The Assistant prompt minus its prefill is exactly the official generation prompt, as ids.
    from replication_checks import body
    official = template_tok.apply_chat_template([{"role": "user", "content": body("assistant", first, second)}],
                                                tokenize=True, add_generation_prompt=True, return_dict=False)
    p = render("assistant", first, second, template_tok, cfg)
    assert template_tok(p.removesuffix("I'll choose Task"), add_special_tokens=False).input_ids == list(official)


def test_persona_adds_exactly_its_block(cfg, template_tok):
    check_persona_blocks(cfg, template_tok, FORMAT, *pairs(tasks(cfg))[0])


def test_max_model_len_holds_the_longest_prompt(cfg, template_tok):
    longest, condition = longest_prompt(cfg, template_tok)
    print(f"longest prompt: {longest} tokens ({condition})")
    assert longest == 557 and condition == "coinflip@evil+roleplay"
    for m in cfg["models"]:
        assert m["vllm"]["max_model_len"] >= longest + 1  # the prompt and the outcome token
