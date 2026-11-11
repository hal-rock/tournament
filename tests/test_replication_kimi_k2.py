"""Kimi K2 in the replication (configs/replication/kimi-k2.yaml): every base and final condition against Kimi's own
template and tokenizer. Needs only small files from the Hub (tokenizer files, the weight index, the file listing).

Kimi's tokenizer is remote code (tokenization_kimi.py): tiktoken behind HF's slow-tokenizer wrapper, which splits out
the added tokens and hands the text between them to tiktoken. These tests don't execute that code. They rebuild the
tokenizer from its data (tiktoken.model, the added tokens in tokenizer_config.json, the split pattern copied verbatim
from tokenization_kimi.py) as an HF fast tokenizer, and check it on every prompt against tiktoken itself, the encoder
tokenization_kimi.py wraps. test_remote_tokenizer_matches_the_stand_in loads the real tokenizers (trust_remote_code)
and compares them with the stand-in; it runs only with KIMI_REMOTE_CODE=1, e.g. on the node, where vLLM runs the remote
code anyway.
"""

import json
import os
from pathlib import Path

import pytest
import tiktoken
from huggingface_hub import hf_hub_download
from tiktoken.load import load_tiktoken_bpe
from transformers import PreTrainedTokenizerFast
from transformers.convert_slow_tokenizer import TikTokenConverter

import prompts
from prompts import outcome_token_ids
from replication_checks import (Format, all_conditions, check_config, check_download, check_persona_blocks,
                                check_prompts, load_config, longest_prompt, pairs, reference, render, tasks)

KEY, FINAL, BASE = "kimi-k2", "moonshotai/Kimi-K2-Instruct", "moonshotai/Kimi-K2-Base"
# Kimi-K2-Instruct at its first upload (2025-07-11): the template the 0711 weights were released with.
RELEASE = "b543046f210643f1a8ccb80b2547261110bc3f92"
# tokenization_kimi.py, TikTokenTokenizer.pat_str, verbatim (the same in both repos).
KIMI_PATTERN = "|".join([
    r"""[\p{Han}]+""",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""\p{N}{1,3}""",
    r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
    r"""\s*[\r\n]+""",
    r"""\s+(?!\S)""",
    r"""\s+""",
])
# Special ids after the 163,584 BPE ids: num_reserved_special_tokens = 256; the base's older tokenization_kimi.py
# makes 2 more (163840 and 163841, past the 163,840 embedding rows; no prompt uses them).
N_SPECIAL = {FINAL: 256, BASE: 258}
DEFAULT_SYSTEM = "<|im_system|>system<|im_middle|>You are Kimi, an AI assistant created by Moonshot AI.<|im_end|>"
MARKERS = {"<|im_user|>": 163587, "<|im_middle|>": 163601, "<|im_end|>": 163586, "<|im_assistant|>": 163588,
           "<|im_system|>": 163594}
FORMAT = Format(bos="", default_system=DEFAULT_SYSTEM + "\n", system="<|im_system|>system<|im_middle|>{}<|im_end|>",
                user="<|im_user|>user<|im_middle|>", user_end="<|im_end|>", reply="<|im_assistant|>assistant<|im_middle|>",
                markers=MARKERS)


def _files(repo, revision):
    cfg = json.loads(Path(hf_hub_download(repo, "tokenizer_config.json", revision=revision)).read_text())
    return cfg, hf_hub_download(repo, "tiktoken.model", revision=revision)


def special_names(repo, revision):
    """The tiktoken special tokens tokenization_kimi.py makes: added_tokens_decoder's names, reserved_token_<id> between."""
    cfg, vocab = _files(repo, revision)
    n = len(load_tiktoken_bpe(vocab))
    added = {int(i): t["content"] for i, t in cfg["added_tokens_decoder"].items()}
    return [added.get(i, f"<|reserved_token_{i}|>") for i in range(n, n + N_SPECIAL[repo])], added


def stand_in(repo, revision, template_revision=None):
    """Kimi's tokenizer rebuilt from its data as an HF fast tokenizer; with the Instruct's chat template if asked."""
    cfg, vocab = _files(repo, revision)
    names, _ = special_names(repo, revision)
    converted = TikTokenConverter(vocab_file=vocab, pattern=KIMI_PATTERN, extra_special_tokens=names).converted()
    tok = PreTrainedTokenizerFast(tokenizer_object=converted, bos_token=cfg["bos_token"], eos_token=cfg["eos_token"],
                                  unk_token=cfg["unk_token"], pad_token=cfg["pad_token"])
    if template_revision is not None:
        tok.chat_template = Path(hf_hub_download(FINAL, "chat_template.jinja", revision=template_revision)).read_text()
    return tok


def tiktoken_encoder(repo, revision):
    """What tokenization_kimi.py computes for tokenizer(text, add_special_tokens=False): HF's wrapper splits out the
    added tokens, and tiktoken encodes the text between them with Kimi's pattern."""
    _, vocab = _files(repo, revision)
    ranks = load_tiktoken_bpe(vocab)
    names, added = special_names(repo, revision)
    enc = tiktoken.Encoding(name=f"kimi-{repo}", pat_str=KIMI_PATTERN, mergeable_ranks=ranks,
                            special_tokens={s: len(ranks) + k for k, s in enumerate(names)})
    return lambda text: enc.encode(text, allowed_special=set(added.values()))


@pytest.fixture(scope="module")
def cfg():
    return load_config(KEY)


@pytest.fixture(scope="module")
def revisions(cfg):
    return {m["name"]: m["revision"] for m in cfg["models"]}


@pytest.fixture(scope="module")
def template_tok(revisions):
    # The final's tokenizer and template, which the base also reads its template from (chat_template_from).
    return stand_in(FINAL, revisions[FINAL], template_revision=revisions[FINAL])


@pytest.fixture(scope="module")
def base_tok(revisions):
    return stand_in(BASE, revisions[BASE])


def test_config(cfg):
    check_config(cfg, KEY, FINAL, BASE)
    for m in cfg["models"]:
        assert m["trust_remote_code"] is True  # tiktoken tokenizer and the kimi_k2 config are remote code


def test_download_fetches_what_loading_needs_and_no_history(cfg):
    """Every shard in the index, the configs, the tokenizer and the remote-code files the configs point to (auto_map);
    nothing else big. Kimi-K2-Base's 2.06 TB of Hub storage is git history (63 shards deleted by the 2025-07-10 "clean
    repo" commit): at the pinned revision the patterns select the 61 current shards, 1029.2 GB."""
    for m in cfg["models"]:
        conf = json.loads(Path(hf_hub_download(m["name"], "config.json", revision=m["revision"])).read_text())
        tconf = json.loads(Path(hf_hub_download(m["name"], "tokenizer_config.json", revision=m["revision"])).read_text())
        code = {v.split(".")[0] + ".py" for v in conf["auto_map"].values()}
        code |= {v.split(".")[0] + ".py" for v in tconf["auto_map"]["AutoTokenizer"] if v}
        assert code == {"configuration_deepseek.py", "modeling_deepseek.py", "tokenization_kimi.py"}
        required = ["model.safetensors.index.json", "config.json", "generation_config.json", "tokenizer_config.json",
                    "tiktoken.model", *sorted(code)] + (["chat_template.jinja"] if m["name"] == FINAL else [])
        total = check_download(m, required)
        assert 1029.1 < total < 1029.3


def test_tokenizer_data_is_the_same_in_base_and_final(revisions):
    (cb, vb), (cf, vf) = _files(BASE, revisions[BASE]), _files(FINAL, revisions[FINAL])
    assert Path(vb).read_bytes() == Path(vf).read_bytes()
    assert cb["added_tokens_decoder"] == cf["added_tokens_decoder"]
    assert (cb["bos_token"], cb["eos_token"]) == (cf["bos_token"], cf["eos_token"]) == ("[BOS]", "[EOS]")
    # The base repo's own template (its default system prompt is "You are a helpful assistant") is not used.
    assert "You are a helpful assistant" in cb["chat_template"]


def test_default_system_block_and_its_newline(template_tok):
    """The template writes its default system block, then a newline, when there's no system message; a system message
    we pass is followed directly by the user turn. prompts.SYSTEM_BLOCKS removes the block with its newline, so "none"
    starts at <|im_user|>."""
    user = [{"role": "user", "content": "X"}]
    tail = "<|im_user|>user<|im_middle|>X<|im_end|><|im_assistant|>assistant<|im_middle|>"
    default = template_tok.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
    assert default == DEFAULT_SYSTEM + "\n" + tail
    given = template_tok.apply_chat_template([{"role": "system", "content": "S"}] + user, tokenize=False,
                                             add_generation_prompt=True)
    assert given == "<|im_system|>system<|im_middle|>S<|im_end|>" + tail
    assert prompts._strip_system(default, "X") == tail


def test_prompts_match_the_template_and_tokenizer(cfg, template_tok, base_tok, revisions):
    """All 44 final conditions (the 4 base ones among them) on every task as Task A and as Task B."""
    longest = check_prompts(cfg, template_tok, base_tok, FORMAT, reference_encode=tiktoken_encoder(FINAL, revisions[FINAL]))
    assert longest < 600
    # The base's own tiktoken path gives the same ids too (it differs from the final's only outside our prompts).
    base_encode = tiktoken_encoder(BASE, revisions[BASE])
    for condition in all_conditions(cfg):
        p = render(condition, *pairs(tasks(cfg))[0], template_tok, cfg)
        assert base_encode(p) == template_tok(p, add_special_tokens=False).input_ids


def test_persona_adds_exactly_its_block(cfg, template_tok):
    check_persona_blocks(cfg, template_tok, FORMAT, *pairs(tasks(cfg))[0])


def test_release_template_renders_the_same_prompts(cfg, template_tok):
    """The 0711 weights were released with an earlier template (default system prompt "You are a helpful assistant",
    no newline after it). Once the default block is gone, it renders every prompt exactly as the current one."""
    release = json.loads(Path(hf_hub_download(FINAL, "tokenizer_config.json", revision=RELEASE)).read_text())
    old = PreTrainedTokenizerFast(tokenizer_object=template_tok.backend_tokenizer)
    old.chat_template = release["chat_template"]
    old_format = Format(**{**FORMAT.__dict__, "default_system":
                           "<|im_system|>system<|im_middle|>You are a helpful assistant<|im_end|>"})
    for condition in all_conditions(cfg):
        for first, second in pairs(tasks(cfg))[:5]:
            p = render(condition, first, second, template_tok, cfg)
            assert reference(condition, first, second, old, old_format) == p
            assert render(condition, first, second, old, cfg) == p


def test_max_model_len_holds_the_longest_prompt(cfg, template_tok):
    longest, condition = longest_prompt(cfg, template_tok)
    print(f"longest prompt: {longest} tokens ({condition})")
    assert longest == 554 and condition == "coinflip@evil+roleplay"
    for m in cfg["models"]:
        assert m["vllm"]["max_model_len"] >= longest + 1  # the prompt and the outcome token


@pytest.mark.skipif(os.environ.get("KIMI_REMOTE_CODE") != "1",
                    reason="executes tokenization_kimi.py (remote code); set KIMI_REMOTE_CODE=1, e.g. on the node")
def test_remote_tokenizer_matches_the_stand_in(cfg, revisions, template_tok):
    from transformers import AutoTokenizer
    real_t = AutoTokenizer.from_pretrained(FINAL, revision=revisions[FINAL], trust_remote_code=True)
    real_b = AutoTokenizer.from_pretrained(BASE, revision=revisions[BASE], trust_remote_code=True)
    assert real_t.bos_token_id == 163584 and real_t("x").input_ids == real_t("x", add_special_tokens=False).input_ids
    for condition in all_conditions(cfg):
        assert outcome_token_ids(condition, real_t) == outcome_token_ids(condition, real_b) == \
               outcome_token_ids(condition, template_tok)
        for first, second in pairs(tasks(cfg)):
            p = render(condition, first, second, real_t, cfg)
            assert p == render(condition, first, second, template_tok, cfg), condition
            ids = template_tok(p, add_special_tokens=False).input_ids
            assert real_t(p, add_special_tokens=False).input_ids == ids, condition
            assert real_b(p, add_special_tokens=False).input_ids == ids, condition
