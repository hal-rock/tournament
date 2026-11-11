"""report_patch.py for the replication's models (finding 8, configs/coin_report/<key>.yaml with `model_from`):
- the config resolves to the replication config's model and prompt format, and the earlier configs resolve unchanged;
- the coin line's positions come out the same from a slow tokenizer (no character offsets: the token-prefix fallback) as
  from the same vocabulary's fast tokenizer (offsets);
- for each model's own tokenizer (tokenizer files only), items build on real pilot pairs for every candidate report
  question with all of build_items' assertions holding, and the prompts are token for token score_replication.py's;
- a tiny random DeepSeek V4 (four residual streams per position) patches as the single-stream models do;
- Kimi K2's config reads as transformers' DeepseekV3Config (tournament.kimi_k2_config).

Kimi K2's real tokenizer is remote code (tokenization_kimi.py), which isn't executed here: KimiSlow below re-types its
encoding path (HF's slow wrapper splits out the added tokens, tiktoken encodes the text between them with Kimi's pattern)
on the same data, the way tests/test_replication_kimi_k2.py's stand-in does for a fast tokenizer. On the node,
KIMI_REMOTE_CODE=1 runs test_kimi_real_tokenizer_builds_the_same_items here, and tests/test_replication_kimi_k2.py compares
the real tokenizer with the stand-in on every prompt.
"""

import os
import sys
from pathlib import Path

import pytest
import tiktoken
import torch
import yaml
from tiktoken.load import load_tiktoken_bpe
from tokenizers import AddedToken
from transformers import AutoTokenizer, DeepseekV3Config, DeepseekV4Config, DeepseekV4ForCausalLM
from transformers.convert_slow_tokenizer import bytes_to_unicode
from transformers.tokenization_python import PythonBackend

import prompts
import score_replication
import tournament
from analyze import read_jsonl
from replication import WORDING_CHECK
from report_patch import (ARMS, HF_KEYS, LINE, build_items, forward, line_positions, load_tokenizers, pilot_pairs,
                          prompt_format, replace, resolve_model, run_chunk)
from test_replication_kimi_k2 import FINAL as KIMI, KIMI_PATTERN, _files, special_names, stand_in

ROOT = Path(__file__).resolve().parent.parent
KEYS = ["kimi-k2", "llama-405b", "deepseek-v3", "deepseek-v4-flash", "trinity-large", "nemotron4-340b"]
LLAMA = "unsloth/Llama-3.3-70B-Instruct"  # test_report_patch.py's tokenizer
TASKS = read_jsonl(ROOT / "data/tasks_n300_max500_seed0.jsonl")
# The first pilot pairs (report_patch.pilot_pairs, as the runs draw them), the ones the wording check scores.
PAIRS = pilot_pairs(len(TASKS), 1000, tournament.select_pairs(len(TASKS), 15000, 0), 0)[:4]


def coin_report_config(key):
    return yaml.safe_load((ROOT / "configs" / "coin_report" / f"{key}.yaml").read_text())


def replication_entry(key):
    rep = yaml.safe_load((ROOT / "configs" / "replication" / f"{key}.yaml").read_text())
    (entry,) = [m for m in rep["models"] if m["stage"] == "instruct"]
    return rep, entry


class SlowOf(PythonBackend):
    """A fast tokenizer's vocabulary behind HF's slow (Python) tokenizer interface, the structure of Kimi K2's remote
    tokenizer: the added tokens are split out in Python and the text between them goes to the inner encoder. No offsets."""

    def __init__(self, fast):
        self._fast, self._vocab = fast, fast.get_vocab()
        self._inverse = {i: t for t, i in self._vocab.items()}
        super().__init__(added_tokens_decoder=dict(fast.added_tokens_decoder), bos_token=fast.bos_token,
                         eos_token=fast.eos_token)

    @property
    def vocab_size(self):
        return len(self._vocab)

    def get_vocab(self):
        return dict(self._vocab)

    def _tokenize(self, text, **kwargs):
        return self._fast.convert_ids_to_tokens(self._fast.backend_tokenizer.encode(text, add_special_tokens=False).ids)

    def _convert_token_to_id(self, token):
        return self._vocab[token]

    def _convert_id_to_token(self, index):
        return self._inverse[index]

    def convert_tokens_to_string(self, tokens):
        return self._fast.convert_tokens_to_string(tokens)


class KimiSlow(PythonBackend):
    """tokenization_kimi.TikTokenTokenizer's encoding and decoding, re-typed from its source (not executed), on its data:
    tiktoken with Kimi's split pattern, the reserved specials named from tokenizer_config.json, tokens as byte-level
    strings, decode() with no arguments straight through tiktoken."""

    def __init__(self, repo, revision):
        cfg, vocab = _files(repo, revision)
        names, _ = special_names(repo, revision)
        ranks = load_tiktoken_bpe(vocab)
        self.model = tiktoken.Encoding(name="kimi", pat_str=KIMI_PATTERN, mergeable_ranks=ranks,
                                       special_tokens={s: len(ranks) + k for k, s in enumerate(names)})
        byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in byte_encoder.items()}
        self.decoder = {i: "".join(byte_encoder[b] for b in self.model.decode_single_token_bytes(i))
                        for i in range(self.model.n_vocab)}
        self.encoder = {t: i for i, t in self.decoder.items()}
        self.unk_id = self.encoder.get(cfg["unk_token"])
        fields = ("content", "single_word", "lstrip", "rstrip", "normalized", "special")
        added = {int(i): AddedToken(**{k: t[k] for k in fields if k in t}) for i, t in cfg["added_tokens_decoder"].items()}
        super().__init__(bos_token=cfg["bos_token"], eos_token=cfg["eos_token"], unk_token=cfg["unk_token"],
                         pad_token=cfg["pad_token"], added_tokens_decoder=added)

    @property
    def vocab_size(self):
        return self.model.n_vocab

    def get_vocab(self):
        return self.encoder

    def _tokenize(self, text, **kwargs):
        return [self.decoder[t] for t in self.model.encode(text, allowed_special="all")]

    def _convert_token_to_id(self, token):
        return self.encoder.get(token, self.unk_id)

    def _convert_id_to_token(self, index):
        return self.decoder.get(index)

    def convert_tokens_to_string(self, tokens):
        return bytearray([self.byte_decoder[c] for c in "".join(tokens)]).decode("utf-8", "replace")

    def decode(self, token_ids, **kwargs):
        if kwargs:
            return super().decode(token_ids, **kwargs)
        return self.model.decode([token_ids] if isinstance(token_ids, int) else list(token_ids))


@pytest.fixture(scope="module")
def kimi_tokenizers():
    """(ids, template): the slow tiktoken stand-in for the ids, the fast stand-in carrying the Instruct's template."""
    rev = resolve_model(coin_report_config("kimi-k2"))["revision"]
    return KimiSlow(KIMI, rev), stand_in(KIMI, rev, template_revision=rev)


def tokenizers_for(key, m, kimi_tokenizers):
    """What report_patch.load_tokenizers gives on the node; for Kimi, the stand-ins (its tokenizer is remote code)."""
    return kimi_tokenizers if key == "kimi-k2" else load_tokenizers(m)


# ---------------------------------------------------------------- configs

@pytest.mark.parametrize("key", KEYS)
def test_model_from_resolves(key):
    cfg = coin_report_config(key)
    rep, entry = replication_entry(key)
    m = resolve_model(cfg)
    assert cfg["model_from"] == {"config": f"configs/replication/{key}.yaml", "stage": "instruct"}
    assert set(cfg.get("model", {})) <= HF_KEYS and cfg["family"] == key and cfg["pairs"] == "pilot"
    assert (cfg["n_pairs"], cfg["seed"], cfg["tasks_file"]) == (1000, 0, rep["tasks_file"]) and "report" not in cfg
    for k in ("name", "revision", "tokenizer_from", "chat_template_from", "trust_remote_code", "turn_format",
              "close_think", "strip_think", "sanity"):
        assert m.get(k) == entry.get(k), k
    assert prompt_format(m) == {k: entry[k] for k in ("turn_format", "close_think", "strip_think") if k in entry}
    # Borrowed repos are read at the revision the replication config pins for them (score_replication.revision_of).
    for repo, rev in m["revisions"].items():
        assert rev == score_replication.revision_of(rep, repo), repo
    assert m.get("loader") == ("kimi_k2" if key == "kimi-k2" else None)
    assert m.get("loader") is None or m["loader"] in tournament.LOADERS
    assert cfg["batch_size"] >= 1 and cfg["chunk_pairs"] >= 1


@pytest.mark.parametrize("path", sorted(p.name for p in (ROOT / "configs/coin_report").glob("*.yaml")
                                        if p.stem not in KEYS))
def test_earlier_configs_resolve_unchanged(path):
    cfg = yaml.safe_load((ROOT / "configs/coin_report" / path).read_text())
    m = resolve_model(cfg)
    assert m == cfg["model"] and prompt_format(m) == {} and "model_from" not in cfg


def test_model_from_takes_only_hf_settings(tmp_path):
    cfg = coin_report_config("trinity-large")
    with pytest.raises(AssertionError):
        resolve_model({**cfg, "model": {"close_think": False}})
    with pytest.raises(AssertionError):
        resolve_model({**cfg, "model": {"name": "someone/else"}})


# ---------------------------------------------------------------- line positions on a slow tokenizer

@pytest.fixture(scope="module")
def llama():
    fast = AutoTokenizer.from_pretrained(LLAMA)
    return fast, SlowOf(fast)


def test_slow_tokenizer_has_no_offsets(llama):
    """transformers 5.17's slow tokenizers ignore return_offsets_mapping (no error, no offsets), so the offset method
    can't run on them; report_patch picks the prefix method by is_fast."""
    fast, slow = llama
    assert fast.is_fast and not slow.is_fast
    assert "offset_mapping" in fast("it came up", return_offsets_mapping=True)
    assert "offset_mapping" not in slow("it came up", return_offsets_mapping=True)
    with pytest.raises((AttributeError, KeyError)):
        line_positions(slow, "it came up", "it came up", method="offsets")


@pytest.mark.parametrize("report", ["coin_report", "coin_report_bold", "coin_predict_think_bold", "coin_report_ab"])
def test_prefix_fallback_matches_offsets(llama, report):
    """The token-prefix method on a slow tokenizer finds the same line, at the same positions, as character offsets on the
    fast one; so do the items built on each. On the fast tokenizer the two methods agree too."""
    fast, slow = llama
    for first, second in [(TASKS[i]["prompt"], TASKS[j]["prompt"]) for i, j in PAIRS]:
        for condition, line in LINE.items():
            text = prompts.build_prompt(condition, first, second, fast)
            by_offsets = line_positions(fast, text, line)
            assert line_positions(fast, text, line, method="prefix") == by_offsets
            assert line_positions(slow, text, line) == by_offsets  # the slow tokenizer defaults to the prefix
            assert fast.decode(by_offsets[0][by_offsets[1][0]:]).strip() == line
    # Items: ids from the slow tokenizer (the template from the fast one, which carries it), against the fast one alone.
    assert build_items(slow, TASKS, PAIRS, report, fast) == build_items(fast, TASKS, PAIRS, report)


def test_prefix_fallback_rejects_a_token_across_the_line_start(llama):
    """If the text before the line doesn't tokenize to a prefix of the whole (a token straddles the line's start), the
    fallback fails rather than guessing."""
    fast, slow = llama
    text = "I would like you to flip a coin.Okay, I'm flipping the coin now ... it came up"
    line = "y, I'm flipping the coin now ... it came up"  # starts inside the token "Okay"
    with pytest.raises(AssertionError):
        line_positions(slow, text, line)


def test_kimi_slow_stand_in_matches_the_fast_one(kimi_tokenizers):
    """KimiSlow (the remote code's path) and the fast stand-in give the same ids on the report prompts, and the items
    built with each (prefix fallback vs offsets) are the same."""
    slow, fast = kimi_tokenizers
    m = resolve_model(coin_report_config("kimi-k2"))
    assert not slow.is_fast and fast.is_fast
    for report in WORDING_CHECK:
        for i, j in PAIRS[:2]:
            text = prompts.build_prompt(report, TASKS[i]["prompt"], TASKS[j]["prompt"], fast, **prompt_format(m))
            assert slow(text, add_special_tokens=False).input_ids == fast(text, add_special_tokens=False).input_ids
        assert (build_items(slow, TASKS, PAIRS[:2], report, fast, **prompt_format(m))
                == build_items(fast, TASKS, PAIRS[:2], report, fast, **prompt_format(m)))
    assert slow.decode(slow("it came up", add_special_tokens=False).input_ids[-1]) == " up"


@pytest.mark.skipif(os.environ.get("KIMI_REMOTE_CODE") != "1",
                    reason="executes tokenization_kimi.py (remote code); set KIMI_REMOTE_CODE=1, e.g. on the node")
def test_kimi_real_tokenizer_builds_the_same_items(kimi_tokenizers):
    """On the node: report_patch.load_tokenizers' real Kimi tokenizer (remote code, slow) builds exactly the items the
    stand-ins build, for every candidate report question."""
    m = resolve_model(coin_report_config("kimi-k2"))
    tok, template = load_tokenizers(m)
    slow, fast = kimi_tokenizers
    assert not tok.is_fast
    for report in WORDING_CHECK:
        assert (build_items(tok, TASKS, PAIRS, report, template, **prompt_format(m))
                == build_items(slow, TASKS, PAIRS, report, fast, **prompt_format(m)))


# ---------------------------------------------------------------- each model's tokenizer

@pytest.mark.parametrize("key", KEYS)
def test_items_build_on_pilot_pairs(key, kimi_tokenizers):
    """Every candidate report question (replication.WORDING_CHECK) on the first pilot pairs, with each model's tokenizer
    and format: build_items' assertions hold (" up" last, the report prompt extends the coinflip token for token, the
    "OK, " line at the same positions), and every prompt is the one score_replication.py renders for that condition."""
    m = resolve_model(coin_report_config(key))
    rep, entry = replication_entry(key)
    tok, template = tokenizers_for(key, m, kimi_tokenizers)
    ordered = [q for i, j in PAIRS for q in ((i, j), (j, i))]
    render = lambda condition: score_replication.render_condition(condition, entry, TASKS, ordered, template, tok,
                                                                  rep["system_prompt"])[0]
    coin, ok = render("coinflip"), render("coinflip_ok")
    # The coin is read at " up": its outcome tokens are the next token after the coinflip prompt.
    text = score_replication.render_condition("coinflip", entry, TASKS, ordered[:1], template, tok, "none")[1]
    for outcome, oid in zip(prompts.CONDITIONS["coinflip"]["outcomes"], prompts.outcome_token_ids("coinflip", tok)):
        assert tok(text + outcome, add_special_tokens=False).input_ids == coin[0] + [oid]
    for report in WORDING_CHECK:
        items = build_items(tok, TASKS, PAIRS, report, template, **prompt_format(m))
        full = render(report)
        assert len(prompts.outcome_token_ids(report, tok)) == 2
        for k, it in enumerate(items):
            assert it["ids"] == full[k] and it["ok_ids"] == ok[k]
            assert it["ids"][:it["up"] + 1] == coin[k] and it["up"] == len(coin[k]) - 1
            assert tok.decode(it["ids"][it["up"]]).strip() == "up"
            assert tok.decode(it["ids"][it["line"][0]]).strip().startswith("Okay")
            assert tok.decode(it["ok_ids"][it["line"][0]]).strip().startswith("OK")
        assert {len(it["line"]) for it in items} == {len(items[0]["line"])}


# ---------------------------------------------------------------- DeepSeek V4: four residual streams

@pytest.fixture(scope="module")
def tiny_v4():
    m = resolve_model(coin_report_config("deepseek-v4-flash"))
    tok, template = load_tokenizers(m)
    tasks = [t for t in TASKS if len(t["prompt"]) < 120][:4]
    items = build_items(tok, tasks, [(0, 1), (2, 3)], "coin_predict_think_bold", template, **prompt_format(m))
    torch.manual_seed(0)
    cfg = DeepseekV4Config(vocab_size=len(tok), hidden_size=64, moe_intermediate_size=32, num_hidden_layers=3,
                           num_attention_heads=4, num_key_value_heads=1, head_dim=32, q_lora_rank=16, o_lora_rank=16,
                           o_groups=2, n_routed_experts=4, num_experts_per_tok=2, compress_ratios=[0, 4, 128],
                           num_hash_layers=1, qk_rope_head_dim=16, index_n_heads=2, index_head_dim=16, index_topk=8,
                           sliding_window=16, initializer_range=0.5, attn_implementation="eager")
    model = DeepseekV4ForCausalLM(cfg).eval()
    return tok, items, model


def test_v4_self_patch_is_the_identity_and_swap_is_not(tiny_v4):
    tok, items, model = tiny_v4
    out_ids = prompts.outcome_token_ids("coin_predict_think_bold", tok)
    coin_ids = prompts.outcome_token_ids("coinflip", tok)
    rows = run_chunk(model, items, tok.pad_token_id, coin_ids, out_ids, 3, batch_size=3)
    for r in rows:
        assert set(r["report"]) == {"none", *ARMS}
        assert r["report"]["self"] == pytest.approx(r["report"]["none"], abs=1e-5)
    assert max(abs(r["report"]["swap"] - r["report"]["none"]) for r in rows) > 1e-4


def test_v4_swap_puts_all_four_streams_of_the_other_order_at_up(tiny_v4):
    tok, items, model = tiny_v4
    a, b = items[0], items[1]
    pos = torch.tensor([[a["up"]]])
    last = lambda it: torch.tensor([len(it["ids"]) - 1])
    _, own_b = forward(model, [b["ids"]], tok.pad_token_id, last(b), ([0, 1, 2], pos))
    assert own_b[1].shape == (1, 1, 4, 64)  # [B, P, hc_mult, D]
    with replace(model, {l: own_b[l] for l in (0, 1, 2)}, pos):
        _, seen = forward(model, [a["ids"]], tok.pad_token_id, last(a), ([0, 1, 2], pos))
    for l in (0, 1, 2):
        assert torch.equal(seen[l], own_b[l])
    _, own_a = forward(model, [a["ids"]], tok.pad_token_id, last(a), ([1], pos))
    assert not torch.allclose(own_a[1], own_b[1])


# ---------------------------------------------------------------- main() and the load check, without a real model

class FakeLM(torch.nn.Module):
    """Logits peaked at one token at every position; records the ids it was fed."""

    def __init__(self, vocab, top):
        super().__init__()
        self.emb, self.vocab, self.top, self.seen = torch.nn.Embedding(1, 1), vocab, top, []

    def get_input_embeddings(self):
        return self.emb

    def forward(self, input_ids, use_cache=False):
        from types import SimpleNamespace
        self.seen.append(input_ids[0].tolist())
        logits = torch.zeros(1, input_ids.shape[1], self.vocab)
        logits[0, -1, self.top] = 10.0
        return SimpleNamespace(logits=logits)


@pytest.mark.parametrize("key", ["llama-405b", "nemotron4-340b"])
def test_sanity_check_feeds_the_pinned_prompt(key):
    import report_patch
    m = resolve_model(coin_report_config(key))
    tok, _ = load_tokenizers(m)
    # Nemotron: "▁Berlin" (replication/onboarding/nemotron4-340b.md); Llama: " Berlin".
    berlin = 12006 if key == "nemotron4-340b" else tok(" Berlin", add_special_tokens=False).input_ids[0]
    model = FakeLM(len(tok), berlin)
    report_patch.sanity_check(model, tok, m, uses_bos=key == "llama-405b")
    if "sanity" in m:  # Nemotron: NeMo's own SentencePiece ids, no BOS
        assert model.seen[-1] == m["sanity"]["ids"]
    else:
        assert model.seen[-1] == [tok.bos_token_id] + tok(report_patch.SANITY[0], add_special_tokens=False).input_ids
    with pytest.raises(SystemExit):
        report_patch.sanity_check(FakeLM(len(tok), berlin + 1), tok, m, uses_bos=False)


def test_main_runs_a_model_from_config(tmp_path, monkeypatch):
    """report_patch.main() end to end on llama-405b's config, with a tiny random Llama in place of the 405B and no W&B:
    the model is loaded with the config's HF settings, the sanity check runs, and every arm is written for every prompt."""
    import report_patch
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = coin_report_config("llama-405b")
    cfg["tasks_file"] = str(ROOT / cfg["tasks_file"])
    cfg["model_from"]["config"] = str(ROOT / cfg["model_from"]["config"])
    path = tmp_path / "llama-405b.yaml"
    path.write_text(yaml.safe_dump(cfg))
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-405B-Instruct")
    loaded, checked = {}, []

    def load_model(name, device, dtype, **kw):
        loaded.update(name=name, **kw)
        torch.manual_seed(0)
        return LlamaForCausalLM(LlamaConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                            num_attention_heads=2, num_key_value_heads=2, initializer_range=0.5)).eval()

    class Run:
        id, summary = "testrun", {}
        save = finish = lambda *a, **k: None

    monkeypatch.setattr(report_patch, "load_model", load_model)
    monkeypatch.setattr(report_patch, "sanity_check", lambda *a, **k: checked.append(k))
    monkeypatch.setattr(report_patch.wandb, "init", lambda **k: Run())
    monkeypatch.setattr(sys, "argv", ["report_patch.py", str(path), "--limit", "2", "--report", "coin_predict_think_bold"])
    monkeypatch.chdir(tmp_path)
    report_patch.main()
    assert loaded == dict(name="meta-llama/Llama-3.1-405B-Instruct", revision=None, max_memory=cfg["model"]["max_memory"],
                          loader=None, attn_implementation=None)
    assert checked == [{"uses_bos": True}]
    rows = read_jsonl(tmp_path / "outputs/coin_report/llama-405b/testrun/rows.jsonl")
    assert len(rows) == 4 and all(set(r["report"]) == {"none", *ARMS} for r in rows)
    assert all(r["report"]["self"] == pytest.approx(r["report"]["none"], abs=1e-5) for r in rows)


def test_main_runs_several_reports_in_one_load_with_fewer_arms(tmp_path, monkeypatch):
    """--report a b c --arms swap ctrl: one model load, one rows file per report, only none/self/swap/ctrl written, and
    coin_report's analysis runs on them (the skipped arms' statistics are NaN)."""
    import math
    import coin_report
    import report_patch
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = coin_report_config("llama-405b")
    cfg["tasks_file"] = str(ROOT / cfg["tasks_file"])
    cfg["model_from"]["config"] = str(ROOT / cfg["model_from"]["config"])
    path = tmp_path / "llama-405b.yaml"
    path.write_text(yaml.safe_dump(cfg))
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-405B-Instruct")
    loads, runs = [], []

    def load_model(name, device, dtype, **kw):
        loads.append(name)
        torch.manual_seed(0)
        return LlamaForCausalLM(LlamaConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                            num_attention_heads=2, num_key_value_heads=2, initializer_range=0.5)).eval()

    class Run:
        def __init__(self, **k):
            self.id, self.summary, self.name = f"run{len(runs)}", {}, k["name"]
            runs.append(self)
        save = finish = lambda *a, **k: None

    reports = ["coin_report_short_coin_bold", "coin_predict_think_bold", "coin_report_ab"]
    monkeypatch.setattr(report_patch, "load_model", load_model)
    monkeypatch.setattr(report_patch, "sanity_check", lambda *a, **k: None)
    monkeypatch.setattr(report_patch.wandb, "init", lambda **k: Run(**k))
    monkeypatch.setattr(sys, "argv", ["report_patch.py", str(path), "--limit", "6", "--report", *reports,
                                      "--arms", "swap", "ctrl"])
    monkeypatch.chdir(tmp_path)
    report_patch.main()
    assert len(loads) == 1 and [r.name for r in runs] == [f"{cfg['run_name']}-{x}" for x in reports]
    for k in range(3):
        rows_path = tmp_path / f"outputs/coin_report/llama-405b/run{k}/rows.jsonl"
        rows = read_jsonl(rows_path)
        assert len(rows) == 12 and all(set(r["report"]) == {"none", "self", "swap", "ctrl"} for r in rows)
        D = coin_report.load(rows_path)
        s = coin_report.stats(D, range(len(D["pair"])), {})
        assert math.isfinite(s["swap: share"]) and math.isfinite(s["kappa (swap+ctrl)"])
        assert math.isnan(s["swap_early: r(d, coin)"]) and math.isnan(s["kappa (swap_line+ctrl_line)"])
        coin_report.plot(D, "test", tmp_path / f"plot{k}.png")


# ---------------------------------------------------------------- Kimi K2 as DeepseekV3

def test_kimi_k2_reads_as_deepseek_v3():
    sys.path.insert(0, str(ROOT / "scripts"))
    from patch_load_check import remote_yarn
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    m = resolve_model(coin_report_config("kimi-k2"))
    config = tournament.kimi_k2_config(m["name"], m["revision"])
    assert isinstance(config, DeepseekV3Config) and config.model_type == "deepseek_v3"
    assert (config.n_routed_experts, config.num_attention_heads, config.first_k_dense_replace, config.n_group,
            config.num_mtp_layers, config.vocab_size) == (384, 64, 1, 1, 0, 163840)
    assert config.quantization_config["weight_block_size"] == [128, 128]
    inv_freq, mult = ROPE_INIT_FUNCTIONS["yarn"](config, "cpu")
    ref_inv, ref_mult, ref_scale = remote_yarn(config)
    assert torch.allclose(inv_freq, torch.tensor(ref_inv), rtol=1e-6) and mult == pytest.approx(ref_mult)
    assert tournament.LOADERS["kimi_k2"] is tournament.load_kimi_k2
