"""Apertus 1.5 70B by training stage (README, "Apertus 1.5 70B by training stage"): its prompts, configs and the text-model
loader. Needs only tokenizer files; the loader test builds a tiny random checkpoint in the release's key layout."""

import json

import numpy as np
import pytest
import torch
import yaml
from safetensors.torch import save_file
from transformers import ApertusConfig, ApertusForCausalLM, AutoTokenizer

from prompts import CONDITIONS, _strip_system, build_prompt, outcome_token_ids
from tournament import load_apertus1p5_text

# Pinned to the commits ../coinflip's Apertus runs used (its configs/apertus-70b-*.yaml, 2026-11-04).
STAGES = {
    "base": ("swiss-ai/Apertus-v1.5-70B-base", "0f3fbdd07bad839bb0c71b23d5adad0433dfcd16"),
    "sft": ("swiss-ai/Apertus-v1.5-70B-SFT", "a97457b345e188fc756f1378f1a37310cab9237d"),
    "rl": ("swiss-ai/Apertus-v1.5-70B-RL", "e7dc4fa2965c3317eea716ec144a6fa1456b83a4"),
    "dpo": ("swiss-ai/Apertus-v1.5-70B", "b212ebf8758bcd8200cc26b83a3f14759a94aa56"),
}
FRAME = ("<s><|developer_start|>Deliberation: disabled\nTool Capabilities: disabled<|developer_end|>"
         "<|user_start|>")


@pytest.fixture(scope="module")
def toks():
    return {s: AutoTokenizer.from_pretrained(name, revision=rev) for s, (name, rev) in STAGES.items()}


def test_post_trained_stages_share_one_template_and_the_base_has_none(toks):
    assert not toks["base"].chat_template
    assert toks["sft"].chat_template == toks["rl"].chat_template == toks["dpo"].chat_template


def test_prompts_drop_the_system_block_keep_the_developer_block_and_read_alike_at_every_stage(toks):
    for condition in ("assistant", "coinflip"):
        p = build_prompt(condition, "Explain gravity.", "Solve $x^2=4$.", toks["sft"], "none")
        assert p.startswith(FRAME) and "Apertus 1.5 Omni" not in p and "<|system_start|>" not in p
        tail = "<|user_end|><|assistant_start|>I'll choose Task" if condition == "assistant" else "it came up"
        assert p.endswith(tail)
        ids = toks["sft"](p, add_special_tokens=False).input_ids
        assert ids[0] == toks["sft"].bos_token_id and ids.count(toks["sft"].bos_token_id) == 1
        for s in STAGES:  # the base tokenizer reads the borrowed template's prompt identically
            assert toks[s](p, add_special_tokens=False).input_ids == ids
            assert ([toks[s].decode([i]) for i in outcome_token_ids(condition, toks[s])]
                    == list(CONDITIONS[condition]["outcomes"]))
    # "default" keeps the system block.
    assert build_prompt("coinflip", "X", "Y", toks["sft"], "default").startswith("<s><|system_start|>You are Apertus")


def test_prefill_continues_like_a_rendered_reply(toks):
    body = CONDITIONS["assistant"]["body"].format(first="FIRST", second="SECOND")
    conv = toks["sft"].apply_chat_template([{"role": "user", "content": body},
                                            {"role": "assistant", "content": "I'll choose Task A."}], tokenize=False)
    assert _strip_system(conv, body).startswith(build_prompt("assistant", "FIRST", "SECOND", toks["sft"], "none"))


def test_configs_pin_the_coinflip_checkpoints_and_share_the_sweep_pairs():
    for s, (name, rev) in STAGES.items():
        cfg = yaml.safe_load(open(f"configs/apertus/{s}.yaml"))
        (m,) = cfg["models"]
        assert (m["name"], m["revision"], m["stage"], m["loader"]) == (name, rev, s, "apertus1p5_text")
        assert m["conditions"] == ["assistant", "coinflip"]
        assert m.get("chat_template_from") == ("swiss-ai/Apertus-v1.5-70B-SFT" if s == "base" else None)
        assert (cfg["max_pairs"], cfg["seed"], cfg["system_prompt"], cfg["group"]) == (15000, 0, "none", "apertus")
        assert cfg["tasks_file"] == "data/tasks_n300_max500_seed0.jsonl"


def _tiny_release(path, drop=None):
    """A random Apertus text model saved in the 1.5 release's layout: text weights under model.language_model., an LM
    head over the first output_vocab_size ids only, and image-tokenizer tensors the text model must skip. Returns the
    full-vocabulary reference model."""
    text = dict(vocab_size=64, output_vocab_size=48, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                num_attention_heads=4, num_key_value_heads=2, hidden_act="xielu", tie_word_embeddings=False,
                bos_token_id=1, eos_token_id=2, pad_token_id=3)
    torch.manual_seed(0)
    ref = ApertusForCausalLM(ApertusConfig(**text)).eval()
    sd = {("lm_head.weight" if k == "lm_head.weight" else "model.language_model." + k.removeprefix("model.")):
          (v[:text["output_vocab_size"]] if k == "lm_head.weight" else v).clone().contiguous()
          for k, v in ref.state_dict().items()}
    assert any(".act_fn." in k for k in sd)  # xIELU's parameters travel with the checkpoint
    sd["model.vision_tokenizer.encoder.conv_in.weight"] = torch.randn(4, 3, 3, 3)
    if drop:
        del sd[drop]
    save_file(sd, str(path / "model.safetensors"), metadata={"format": "pt"})
    json.dump({"model_type": "apertus1p5", "architectures": ["Apertus1p5ForConditionalGeneration"],
               "text_config": {**text, "model_type": "apertus1p5_text", "qk_norm": True, "post_norm": False}},
              open(path / "config.json", "w"))
    return ref, text["output_vocab_size"]


def test_loader_reproduces_the_text_model(tmp_path):
    ref, n_out = _tiny_release(tmp_path)
    model = load_apertus1p5_text(str(tmp_path), None, torch.float32, "cpu")
    ids = torch.tensor([[1, 5, 9, 60, 33, 2, 17]])  # includes an id past the LM head (an image/audio id as input)
    with torch.no_grad():
        got, want = model(ids).logits, ref(ids).logits[..., :n_out]
    assert got.shape == want.shape and torch.allclose(got, want, atol=1e-5)
    # logits_to_keep, which tournament.score uses, reads the same last position.
    with torch.no_grad():
        assert torch.allclose(model(ids, logits_to_keep=1).logits[:, -1], got[:, -1])


def test_loader_fails_on_a_missing_text_weight(tmp_path):
    _tiny_release(tmp_path, drop="model.language_model.layers.1.self_attn.q_norm.weight")
    with pytest.raises(AssertionError):
        load_apertus1p5_text(str(tmp_path), None, torch.float32, "cpu")


def _synthetic_stages(n_per=40, seed=0):
    """Utilities for the four stages over five origins: the coinflip is align[s] x A + noise, plus a harm contrast."""
    from analyze import ORIGIN_STYLE
    import apertus
    rng = np.random.default_rng(seed)
    origins = np.repeat(list(ORIGIN_STYLE), n_per)
    benign = origins != "BAILBENCH"
    loglen = np.log(rng.integers(20, 500, len(origins)))
    a = rng.normal(0, 2, len(origins)) + np.where(benign, 1.0, -3.0)
    align = dict(zip(apertus.STAGES, (0.0, 0.02, 0.04, 0.08)))
    U = {}
    for s in apertus.STAGES:
        U[f"{s}/assistant"] = a + 0.05 * rng.normal(0, 1, len(a))
        U[f"{s}/coinflip"] = align[s] * a + 0.05 * rng.normal(0, 1, len(a))
    return U, origins, benign, loglen


def test_group_rs_matches_direct_correlations():
    import apertus
    from analyze import pearson
    U, origins, benign, loglen = _synthetic_stages()
    a, c = U["dpo/assistant"], U["dpo/coinflip"]
    rs = apertus.group_rs(a, c, origins, benign)
    assert len(rs) == len(apertus.GROUPS)
    assert rs[0] == pytest.approx(pearson(a, c)) and rs[1] == pytest.approx(pearson(a[benign], c[benign]))
    for j, g in enumerate(apertus.GROUPS[3:], start=3):
        assert rs[j] == pytest.approx(pearson(a[origins == g], c[origins == g]))
    assert rs[0] > rs[3] > 0.5  # the harm contrast adds to r over all tasks


def test_analysis_runs_end_to_end_on_synthetic_stages(tmp_path):
    import apertus
    from report import stratified_resamples
    U, origins, benign, loglen = _synthetic_stages()
    cells = apertus.stage_cells(U, benign, origins, loglen, stratified_resamples(origins, np.random.default_rng(1), 50))
    summary = {f"{s}/{c}/{k}": v for s in apertus.STAGES for c in ("assistant", "coinflip")
               for k, v in (("reliability", 0.97), ("outcome_mass_mean", 0.7), ("slot_bias_mean", 0.0),
                            ("additive_r2", 0.9))}
    table = apertus.stage_table(cells, summary)
    origin_rows = apertus.origin_table(cells, U, origins)
    assert len(origin_rows) == 4 * len(apertus.GROUPS)
    assert all(r["r_lo"] <= r["r"] + 1e-9 or np.isnan(r["r"]) for r in origin_rows)
    assert [r["stage"] for r in table] == list(apertus.STAGES)
    assert table[0]["r"] < 0.5 < table[-1]["r"]  # base coinflip is noise; the DPO one follows A
    exp = dict(apertus.expectations(table, origin_rows))
    assert exp["6. coin mass in [0.5, 0.8] at every stage"]
    assert exp["5. Assistant reliability >= 0.95 and benign above harmful at every stage"]
    assert len(apertus.step_table(cells)) == 4 * len(apertus.STATS)
    apertus.plot_scatter(U, origins, cells, tmp_path / "s.png", sharey=False)
    apertus.plot_r_by_stage(cells, tmp_path / "r.png")
    assert (tmp_path / "s.png").stat().st_size > 0 and (tmp_path / "r.png").stat().st_size > 0
