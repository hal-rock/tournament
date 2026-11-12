"""olmo2_attr.py on a tiny random OLMo 2 (real DPO tokenizer, CPU, fp32, lattice stride 1 so every sum is exact): the SFT
label mask is every assistant turn's content and EOS, per-pair coin accumulation and batched queries equal the gradient of
the whole metric taken in one backward, and a unit's score is the brute-force (grad f / rms) . g."""

import json
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoTokenizer, Olmo2Config, Olmo2ForCausalLM

import olmo2_attr as oa

TOK = "allenai/OLMo-2-0325-32B-DPO"


@pytest.fixture(scope="module")
def setup():
    tok = AutoTokenizer.from_pretrained(TOK)
    torch.manual_seed(0)
    cfg = Olmo2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                      num_key_value_heads=2, max_position_embeddings=2048, initializer_range=0.2)
    model = Olmo2ForCausalLM(cfg).float()
    model.gradient_checkpointing_enable()
    model.train()
    lat = oa.Lattice([(n, p) for n, p in model.named_parameters() if p.requires_grad], 1)
    tasks = {t["id"]: t["prompt"] for t in map(json.loads, open("data/tasks_n300_max500_seed0.jsonl")) if len(t["prompt"]) < 80}
    ids = list(tasks)[:5]
    rng = torch.Generator().manual_seed(1)
    pairs = []
    for a, b in [(ids[0], ids[1]), (ids[2], ids[3]), (ids[0], ids[4]), (ids[1], ids[3])]:
        w = {m: float(torch.randn(1, generator=rng)) for m in oa.COIN}
        st = dict(coin_ab=0.0, coin_ba=0.0, asst_ab=0.0, asst_ba=0.0)
        pairs.append(dict(a=a, b=b, w=w, stored={"base": st}))
    args = SimpleNamespace(point="base", query_limit=None, load_tol=1e9, batch=3, check_units=100, score_limit=None, estimate_limit=None)
    return tok, model, lat, tasks, dict(pairs=pairs), args


def full_grad(model, loss):
    model.zero_grad(set_to_none=True)
    loss.backward(retain_graph=True)
    g = torch.cat([p.grad.reshape(-1) for p in model.parameters()])
    model.zero_grad(set_to_none=True)
    return g


def flat(vectors):
    return torch.cat([v.float().reshape(-1) for v in vectors])


def test_sft_mask_is_assistant_content_and_eos(setup):
    tok = setup[0]
    msgs = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi there"}, {"role": "assistant", "content": "Hello! How can I help?"},
            {"role": "user", "content": "Name a color."}, {"role": "assistant", "content": "Blue."}]
    ids, mask = oa.encode_sft(tok, msgs)
    labelled = tok.decode([i for i, m in zip(ids, mask) if m])
    assert labelled == "Hello! How can I help?" + tok.eos_token + "Blue." + tok.eos_token
    assert ids[0] == tok.bos_token_id and not mask[0]


def test_coin_accumulation_is_the_metric_gradient(setup, tmp_path):
    tok, model, lat, tasks, spec, args = setup
    oa.stage_coin(model, lat, tok, tasks, spec, args, tmp_path)
    heads, tails = oa.outcome_ids("coinflip", tok)
    total = {m: 0.0 for m in oa.COIN}
    for r in spec["pairs"]:
        enc = oa.encode_batch(tok, oa.pair_prompts("coinflip", tasks, r["a"], r["b"], tok), "cpu")
        lg = oa.last_logits(model, enc)
        c = ((lg[0, heads] - lg[0, tails]) - (lg[1, heads] - lg[1, tails])) / 2
        for m in oa.COIN:
            total[m] = total[m] + r["w"][m] * c
    for m in ("benign_odd", "t1", "harm_gap"):
        q = flat(oa.load_vectors(lat, tmp_path / f"query_{m}.safetensors"))
        g = full_grad(model, total[m])
        assert torch.allclose(q, g, rtol=2e-2, atol=1e-5 * g.abs().max().item()), m  # saved in bf16
    benign = flat(oa.load_vectors(lat, tmp_path / "query_benign.safetensors"))
    g = full_grad(model, total["benign_odd"] + total["benign_even"])
    assert torch.allclose(benign, g, rtol=2e-2, atol=1e-5 * g.abs().max().item())
    info = json.load(open(tmp_path / "queries.json"))
    assert info["coin"]["values"]["t1"] == pytest.approx(total["t1"].item(), abs=1e-4)


def test_batched_query_is_batch_independent(setup, tmp_path):
    tok, model, lat, tasks, spec, args = setup
    ids = tuple(oa.outcome_ids("assistant", tok))
    items = []
    for r in spec["pairs"]:
        for sign, text in zip((0.5, -0.5), oa.pair_prompts("assistant", tasks, r["a"], r["b"], tok)):
            items.append((text, sign * r["w"]["perm1"], ids, None))
    for batch in (1, 3):
        d = tmp_path / f"b{batch}"
        d.mkdir()
        oa.batched_query(model, lat, tok, items, "asst", d, SimpleNamespace(**{**vars(args), "batch": batch}), check=False)
    a, b = (flat(oa.load_vectors(lat, tmp_path / f"b{x}" / "query_asst.safetensors")) for x in (1, 3))
    assert torch.allclose(a, b, rtol=2e-2, atol=1e-6 * a.abs().max().item())
    fa, fb = (json.load(open(tmp_path / f"b{x}" / "queries.json"))["asst"]["f"] for x in (1, 3))
    assert fa == pytest.approx(fb, abs=1e-4)


def test_standard_prompts_are_the_coinflip_layout(setup):
    tok = setup[0]
    std = json.load(open("data/olmo2_attr/standard.json"))
    texts, signs = oa.standard_prompts(tok, std)
    assert len(texts) == 50 and sum(signs) == 0
    assert texts[0].endswith("Okay, I'm flipping the coin now ... it came up")
    assert f"Task A (if heads): {std['preferred'][0]}" in texts[0] and f"Task B (if tails): {std['dispreferred'][0]}" in texts[0]
    assert f"Task A (if heads): {std['dispreferred'][0]}" in texts[1]


def test_unit_score_is_brute_force(setup, tmp_path):
    tok, model, lat, tasks, spec, args = setup
    torch.manual_seed(3)
    for n in oa.SCORE_QUERIES:
        oa.save_vectors(lat, [torch.randn_like(p).reshape(-1).to(torch.bfloat16) for p in lat.params], tmp_path / f"query_{n}.safetensors")
    for h in "ab":
        oa.save_vectors(lat, [torch.rand_like(p).reshape(-1) + 0.1 for p in lat.params], tmp_path / f"second_{h}.safetensors")
    rows = tmp_path / "rows.jsonl"
    msgs = [{"role": "user", "content": "What is 2 + 2?"}, {"role": "assistant", "content": "4."}]
    rows.write_text(json.dumps(dict(idx=7, id="x7", source="s", messages=msgs)) + "\n")
    sample = [dict(idx=7, id="x7", source="s", category="math", set="strat")]
    oa.stage_score(model, lat, oa.Units(tok, rows), sample, args, tmp_path)
    row = json.loads(open(tmp_path / "scores.jsonl").readline())
    u = oa.Units(tok, rows).get(sample[0])
    g = full_grad(model, oa.logprob_masked(model, u["ids"], u["mask"]))
    second = [(a + b) / 2 for a, b in zip(*(oa.load_vectors(lat, tmp_path / f"second_{h}.safetensors") for h in "ab"))]
    w = flat([x.to(torch.bfloat16) for x in oa.rms_weights(lat, second)])
    for n in ("benign", "standard", "perm2"):
        q = flat(oa.load_vectors(lat, tmp_path / f"query_{n}.safetensors"))
        assert row["adam"][n] == pytest.approx(torch.dot(g * w, q).item(), rel=1e-3)
        assert row["raw"][n] == pytest.approx(torch.dot(g, q).item(), rel=1e-3)
    assert row["g_adam_sq"] == pytest.approx(torch.dot(g * w, g * w).item(), rel=1e-3)
    assert row["lattice_even"] + row["lattice_odd"] == pytest.approx(2 * row["adam"]["benign"], rel=0.5)  # two halves, each x2
    assert sum(row["group_adam_benign"]) == pytest.approx(row["adam"]["benign"], rel=1e-6)
