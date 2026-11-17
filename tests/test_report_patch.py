"""report_patch.py on a tiny random Llama-architecture model (real tokenizer, CPU, fp32): the prompts extend the coinflip
token for token, a self-patch changes nothing, batching changes nothing, and a swap does change the report."""

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

import prompts
from analyze import read_jsonl
from report_patch import (ARMS, arm_blocks, arm_specs, build_items, coin_grads, forward, lens_dirs, mix,
                          outcome, rand_dirs, run_chunk, unit)

LLAMA = "unsloth/Llama-3.3-70B-Instruct"


@pytest.fixture(scope="module")
def setup():
    tok = AutoTokenizer.from_pretrained(LLAMA)
    tasks = [t for t in read_jsonl("data/tasks_n300_max500_seed0.jsonl") if len(t["prompt"]) < 120][:4]
    items = build_items(tok, tasks, [(0, 1), (2, 3)])
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=1024,
                      initializer_range=0.5)  # large weights, so a random model's report depends on its context
    model = LlamaForCausalLM(cfg).eval()
    return tok, items, model


def test_prompt_extends_the_coinflip(setup):
    tok, items, _ = setup
    it = items[0]
    text = tok.decode(it["ids"])
    assert "it came up. But before I tell you which way it landed" in text and text.endswith("I would have preferred")
    assert tok.decode(it["ids"][it["up"]]).strip() == "up"
    assert tok.decode(it["ok_ids"][it["line"][0]]).strip() == "OK"
    assert prompts.CONDITIONS["coin_report"]["body"].startswith(prompts.CONDITIONS["coinflip"]["body"])
    assert items[0]["a"] == items[1]["b"] and items[0]["b"] == items[1]["a"]


def test_self_patch_is_the_identity_and_swap_is_not(setup):
    tok, items, model = setup
    out_ids = prompts.outcome_token_ids("coin_report", tok)
    rows = run_chunk(model, items, tok.eos_token_id, out_ids, out_ids, 2, batch_size=3)
    for r in rows:
        assert r["report"]["self"] == pytest.approx(r["report"]["none"], abs=1e-5)
        assert set(r["report"]) == {"none", *ARMS}
    assert max(abs(r["report"]["swap"] - r["report"]["none"]) for r in rows) > 1e-4  # self is exact; a swap is not
    for it, r in zip(items, rows):  # batched rows match a forward of the prompt alone
        lp, _ = forward(model, [it["ids"]], tok.eos_token_id, torch.tensor([[it["up"], len(it["ids"]) - 1]]))
        coin, rep = outcome(lp[0], out_ids)[0].tolist()  # same outcome tokens for coin and report here
        assert coin == pytest.approx(r["coin"], abs=1e-4) and rep == pytest.approx(r["report"]["none"], abs=1e-4)


def test_swap_puts_the_other_orders_residual_at_up(setup):
    """Inside a patched forward, the residual entering every block at " up" is the other order's."""
    from matskit import acts
    tok, items, model = setup
    a, b = items[0], items[1]
    pos = torch.tensor([[a["up"]]])
    _, own_b = forward(model, [b["ids"]], tok.eos_token_id, torch.tensor([len(b["ids"]) - 1]), ([0, 1], pos))
    values = {l: own_b[l] for l in (0, 1)}  # [1, 1, D]
    with acts.replace(model, values, at="pre", positions=pos):
        _, seen = forward(model, [a["ids"]], tok.eos_token_id, torch.tensor([len(a["ids"]) - 1]), ([0, 1], pos))
    for l in (0, 1):
        assert torch.equal(seen[l], own_b[l])
    _, own_a = forward(model, [a["ids"]], tok.eos_token_id, torch.tensor([len(a["ids"]) - 1]), ([1], pos))
    assert not torch.allclose(own_a[1], own_b[1])  # the two orders differ at " up" after block 0


@pytest.mark.parametrize("report", ["coin_report_bold", "coin_report_ab", "coin_report_short", "coin_report_short_bold",
                                    "coin_report_short_land", "coin_report_short_land_bold",
                                    "coin_report_short_coin", "coin_report_short_coin_bold",
                                    "coin_predict", "coin_predict_bold", "coin_predict_think", "coin_predict_think_bold",
                                    "coin_predict_my", "coin_predict_my_bold", "coin_choose_ab", "coin_choose_ht",
                                    "coin_choose_ht_bold"])
def test_report_variants_extend_the_coinflip(setup, report):
    tok, _, _ = setup
    tasks = [t for t in read_jsonl("data/tasks_n300_max500_seed0.jsonl") if len(t["prompt"]) < 120][:2]
    (it, _) = build_items(tok, tasks, [(0, 1)], report)
    text = tok.decode(it["ids"])
    assert text.endswith(prompts.CONDITIONS[report]["prefill"])
    assert len(prompts.outcome_token_ids(report, tok)) == 2
    if report.startswith(("coin_report_short", "coin_predict")):  # the question doesn't name the outcomes again
        question = text[text.index("it came up. But before"):]
        assert "heads" not in question.lower() and "tails" not in question.lower(), question



def test_suffix_is_shared_and_starts_the_line(setup):
    tok, items, _ = setup
    for it in items:
        assert it["suffix"][0] == it["line"][0] and it["suffix"][-1] == len(it["ids"]) - 1
        assert it["suffix"][len(it["line"]) - 1] == it["up"]


def test_coin_grads_match_a_full_forward_gradient(setup):
    """The cached-prefix gradient at " up" equals autograd through a full forward of the coinflip prompt."""
    tok, items, model = setup
    coin_ids = prompts.outcome_token_ids("coinflip", tok)
    it = items[0]
    (g, y), = coin_grads(model, [it], coin_ids)
    ids = torch.tensor([it["ids"][:it["up"] + 1]])
    from matskit.models import layers as get_layers
    blocks, z = get_layers(model), {}

    def hook(m, a, kw, l):
        h = a[0] if a else kw["hidden_states"]
        z[l] = torch.zeros(h.shape[-1], requires_grad=True)
        h = h.clone()
        h[0, -1] = h[0, -1] + z[l]
        return ((h,) + tuple(a[1:]), kw) if a else (a, {**kw, "hidden_states": h})

    hs = [b.register_forward_pre_hook(lambda m, a, kw, l=l: hook(m, a, kw, l), with_kwargs=True) for l, b in enumerate(blocks)]
    try:
        logits = model(input_ids=ids, use_cache=False).logits[0, -1]
        yy = logits[coin_ids[0]] - logits[coin_ids[1]]
        yy.backward()
    finally:
        for h in hs:
            h.remove()
    assert y == pytest.approx(yy.item(), abs=1e-4)
    for l in range(len(blocks)):
        torch.testing.assert_close(g[l], z[l].grad, rtol=1e-3, atol=1e-5)
    assert g.abs().sum() > 0


def test_mix_splits_the_swap_into_potent_and_null_parts():
    torch.manual_seed(0)
    own, src = torch.randn(1, 8), torch.randn(1, 8)
    u, rho = unit(torch.randn(8)), unit(torch.randn(8))
    pot, null = mix(own, src, "pot_grad", u), mix(own, src, "null_grad", u)
    torch.testing.assert_close(pot + null - own, src)  # the two parts add up to the swap
    torch.testing.assert_close(((pot - own) * u).sum(), ((src - own) * u).sum())  # pot moves own along u only
    torch.testing.assert_close(((null - own) * u).sum(), torch.tensor(0.0), atol=1e-6, rtol=0)
    rnd = mix(own, src, "rand", u, rho)
    torch.testing.assert_close((rnd - own).norm(), ((src - own) * u).sum().abs(), rtol=1e-5, atol=1e-6)  # matched size
    assert mix(own, src, None) is src


def test_lens_dir_raises_the_lens_log_odds(setup):
    from matskit import acts
    tok, items, model = setup
    coin_ids = prompts.outcome_token_ids("coinflip", tok)
    H = torch.randn(3, model.config.hidden_size)
    U = lens_dirs(model, H, coin_ids)
    f = lambda h: (lambda x: x[..., 0] - x[..., 1])(acts.logit_lens(model, h, token_ids=coin_ids))
    eps = 1e-3 * H.norm(dim=-1, keepdim=True)
    assert (f(H + eps * U) > f(H)).all() and (f(H - eps * U) < f(H)).all()


def test_followup_arms(setup):
    """line3 and line4 arm sets: self is exact; pos at " up" is the swap; the potent and null parts move the report;
    a single-block arm patches one block; the lens readout lines up with the coin at " up"."""
    tok, items, model = setup
    coin_ids = prompts.outcome_token_ids("coinflip", tok)
    out_ids = prompts.outcome_token_ids("coin_report", tok)
    n_line, n_suffix = len(items[0]["line"]), len(items[0]["suffix"])
    specs = arm_specs(["base", "line3", "line4"], n_line, n_suffix)
    assert {f"pos{k:02d}" for k in range(n_suffix)} <= set(specs) and f"loo{n_line - 1:02d}" in specs
    assert arm_blocks(specs["v90_full"].blocks, 10) == [9] and arm_blocks(specs["v50_pot"].blocks, 10) == [5]
    lens_pairs = [tuple(coin_ids), tuple(out_ids)]
    rows = run_chunk(model, items, tok.eos_token_id, coin_ids, out_ids, 2, batch_size=3, specs=specs, lens_pairs=lens_pairs)
    for r in rows:
        assert r["report"]["self"] == pytest.approx(r["report"]["none"], abs=1e-5)
        assert r["report"][f"pos{n_line - 1:02d}"] == pytest.approx(r["report"]["swap"], abs=1e-5)  # the same patch
        assert r["coin_grad_path"] == pytest.approx(r["coin"], abs=1e-4)
        assert len(r["lens"]) == 2 and len(r["lens"][0]) == n_suffix and len(r["out"]) == n_suffix
        assert r["out"][n_line - 1][0] == pytest.approx(r["coin"], abs=1e-5)  # the output read at " up" is the coin
        # swapping all blocks' " up" makes the coin read at " up" the other order's (up to the last block's own attention)
        assert r["coin_arm"]["self"] == pytest.approx(r["coin"], abs=1e-5)
    # every arm moves the coin read at " up" (this random model's report barely reads " up" at all: the swap moves it
    # by ~3e-4 nats), and the potent and null parts each move it
    for arm in ("swap_pot_grad", "swap_null_grad", "swap_pot_lens", "swap_rand", "v50_full", "v50_pot", "loo00"):
        assert max(abs(r["coin_arm"][arm] - r["coin"]) for r in rows) > 1e-3, arm
    # positions after " up" can't move the coin there; the readout position itself moves the report a lot
    for k in range(n_line, n_suffix):
        assert all(r["coin_arm"][f"pos{k:02d}"] == r["coin"] for r in rows)
    assert max(abs(r["report"][f"pos{n_suffix - 1:02d}"] - r["report"]["none"]) for r in rows) > 1.0
    # the whole line without its last token is the line minus " up": it differs from the whole-line swap
    assert max(abs(r["report"][f"loo{n_line - 1:02d}"] - r["report"]["swap_line"]) for r in rows) > 1e-6


def test_push_arms(setup):
    """Pushes of the heads/tails lean: a zero dose leaves everything unchanged; a small push moves the lens log-odds by the
    dose to first order; + and - pushes at " up" move the coin there in opposite directions; random-direction pushes are
    not the lens pushes; a transplant with zero slopes is the unpatched pass; inject_positions finds the tokens."""
    from report_patch import Arm, inject_positions
    tok, items, model = setup
    coin_ids = prompts.outcome_token_ids("coinflip", tok)
    out_ids = prompts.outcome_token_ids("coin_report", tok)
    ks = inject_positions(tok, items[0], [",", " the", " ...", " it", " came", " up", "."])
    assert [tok.decode([items[0]["ids"][items[0]["suffix"][k]]]) for k in ks.values()] == list(ks)
    up, ctx = ks[" up"], sorted(k for t, k in ks.items() if t != " up")
    specs = arm_specs(["inject"], len(items[0]["line"]), len(items[0]["suffix"]),
                      dict(up=up, ctx=ctx, onset=1, dose=0.05, seeds=[0]))
    specs["zero"] = Arm("own", ("tokens", tuple(sorted((up, *ctx)))), ("from", 1), "push_lens", 0.0)
    A = {it["a"]: 0.3 * i for i, it in enumerate(items)} | {it["b"]: -0.2 * i for i, it in enumerate(items)}
    tp = {"A": A, "slope": torch.zeros(2, len(ctx) + 1)}
    rows = run_chunk(model, items, tok.eos_token_id, coin_ids, out_ids, 2, 3, specs=specs, transplant=tp)
    for r in rows:
        assert r["report"]["zero"] == pytest.approx(r["report"]["none"], abs=1e-5)
        assert r["report"]["transplant_all"] == pytest.approx(r["report"]["none"], abs=1e-5)  # zero slopes
        assert r["push_check"]["push_all_pos"] == pytest.approx(1.0, abs=0.05)  # first order, small dose
    assert all(r["coin_arm"]["push_up_pos"] > r["coin_arm"]["push_up_neg"] for r in rows)
    assert max(abs(r["report"]["rand0_all_pos"] - r["report"]["push_all_pos"]) for r in rows) > 1e-6
    # a transplant with slopes pushes toward the first task when the first task has the higher utility
    tp["slope"] = torch.ones(2, len(ctx) + 1) * 0.5
    rows = run_chunk(model, items, tok.eos_token_id, coin_ids, out_ids, 2, 3,
                     specs={"self": specs["self"], "transplant_all": specs["transplant_all"]}, transplant=tp)
    for r, it in zip(rows, items):
        shift = r["coin_arm"]["transplant_all"] - r["coin"]
        assert np.sign(shift) == np.sign(A[it["a"]] - A[it["b"]]) or abs(A[it["a"]] - A[it["b"]]) < 1e-9
