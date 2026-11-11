"""report_patch.py on a tiny random Llama-architecture model (real tokenizer, CPU, fp32): the prompts extend the coinflip
token for token, a self-patch changes nothing, batching changes nothing, and a swap does change the report."""

import pytest
import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

import prompts
from analyze import read_jsonl
from report_patch import ARMS, build_items, forward, outcome, run_chunk

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
                                    "coin_predict", "coin_predict_bold", "coin_predict_think", "coin_predict_think_bold",
                                    "coin_predict_my", "coin_predict_my_bold"])
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
