"""rollouts.py on a tiny random Llama-architecture model (real tokenizer, CPU, fp32, eager attention): the prompts extend the
coinflip through " up", coupled sampling repeats itself, a none/swap pair shares its noise, and teacher-forced scoring
matches the generation."""

import pytest
import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

import prompts
from analyze import read_jsonl
from rollouts import generate_group, prefix, score, up_residuals

LLAMA = "unsloth/Llama-3.3-70B-Instruct"


@pytest.fixture(scope="module")
def setup():
    tok = AutoTokenizer.from_pretrained(LLAMA)
    tasks = [t for t in read_jsonl("data/tasks_n300_max500_seed0.jsonl") if len(t["prompt"]) < 100][:2]
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                      num_key_value_heads=2, max_position_embeddings=1024, initializer_range=0.5, attn_implementation="eager")
    model = LlamaForCausalLM(cfg).eval()
    return tok, tasks, model


@pytest.mark.parametrize("outcome", ["heads", "tails"])
def test_task_prompt_extends_the_coinflip(setup, outcome):
    tok, tasks, _ = setup
    first, second = tasks[0]["prompt"], tasks[1]["prompt"]
    ids_pre, up = prefix(tok, first, second)
    ids = tok(prompts.build_prompt(f"coin_task_{outcome}", first, second, tok), add_special_tokens=False).input_ids
    assert ids[:up + 1] == ids_pre
    text = tok.decode(ids)
    assert f"it came up {outcome}, so please do Task {'A' if outcome == 'heads' else 'B'}." in text
    assert text.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")


def test_coupled_generation_and_scoring(setup):
    tok, tasks, model = setup
    first, second = tasks[0]["prompt"], tasks[1]["prompt"]
    ids = tok(prompts.build_prompt("coin_task_heads", first, second, tok), add_special_tokens=False).input_ids
    _, up = prefix(tok, first, second)
    source = up_residuals(model, prefix(tok, second, first)[0], up, 2)
    res = generate_group(model, tok, ids, up, source, [11, 12], 12, 2)
    by = {(c, s): rep for c, s, rep, _ in res}
    assert by["none_dup", 11] == by["none", 11]  # same noise, same rows: identical
    assert by["none", 11] != by["none", 12]  # different noise: different samples
    rep = by["none", 11]
    logp, attn = score(model, ids, up, rep, source, [0, 1], 2)
    assert logp.shape == (2, len(rep)) and attn.shape == (2, len(ids) + len(rep) - up - 1)
    assert (attn >= 0).all() and (attn <= 1).all() and attn.sum() > 0
    # the reply is the argmax of logits + noise, so each of its tokens has finite log-prob; none and swap differ somewhere
    assert torch.isfinite(logp).all()
    assert not torch.equal(logp[0], logp[1])


def test_page_from_tiny_rows(setup, tmp_path):
    """Rows as rollouts.main writes them, from the tiny model, make a page and a summary."""
    import json
    from matskit import io
    from rollouts_page import build
    tok, tasks, model = setup
    first, second = tasks[0]["prompt"], tasks[1]["prompt"]
    _, up = prefix(tok, first, second)
    source = up_residuals(model, prefix(tok, second, first)[0], up, 2)
    run = tmp_path / "llama-70b" / "run1"
    run.mkdir(parents=True)
    (run / "lens.json").write_text(json.dumps({"r": [0, 1, 1], "onset": 1, "rel_layers": [1]}))
    for outcome in ("heads", "tails"):
        ids = tok(prompts.build_prompt(f"coin_task_{outcome}", first, second, tok), add_special_tokens=False).input_ids
        for cond, seed, reply, finished in generate_group(model, tok, ids, up, source, [5], 8, 2):
            if cond == "none_dup":
                continue
            logp, attn = score(model, ids, up, reply, source, [1], 2)
            io.append_jsonl(run / "rows.jsonl", dict(pair=0, a=tasks[0]["id"], b=tasks[1]["id"], outcome=outcome,
                            condition=cond, sample=0, seed=seed, up=up, prompt_ids=ids, reply_ids=reply, finished=finished,
                            logp_none=logp[0].tolist(), logp_swap=logp[1].tolist(), attn_none=attn[0].tolist(),
                            attn_swap=attn[1].tolist()))
    out = build(run / "rows.jsonl", tmp_path / "pages")
    assert out.exists() and out.stat().st_size > 1000
