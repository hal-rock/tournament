"""Preference probe: prompt bookkeeping (real OLMo 3.1 tokenizer), activation capture and interventions on a
tiny random OLMo 3 (pipeline only; don't read numbers into it)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from probe_prompts import (all_tasks, extraction_items, load_design, outcome_ids, pair_item, single_item,
                           test_pairs as make_test_pairs)

REAL = "allenai/Olmo-3.1-32B-Instruct-DPO"
COINFLIP = Path(__file__).resolve().parents[2] / "coinflip"


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(REAL)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory, tok):
    """A tiny random OLMo 3 with the real tokenizer (as ../coinflip/circuits/tests/make_tiny.py)."""
    from transformers import AutoConfig, AutoModelForCausalLM
    torch.manual_seed(0)
    cfg = AutoConfig.from_pretrained(REAL)
    cfg.update(dict(hidden_size=64, intermediate_size=96, num_hidden_layers=4, num_attention_heads=4,
                    num_key_value_heads=2, head_dim=16, layer_types=["sliding_attention"] * 3 + ["full_attention"],
                    vocab_size=len(tok), tie_word_embeddings=False))
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name:
                p.add_(0.1 * torch.randn_like(p))
    path = tmp_path_factory.mktemp("tiny")
    model.save_pretrained(path)
    tok.save_pretrained(path)
    return str(path)


def awkward_tasks():
    """Tasks ending in "?", "$", "]", a letter, and ones with newlines inside."""
    texts = all_tasks()
    picked = {}
    for t, x in texts.items():
        key = "newline" if "\n" in x else x[-1]
        picked.setdefault(key, t)
    return [texts[t] for t in picked.values()]


@pytest.mark.parametrize("kind", ["assistant", "coinflip", "choice"])
def test_pair_reads(tok, kind):
    tasks = awkward_tasks()
    first_ids, second_ids = outcome_ids(tok, kind)
    for k in range(len(tasks)):
        first, second = tasks[k], tasks[(k + 3) % len(tasks)]
        it = pair_item(tok, kind, first, second)
        enc = tok.decode(it.ids)
        offs = tok(enc, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        for slot, task in (("a", first), ("b", second)):
            n = it.read[f"task_{slot}"]
            a, b = offs[n]
            # the read token closes the task: the text up to it ends with the task (plus newlines), and the next
            # token is the next segment's
            assert enc[:b].rstrip("\n").endswith(task.strip()), (kind, slot, enc[a:b])
            assert enc[a:b].strip("\n") == "" or task.strip().endswith(enc[a:b].rstrip("\n")), enc[a:b]
            assert it.seg[n] == f"task_{slot}" and it.seg[n + 1] != f"task_{slot}"
        assert it.ids[it.read["label_a"]] == first_ids and it.ids[it.read["label_b"]] == second_ids
        assert it.seg[-1] == "ending" and it.read["last"] == len(it.ids) - 1
        segs = [s for i, s in enumerate(it.seg) if i == 0 or it.seg[i - 1] != s]
        assert segs == ["prefix", "label_a", "task_a", "label_b", "task_b", "ending"], segs
        if kind == "assistant":
            assert tok.decode([it.ids[it.read["task_b"] + 1]]) == "<|im_end|>"


def test_single_item(tok):
    for task in awkward_tasks():
        it = single_item(tok, task)
        text = tok.decode(it.ids)
        assert text.startswith("<|im_start|>user\n" + task.strip()) and text.endswith("<|im_start|>assistant\n")
        assert tok.decode([it.ids[it.read["eot"]]]) == "<|im_end|>"
        assert it.read["eot"] == len(it.ids) - 5 or tok.decode(it.ids[it.read["eot"] + 1:]) == "\n<|im_start|>assistant\n"


def test_extraction_items_cover_every_task(tok):
    design = load_design()
    items = extraction_items(tok, design, limit=6)
    tasks = list(all_tasks(design))[:6]
    for t in tasks:
        mine = [i for i in items if i.meta["task"] == t]
        n_partners = sum(p != t for p in design["partners"])
        assert sum(i.kind == "single" for i in mine) == 1
        for kind in ("assistant", "coinflip"):
            for slot in "ab":
                assert sum(i.kind == kind and i.meta["slot"] == slot for i in mine) == n_partners
    a = next(i for i in items if i.kind == "coinflip" and i.meta["slot"] == "a")
    assert set(a.read) == {"task", "label_b"}


@pytest.mark.skipif(not COINFLIP.exists(), reason="needs ../coinflip")
@pytest.mark.parametrize("prompt", ["coinflip", "choice"])
def test_standard_pairs_match_coinflip(tok, prompt):
    """Our rebuild of the 25 standard pairs is token-identical to ../coinflip/circuits/pairs.py, segments included."""
    sys.path[:0] = [str(COINFLIP), str(COINFLIP / "circuits")]
    try:
        from pairs import make_pairs
        theirs = make_pairs(tok, root=str(COINFLIP), prompt=prompt)
    finally:
        sys.path[:2] = []
    mine = make_test_pairs(tok, prompt, which="standard")
    assert len(mine) == len(theirs) == 25
    for (p, d, hp, tp), pair in zip(mine, theirs):
        assert (p, d) == (f"std_pref_{pair.pref_idx}", f"std_dispref_{pair.dispref_idx}")
        for a, b in ((hp, pair.hp), (tp, pair.tp)):
            assert a.ids == b.ids
            assert a.seg == b.seg


def test_extract_matches_hidden_states(tok, tiny):
    """Saved partner means equal hidden_states[b+1] at the read positions, averaged by hand."""
    from matskit import models
    from probe_acts import extract
    model, _ = models.load(tiny, device_map="cpu", dtype="float32", check=False)
    design = load_design()
    task_ids = list(all_tasks(design))[:2]
    items = extraction_items(tok, design, limit=2)
    means, records, info = extract(model, tok, items, task_ids, progress_every=0)
    L = models.n_layers(model)
    assert means["assistant_task"].shape == (2, 2, L, 64) and means["single_eot"].shape == (2, L, 64)
    with torch.no_grad():
        def hs(it, pos):
            out = model(input_ids=torch.tensor([it.ids]), output_hidden_states=True)
            # hidden_states[L] has the final norm applied, so compare blocks 0..L-2 there and the last block via
            # its pre-norm output captured separately
            return torch.stack([out.hidden_states[b + 1][0, pos] for b in range(L - 1)])
        t = task_ids[1]
        mine = [i for i in items if i.meta["task"] == t and i.kind == "coinflip" and i.meta["slot"] == "b"]
        manual = torch.stack([hs(i, i.read["task"]) for i in mine]).mean(0)
        torch.testing.assert_close(means["coinflip_task"][1, 1, :L - 1], manual, rtol=1e-4, atol=1e-4)
        single = next(i for i in items if i.meta["task"] == t and i.kind == "single")
        torch.testing.assert_close(means["single_eot"][1, :L - 1], hs(single, single.read["eot"]), rtol=1e-4, atol=1e-4)
    # L matches log-softmax of the full logits at the last token
    it = next(i for i in items if i.kind == "assistant")
    rec = records[items.index(it)]
    lp = torch.log_softmax(model(input_ids=torch.tensor([it.ids])).logits[0, -1], -1)
    a, b = outcome_ids(tok, "assistant")
    assert abs(rec["L"] - (lp[a] - lp[b]).item()) < 1e-4


def test_bits_roundtrip():
    from probe_acts import as_float, to_bits
    x = torch.randn(3, 5) * 1e4
    np.testing.assert_array_equal(as_float(to_bits(x)), x.to(torch.bfloat16).float().numpy())


def test_design_file():
    d = load_design()
    assert len(d["partners"]) == 4 and len(d["standard"]["pref"]) == len(d["standard"]["dispref"]) == 5
    assert len({tuple(p) for p in d["benign_pairs"]}) == len(d["benign_pairs"])
    json.dumps(d)
