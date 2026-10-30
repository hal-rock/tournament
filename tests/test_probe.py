"""Preference probe: prompt bookkeeping (real OLMo 3.1 tokenizer), activation capture and interventions on a
tiny random OLMo 3 (pipeline only; don't read numbers into it)."""

import contextlib
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


def test_ridge_weights_reproduce_predictions_and_recover_a_planted_signal():
    from probe_fit import Gram, cv_predict, pearson
    rng = np.random.default_rng(0)
    n, d = 120, 400
    # low-rank activations (10 latent factors) with uneven per-dimension scales and offsets; y reads factor 0
    F = rng.normal(size=(n, 10))
    X = (F @ rng.normal(size=(10, d)) + rng.normal(0, 1, (n, d))) * rng.uniform(0.1, 30, d) + rng.normal(0, 50, d)
    y = F[:, 0] + rng.normal(0, 0.3, n)
    gram = Gram(X)
    m = gram.fit(np.arange(100), y[:100])
    w, b = m.weights(1.0)
    np.testing.assert_allclose(X[100:] @ w + b, m.predict(np.arange(100, n), [1.0])[:, 0], rtol=1e-8, atol=1e-8)
    pred, _ = cv_predict(gram, y, np.arange(n))
    assert pearson(pred, y) > 0.8
    null, _ = cv_predict(gram, rng.permutation(y), np.arange(n))
    assert abs(pearson(null, y)) < 0.4


def test_fit_one_on_synthetic_acts(tmp_path):
    """End to end on fake activations whose layer-1 slot readouts carry the DPO utility: the fit finds it there,
    not at layer 0, and the benign probe separates harmful tasks (which have low utility)."""
    import csv
    from probe_acts import to_bits
    from probe_fit import Data, fit_one
    design = load_design()
    task_ids = list(all_tasks(design))
    rows = {r["id"]: r for r in csv.DictReader(open("outputs/d8ysbybo/analysis/utilities.csv"))} \
        if Path("outputs/d8ysbybo/analysis/utilities.csv").exists() else None
    if rows is None:
        pytest.skip("needs outputs/d8ysbybo/analysis/utilities.csv")
    rng = np.random.default_rng(1)
    u = np.array([float(rows[t]["dpo/assistant"]) if t in rows else (5.0 if "pref_" in t[:9] else -15.0)
                  for t in task_ids])
    n, L, D = len(task_ids), 2, 32
    direction = rng.normal(size=D)
    base = rng.normal(size=(n, L, D))
    base[:, 1] += 0.3 * u[:, None] * direction
    slotted = np.stack([base, base + rng.normal(0, 0.3, size=base.shape)], 1)
    acts = tmp_path / "acts"
    acts.mkdir()
    for name, arr in dict(single_eot=base, single_last=base, coinflip_label_b=base, assistant_task=slotted,
                          coinflip_task=slotted).items():
        np.save(acts / f"{name}.npy", to_bits(torch.tensor(arr, dtype=torch.float32)))
    (acts / "tasks.json").write_text(json.dumps(task_ids))
    data = Data(acts)
    assert len(data.idx) == 300 and data.harm.sum() == 43
    good, *_ = fit_one(data, "assistant_task", 1, "dpo", "benign", n_null=1)
    bad, *_ = fit_one(data, "assistant_task", 0, "dpo", "all", n_null=1)
    assert good["r_cv"] > 0.9 and abs(bad["r_cv"]) < 0.3
    assert good["harm_auroc"] > 0.9 and good["std_acc"] == 1.0


def test_swap_arms_set_the_other_orders_projections(tok, tiny, monkeypatch):
    """After a single-block swap arm, each slot's mean projection at that block equals the other prompt's."""
    from matskit import acts, models
    import probe_tests as pt
    model, _ = models.load(tiny, device_map="cpu", dtype="float32", check=False)
    monkeypatch.setattr(pt, "SWAP_BLOCKS", [1, 2])
    dirs = pt.load_directions("nonexistent.npz", models.n_layers(model), 64)
    _, _, hp, tp = make_test_pairs(tok, "coinflip", which="standard")[0]
    _, h_hp = pt.baseline(model, hp, [1, 2])
    _, h_tp = pt.baseline(model, tp, [1, 2])
    arms = pt.swap_arms(hp, tp, h_hp, h_tp, {"rand0": dirs["rand0"]}, [2])
    (label, add, clamp), = arms
    assert label == dict(arm="swap", direction="rand0", blocks="2") and clamp == {}
    with acts.steer_tokens(model, add, at="post"):
        after = acts.residuals(model, torch.tensor([hp.ids]), layers=[2], at="post")[2][0]
    u = dirs["rand0"][2]
    for s in ("task_a", "task_b"):
        m_hp = (after @ u)[torch.tensor([x == s for x in hp.seg])].mean()
        m_tp = (h_tp[2] @ u)[torch.tensor([x == s for x in tp.seg])].mean()
        torch.testing.assert_close(m_hp, m_tp, rtol=1e-4, atol=1e-4)
    # other directions' components at the task tokens, and every other token, are untouched
    keep = torch.tensor([x not in ("task_a", "task_b") for x in hp.seg])
    torch.testing.assert_close(after[keep], h_hp[2][keep], rtol=1e-5, atol=1e-5)
    # "together": the slot means equal the other prompt's at every swap block (no stacking), via run_arms
    (label, add, clamp), = pt.swap_arms(hp, tp, h_hp, h_tp, {"rand0": dirs["rand0"]}, ["together"])
    assert add == {} and sorted(clamp) == [(1, "task"), (2, "task")]
    pos = pt.task_positions(hp)
    from matskit import acts as A
    with A.clamp(model, {b: clamp[b, w][0] for b, w in clamp}, {b: clamp[b, w][1] for b, w in clamp}, at="post",
                 positions=pos):
        both = A.residuals(model, torch.tensor([hp.ids]), layers=[1, 2], at="post")
    for b in (1, 2):
        u = dirs["rand0"][b]
        for s in ("task_a", "task_b"):
            m_hp = (both[b][0] @ u)[torch.tensor([x == s for x in hp.seg])].mean()
            m_tp = (h_tp[b] @ u)[torch.tensor([x == s for x in tp.seg])].mean()
            torch.testing.assert_close(m_hp, m_tp, rtol=1e-4, atol=1e-4)
    out_ids = outcome_ids(tok, "coinflip")
    mixed, _ = pt.run_arms(model, hp, arms + [(label, add, clamp)], out_ids, batch_size=4)
    alone, _ = pt.run_arms(model, hp, [(label, add, clamp)], out_ids, batch_size=2)
    np.testing.assert_allclose(mixed[-1], alone[0], rtol=1e-4, atol=1e-4)


def test_run_arms_batches_match_single_rows(tok, tiny, monkeypatch):
    """A steered row in a shared batch gives the same L as that arm alone, and the unsteered row the baseline."""
    from matskit import models
    import probe_tests as pt
    model, _ = models.load(tiny, device_map="cpu", dtype="float32", check=False)
    monkeypatch.setattr(pt, "STEER_BLOCKS", [1, 2, 3])
    dirs = pt.load_directions("nonexistent.npz", models.n_layers(model), 64)
    _, _, hp, _ = make_test_pairs(tok, "choice", which="standard")[0]
    out_ids = outcome_ids(tok, "choice")
    norms = {b: 10.0 for b in (1, 2, 3)}
    arms = pt.steer_arms(hp, {"rand0": dirs["rand0"]}, norms, [0.5, 1.0], [1, "together"])
    Ls, bases = pt.run_arms(model, hp, arms, out_ids, batch_size=3)
    alone = [pt.run_arms(model, hp, [arm], out_ids, batch_size=2)[0][0] for arm in arms]
    np.testing.assert_allclose(Ls, alone, rtol=1e-4, atol=1e-4)
    logits, _ = pt.baseline(model, hp, [1])
    np.testing.assert_allclose(bases, pt.logit_diff(logits, out_ids).item(), rtol=1e-4, atol=1e-4)
    assert len(set(np.round(Ls, 5))) == len(Ls)  # every arm moved L differently


def test_label_arms(tok, tiny, monkeypatch):
    """swap_label sets the label B token's projection to the other order's at every swap block, swap_both also
    swaps the task tokens, and steer_label adds only at the label token."""
    from matskit import acts as A, models
    import probe_tests as pt
    model, _ = models.load(tiny, device_map="cpu", dtype="float32", check=False)
    monkeypatch.setattr(pt, "SWAP_BLOCKS", [1, 2])
    monkeypatch.setattr(pt, "STEER_BLOCKS", [1, 2])
    tdirs = pt.load_directions("nonexistent.npz", models.n_layers(model), 64)
    ldirs = pt.load_directions("nonexistent.npz", models.n_layers(model), 64, seed=1, prefix="label_choice_")
    assert not torch.allclose(tdirs["rand0"], ldirs["rand0"])
    _, _, hp, tp = make_test_pairs(tok, "choice", which="standard")[0]
    _, h_hp = pt.baseline(model, hp, [1, 2])
    _, h_tp = pt.baseline(model, tp, [1, 2])
    arms = pt.label_swap_arms(hp, tp, h_hp, h_tp, {"rand0": ldirs["rand0"]}, {"rand0": tdirs["rand0"]}, [2, "together"])
    assert [a[0]["arm"] for a in arms] == ["swap_label", "swap_both", "swap_label", "swap_both"]
    (_, _, only), (_, _, both) = arms[2], arms[3]
    assert sorted(only) == [(1, "label"), (2, "label")] and sorted(both) == [(1, "label"), (1, "task"), (2, "label"),
                                                                            (2, "task")]
    lab = hp.read["label_b"]
    with contextlib.ExitStack() as st:
        for (b, w), (u, v) in both.items():
            st.enter_context(A.clamp(model, {b: u}, {b: v}, at="post", positions=pt.clamp_positions(hp, w)))
        after = A.residuals(model, torch.tensor([hp.ids]), layers=[1, 2], at="post")
    for b in (1, 2):
        u = ldirs["rand0"][b]
        torch.testing.assert_close(after[b][0, lab] @ u, h_tp[b][tp.read["label_b"]] @ u, rtol=1e-4, atol=1e-4)
    out_ids = outcome_ids(tok, "choice")
    mixed, _ = pt.run_arms(model, hp, arms, out_ids, batch_size=3)
    alone = [pt.run_arms(model, hp, [arm], out_ids, batch_size=2)[0][0] for arm in arms]
    np.testing.assert_allclose(mixed, alone, rtol=1e-4, atol=1e-4)
    # steer_label: only the label token moves at the steered block, by c * norm * u
    (_, add, _), = pt.label_steer_arms(hp, {"rand0": ldirs["rand0"]}, {1: 10.0, 2: 10.0}, [0.5], [1])
    with A.steer_tokens(model, add, at="post"):
        moved = A.residuals(model, torch.tensor([hp.ids]), layers=[1], at="post")[1][0]
    torch.testing.assert_close(moved[lab], h_hp[1][lab] + 5.0 * ldirs["rand0"][1], rtol=1e-4, atol=1e-4)
    keep = torch.arange(len(hp.ids)) != lab
    torch.testing.assert_close(moved[keep], h_hp[1][keep], rtol=1e-5, atol=1e-5)


def test_choice_extraction_reads(tok):
    items = extraction_items(tok, limit=3, kinds=("choice",))
    assert len(items) == 3 and all(i.kind == "choice" and i.meta["slot"] == "a" for i in items)
    for it in items:
        assert tok.decode([it.ids[it.read["label_b"]]]) == " B"
        assert it.read["task"] < it.read["label_b"]
