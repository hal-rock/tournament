"""up_lens.py on rows and lens files from a tiny random model (report_patch.run_chunk with --lens-lp, saved as main()
saves them): the maps have the right shapes, the output row's probability is the stored output mass, the figure draws."""

import numpy as np
import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

import prompts
import up_lens as ul
from analyze import read_jsonl
from matskit import io
from report_patch import arm_specs, build_items, run_chunk

LLAMA = "unsloth/Llama-3.3-70B-Instruct"


def test_maps_and_figure(tmp_path, monkeypatch):
    tok = AutoTokenizer.from_pretrained(LLAMA)
    tasks = [t for t in read_jsonl("data/tasks_n300_max500_seed0.jsonl") if len(t["prompt"]) < 150][:6]
    pairs = [(i, j) for i in range(6) for j in range(i + 1, 6)][:8]
    items = build_items(tok, tasks, pairs, "coin_report_bold")
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=2,
                      num_key_value_heads=2, max_position_embeddings=1024, initializer_range=0.5)
    model = LlamaForCausalLM(cfg).eval()
    coin = prompts.outcome_token_ids("coinflip", tok)
    rep = prompts.outcome_token_ids("coin_report_bold", tok)
    specs = arm_specs(["none"], len(items[0]["line"]), len(items[0]["suffix"]))
    suffix_tokens = [tok.decode([items[0]["ids"][p]]) for p in items[0]["suffix"]]
    M = {}
    for fam in ("llama-70b", "llama-70b-base"):
        path = tmp_path / fam
        path.mkdir()
        lens_rows = []
        for row in run_chunk(model, items, tok.eos_token_id, coin, rep, 3, 4, specs=specs,
                             lens_pairs=[tuple(coin), tuple(rep)], lens_lp=True):
            lens_rows.append((row["pair"], row["order"], row.pop("lens"), row.pop("out"), row.pop("out_mass"),
                              row.pop("lens_lp"), row.pop("out_lp")))
            io.append_jsonl(path / "rows.jsonl", row)
        np.savez_compressed(path / "lens_00000.npz", pair=np.array([x[0] for x in lens_rows]),
                            order=np.array([x[1] for x in lens_rows]),
                            **{k: np.array([x[i] for x in lens_rows], dtype=np.float32)
                               for i, k in enumerate(("lens", "out", "out_mass", "lens_lp", "out_lp"), start=2)})
        M[fam] = m = ul.maps(dict(path=path, config=dict(suffix_tokens=suffix_tokens)))
        P = len(suffix_tokens)
        assert m["prob"].shape == m["corr"].shape == (4, P) and m["n_pairs"] == len(pairs)
        assert (m["prob"] <= 1e-6).all()
        out_mass = np.array([x[4] for x in lens_rows])[..., 0]  # [n, P]
        np.testing.assert_allclose(m["prob"][-1], np.log10(out_mass.mean(0)), atol=1e-5)
        np.testing.assert_allclose(m["lean_check"], m["lean_lens"], atol=1e-4)  # fp32 on CPU: the two lenses agree
        assert suffix_tokens[m["up"]] == " up"
    saved = []
    monkeypatch.setattr(ul.ps, "save", lambda fig, name: saved.append(name))
    ul.ps.setup()
    ul.plot("llama-70b", M)
    assert saved == ["up_lens_llama-70b"]
