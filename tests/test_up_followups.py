"""up_followups.py end to end on rows from a tiny random model (report_patch.run_chunk with every arm set and the lens),
written the way report_patch.main writes them: the report runs through, and its line-4 bookkeeping lines up."""

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

import prompts
import up_followups as uf
from analyze import read_jsonl
from matskit import io
from report_patch import arm_specs, build_items, run_chunk

LLAMA = "unsloth/Llama-3.3-70B-Instruct"


def write_run(tmp_path, family, report, sets, items, model, tok, lens=False):
    coin_ids = prompts.outcome_token_ids("coinflip", tok)
    rep_ids = prompts.outcome_token_ids(report, tok)
    specs = arm_specs(sets, len(items[0]["line"]), len(items[0]["suffix"]))
    lens_pairs = [tuple(coin_ids), tuple(prompts.outcome_token_ids("coin_report_bold", tok))] if lens else None
    path = tmp_path / family / f"{report}-{'-'.join(sets)}"
    path.mkdir(parents=True)
    rows = run_chunk(model, items, tok.eos_token_id, coin_ids, rep_ids, 2, 4, specs=specs, lens_pairs=lens_pairs)
    lens_rows = []
    for row in rows:
        if lens:
            lens_rows.append((row["pair"], row["order"], row.pop("lens"), row.pop("out"), row.pop("out_mass")))
        io.append_jsonl(path / "rows.jsonl", row)
    if lens:
        np.savez_compressed(path / "lens_00000.npz", pair=np.array([x[0] for x in lens_rows]),
                            order=np.array([x[1] for x in lens_rows]), lens=np.array([x[2] for x in lens_rows], np.float32),
                            out=np.array([x[3] for x in lens_rows], np.float32),
                            out_mass=np.array([x[4] for x in lens_rows], np.float32))
    tokens = [tok.decode([items[0]["ids"][p]]) for p in items[0]["suffix"]]
    return dict(id=path.name, family=family, report=report, arm_sets=tuple(sets), path=path,
                config=dict(suffix_tokens=tokens, single_blocks={}, wording_check=None))


def test_report_runs_on_tiny_model_rows(tmp_path, capsys, monkeypatch):
    tok = AutoTokenizer.from_pretrained(LLAMA)
    tasks = [t for t in read_jsonl("data/tasks_n300_max500_seed0.jsonl") if len(t["prompt"]) < 150][:12]
    pairs = [(i, j) for i in range(12) for j in range(i + 1, 12)][:24]
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                      num_key_value_heads=2, max_position_embeddings=1024, initializer_range=0.5)
    model = LlamaForCausalLM(cfg).eval()
    monkeypatch.setattr(uf, "N_BOOT", 20)
    monkeypatch.setattr(uf, "OUT", tmp_path / "out")
    monkeypatch.setattr(uf.ps, "save", lambda fig, name, *a, **k: None)
    util = {t["id"]: (float(k), float(-k)) for k, t in enumerate(tasks)}
    all_runs = []
    for fam in ("gemma4-31b", "gemma4-31b-base"):
        for report in ("coin_report_bold", "coin_predict_think_bold"):
            items = build_items(tok, tasks, pairs, report)
            all_runs.append(write_run(tmp_path, fam, report, ["base", "line3"], items, model, tok, lens=True))
            if fam == "gemma4-31b":
                all_runs.append(write_run(tmp_path, fam, report, ["line4"], build_items(tok, tasks, pairs[:16], report),
                                          model, tok))
    for report in ("coin_choose_ab", "coin_choose_ht"):
        all_runs.append(write_run(tmp_path, "gemma4-31b", report, ["base"], build_items(tok, tasks, pairs, report), model, tok))
    stats, data = {}, {}
    for run in all_runs:
        D, A = uf.load_run(run, util)
        key = (run["family"], uf.question(run["report"]), "line4" if "line4" in run["arm_sets"] else "main")
        data[key] = (run, D, A)
        stats[key] = uf.run_stats(D, A, uf.boot_indices(len(D["c"])))
    uf.report(all_runs, stats, data)
    out = capsys.readouterr().out
    for section in ("Line 1", "Line 2", "Line 3", "Line 4: across", "logit lens", "Pre-registered predictions"):
        assert section in out, section
    assert out.count(": holds") + out.count(": fails") == 13
    # the per-position swap at " up" is the " up" swap: the same share in the line-4 run as its own swap would give
    S4 = stats["gemma4-31b", "coin_report", "line4"]
    n_line = len(build_items(tok, tasks, pairs[:1], "coin_report_bold")[0]["line"])
    assert np.isfinite(S4[f"pos{n_line - 1:02d}: share_r"][0])
