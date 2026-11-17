"""up_inject.py on rows from a tiny random model (report_patch.run_chunk with the inject arm set): the gains, controls,
transplant and predictions report runs through, and the coin at " up" follows a push at " up" (positive coin gain)."""

import numpy as np
import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

import prompts
import up_inject as ui
from analyze import read_jsonl
from coin_report import load
from matskit import io
from report_patch import arm_specs, build_items, inject_positions, run_chunk
from up_followups import boot_indices

LLAMA = "unsloth/Llama-3.3-70B-Instruct"


def test_report_runs(tmp_path, capsys, monkeypatch):
    tok = AutoTokenizer.from_pretrained(LLAMA)
    tasks = [t for t in read_jsonl("data/tasks_n300_max500_seed0.jsonl") if len(t["prompt"]) < 150][:10]
    pairs = [(i, j) for i in range(10) for j in range(i + 1, 10)][:16]
    items = build_items(tok, tasks, pairs, "coin_report_bold")
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=2,
                      num_key_value_heads=2, max_position_embeddings=1024, initializer_range=0.5)
    model = LlamaForCausalLM(cfg).eval()
    ks = inject_positions(tok, items[0], [",", " the", " ...", " it", " came", " up", "."])
    inject = dict(up=ks[" up"], ctx=sorted(k for t, k in ks.items() if t != " up"), onset=1, dose=1.0, seeds=[0, 1, 2])
    specs = arm_specs(["inject"], len(items[0]["line"]), len(items[0]["suffix"]), inject)
    A = {t["id"]: float(k) for k, t in enumerate(tasks)}
    tp = {"A": A, "slope": torch.full((3, 7), 0.2)}
    coin = prompts.outcome_token_ids("coinflip", tok)
    rep = prompts.outcome_token_ids("coin_report_bold", tok)
    stats, info = {}, {}
    for fam in ("gemma4-31b", "gemma4-31b-base"):
        path = tmp_path / fam / "rows.jsonl"
        path.parent.mkdir()
        for row in run_chunk(model, items, tok.eos_token_id, coin, rep, 3, 4, specs=specs, transplant=tp):
            io.append_jsonl(path, row)
        D = load(path)
        rows = io.read_jsonl(path)
        Ad = np.array([A[a] - A[b] for a, b in zip(D["a"], D["b"])])
        monkeypatch.setattr(ui, "boot_indices", lambda n, seed=0: boot_indices(n)[:21])
        stats[fam] = ui.run_stats(D, rows, Ad, 1.0, ui.boot_indices(len(D["c"])))
        checks = {}
        for r in rows:
            for arm, v in r.get("push_check", {}).items():
                checks.setdefault(arm, []).append(v)
        info[fam] = dict(run=dict(id=fam), D=D, checks={a: float(np.median(v)) for a, v in checks.items()},
                         mass=float(np.median(D["mass"]["none"])), self_max=0.0)
    monkeypatch.setattr(ui, "OUT", tmp_path / "out")
    (tmp_path / "out").mkdir()
    monkeypatch.setattr(ui.ps, "save", lambda *a, **k: None)
    ui.report(stats, info)
    out = capsys.readouterr().out
    assert "Pre-registered predictions" in out and out.count("  1 finals") == 1 and "transplant" in out
    assert stats["gemma4-31b"]["push_up: coin gain"][0] > 0
