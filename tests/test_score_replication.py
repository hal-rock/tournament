"""score_replication.py's bookkeeping, with a fake engine (no model): the pairs, the rows, resuming, the pilot files."""

import gzip
import csv
import json
import math

import yaml

import score_replication as sr
import tournament
from prompts import build_prompt, outcome_token_ids
from replication import FINAL_CONDITIONS, BASE_CONDITIONS, expand_conditions

TOK = "allenai/OLMo-2-0425-1B-Instruct"  # small tokenizer, already used by test_prompts


def test_select_pairs_matches_tournament():
    for n, k in ((300, 15000), (300, None), (10, 7)):
        assert sr.select_pairs(n, k, 0) == tournament.select_pairs(n, k, 0)


def test_condition_sets():
    assert len(BASE_CONDITIONS) == 4 and len(FINAL_CONDITIONS) == 44 == len(set(FINAL_CONDITIONS))
    assert expand_conditions("final") == FINAL_CONDITIONS and expand_conditions(["coinflip"]) == ["coinflip"]
    for p in ("evil", "slacker", "contrarian", "strategist"):
        for c in (f"coinflip@{p}", f"coinflip@{p}+user", f"predict_user@{p}+self", f"assistant@{p}+roleplay"):
            assert c in FINAL_CONDITIONS


class FakeEngine:
    """log P(first) = -1 - (len(prompt) % 7) / 10, log P(second) = -2; the top token is the first outcome."""
    calls = 0

    def __init__(self, m, top_k):
        pass

    def score(self, prompts, outcome_ids):
        FakeEngine.calls += 1
        return [(-1 - (len(p) % 7) / 10, -2.0, outcome_ids[0], -1.0) for p in prompts]

    def top_tokens(self, prompts, k):
        return [[(42, -0.5), (43, -1.5)][:k] for _ in prompts]


def _config(tmp_path, conditions):
    cfg = dict(wandb_project="tournament", group="replication", run_name="fake", tasks_file="data/tasks_n300_max500_seed0.jsonl",
               seed=0, max_pairs=15000, system_prompt="none", top_k_tokens=2,
               models=[dict(name=TOK, stage="instruct", conditions=conditions)])
    path = tmp_path / "fake.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def _rows(path):
    with gzip.open(path, "rt") as f:
        return list(csv.DictReader(f))


def test_pilot_rows_and_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(sr, "OUT_ROOT", tmp_path / "out")
    monkeypatch.setattr(sr, "VllmEngine", FakeEngine)
    monkeypatch.setattr(sr, "sanity_check", lambda *a: None)
    conds = ["assistant", "coinflip@contrarian+self"]
    cfg = _config(tmp_path, conds)
    sr.main([str(cfg), "--stage", "instruct", "--pilot", "3", "--no-wandb", "--wording", "0"])

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK)
    tasks = sr.read_jsonl("data/tasks_n300_max500_seed0.jsonl")
    pairs = sr.select_pairs(300, 15000, 0)[:3]
    ordered = pairs + [(j, i) for i, j in pairs]
    for c in conds:
        rows = _rows(tmp_path / "out" / "fake" / "pilot" / f"instruct__{c}.csv.gz")
        assert len(rows) == 6
        for (i, j), r in zip(ordered, rows):
            assert (r["first"], r["second"]) == (tasks[i]["id"], tasks[j]["id"]) and r["condition"] == c
            ids = tok(build_prompt(c, tasks[i]["prompt"], tasks[j]["prompt"], tok), add_special_tokens=False).input_ids
            assert int(r["prompt_tokens"]) == len(ids)
            assert math.isclose(float(r["log_odds"]), -1 - (len(ids) % 7) / 10 + 2)
            assert math.isclose(float(r["outcome_mass"]), math.exp(float(r["logp_first"])) + math.exp(-2))
            assert len(json.loads(r["top_tokens"])) == 2  # pilot rows all get the top-k pass
    progress = [json.loads(l) for l in open(tmp_path / "out" / "fake" / "progress.jsonl")]
    assert [p["condition"] for p in progress] == conds and all(p["pilot"] for p in progress)

    # A second launch finds every file and loads nothing.
    FakeEngine.calls = 0
    sr.main([str(cfg), "--stage", "instruct", "--pilot", "3", "--no-wandb", "--wording", "0"])
    assert FakeEngine.calls == 0


def test_full_run_keeps_top_k_only_on_the_sample(tmp_path, monkeypatch):
    monkeypatch.setattr(sr, "OUT_ROOT", tmp_path / "out")
    monkeypatch.setattr(sr, "VllmEngine", FakeEngine)
    monkeypatch.setattr(sr, "sanity_check", lambda *a: None)
    cfg = _config(tmp_path, ["coinflip"])
    sr.main([str(cfg), "--stage", "instruct", "--no-wandb", "--top-sample", "5", "--chunk", "7000"])
    rows = _rows(tmp_path / "out" / "fake" / "results" / "instruct__coinflip.csv.gz")
    assert len(rows) == 30000
    assert [len(json.loads(r["top_tokens"])) for r in rows[:6]] == [2] * 5 + [1]
    from transformers import AutoTokenizer
    heads, tails = outcome_token_ids("coinflip", AutoTokenizer.from_pretrained(TOK))
    assert json.loads(rows[10]["top_tokens"])[0][0] == AutoTokenizer.from_pretrained(TOK).decode([heads])


def test_with_pilot_scores_the_pilot_then_everything_in_one_load(tmp_path, monkeypatch):
    monkeypatch.setattr(sr, "OUT_ROOT", tmp_path / "out")
    monkeypatch.setattr(sr, "VllmEngine", FakeEngine)
    monkeypatch.setattr(sr, "sanity_check", lambda *a: None)
    loads = []
    monkeypatch.setattr(FakeEngine, "__init__", lambda self, m, k: loads.append(m["name"]))
    cfg = _config(tmp_path, ["coinflip", "assistant@strategist+roleplay"])
    sr.main([str(cfg), "--stage", "instruct", "--no-wandb", "--with-pilot", "4", "--wording", "0"])
    out = tmp_path / "out" / "fake"
    assert len(loads) == 1
    for c in ("coinflip", "assistant@strategist+roleplay"):
        assert len(_rows(out / "pilot" / f"instruct__{c}.csv.gz")) == 8
        assert len(_rows(out / "results" / f"instruct__{c}.csv.gz")) == 30000
    progress = [json.loads(l) for l in open(out / "progress.jsonl")]
    assert [p["pilot"] for p in progress] == [True, True, False, False]


def test_report_pilot_pairs_match_report_patch():
    import report_patch
    from replication import report_pilot_pairs
    assert report_pilot_pairs(300, 1000, 0) == report_patch.pilot_pairs(300, 1000, tournament.select_pairs(300, 15000, 0), 0)


def test_final_pilot_adds_the_wording_check(tmp_path, monkeypatch):
    from replication import WORDING_CHECK, report_pilot_pairs
    monkeypatch.setattr(sr, "OUT_ROOT", tmp_path / "out")
    monkeypatch.setattr(sr, "VllmEngine", FakeEngine)
    monkeypatch.setattr(sr, "sanity_check", lambda *a: None)
    cfg = _config(tmp_path, ["coinflip"])
    sr.main([str(cfg), "--stage", "instruct", "--no-wandb", "--with-pilot", "3"])
    out = tmp_path / "out" / "fake"
    tasks = sr.read_jsonl("data/tasks_n300_max500_seed0.jsonl")
    i, j = report_pilot_pairs(300, 1000, 0)[0]
    for c in WORDING_CHECK:
        rows = _rows(out / "pilot" / f"instruct__wording__{c}.csv.gz")
        assert len(rows) == 6 and (rows[0]["first"], rows[0]["second"]) == (tasks[i]["id"], tasks[j]["id"])
    progress = [json.loads(l) for l in open(out / "progress.jsonl")]
    # the pilot, then the wording check, then the full run
    assert [(p["pilot"], p["wording"]) for p in progress] == ([(True, False)] + [(True, True)] * len(WORDING_CHECK)
                                                              + [(False, False)])
    # a base stage gets none; --wording-only runs only the check
    FakeEngine.calls = 0
    sr.main([str(cfg), "--stage", "instruct", "--no-wandb", "--wording", "3", "--wording-only"])
    assert FakeEngine.calls == 0  # all done already
