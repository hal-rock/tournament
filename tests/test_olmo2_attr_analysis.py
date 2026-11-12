"""olmo2_attr_analysis.py on synthetic scores: the pipeline runs end to end, the rules call a planted diffuse push diffuse and
a planted one-category push concentrated, the refusal test sees a planted harm-only push, and label contrasts find a planted
label effect and not a shuffled one."""

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pandas")
pytest.importorskip("scipy")
import olmo2_attr_analysis as an  # noqa: E402

CATS = ["chat", "math", "code", "safety", "if", "science", "multilingual", "other"]


def fake_point(tmp, point, push, seed=0, n=120):
    rng = np.random.default_rng(seed)
    d = tmp / point
    d.mkdir(parents=True)
    rows, labels = [], []
    idx = 0
    sets = [("strat", c, n) for c in CATS] + [("strat", "identity", 40)] + [(s, "safety", 80) for s in ("refusal", "refusal_ctrl")] + \
        [(s, "chat", 80) for s in ("selfref", "selfref_ctrl")]
    for st, cat, k in sets:
        for _ in range(k):
            base = push(cat, st, rng)
            view = bool(rng.random() < 0.3)
            adam = {m: float(rng.normal()) for m in an.QUERIES}
            adam["benign"] = base["benign"] + (0.8 if view else 0.0) + float(rng.normal())
            adam["t1"] = adam["benign"] + float(rng.normal())
            adam["t2"] = adam["benign"] + float(rng.normal())
            adam["standard"] = base.get("standard", 0.0) + float(rng.normal())
            adam["harm_gap"] = base.get("standard", 0.0) * 0.8 + float(rng.normal())
            row = dict(idx=idx, set=st, category=cat, source="s", n=100, tokens=150, lp=-10.0, g_sq=1.0, g_adam_sq=1.0, raw=adam, adam=adam,
                       group_adam_benign=[0.0])
            if idx < 300:
                row.update(rms_half_a=adam["benign"] + 0.01, rms_half_b=adam["benign"], lattice_even=adam["benign"], lattice_odd=adam["benign"] + 0.05,
                           lattice_std=[adam["standard"], adam["standard"]])
            rows.append(row)
            labels.append(dict(idx=idx, labelled_by="m", task_type="math" if cat == "math" else "advice", response_stance="complies",
                               voice="assistant", own_view=view, mentions_ai=False, moral_language=bool(rng.random() < 0.2), missing_reason=""))
            idx += 1
    with open(d / "scores.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    norms = {m: 1.0 for m in an.QUERIES}
    info = {"coin": {"lattice_norms": norms, "cosines": {"benign_odd|benign_even": 0.9, **{k: 0.5 for k in ("benign|t1", "benign|t2", "t1|t2", "benign|perm1", "benign|perm2", "perm1|perm2", "benign|ordinary", "benign|harm_gap")}},
                     "load_median_abs_diff": 0.03, "values": {"benign": 0.02}},
            **{m: {"lattice_norm": 1.0, "load_median_abs_diff": 0.03, "f": 0.1} for m in ("heads_bias", "asst_benign", "standard")}}
    json.dump(info, open(d / "queries.json", "w"))
    json.dump({"R": 100.0}, open(d / "pilot.json", "w"))
    return pd.DataFrame(labels)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(an, "OUT", tmp_path / "out")
    monkeypatch.setattr(an, "DATA", tmp_path / "data")
    monkeypatch.setattr(an, "YESTERDAY", tmp_path / "none")
    (tmp_path / "data").mkdir()
    stats = {"category": {c: 100000 for c in CATS} | {"identity": 240}}
    return tmp_path, stats


def test_diffuse_push_and_refusal_specific_to_harm(env):
    tmp, stats = env
    lab = fake_point(tmp / "out", "base", lambda c, s, r: {"benign": 1.0, "standard": 3.0 if s == "refusal" else 0.0})
    lab.set_index("idx").to_csv(tmp / "data" / "labels.csv")
    res, u, w = an.analyse("base", stats, an.load_labels())
    assert res["benign"]["verdict"] == "diffuse"
    assert res["refusal"]["call"] == "specific to harm"
    own = next(r for r in res["labels"] if r["field"] == "own_view")
    assert own["finding"] and own["benign"]["d"] > 0.3
    moral = next(r for r in res["labels"] if r["field"] == "moral_language")
    assert not moral["finding"]
    assert res["checks"]["pass"]["4_specificity"]


def test_concentrated_and_opposing(env):
    tmp, stats = env
    fake_point(tmp / "out", "base", lambda c, s, r: {"benign": 6.0 if c == "if" else (-1.0 if c == "math" else 0.05)})
    res, u, w = an.analyse("base", stats, None)
    assert res["benign"]["opposing_pushes"]
    assert res["benign"]["verdict"] in ("concentrated", "mixed")
    assert not res["benign"]["rules"]["R3_proportionality"]


def test_main_writes_summary_and_figures(env, monkeypatch):
    tmp, stats = env
    for p, seed in (("base", 0), ("sft", 1)):
        lab = fake_point(tmp / "out", p, lambda c, s, r: {"benign": 0.5}, seed=seed)
    lab.set_index("idx").to_csv(tmp / "data" / "labels.csv")
    json.dump(stats, open(tmp / "data" / "sft_stats.json", "w"))
    from matskit import plotstyle
    monkeypatch.setattr(plotstyle, "save", lambda fig, name, **kw: None)
    an.main()
    s = json.load(open(tmp / "data" / "summary.json"))
    assert {"base", "sft", "base_vs_sft"} <= set(s)
    assert (tmp / "data" / "units_base.csv").exists()
