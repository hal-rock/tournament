"""round2_analysis.py on synthetic utilities with known answers (README, "Replication round 2", B-D): a new model where the
tool's coin follows the roleplay persona, both coins follow the length rules (the tool's less), and the stories are absent
while the first-person raw text tracks the chat coin; plus a first-round layout whose gate fails, so A_0 is re-scored."""

import json

import numpy as np
import pytest
import yaml

import replication_analysis as ra
import round2_analysis as r2
from test_replication_analysis import SOURCES, write_results

N = 60


@pytest.fixture
def tasks(tmp_path):
    rng = np.random.default_rng(0)
    lengths = rng.integers(20, 500, N)
    tasks = [dict(id=f"t{i:02d}", origin=SOURCES[i % 5], topic="other", prompt="x" * int(n)) for i, n in enumerate(lengths)]
    path = tmp_path / "tasks.jsonl"
    path.write_text("".join(json.dumps(t) + "\n" for t in tasks))
    return path, tasks


def layout(tmp_path, monkeypatch, tasks_file, utils, key="n"):
    root = tmp_path / "replication"
    monkeypatch.setattr(ra, "OUT", root)
    monkeypatch.setattr(r2, "OUT", root)
    for k, conds in utils.items():
        (root / k).mkdir(parents=True, exist_ok=True)
        (root / k / "run_config.yaml").write_text(yaml.safe_dump(dict(
            run_name=k, tasks_file=str(tasks_file), models=[dict(name=f"org/{k}", stage="instruct", conditions=[])])))
        ids = [f"t{i:02d}" for i in range(N)]
        for n, (cond, u) in enumerate(conds.items()):
            write_results(root / k / "results" / f"instruct__{cond}.csv.gz", "instruct", cond, ids, u, seed=n)
    return root


def synthetic(tasks):
    rng = np.random.default_rng(1)
    e = lambda sd: rng.normal(0, sd, N)
    loglen = np.log([len(t["prompt"]) for t in tasks])
    zl = (loglen - loglen.mean()) / loglen.std()
    a0 = e(2)
    a_rp = 0.3 * a0 + e(2)  # separable from A_0 (r ~ 0.3)
    a_sys = 0.3 * a0 + e(2)
    U = {"assistant": a0, "coinflip": 0.5 * a0 + e(0.2), "coinflip_tool": 0.5 * a0 + e(0.2)}
    for p in ("evil", "slacker", "contrarian", "strategist"):
        U[f"assistant@{p}+roleplay"] = a_rp
        U[f"assistant@{p}"] = a_sys
        U[f"coinflip@{p}+roleplay"] = 0.5 * a0 + e(0.2)  # the user-reported coin stays with the default
        U[f"coinflip@{p}"] = 0.5 * a_sys + e(0.2)
        U[f"coinflip_tool@{p}"] = 0.5 * a_sys + e(0.2)
        U[f"coinflip_tool@{p}+roleplay"] = 0.5 * a_rp + e(0.2)  # the tool's coin follows the roleplay persona
    U["assistant_ifchoose_shorter"] = -3 * zl + 0.3 * a0 + e(0.2)
    U["assistant_ifchoose_longer"] = 3 * zl + 0.3 * a0 + e(0.2)
    U["coinflip_ifchoose_shorter"] = 0.5 * a0 - 1.0 * zl + e(0.2)
    U["coinflip_ifchoose_longer"] = 0.5 * a0 + 1.0 * zl + e(0.2)
    U["coinflip_tool_ifchoose_shorter"] = 0.5 * a0 - 0.3 * zl + e(0.8)  # a weaker, noisier move than the user-reported coin
    U["coinflip_tool_ifchoose_longer"] = 0.5 * a0 + 0.3 * zl + e(0.8)
    U["raw_okay"] = 0.5 * a0 + e(0.2)
    U["raw_no_okay"] = 0.45 * a0 + e(0.2)
    U["raw_story_ai"] = 0.02 * a0 + e(0.3)
    U["raw_story_person"] = e(0.3)
    return U


def test_new_model_calls(tmp_path, monkeypatch, tasks):
    path, tl = tasks
    layout(tmp_path, monkeypatch, path, {"n": synthetic(tl)})
    monkeypatch.setattr(r2, "NEW", ["n"])
    S, rows = r2.analyze("n", n_boot=40)
    got = {(r["experiment"], r["unit"], r["measure"]): r["call"] for r in rows}
    for p in ("evil", "slacker"):
        assert got["B", p, "T_rp fit on z(A_0), z(A_rp)"] == "persona"
        assert got["B", p, "w_rp(T) - w_rp(C)"] == "more"
    for k in ("shorter", "longer"):
        assert got["C", k, "following: agreement(A_k) - agreement(A_0)"] == "follows"
        assert got["C", k, "d_pr (C)"] == "toward the rule" and got["C", k, "d_pr (T)"] in ("toward the rule", "no change")
        assert got["C", k, "d_pr, tool minus user-reported"] == "the tool's coin follows less"
    assert got["C", "both", "mirrored length contrast (C)"] == "follows the length instruction"
    assert got["D", "raw_okay", "relative slope"] == "tracks"
    assert got["D", "raw_story_ai", "relative slope"] == got["D", "raw_story_person", "relative slope"] == "absent"
    assert got["D", "raw_story_ai", "relative slope minus raw_no_okay"] == "below"
    assert S["D:raw_okay:rel_slope"][0] == pytest.approx(1.0, abs=0.15)
    assert (tmp_path / "replication" / "round2" / "n" / "calls.csv").exists()


def test_first_round_model_uses_round1_or_rescored_baselines(tmp_path, monkeypatch, tasks):
    """A first-round model: round-2 conditions from r2-<key>; A_0, C_0 and T from round 1 if the gate passes (here round 2's
    re-scored A_0 is a different ranking, so it fails and the re-scored ones are used)."""
    path, tl = tasks
    U = synthetic(tl)
    round1 = {c: u for c, u in U.items() if not c.startswith(("coinflip_tool", "assistant_ifchoose", "coinflip_ifchoose",
                                                               "raw_"))}
    tool1 = {c: u for c, u in U.items() if c in ("coinflip_tool",) or (c.startswith("coinflip_tool@") and "+" not in c)}
    rescored = {"assistant": U["assistant"][::-1].copy(), "coinflip": U["coinflip"], "coinflip_tool": U["coinflip_tool"]}
    new = {c: u for c, u in U.items() if c.startswith(("coinflip_tool@", "assistant_ifchoose", "coinflip_ifchoose",
                                                         "coinflip_tool_ifchoose", "raw_")) and (
        "+roleplay" in c or not c.startswith("coinflip_tool@"))}
    layout(tmp_path, monkeypatch, path, {"m": round1, "tool-m": tool1, "r2-m": {**rescored, **new}})
    assert r2.gate_passes("m") is False
    Uread, info, *_ , note = r2.readouts("m")
    assert note.startswith("gate fails") and info["A0"]["source"] == "r2-m/assistant"
    assert info["A_rp@evil"]["source"] == "m/assistant@evil+roleplay" and info["T_sys@evil"]["source"] == "tool-m/coinflip_tool@evil"
    assert info["T_rp@evil"]["source"] == "r2-m/coinflip_tool@evil+roleplay"
    # With the round-1 Assistant re-scored identically, the gate passes and round 1's cells are used.
    layout(tmp_path, monkeypatch, path, {"r2-m": {**rescored, "assistant": U["assistant"]}})
    assert r2.gate_passes("m") is True
    assert r2.readouts("m")[1]["A0"]["source"] == "m/assistant"
