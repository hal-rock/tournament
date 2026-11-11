"""replication_analysis.py on synthetic utilities where each finding's call is known, the same formulas as the earlier
sections' scripts, the results-file layout end to end, and replication_status.py's grid."""

import csv
import gzip
import itertools
import json

import numpy as np
import pytest
import yaml

import persona_placement
import replication_analysis as ra
import replication_status
import roleplay
import user_choice
from personas import z
from report import stratified_resamples

SOURCES = ("MATH", "ALPACA", "WILDCHAT", "STRESS_TEST", "BAILBENCH")
N_BOOT = 60


class FakeModel:
    """What the call functions read from a Model: utilities, cell diagnostics, pilots."""

    def __init__(self, U, origins, nonharm, bad=None):
        self.key, self.label, self.U, self.origins, self.nonharm, self.pilots = "fake", "Fake", U, origins, nonharm, set()
        self.cells = {k: dict(reliability=0.99, median_mass=0.9) for k in U}
        for k, v in (bad or {}).items():
            self.cells[k].update(v)

    def personas(self):
        return [p for p in ra.PERSONAS if any(f"@{p}" in k for k in self.U)]


def k(x, place="none", p=None):
    """A final cell's key: k("C") the plain coin, k("C", "user", "evil") the instructed evil coin."""
    return ra.condition_key(x, "instruct", place, p)


def kb(x):
    return ra.condition_key(x, "base")


def synthetic(n=300, seed=0):
    """Utilities with known outcomes:
    - plain readouts: the final coin, user's choice and prediction follow the final Assistant; at base nothing does;
    - evil: separable from A_0; the instructed coin follows it, the self-described coin barely, the roleplay coin follows
      its roleplay Assistant, and the prediction under the self-description moves to the persona while A stays;
    - slacker: separable; the instructed coin ignores it, the self-described coin follows it more than the instructed one,
      the roleplay coin follows the default only, and the self-described prediction is the self-described Assistant;
    - contrarian: its Assistant is the default (not separable);
    - strategist: not run."""
    rng = np.random.default_rng(seed)
    e = lambda sd: rng.normal(0, sd, n)
    origins = np.array(SOURCES * (n // 5))
    nonharm = origins != "BAILBENCH"
    a0 = e(8)
    U = {kb("A"): e(0.2), k("A"): a0}
    U[kb("C")], U[k("C")] = e(0.05), 0.3 * z(a0) + e(0.05)
    U[kb("U")], U[k("U")] = U[kb("A")] + e(0.05), 0.5 * z(a0) + e(0.1)
    U[kb("P")], U[k("P")] = e(0.1), 2 * z(a0) + e(0.3)
    for p, scale in (("evil", 7), ("slacker", 6)):
        ap = e(scale)
        U[k("A", "sys", p)] = ap
        U[k("A", "user", p)] = ap + e(1)
        U[k("A", "self", p)] = a0 + e(1)
        U[k("A", "roleplay", p)] = ap + e(1)
        U[k("C", "sys", p)] = 0.3 * z(a0) + 0.5 * z(ap) + e(0.05)
    ev, sl = U[k("A", "sys", "evil")], U[k("A", "sys", "slacker")]
    U[k("C", "user", "evil")] = 0.2 * z(a0) + 0.4 * z(ev) + e(0.05)
    U[k("C", "self", "evil")] = 0.6 * z(a0) + 0.05 * z(ev) + e(0.05)
    U[k("C", "roleplay", "evil")] = 0.2 * z(a0) + 0.5 * z(U[k("A", "roleplay", "evil")]) + e(0.05)
    U[k("P", "self", "evil")] = 1.5 * z(ev) + 0.5 * z(a0) + e(0.2)
    U[k("U", "self", "evil")] = U[k("A", "self", "evil")] + e(1)
    U[k("C", "user", "slacker")] = 0.5 * z(a0) - 0.1 * z(sl) + e(0.3)
    U[k("C", "self", "slacker")] = 0.3 * z(a0) + 0.3 * z(sl) + e(0.05)
    U[k("C", "roleplay", "slacker")] = 0.5 * z(a0) - 0.2 * z(U[k("A", "roleplay", "slacker")]) + e(0.05)
    U[k("P", "self", "slacker")] = U[k("A", "self", "slacker")] - 2 * z(sl) + e(0.5)
    U[k("U", "self", "slacker")] = U[k("A", "self", "slacker")] + e(1)
    ac = a0 + e(0.5)
    U[k("A", "sys", "contrarian")], U[k("A", "user", "contrarian")] = ac, ac + e(0.5)
    U[k("A", "self", "contrarian")], U[k("P", "self", "contrarian")] = a0 + e(0.5), z(ac) + e(0.1)
    for place in ("sys", "user", "self"):
        U[k("C", place, "contrarian")] = 0.3 * z(a0) + e(0.05)
    return U, origins, nonharm


@pytest.fixture(scope="module")
def fitted():
    U, origins, nonharm = synthetic()
    S = ra.bootstrap(U, origins, nonharm, ra.resamples_for(origins, N_BOOT))
    return U, origins, nonharm, S


def calls(rows):
    return {(r["finding"], r["persona"] or None): r["call"] for r in rows}


def test_calls_follow_the_construction(fitted):
    U, origins, nonharm, S = fitted
    got = calls(ra.findings(S, FakeModel(U, origins, nonharm)))
    assert got[1, None] == got[2, None] == got[3, None] == "replicates"
    assert got[4, "evil"] == got[5, "evil"] == got[6, "evil"] == got[7, "evil"] == "replicates"
    assert got[4, "slacker"] == "doesn't"  # the instructed coin ignores the persona
    assert got[5, "slacker"] == "doesn't"  # the self-described coin follows it more than the instructed one
    assert got[6, "slacker"] == "doesn't"  # the roleplay coin follows the default Assistant only
    assert got[7, "slacker"] == "doesn't"  # the prediction is the Assistant's own choice
    assert got[4, "contrarian"] == got[5, "contrarian"] == got[7, "contrarian"] == "not separable"
    assert got[6, "contrarian"] == "not run"
    assert all(got[f, "strategist"] == "not run" for f in (4, 5, 6, 7))


def test_known_values(fitted):
    U, origins, nonharm, S = fitted
    assert S["r:C_f~A_f"][0] > 0.98 and abs(S["r:C_b~A_b"][0]) < 0.2 and S["shift:C~A"][0] > 0.95
    assert abs(S["f4:evil:rho_user"][0] - 0.8) < 0.05  # 0.4 / 0.5
    assert abs(S["f5:evil:rho_self"][0] - 0.1) < 0.05
    assert abs(S["f6:evil:w"][0] - 0.5 / 0.7) < 0.05  # 0.5 / (0.2 + 0.5)
    assert abs(S["fit:C_sys@evil:w"][0] - 0.5 / 0.8) < 0.02
    assert abs(S["fit:A_sys@evil:w"][0] - 1) < 1e-9 and abs(S["fit:A_none@evil:w"][0]) < 1e-9
    assert S["f7:evil:dw_P"][0] > 0.5
    # Breakdowns exist for every r, and per-source r is within that source only.
    m = origins == "MATH"
    assert S["r:C_f~A_f|src:MATH"][0] == pytest.approx(np.corrcoef(U[k("A")][m], U[k("C")][m])[0, 1])
    bs = np.isin(origins, ra.BENIGN_SOURCES)
    assert S["r:U_f~A_f|benign_sources"][0] == pytest.approx(np.corrcoef(U[k("A")][bs], U[k("U")][bs])[0, 1])


def test_same_formulas_as_the_sections(fitted):
    """The replication's numbers are the sections' scripts' numbers on the same arrays."""
    U, origins, nonharm, S = fitted
    res = stratified_resamples(origins, np.random.default_rng(0), N_BOOT)
    arrays = (U[kb("A")], U[k("A")], U[kb("U")], U[k("U")], U[kb("C")], U[k("C")])
    cell = user_choice.model_cell(arrays, nonharm, origins, res)
    d = user_choice.contrast(cell, "target")
    assert np.allclose(S["f2:d"], d) and S["r:U_f~A_f|within"][0] == pytest.approx(
        cell["est"][user_choice.K["r_final_within"]])
    assert user_choice.classify(cell) == "more Assistant-like"
    V = {"A0": U[k("A")], "C0": U[k("C")], "Ae": U[k("A", "sys", "evil")], "Csys": U[k("C", "sys", "evil")],
         "Cuser": U[k("C", "user", "evil")], "Cself": U[k("C", "self", "evil")], "Auser": U[k("A", "user", "evil")],
         "Aself": U[k("A", "self", "evil")], "Bsys": U[kb("C")], "Buser": U[kb("C")],
         "Bself": U[kb("C")]}
    loglen = np.zeros(len(origins))
    est = persona_placement.stats(V, nonharm, loglen, origins)
    K = persona_placement.K
    assert S["f4:evil:rho_user"][0] == pytest.approx(est[K["rho_user"]], abs=1e-12)
    assert S["f5:evil:drho"][0] == pytest.approx(est[K["rho_self_minus_user"]], abs=1e-12)
    boot = np.array([persona_placement.stats({a: v[i] for a, v in V.items()}, nonharm[i], loglen[i], origins[i])
                     for i in res])
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    assert S["f4:evil:rho_user"][1:] == pytest.approx((lo[K["rho_user"]], hi[K["rho_user"]]), abs=1e-12)
    R = {"A0": U[k("A")], "C0": U[k("C")]}
    for p in roleplay.PERSONAS:
        R |= {f"As_{p}": U[k("A", "sys", p)], f"Cs_{p}": U[k("C", "sys", p)], f"Ar_{p}": U[k("A", "roleplay", p)],
              f"Cr_{p}": U[k("C", "roleplay", p)]}
    est = roleplay.stats(R, nonharm, loglen, origins)
    assert S["f6:evil:w"][0] == pytest.approx(est[roleplay.K["evil:w_r"]], abs=1e-12)
    assert S["f6:slacker:dw"][0] == pytest.approx(est[roleplay.K["slacker:dw"]], abs=1e-12)


def test_gates_unstable_and_missing_cells(fitted):
    U, origins, nonharm, S = fitted
    low = FakeModel(U, origins, nonharm, bad={k("C"): dict(median_mass=0.1), k("P", "self", "evil"): dict(reliability=0.8)})
    rows = {(r["finding"], r["persona"] or None): r for r in ra.findings(S, low)}
    assert rows[1, None]["call"] == "not interpretable" and "C_f median mass 0.10" in rows[1, None]["detail"]
    assert rows[1, None]["key_number"]  # reported, with no call
    assert rows[7, "evil"]["call"] == "not interpretable" and rows[2, None]["call"] == "replicates"
    # The prediction under the self-description carries no preferences: b_def + b_per includes 0, so w is unstable.
    V = dict(U)
    V[k("P", "self", "evil")] = np.random.default_rng(5).normal(0, 0.01, len(origins))
    S2 = ra.bootstrap(V, origins, nonharm, ra.resamples_for(origins, N_BOOT))
    assert calls(ra.findings(S2, FakeModel(V, origins, nonharm)))[7, "evil"] == "unstable"
    # A model with only its plain final readouts: calls 1-3 need the base, so nothing is called and nothing fails.
    plain = {key: u for key, u in U.items() if key.startswith("instruct/") and "@" not in key}
    S3 = ra.bootstrap(plain, origins, nonharm, ra.resamples_for(origins, 20))
    got = calls(ra.findings(S3, FakeModel(plain, origins, nonharm)))
    assert set(got.values()) == {"not run"}


def test_cell_names_round_trip():
    for x in "ACUP":
        for s in ("base", "instruct"):
            assert ra.short_of(ra.condition_key(x, s)) == ra.cell_name(x, s)
        for place in ("sys", "user", "self", "roleplay"):
            assert ra.short_of(ra.condition_key(x, "instruct", place, "evil")) == f"{x}_{place}@evil"
    assert ra.condition_key("C", "instruct", "user", "evil") == "instruct/coinflip@evil+user"


def write_results(path, stage, condition, ids, u, mass=0.9, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "stage", "condition", "first", "second", "logp_first", "logp_second",
                                           "log_odds", "outcome_mass", "prompt_tokens", "top_tokens"])
        w.writeheader()
        for (i, a), (j, b) in itertools.permutations(enumerate(ids), 2):
            lo = 0.3 + u[i] - u[j] + rng.normal(0, noise)
            w.writerow(dict(model="m", stage=stage, condition=condition, first=a, second=b, logp_first=0, logp_second=0,
                            log_odds=lo, outcome_mass=mass, prompt_tokens=10, top_tokens="[]"))


@pytest.fixture
def layout(tmp_path, monkeypatch):
    """A 30-task model on disk: plain readouts at both stages (the base coin with low mass), and a pilot-only cell."""
    rng = np.random.default_rng(3)
    tasks = [dict(id=f"t{i:02d}", origin=SOURCES[i % 5], topic="harmful_request" if i % 5 == 4 else "other",
                  prompt="x" * (i + 1)) for i in range(30)]
    tasks_file = tmp_path / "tasks.jsonl"
    tasks_file.write_text("".join(json.dumps(t) + "\n" for t in tasks))
    root = tmp_path / "replication"
    monkeypatch.setattr(ra, "OUT", root)
    monkeypatch.setattr(ra, "FIGURES", tmp_path / "figures")
    monkeypatch.setattr(ra, "CONFIGS", tmp_path / "configs")
    (root / "m").mkdir(parents=True)
    (root / "m" / "run_config.yaml").write_text(yaml.safe_dump(dict(
        run_name="m", tasks_file=str(tasks_file), models=[dict(name="org/M-Base", stage="base", conditions="base"),
                                                          dict(name="org/M-Instruct", stage="instruct", conditions="final")])))
    ids = [t["id"] for t in tasks]
    truth = {}
    for n, (stage, cond) in enumerate(itertools.product(("base", "instruct"), ra.READOUT.values())):
        truth[f"{stage}/{cond}"] = u = rng.normal(0, 2, len(ids))
        write_results(root / "m" / "results" / f"{stage}__{cond}.csv.gz", stage, cond, ids, u,
                      mass=0.1 if (stage, cond) == ("base", "coinflip") else 0.9, seed=n)
    write_results(root / "m" / "pilot" / "instruct__coinflip@evil.csv.gz", "instruct", "coinflip@evil", ids[:4],
                  np.zeros(4))
    return root, ids, truth


def test_layout_end_to_end(layout):
    root, ids, truth = layout
    model = ra.Model("m")
    assert model.label == "M-Instruct" and model.ids == ids and model.personas() == []
    for key, u in truth.items():
        assert np.corrcoef(model.U[key], u)[0, 1] > 0.99 and model.cells[key]["reliability"] > 0.9
    assert model.cells["base/coinflip"]["median_mass"] == pytest.approx(0.1)
    assert model.cells["instruct/assistant"]["slot_bias"] == pytest.approx(0.3, abs=0.05)
    assert "instruct/coinflip@evil" in model.pilots
    _, S, rows = ra.analyze_model("m", n_boot=20)
    got = {(r["finding"], r["persona"] or None): r for r in rows}
    assert got[1, None]["call"] == "not interpretable"  # the base coin's mass
    assert got[5, "evil"]["call"] == "not run" and "C_sys@evil (pilot only)" in got[5, "evil"]["detail"]
    out = root / "m" / "analysis"
    assert {p.name for p in out.iterdir()} >= {"findings.csv", "stats.csv", "cells.csv", "utilities.csv", "report.txt"}
    figs = {p.name for p in (ra.FIGURES / "m").iterdir()}
    assert {"summary.png", "coin.png", "user_predict.png", "sources_plain.png"} <= figs
    assert not any(f.startswith(("persona_", "sources_evil")) for f in figs)  # no persona data, no persona figures
    assert ra.read_findings("m")[1, None]["call"] == "not interpretable"
    assert ra.read_stats("m")["r:C_f~A_f"] == pytest.approx(S["r:C_f~A_f"])
    # A second load reads the cache and gives the same utilities.
    again = ra.Model("m")
    assert all(np.array_equal(again.U[key], model.U[key]) for key in model.U)


def test_status_grid(layout, monkeypatch, tmp_path):
    root, _, _ = layout
    configs = tmp_path / "configs"
    configs.mkdir()
    cfg = yaml.safe_load((root / "m" / "run_config.yaml").read_text())
    (configs / "m.yaml").write_text(yaml.safe_dump(cfg))
    (configs / "validate-x.yaml").write_text(yaml.safe_dump(dict(run_name="validate-x")))
    assert [key for key, _ in replication_status.configured()] == ["m"]
    model = ra.Model("m")
    keys = replication_status.planned(cfg)
    assert len(keys) == 4 + 44 and keys[0] == "base/assistant"
    text = replication_status.render({"m": (model, keys)}, now="now")
    assert "| M-Instruct | `m` | 4/4 | 4/44 | 1 |" in text
    assert "| `instruct/coinflip@evil` | *pilot* |" in text and "| `instruct/coinflip@evil+user` | – |" in text
    assert "low mass" in text.split("`base/coinflip`")[1].split("\n")[0]
    assert "No replication models" in replication_status.render({}, now="now")
