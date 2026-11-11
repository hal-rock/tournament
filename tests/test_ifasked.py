"""Shallow instructions (README, "Shallow instructions: 'if asked what you prefer in the next message'"): the prompts add
only the instruction at the start of the user's message, the configs cover every rule once per model, and ifasked.py's
statistics recover known pulls on synthetic utilities."""

from pathlib import Path

import numpy as np
import pytest
import yaml
from transformers import AutoTokenizer

import prompts
from prompts import IFASKED_LEAD, IFASKED_RULES, build_prompt, outcome_token_ids

LLAMA, GEMMA, GLM = "unsloth/Llama-3.3-70B-Instruct", "google/gemma-4-31B-it", "zai-org/GLM-4.5-Air"
RULES = tuple(IFASKED_RULES)


def test_rule_wording():
    assert IFASKED_LEAD.format(rule=IFASKED_RULES["shorter"]) == (
        "If asked what you prefer in the next message, please respond according to the shorter task description.\n\n")
    for base in ("assistant", "coinflip"):
        for rule in RULES:
            spec, plain = prompts.CONDITIONS[f"{base}_ifasked_{rule}"], prompts.CONDITIONS[base]
            assert spec["body"] == IFASKED_LEAD.format(rule=IFASKED_RULES[rule]) + plain["body"]
            assert {k: v for k, v in spec.items() if k != "body"} == {k: v for k, v in plain.items() if k != "body"}


@pytest.mark.parametrize("name", [LLAMA, GEMMA, GLM])
@pytest.mark.parametrize("base", ["assistant", "coinflip"])
def test_instruction_only_adds_the_lead(name, base):
    tok = AutoTokenizer.from_pretrained(name)
    plain = build_prompt(base, "FIRST", "SECOND", tok)
    i = plain.index("I would like you")
    for rule in RULES:
        cond = f"{base}_ifasked_{rule}"
        lead = IFASKED_LEAD.format(rule=IFASKED_RULES[rule])
        assert build_prompt(cond, "FIRST", "SECOND", tok) == plain[:i] + lead + plain[i:]
        assert outcome_token_ids(cond, tok) == outcome_token_ids(base, tok)
    # The lead's tokens don't merge with what follows: the plain prompt's tokens from "I would like you" on are unchanged.
    ids = lambda s: tok(s, add_special_tokens=False).input_ids
    tail = ids(plain[i:])
    assert ids(build_prompt(f"{base}_ifasked_alpha", "FIRST", "SECOND", tok))[-len(tail):] == tail


def test_configs_cover_every_rule_once_per_model():
    want = sorted([f"{b}_ifasked_{r}" for b in ("coinflip", "assistant") for r in RULES])
    seen, plain = {}, {}
    for path in sorted(Path("configs/ifasked").glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text())
        assert (cfg["max_pairs"], cfg["seed"], cfg["system_prompt"], cfg["group"]) == (15000, 0, "none", "ifasked")
        assert cfg["tasks_file"] == "data/tasks_n300_max500_seed0.jsonl" and cfg["run_name"] == f"ifasked-{path.stem}"
        (m,) = cfg["models"]
        assert m["stage"] == "instruct" and m["conditions"][-1] == "coinflip"
        seen.setdefault(m["name"], []).extend(c for c in m["conditions"] if c != "coinflip")
        plain[m["name"]] = plain.get(m["name"], 0) + m["conditions"].count("coinflip")
    assert {k: sorted(v) for k, v in seen.items()} == {LLAMA: want, GEMMA: want, GLM: want}
    assert plain == {LLAMA: 2, GEMMA: 2, GLM: 1}


def synthetic(w, n=300, seed=0):
    """A0 and C0 = z(A0); each rule's Assistant follows its rule score plus noise, and its coin pulls (1 - w[rule]) toward the
    default and w[rule] toward the instructed Assistant."""
    from personas import z
    import ifasked
    rng = np.random.default_rng(seed)
    a0 = rng.normal(0, 8, n)
    origins = np.array(["MATH", "ALPACA", "WILDCHAT", "STRESS_TEST", "BAILBENCH"] * (n // 5))
    loglen = rng.normal(4.5, 1, n)
    prompts_ = [f"{chr(97 + rng.integers(26))}{chr(97 + rng.integers(26))} task {i}" for i in range(n)]
    S = ifasked.rule_scores(prompts_, loglen, a0)
    V = {"A0": a0, "C0": z(a0) + rng.normal(0, 0.05, n)}
    for rule in RULES:
        ak = 6 * z(S[rule]) + rng.normal(0, 2, n)
        V |= {f"A_{rule}": ak, f"C_{rule}": (1 - w[rule]) * z(a0) + w[rule] * z(ak) + rng.normal(0, 0.05, n)}
    return V, S, origins, origins != "BAILBENCH", loglen


def test_stats_recover_pulls_and_calls():
    import ifasked
    from report import stratified_resamples
    from user_choice import ci
    w = {"shorter": 0.5, "longer": 0.0, "alpha": 1.0, "enjoy_less": 0.3}
    V, S, origins, benign, loglen = synthetic(w)
    K = ifasked.K
    est = ifasked.stats(V, S, benign, loglen, origins)
    for rule in ("shorter", "longer", "alpha"):
        assert abs(est[K[f"{rule}:w"]] - w[rule]) < 0.05, rule
        assert est[K[f"{rule}:agree_a"]] > 0.85 and abs(est[K[f"{rule}:agree_a0"]] - 0.5) < 0.1
    # enjoy_less's Assistant is -A0 plus noise: not separable, so the call falls back to the coin's slope on A0 against b_0.
    assert est[K["enjoy_less:r_ak_a0"]] < -0.9
    assert abs(est[K["enjoy_less:q"]] - (1 - w["enjoy_less"] - w["enjoy_less"] * 0.9)) < 0.15
    # The mirrored length contrast: shorter's coin moves toward short tasks, longer's doesn't move, so D_C follows -log length.
    assert est[K["len:b_dc"]] > 0.15 and est[K["len:b_da"]] > 2
    res = stratified_resamples(origins, np.random.default_rng(0), 200)
    boot = np.array([ifasked.stats({k: v[i] for k, v in V.items()}, {k: v[i] for k, v in S.items()}, benign[i], loglen[i],
                                   origins[i]) for i in res])
    lo, hi = ci(boot)
    assert ifasked.call(est, lo, hi, "shorter") == "mix"
    assert ifasked.call(est, lo, hi, "longer") == "default Assistant"
    assert ifasked.call(est, lo, hi, "alpha") == "instructed choices"
    assert ifasked.call(est, lo, hi, "enjoy_less") == "not separable: weakened"
    assert ifasked.length_call(lo, hi) == "follows the length instruction"


def test_agreement_counts_only_pairs_the_rule_orders():
    import ifasked
    s = np.array([1.0, 2.0, 2.0, 3.0])
    assert ifasked.agreement(np.array([0.0, 1.0, 5.0, 2.0]), s) == pytest.approx(4 / 5)
    assert ifasked.agreement(-s, s) == 0.0 and ifasked.agreement(s, s) == 1.0


def test_alphabetical_key_ignores_case_and_non_letters():
    import ifasked
    S = ifasked.rule_scores(["$f(x)$ is", "Apple", "banana", "  apple pie"], np.zeros(4), np.zeros(4))
    # "apple" < "applepie" < "banana" < "fxis": higher score = earlier.
    assert list(np.argsort(-S["alpha"])) == [1, 3, 2, 0]
