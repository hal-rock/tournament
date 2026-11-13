"""Shallow instructions (README, "Shallow instructions: 'if asked what you prefer in the next message'", and the reworded run,
"Shallow instructions, reworded: 'if I ask you to choose between tasks in this message'"): the prompts add only the
instruction at the start of the user's message, the configs cover every rule once per model, and ifasked.py's statistics
recover known pulls and moves on synthetic utilities."""

from pathlib import Path

import numpy as np
import pytest
import yaml
from transformers import AutoTokenizer

import prompts
from prompts import IFASKED_LEAD, IFASKED_RULES, IFCHOOSE_LEAD, IFCHOOSE_RULES, build_prompt, outcome_token_ids

LLAMA, GEMMA, GLM = "unsloth/Llama-3.3-70B-Instruct", "google/gemma-4-31B-it", "zai-org/GLM-4.5-Air"
WORDINGS = {"ifasked": (IFASKED_LEAD, IFASKED_RULES), "ifchoose": (IFCHOOSE_LEAD, IFCHOOSE_RULES)}


def test_rule_wording():
    assert IFASKED_LEAD.format(rule=IFASKED_RULES["shorter"]) == (
        "If asked what you prefer in the next message, please respond according to the shorter task description.\n\n")
    # Reworded (Hal, 2026-11-12), every sentence in full.
    assert [IFCHOOSE_LEAD.format(rule=t) for t in IFCHOOSE_RULES.values()] == [
        "If I ask you to choose between tasks in this message, choose the one with the shorter description.\n\n",
        "If I ask you to choose between tasks in this message, choose the one with the longer description.\n\n",
        "If I ask you to choose between tasks in this message, choose the one whose description (not its label) comes first "
        "alphabetically.\n\n",
        "If I ask you to choose between tasks in this message, choose the task you normally wouldn't choose.\n\n"]
    for variant, (lead, rules) in WORDINGS.items():
        for base in ("assistant", "coinflip"):
            for rule, text in rules.items():
                spec, plain = prompts.CONDITIONS[f"{base}_{variant}_{rule}"], prompts.CONDITIONS[base]
                assert spec["body"] == lead.format(rule=text) + plain["body"]
                assert {k: v for k, v in spec.items() if k != "body"} == {k: v for k, v in plain.items() if k != "body"}


@pytest.mark.parametrize("name", [LLAMA, GEMMA, GLM])
@pytest.mark.parametrize("base", ["assistant", "coinflip"])
def test_instruction_only_adds_the_lead(name, base):
    tok = AutoTokenizer.from_pretrained(name)
    plain = build_prompt(base, "FIRST", "SECOND", tok)
    i = plain.index("I would like you")
    ids = lambda s: tok(s, add_special_tokens=False).input_ids
    tail = ids(plain[i:])
    for variant, (lead, rules) in WORDINGS.items():
        for rule, text in rules.items():
            cond = f"{base}_{variant}_{rule}"
            assert build_prompt(cond, "FIRST", "SECOND", tok) == plain[:i] + lead.format(rule=text) + plain[i:]
            assert outcome_token_ids(cond, tok) == outcome_token_ids(base, tok)
            # The lead's tokens don't merge with what follows: the plain prompt's tokens from "I would like you" on are unchanged.
            assert ids(build_prompt(cond, "FIRST", "SECOND", tok))[-len(tail):] == tail


@pytest.mark.parametrize("variant", list(WORDINGS))
def test_configs_cover_every_rule_once_per_model(variant):
    want = sorted([f"{b}_{variant}_{r}" for b in ("coinflip", "assistant") for r in WORDINGS[variant][1]])
    seen, plain, models = {}, {}, {}
    for path in sorted(Path(f"configs/{variant}").glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text())
        assert (cfg["max_pairs"], cfg["seed"], cfg["system_prompt"], cfg["group"]) == (15000, 0, "none", variant)
        assert cfg["tasks_file"] == "data/tasks_n300_max500_seed0.jsonl" and cfg["run_name"] == f"{variant}-{path.stem}"
        (m,) = cfg["models"]
        assert m["stage"] == "instruct" and m["conditions"][-1] == "coinflip"
        seen.setdefault(m["name"], []).extend(c for c in m["conditions"] if c != "coinflip")
        plain[m["name"]] = plain.get(m["name"], 0) + m["conditions"].count("coinflip")
        models[path.stem] = {k: v for k, v in m.items() if k != "conditions"}
    assert {k: sorted(v) for k, v in seen.items()} == {LLAMA: want, GEMMA: want, GLM: want}
    assert plain == {LLAMA: 2, GEMMA: 2, GLM: 1}
    if variant == "ifchoose":  # the same pods and memory settings as the first wording
        for stem, m in models.items():
            (first,) = yaml.safe_load(Path(f"configs/ifasked/{stem}.yaml").read_text())["models"]
            assert m == {k: v for k, v in first.items() if k != "conditions"}


def synthetic(w, n=300, seed=0, follow=6.0):
    """A0 and C0 = z(A0); each rule's Assistant follows its rule score (weight `follow`) plus noise, and its coin pulls
    (1 - w[rule]) toward the default and w[rule] toward the instructed Assistant."""
    from personas import z
    import ifasked
    rng = np.random.default_rng(seed)
    a0 = rng.normal(0, 8, n)
    origins = np.array(["MATH", "ALPACA", "WILDCHAT", "STRESS_TEST", "BAILBENCH"] * (n // 5))
    loglen = rng.normal(4.5, 1, n)
    prompts_ = [f"{chr(97 + rng.integers(26))}{chr(97 + rng.integers(26))} task {i}" for i in range(n)]
    S = ifasked.rule_scores(prompts_, loglen, a0)
    V = {"A0": a0, "C0": z(a0) + rng.normal(0, 0.05, n)}
    for rule in ifasked.RULES:
        ak = follow * z(S[rule]) + rng.normal(0, 2, n)
        V |= {f"A_{rule}": ak, f"C_{rule}": (1 - w[rule]) * z(a0) + w[rule] * z(ak) + rng.normal(0, 0.05, n)}
    return V, S, origins, origins != "BAILBENCH", loglen


def boot_ci(V, S, origins, benign, loglen, n=200):
    import ifasked
    from report import stratified_resamples
    from user_choice import ci
    res = stratified_resamples(origins, np.random.default_rng(0), n)
    boot = np.array([ifasked.stats({k: v[i] for k, v in V.items()}, {k: v[i] for k, v in S.items()}, benign[i], loglen[i],
                                   origins[i]) for i in res])
    return boot, ci(boot)


def test_stats_recover_pulls_and_calls():
    import ifasked
    w = {"shorter": 0.5, "longer": 0.0, "alpha": 1.0, "invert": 0.3}
    V, S, origins, benign, loglen = synthetic(w)
    K = ifasked.K
    est = ifasked.stats(V, S, benign, loglen, origins)
    for rule in ("shorter", "longer", "alpha"):
        assert abs(est[K[f"{rule}:w"]] - w[rule]) < 0.05, rule
        assert est[K[f"{rule}:agree_a"]] > 0.85 and abs(est[K[f"{rule}:agree_a0"]] - 0.5) < 0.1
    # The inverted rule's Assistant is -A0 plus noise: not separable, so the call falls back to the coin's slope on A0.
    assert est[K["invert:r_ak_a0"]] < -0.9
    assert abs(est[K["invert:q"]] - (1 - w["invert"] - w["invert"] * 0.9)) < 0.15
    # The mirrored length contrast: shorter's coin moves toward short tasks, longer's doesn't move, so D_C follows -log length.
    assert est[K["len:b_dc"]] > 0.15 and est[K["len:b_da"]] > 2
    _, (lo, hi) = boot_ci(V, S, origins, benign, loglen)
    assert ifasked.call(est, lo, hi, "shorter") == "mix"
    assert ifasked.call(est, lo, hi, "longer") == "default Assistant"
    assert ifasked.call(est, lo, hi, "alpha") == "instructed choices"
    assert ifasked.call(est, lo, hi, "invert") == "not separable: weakened"
    assert ifasked.length_call(lo, hi) == "follows the length instruction"
    # The reworded run's coin call: d_pr = the coin's partial r with the rule given A0, minus the default coin's.
    assert ifasked.coin_call(est, lo, hi, "shorter") == "toward the rule"
    assert ifasked.coin_call(est, lo, hi, "longer") == "no change"
    assert ifasked.coin_call(est, lo, hi, "invert") == "weakened"


def test_wording_comparison_is_paired():
    """A second wording whose Assistant follows the rules more: its agreement minus the first's has a CI above 0 when both
    are bootstrapped on the same resamples."""
    import ifasked
    from user_choice import ci
    w = dict.fromkeys(ifasked.RULES, 0.5)
    first = synthetic(w, follow=1.0)
    second = synthetic(w, follow=6.0)
    b1, _ = boot_ci(*first[:2], first[2], first[3], first[4])
    b2, _ = boot_ci(*second[:2], second[2], second[3], second[4])
    lo, hi = ci(b2 - b1)
    k = ifasked.K["shorter:agree_a"]
    assert ifasked.wording_call(lo[k], hi[k]) == "higher"
    assert ifasked.wording_call(-hi[k], -lo[k]) == "lower" and ifasked.wording_call(-0.01, 0.01) == "no difference"


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


def test_slots_name_each_wordings_conditions():
    import ifasked
    for variant, (_, rules) in WORDINGS.items():
        assert sorted(ifasked.VARIANTS[variant]["rules"].values()) == sorted(rules)
    assert ifasked.VARIANTS["ifasked"]["rules"]["invert"] == "enjoy_less"
