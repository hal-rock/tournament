"""Lazy personas (README, "Lazy personas"): the prompts add only the lead at the start of the user's message, the configs cover
every persona once per model, and lazy.py's statistics recover known transfers on synthetic utilities, including two nulls:
a persona that only flattens both the Assistant and the coin, and one that changes nothing (A_p and A_0 then differ only by
noise, which a two-pull fit reads as a pull toward A_p)."""

from pathlib import Path

import numpy as np
import pytest
import yaml
from transformers import AutoTokenizer

import prompts
from prompts import LAZY_LEADS, LAZY_PERSONAS, build_prompt, outcome_token_ids

OLMO, LLAMA, GEMMA, GLM = ("allenai/Olmo-3.1-32B-Instruct", "unsloth/Llama-3.3-70B-Instruct", "google/gemma-4-31B-it",
                           "zai-org/GLM-4.5-Air")


def test_wording():
    assert LAZY_LEADS["youare"].format(persona=LAZY_PERSONAS["pirate"]) == "You are a pirate.\n\n"
    assert LAZY_LEADS["answeras"].format(persona=LAZY_PERSONAS["ghost"]) == "In your next message, answer as a ghost.\n\n"
    assert set(LAZY_PERSONAS) == {"pirate", "ghost", "cat", "wizard", "toddler"}
    for base in ("assistant", "coinflip"):
        for f, lead in LAZY_LEADS.items():
            for p, persona in LAZY_PERSONAS.items():
                spec, plain = prompts.CONDITIONS[f"{base}_{f}_{p}"], prompts.CONDITIONS[base]
                assert spec["body"] == lead.format(persona=persona) + plain["body"]
                assert {k: v for k, v in spec.items() if k != "body"} == {k: v for k, v in plain.items() if k != "body"}


@pytest.mark.parametrize("name", [OLMO, LLAMA, GEMMA, GLM])
@pytest.mark.parametrize("base", ["assistant", "coinflip"])
def test_lead_is_the_only_change(name, base):
    tok = AutoTokenizer.from_pretrained(name)
    plain = build_prompt(base, "FIRST", "SECOND", tok)
    i = plain.index("I would like you")
    ids = lambda s: tok(s, add_special_tokens=False).input_ids
    tail = ids(plain[i:])
    for f, lead in LAZY_LEADS.items():
        for p, persona in LAZY_PERSONAS.items():
            cond = f"{base}_{f}_{p}"
            text = build_prompt(cond, "FIRST", "SECOND", tok)
            assert text == plain[:i] + lead.format(persona=persona) + plain[i:]
            assert outcome_token_ids(cond, tok) == outcome_token_ids(base, tok)
            # The lead's tokens don't merge with what follows.
            assert ids(text)[-len(tail):] == tail


def test_configs_cover_every_persona_once_per_model():
    want = sorted(f"{b}_{f}_{p}" for b in ("coinflip", "assistant") for f in LAZY_LEADS for p in LAZY_PERSONAS)
    seen, plain = {}, {}
    for path in sorted(Path("configs/lazy").glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text())
        assert (cfg["max_pairs"], cfg["seed"], cfg["system_prompt"], cfg["group"]) == (15000, 0, "none", "lazy")
        assert cfg["tasks_file"] == "data/tasks_n300_max500_seed0.jsonl" and cfg["run_name"] == f"lazy-{path.stem}"
        (m,) = cfg["models"]
        persona = path.stem.removeprefix(f"{cfg['tags'][0]}-")
        assert m["stage"] == "instruct" and m["conditions"][-1] == "coinflip"
        assert sorted(m["conditions"][:-1]) == sorted(f"{b}_{f}_{persona}" for b in ("coinflip", "assistant")
                                                      for f in LAZY_LEADS)
        seen.setdefault(m["name"], []).extend(m["conditions"][:-1])
        plain[m["name"]] = plain.get(m["name"], 0) + 1
    assert {k: sorted(v) for k, v in seen.items()} == {n: want for n in (OLMO, LLAMA, GEMMA, GLM)}
    assert plain == {n: 5 for n in (OLMO, LLAMA, GEMMA, GLM)}


def synthetic(kind, n=300, seed=0, noise_sd=0.1, answeras=1.0):
    """A0, C0 and every persona cell, all with measurement noise of SD noise_sd. Per persona:
    - "follow": the Assistant moves along a new direction P, and the coin follows it at half its default slope (tau 0.5);
    - "ignore": the Assistant moves along P, the coin doesn't;
    - "flatten": both shrink toward 0 by half, nothing else;
    - "none": nothing changes (A_p, C_p are fresh noisy copies of the defaults).
    Both phrasings share P; "answer as" gets answeras times the persona's change."""
    rng = np.random.default_rng(seed)
    origins = np.array(["ALPACA", "MATH", "WILDCHAT", "STRESS_TEST", "BAILBENCH"] * (n // 5))
    benign = ~np.isin(origins, ["BAILBENCH"])
    a = rng.normal(size=n)
    c = 0.8 * a + 0.4 * rng.normal(size=n)  # the default coin: slope 0.8 on the true A
    e = lambda: noise_sd * rng.normal(size=n)
    V = {"A0": a + e(), "C0": c + e(), "loglen": rng.normal(size=n)}
    for p, k in zip(LAZY_PERSONAS, kind):
        P = rng.normal(size=n)
        for f in LAZY_LEADS:
            m = answeras if f == "answeras" else 1.0
            ap, cp = {"follow": (a + m * P, c + 0.4 * m * P), "ignore": (a + m * P, c), "flatten": (a - m * a / 2, c - m * c / 2),
                      "none": (a, c)}[k]
            V[f"A:{p}:{f}"], V[f"C:{p}:{f}"] = ap + e(), cp + e()
    noise = {key: noise_sd ** 2 for key in V}
    return V, noise, origins, benign


def test_transfer_recovers_known_cases():
    import lazy
    kind = ("follow", "ignore", "flatten", "none", "follow")
    V, noise, origins, benign = synthetic(kind)
    est = lazy.stats(V, noise, origins, benign)
    K = lazy.K
    for p, k in zip(lazy.PERSONAS, kind):
        for f in lazy.PHRASINGS:
            g = lambda s: est[K[f"{p}:{f}:{s}"]]
            if k == "follow":
                assert g("s_a") > 0.9 and g("rho") > 0.4 and abs(g("tau") - 0.5) < 0.15
            if k == "ignore":
                assert g("s_a") > 0.9 and abs(g("rho")) < 0.2 and abs(g("tau")) < 0.15
            if k in ("flatten", "none"):  # nothing persona-specific: the residual change is noise
                assert g("s_a") < 0.5 and abs(g("rho")) < 0.25
    # The two-pull fit, by contrast, reads a pull toward the noisy copy in the "none" case.
    assert est[K["wizard:youare:b_per"]] > 0.2


def test_calls():
    import lazy
    kind = ("follow", "ignore", "flatten", "none", "follow")
    V, noise, origins, benign = synthetic(kind)
    rng = np.random.default_rng(1)
    from report import stratified_resamples
    boot = np.array([lazy.stats({k: v[i] for k, v in V.items()}, noise, origins[i], benign[i])
                     for i in stratified_resamples(origins, rng, 200)])
    est = lazy.stats(V, noise, origins, benign)
    lo, hi = np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)
    want = {"follow": "coin follows the persona", "ignore": "no transfer", "flatten": "Assistant unmoved",
            "none": "Assistant unmoved"}
    for p, k in zip(lazy.PERSONAS, kind):
        for f in lazy.PHRASINGS:
            assert lazy.cell_call(est, lo, hi, p, f) == want[k], (p, f)
    assert lazy.phrasing_call(lo, hi, "ma") == "no difference"  # the phrasings differ only by noise


def test_phrasing_call():
    import lazy
    from report import stratified_resamples
    V, noise, origins, benign = synthetic(("follow",) * 5, answeras=0.5)
    boot = np.array([lazy.stats({k: v[i] for k, v in V.items()}, noise, origins[i], benign[i])
                     for i in stratified_resamples(origins, np.random.default_rng(1), 200)])
    lo, hi = np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)
    assert lazy.phrasing_call(lo, hi, "ma") == '"you are" moves it more'
    assert lazy.phrasing_call(lo, hi, "mc") == '"you are" moves it more'
