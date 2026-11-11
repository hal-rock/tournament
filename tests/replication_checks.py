"""Checks shared by the replication onboarding tests (tests/test_replication_<key>.py; PLAN_replication.md).

Each test reads configs/replication/<key>.yaml, so it checks what will run: every base and final condition, rendered by
prompts.build_prompt on the real task file, against
- the model's own chat template rendered directly (apply_chat_template, not prompts._render), and
- the format written out by hand (`Format`), so the exact markers and newlines are on record;
and the token ids against what the model will be fed (tokenized with add_special_tokens=False, as the scorer does).
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import filter_repo_objects

import prompts
from prompts import CONDITIONS, build_prompt, outcome_token_ids, parse_condition
from replication import BASE_CONDITIONS, FINAL_CONDITIONS, expand_conditions

ROOT = Path(__file__).resolve().parent.parent


def load_config(key: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / "replication" / f"{key}.yaml").read_text())


def tasks(cfg: dict) -> list[str]:
    return [json.loads(line)["prompt"] for line in open(ROOT / cfg["tasks_file"])]


def pairs(task_list: list[str]) -> list[tuple[str, str]]:
    """Every task once as Task A and once as Task B, so each one's text meets both of its neighbours in the layout."""
    return [(task_list[i], task_list[(i + 1) % len(task_list)]) for i in range(len(task_list))]


@dataclass
class Format:
    """A model's official chat format, written out by hand from its template."""
    bos: str              # opens every prompt ("" if the format has none)
    default_system: str   # what the template writes when there is no system message ("" if nothing)
    system: str           # a system message; "{}" stands for its text
    user: str             # opens the user turn
    user_end: str         # closes it ("" if the next turn's opening does)
    reply: str            # opens the Assistant's turn (the generation prompt)
    markers: dict[str, int] = field(default_factory=dict)  # every special token the prompts use -> its id


def persona_parts(condition: str) -> tuple[str | None, str]:
    """(system message or None, text that opens the user turn) for a condition, straight from prompts' tables."""
    _, persona, placement = parse_condition(condition)
    system = {"system": prompts.PERSONAS, "roleplay": prompts.ROLEPLAY}.get(placement, {}).get(persona)
    lead = {"user": prompts.PERSONAS, "self": prompts.PERSONAS_SELF}.get(placement, {}).get(persona)
    return system, (lead + "\n\n" if lead else "")


def body(condition: str, first: str, second: str) -> str:
    _, lead = persona_parts(condition)
    return lead + CONDITIONS[parse_condition(condition)[0]]["body"].format(first=first.strip(), second=second.strip())


def has_reply(condition: str) -> bool:
    return CONDITIONS[parse_condition(condition)[0]]["prefill"] is not None


def reference(condition: str, first: str, second: str, template_tok, fmt: Format) -> str:
    """The prompt rendered with the official template directly: the template's default system block removed when
    there is no system message, the user turn cut where its text ends (user-turn conditions) or the generation prompt
    plus the prefill (Assistant conditions)."""
    system, _ = persona_parts(condition)
    text = body(condition, first, second)
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": text}]
    reply = has_reply(condition)
    rendered = template_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=reply)
    if system is None:
        assert rendered.startswith(fmt.bos + fmt.default_system), rendered[:200]
        rendered = fmt.bos + rendered[len(fmt.bos + fmt.default_system):]
    if reply:
        return rendered + CONDITIONS[parse_condition(condition)[0]]["prefill"]
    assert rendered.endswith(text + fmt.user_end), rendered[-200:]
    return rendered[:len(rendered) - len(fmt.user_end)]


def by_hand(condition: str, first: str, second: str, fmt: Format) -> str:
    system, _ = persona_parts(condition)
    head = fmt.bos + (fmt.system.replace("{}", system) if system else "") + fmt.user + body(condition, first, second)
    if not has_reply(condition):
        return head
    return head + fmt.user_end + fmt.reply + CONDITIONS[parse_condition(condition)[0]]["prefill"]


def render(condition: str, first: str, second: str, template_tok, cfg: dict) -> str:
    """What the scorer renders (prompts.build_prompt with the config's settings)."""
    return build_prompt(condition, first, second, template_tok, cfg["system_prompt"])


def all_conditions(cfg: dict) -> list[str]:
    """The union of the config's conditions, final's order; checks the base runs exactly the base set, final the final."""
    (base,), (final,) = ([m for m in cfg["models"] if m["stage"] == s] for s in ("base", "instruct"))
    assert expand_conditions(base["conditions"]) == list(BASE_CONDITIONS)
    assert expand_conditions(final["conditions"]) == list(FINAL_CONDITIONS)
    assert set(BASE_CONDITIONS) <= set(FINAL_CONDITIONS)
    return list(FINAL_CONDITIONS)


def check_config(cfg: dict, key: str, final_name: str, base_name: str):
    assert (cfg["group"], cfg["run_name"], cfg["max_pairs"], cfg["seed"], cfg["system_prompt"]) == \
           ("replication", key, 15000, 0, "none")
    assert cfg["tasks_file"] == "data/tasks_n300_max500_seed0.jsonl"
    base, final = cfg["models"]  # base first, then the final model
    assert (base["name"], base["stage"], base["conditions"]) == (base_name, "base", "base")
    assert (final["name"], final["stage"], final["conditions"]) == (final_name, "instruct", "final")
    assert base["chat_template_from"] == final_name and "chat_template_from" not in final
    for m in (base, final):
        assert len(m["revision"]) == 40, "pin the commit the tests checked"
        assert m["vllm"]["tensor_parallel_size"] == 8 and 0.85 <= m["vllm"]["gpu_memory_utilization"] <= 0.95
    assert base["vllm"] == final["vllm"], "base and final run with the same engine settings"
    all_conditions(cfg)


def check_download(m: dict, required: list[str]):
    """allow_patterns, applied the way snapshot_download applies them to the repo at the pinned revision, fetch every
    shard in the index plus `required`, nothing else big, and `gb` is the total."""
    files = list(HfApi().list_repo_tree(m["name"], revision=m["revision"], recursive=True))
    files = [f for f in files if hasattr(f, "size")]  # files, not folders
    chosen = {f.path: f.size for f in filter_repo_objects(files, allow_patterns=m["download"]["allow_patterns"],
                                                         key=lambda f: f.path)}
    index = json.loads(Path(hf_hub_download(m["name"], "model.safetensors.index.json", revision=m["revision"]))
                       .read_text())["weight_map"]
    shards = set(index.values())
    assert {p for p in chosen if p.endswith(".safetensors")} == shards
    assert set(required) <= set(chosen), set(required) - set(chosen)
    assert set(chosen) == shards | set(required), set(chosen) - shards - set(required)
    total = sum(chosen.values()) / 1e9
    assert abs(total - m["download"]["gb"]) < 0.1, total
    return total


def check_prompts(cfg: dict, template_tok, base_tok, fmt: Format, reference_encode=None, max_pairs=None):
    """Every condition on every task pair: build_prompt equals the template's own rendering and the hand-written
    format; the ids are the same under the base and the template tokenizer (and `reference_encode`, an independent
    tokenizer implementation, if given); markers are single ids; BOS where the format has it; the outcome token is the
    next token after the prompt. Returns the longest prompt in tokens."""
    enc_t = lambda s: template_tok(s, add_special_tokens=False).input_ids
    enc_b = lambda s: base_tok(s, add_special_tokens=False).input_ids
    marker_ids = set(fmt.markers.values())
    for s, i in fmt.markers.items():
        assert enc_t(s) == [i] and enc_b(s) == [i], f"{s!r} is not the single token {i}"
    bos_id = fmt.markers.get(fmt.bos) if fmt.bos else None
    added = set(template_tok.get_added_vocab().values())
    longest = 0
    for condition in all_conditions(cfg):
        outcomes = CONDITIONS[parse_condition(condition)[0]]["outcomes"]
        out_t, out_b = outcome_token_ids(condition, template_tok), outcome_token_ids(condition, base_tok)
        assert out_t == out_b and len(set(out_t)) == 2 and not set(out_t) & added, (condition, out_t, out_b)
        system, _ = persona_parts(condition)
        for first, second in pairs(tasks(cfg))[:max_pairs]:
            p = render(condition, first, second, template_tok, cfg)
            assert p == reference(condition, first, second, template_tok, fmt), condition
            assert p == by_hand(condition, first, second, fmt), condition
            ids = enc_t(p)
            assert enc_b(p) == ids, condition
            if reference_encode is not None:
                assert reference_encode(p) == ids, condition
            longest = max(longest, len(ids))
            # Markers: each occurrence in the text is its one id, and no other added token appears.
            for s, i in fmt.markers.items():
                assert p.count(s) == ids.count(i), (condition, s)
            assert {i for i in ids if i in added} <= marker_ids, condition
            # BOS exactly where the format has it.
            if bos_id is not None:
                assert ids[0] == bos_id and ids.count(bos_id) == 1, condition
            else:
                assert template_tok.bos_token_id not in ids, condition
            # No system block unless the condition has one.
            if system is None:
                assert p.startswith(fmt.bos + fmt.user), condition
            # Where the prompt ends: inside the open user turn, or the reply header plus the prefill.
            if has_reply(condition):
                prefill = CONDITIONS[parse_condition(condition)[0]]["prefill"]
                assert p.endswith(fmt.user_end + fmt.reply + prefill)
                assert ids == enc_t(p[:-len(prefill)]) + enc_t(prefill), condition
            else:
                last = p.rindex(fmt.user)
                assert fmt.reply not in p and (not fmt.user_end or fmt.user_end not in p[last:]), condition
            # The outcome is the next token: prompt + outcome string tokenizes to the prompt's ids + the outcome id.
            # (A prompt ending in a space would pass that too, with a lone " " token before " A": rule it out.)
            assert not p[-1].isspace(), condition
            for o, oid in zip(outcomes, out_t):
                assert enc_t(p + o) == ids + [oid] and enc_b(p + o) == ids + [oid], (condition, o)
    return longest


def check_persona_blocks(cfg: dict, template_tok, fmt: Format, first: str, second: str):
    """A persona condition is the plain prompt with exactly its system block (system, roleplay) or its lead (user,
    self) inserted, in text and in tokens; the plain prompt has neither."""
    enc = lambda s: template_tok(s, add_special_tokens=False).input_ids
    for condition in all_conditions(cfg):
        plain_name, persona, placement = parse_condition(condition)
        if persona is None:
            continue
        plain = render(plain_name, first, second, template_tok, cfg)
        placed = render(condition, first, second, template_tok, cfg)
        system, lead = persona_parts(condition)
        if system is not None:
            head = fmt.bos + fmt.system.replace("{}", system)
            assert placed == head + plain[len(fmt.bos):], condition
            assert enc(placed) == enc(head) + enc(plain[len(fmt.bos):]), condition
        else:
            cut = len(fmt.bos + fmt.user)
            assert plain.startswith(fmt.bos + fmt.user) and placed == plain[:cut] + lead + plain[cut:], condition
        assert outcome_token_ids(condition, template_tok) == outcome_token_ids(plain_name, template_tok)


def longest_prompt(cfg: dict, template_tok, top: int = 12) -> tuple[int, str]:
    """The longest prompt in tokens over every condition, among all ordered pairs of the `top` longest tasks (prompt
    length is the condition's frame plus the two tasks, so the longest pair is among them)."""
    task_list = tasks(cfg)
    n = [len(template_tok(t.strip(), add_special_tokens=False).input_ids) for t in task_list]
    longest = sorted(range(len(task_list)), key=lambda i: -n[i])[:top]
    best = (0, "")
    for condition in all_conditions(cfg):
        for a, b in itertools.permutations(longest, 2):
            p = render(condition, task_list[a], task_list[b], template_tok, cfg)
            best = max(best, (len(template_tok(p, add_special_tokens=False).input_ids), condition))
    return best
