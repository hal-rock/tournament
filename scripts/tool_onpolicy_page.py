"""The tool run's on-policy check as transcript pages (README, "The coin as a tool call"): the Assistant's own greedy replies
to each tool-call condition's user turn (score_replication.py --onpolicy), on the prompt as the model read it.

    uv run python scripts/tool_onpolicy_page.py [key ...]

Per model, outputs/replication/tool-<key>/onpolicy.html (gitignored: the prompts hold WildChat tasks). Picked per condition,
in pair order: the plain tool coin's first 3 replies that call flip_coin and first 3 that don't; each persona condition's
first reply that calls it and first that doesn't. The prompt is re-rendered from the config (build_prompt with
prefill=False, as the scorer rendered it); the reply is the node's decoded greedy reply (special tokens kept).
"""

import json
import sys
from pathlib import Path

import yaml
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from matskit import view  # noqa: E402
from prompts import CONDITIONS, PERSONAS, PERSONAS_SELF, build_prompt, parse_condition  # noqa: E402
from score_replication import read_jsonl, select_pairs  # noqa: E402

KEYS = ("llama-405b", "deepseek-v3", "kimi-k2", "deepseek-v4-flash", "trinity-large")


def tokenizer(m):
    if m["tool_family"] == "kimi":  # a fast stand-in for Kimi's remote-code tiktoken tokenizer, which has no offsets
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))  # (tests/test_replication_kimi_k2.py)
        from test_replication_kimi_k2 import stand_in
        return stand_in(m["name"], m["revision"], m["revision"])
    return AutoTokenizer.from_pretrained(m["name"], revision=m.get("revision"),
                                         trust_remote_code=m.get("trust_remote_code", False))


def pick(rows, n):
    calls = [r for r in rows if r["calls"]][:n]
    other = [r for r in rows if not r["calls"]][:n]
    return calls + other


def page(key):
    cfg = yaml.safe_load(open(f"configs/replication/tool/{key}.yaml"))
    (m,) = cfg["models"]
    tok = tokenizer(m)
    tasks = {t["id"]: t for t in read_jsonl(cfg["tasks_file"])}
    out = Path("outputs/replication") / f"tool-{key}"
    transcripts = []
    for path in sorted((out / "onpolicy").glob("instruct__*.jsonl"), key=lambda p: ("@" in p.name, p.name)):
        rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        condition = rows[0]["condition"]
        _, persona, placement = parse_condition(condition)
        for r in pick(rows, 1 if persona else 3):
            first, second = tasks[r["first"]]["prompt"], tasks[r["second"]]["prompt"]
            prompt = build_prompt(condition, first, second, tok, cfg.get("system_prompt", "none"), False,
                                  close_think=m.get("close_think", False), tool_family=m["tool_family"],
                                  tool_repo=m.get("tool_repo", m["name"]))
            lead = ""
            parts = []
            if placement == "system":
                parts.append(dict(text=PERSONAS[persona], role="system", turn=0))
            elif placement in ("user", "self"):
                lead = (PERSONAS if placement == "user" else PERSONAS_SELF)[persona] + "\n\n"
            body = lead + CONDITIONS["coinflip_tool"]["body"].format(first=first.strip(), second=second.strip())
            parts.append(dict(text=body, role="user", turn=1))
            parts.append(dict(text=r["reply"], role="assistant", turn=2))
            text = prompt + r["reply"]
            try:
                t = view.from_exact(tok, text, parts, sampled_text=[r["reply"]], title=f"{condition}: {r['first']} vs "
                                    f"{r['second']}", group=condition,
                                    meta=dict(condition=condition, calls_flip_coin=r["calls"],
                                              call_before_any_text=r["call_first"]))
            except ValueError:  # a template that rewrites the message text: show it all as the exact string
                t = view.from_exact(tok, text, [dict(text=r["reply"], role="assistant", turn=2)],
                                    sampled_text=[r["reply"]], title=f"{condition}: {r['first']} vs {r['second']}",
                                    group=condition, meta=dict(condition=condition, calls_flip_coin=r["calls"],
                                                               call_before_any_text=r["call_first"]))
            transcripts.append(t)
    counts = "; ".join(f"{p.name.removeprefix('instruct__').removesuffix('.jsonl')} "
                       f"{sum(json.loads(x)['calls'] for x in p.read_text().splitlines())}/50"
                       for p in sorted((out / "onpolicy").glob("instruct__*.jsonl")))
    view.write(out / "onpolicy.html", transcripts, title=f"{key}: does the Assistant call flip_coin itself?",
               note=("Greedy replies (64 tokens, thinking off as read) to each tool-call condition's user turn, on the first "
                     "50 sweep pairs. Shown per condition in pair order: the plain tool coin's first 3 replies that call "
                     "flip_coin and first 3 that don't; each persona condition's first of each. Prompts re-rendered from "
                     f"the config. Calls per condition: {counts}."))


if __name__ == "__main__":
    for key in sys.argv[1:] or KEYS:
        if (Path("outputs/replication") / f"tool-{key}" / "onpolicy").exists():
            page(key)
