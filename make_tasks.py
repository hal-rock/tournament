"""Build the task files used in the tournament.

Tasks come from the canonical 6,000-task set of Gilg et al., "Probing
Persona-Dependent Preferences in Language Models"
(https://github.com/oscar-gilg/probing-persona-preferences), drawn from
Alpaca, MATH, WildChat, BailBench and STRESS-TEST.

    python make_tasks.py extract   # needs the upstream repo cloned at ./probing-persona-preferences
    python make_tasks.py sample --n 100 --max-chars 500 --seed 0

`extract` writes data/canonical_6000.jsonl (committed, so nothing downstream
needs the clone). `sample` draws a subsample from it with the upstream source
quotas. Within a seed the draws are nested: the n=100 tasks are a subset of the
n=300 tasks, so a larger run extends a smaller one.
"""

import argparse
import csv
import json
import random
from pathlib import Path

UPSTREAM = Path("probing-persona-preferences")
CANONICAL = Path("data/canonical_6000.jsonl")

# Source quotas of the upstream canonical splits (data/canonical_splits/README.md).
QUOTAS = {"MATH": 0.25, "ALPACA": 0.25, "WILDCHAT": 0.25, "STRESS_TEST": 0.15, "BAILBENCH": 0.10}


def _jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def extract():
    """Join the canonical task IDs to their texts, mirroring upstream src/task_data/loader.py.
    (Their loader imports the whole package and its API-client dependencies, hence the copy.)"""
    d = UPSTREAM / "src/task_data/data"
    tasks = {}
    for r in _jsonl(d / "wildchat_en_8k.jsonl"):
        tasks[r["id"]] = ("WILDCHAT", r["text"], {"type": r["type"]})
    for r in _jsonl(d / "alpaca_tasks_nemocurator.jsonl"):
        tasks[r["task_id"]] = ("ALPACA", r["task_text"], {"nemo_analysis": r.get("nemo_analysis", {})})
    for r in _jsonl(d / "math.jsonl"):
        tasks[r["id"]] = ("MATH", r["text"], r.get("q_metadata", {}).get("competition_math", {}))
    with open(d / "bailBench.csv", newline="", encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f)):
            tasks[f"bailbench_{i}"] = ("BAILBENCH", r["content"], {"category": r["category"], "subcategory": r["subcategory"]})
    for r in _jsonl(d / "stress_testing_model_spec.jsonl"):
        tid = "stresstest_{chunk_index}_{entry_idx}_{nudge_direction}".format(**r)
        tasks[tid] = ("STRESS_TEST", r["query"], {k: r[k] for k in ("value1", "value2", "nudge_direction")})

    # Topic labels were assigned by Claude Sonnet 4.5 upstream.
    topics = json.load(open(UPSTREAM / "data/topics/topics.json"))
    ids = (UPSTREAM / "data/canonical_splits/all_6000_task_ids.txt").read_text().split()
    CANONICAL.parent.mkdir(exist_ok=True)
    with open(CANONICAL, "w") as f:
        for tid in ids:
            origin, prompt, metadata = tasks[tid]  # KeyError here means upstream data changed
            label = topics.get(tid, {}).get("anthropic/claude-sonnet-4.5", {})
            f.write(json.dumps(dict(
                id=tid, origin=origin, topic=label.get("primary"), topic_secondary=label.get("secondary"),
                n_chars=len(prompt), prompt=prompt, metadata=metadata,
            )) + "\n")
    print(f"wrote {len(ids)} tasks to {CANONICAL}")


def quota_counts(n):
    """Split n across origins by QUOTAS, largest-remainder rounding."""
    raw = {o: n * q for o, q in QUOTAS.items()}
    counts = {o: int(r) for o, r in raw.items()}
    for o in sorted(raw, key=lambda o: raw[o] - counts[o], reverse=True)[: n - sum(counts.values())]:
        counts[o] += 1
    return counts


def sample(n, max_chars, seed):
    tasks = [t for t in _jsonl(CANONICAL) if t["n_chars"] <= max_chars]
    rng = random.Random(seed)
    out = []
    for origin, k in quota_counts(n).items():
        pool = sorted((t for t in tasks if t["origin"] == origin), key=lambda t: t["id"])
        rng_o = random.Random(f"{seed}-{origin}")  # per-origin stream keeps draws nested across n
        rng_o.shuffle(pool)
        assert k <= len(pool), f"only {len(pool)} {origin} tasks under {max_chars} chars, need {k}"
        out.extend(pool[:k])
    rng.shuffle(out)
    path = Path(f"data/tasks_n{n}_max{max_chars}_seed{seed}.jsonl")
    with open(path, "w") as f:
        f.writelines(json.dumps(t) + "\n" for t in out)
    print(f"wrote {len(out)} tasks to {path}: {quota_counts(n)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("extract")
    s = sub.add_parser("sample")
    s.add_argument("--n", type=int, default=100)
    s.add_argument("--max-chars", type=int, default=500)
    s.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    extract() if a.cmd == "extract" else sample(a.n, a.max_chars, a.seed)
