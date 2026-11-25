"""Choice rules (README, "Choice rules: tickets, languages and position"; Hal, 2026-11-25).

Rules that tell the Assistant how to choose by something unrelated to its preferences, so its choice can be moved without
moving what it likes. Three kinds:
- tickets: every task carries a random 3-digit ticket number ("Task A (ticket #417): ..."), and the rule is "the lower /
  higher ticket number". The number is a per-task utility we set ourselves.
- mixed: a random half of the tasks is shown in Spanish (translate.py's translations), and the rule is "the one in Spanish
  / English". A two-level per-task utility.
- position: "the one listed first / second". No task utility: the planned choice says nothing about the tasks.
Each rule goes in the user turn (the reworded "if I ask you to choose" lead) or in the system prompt.

This module draws the per-task assignments (`python choice_rules.py assign`), which tournament.py adds to the tasks
(`decorate`), and analyzes the pilot (`python choice_rules.py pilot`): how well each Assistant follows each rule.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

ASSIGN_PATH = Path("data/choice_rules_assign.json")
TASKS_PATH = Path("data/tasks_n300_max500_seed0.jsonl")
TASKS_ES_PATH = Path("data/tasks_n300_max500_seed0_es.jsonl")
# The model sweep's runs for the four pilot models (README, "Model sweep"): the default Assistant and coin, which the
# assignments are drawn to be uncorrelated with.
SWEEP_RUNS = {"llama-70b": "tvlqj8a0", "gemma4-31b": "jrl62npf", "glm-4.5-air": "i5vejs3b", "deepseek-v4-flash": "lszrxqyq"}
MAX_ABS_R = 0.05


def _read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def is_harmful(task):
    from analyze import HARM_ORIGINS, HARM_TOPICS
    return task["topic"] in HARM_TOPICS or task["origin"] in HARM_ORIGINS


def balance_columns(tasks):
    """{name: per-task values} that the assignments must not correlate with."""
    import csv
    cols = {"log_length": np.log([t["n_chars"] for t in tasks]), "harmful": np.array([is_harmful(t) for t in tasks], float)}
    ids = [t["id"] for t in tasks]
    for key, run in SWEEP_RUNS.items():
        with open(f"outputs/{run}/analysis/utilities.csv") as f:
            by_id = {r["id"]: r for r in csv.DictReader(f)}
        for cond in ("assistant", "coinflip"):
            cols[f"{key}/{cond}"] = np.array([float(by_id[i][f"instruct/{cond}"]) for i in ids])
    return cols


def _max_abs_r(x, cols):
    return max(abs(np.corrcoef(x, v)[0, 1]) for v in cols.values())


def draw_assignments(tasks, cols, seed=0, max_tries=100_000):
    """Tickets: distinct integers in 100-999, redrawn until |r| < MAX_ABS_R with every balance column and each source's mean
    ticket rank is within 0.42-0.58. Spanish: exactly half of each (source, harmful) stratum (an odd stratum's extra task
    goes either way at random), redrawn until |r| < MAX_ABS_R with every balance column."""
    rng = np.random.default_rng(seed)
    n = len(tasks)
    origins = np.array([t["origin"] for t in tasks])
    harmful = cols["harmful"].astype(bool)
    for tries in range(1, max_tries + 1):
        tickets = rng.choice(np.arange(100, 1000), size=n, replace=False)
        ranks = tickets.argsort().argsort() / (n - 1)
        if _max_abs_r(tickets, cols) < MAX_ABS_R and all(0.42 <= ranks[origins == o].mean() <= 0.58 for o in set(origins)):
            break
    else:
        raise RuntimeError("no ticket draw met the balance constraints")
    ticket_tries = tries
    strata = sorted({(o, h) for o, h in zip(origins, harmful)})
    for tries in range(1, max_tries + 1):
        spanish = np.zeros(n, bool)
        for o, h in strata:
            idx = np.flatnonzero((origins == o) & (harmful == h))
            k = len(idx) // 2 + (rng.random() < 0.5 if len(idx) % 2 else 0)
            spanish[rng.choice(idx, size=k, replace=False)] = True
        if _max_abs_r(spanish.astype(float), cols) < MAX_ABS_R:
            break
    else:
        raise RuntimeError("no language draw met the balance constraints")
    return {t["id"]: {"ticket": int(tk), "lang": "es" if es else "en"} for t, tk, es in zip(tasks, tickets, spanish)}, (
        ticket_tries, tries)


def decorate(tasks, assign_path=ASSIGN_PATH, tasks_es_path=TASKS_ES_PATH):
    """The tasks with what the choice-rule conditions read: tags["ticket"] (formatted into "{first_ticket}" /
    "{second_ticket}") and prompt_mixed (the Spanish translation for the tasks assigned Spanish, the English otherwise)."""
    assign = json.loads(Path(assign_path).read_text())["tasks"]
    es = {t["id"]: t for t in _read_jsonl(tasks_es_path)}
    out = []
    for t in tasks:
        a = assign[t["id"]]
        if a["lang"] == "es":
            assert es[t["id"]]["prompt_en"] == t["prompt"], t["id"]
        out.append({**t, "tags": {"ticket": a["ticket"]}, "lang": a["lang"],
                    "prompt_mixed": es[t["id"]]["prompt"] if a["lang"] == "es" else t["prompt"]})
    return out


def cmd_assign(args):
    tasks = _read_jsonl(TASKS_PATH)
    cols = balance_columns(tasks)
    assign, tries = draw_assignments(tasks, cols, seed=args.seed)
    tickets = np.array([assign[t["id"]]["ticket"] for t in tasks], float)
    spanish = np.array([assign[t["id"]]["lang"] == "es" for t in tasks], float)
    report = {name: {"ticket": round(float(np.corrcoef(tickets, v)[0, 1]), 4),
                     "spanish": round(float(np.corrcoef(spanish, v)[0, 1]), 4)} for name, v in cols.items()}
    out = {"seed": args.seed, "max_abs_r": MAX_ABS_R, "tries": {"ticket": tries[0], "lang": tries[1]},
           "correlations": report, "tasks": assign}
    ASSIGN_PATH.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "tasks"}, indent=1))
    print(f"{int(spanish.sum())} of {len(tasks)} tasks in Spanish; wrote {ASSIGN_PATH}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("assign", help="draw the ticket numbers and languages")
    a.add_argument("--seed", type=int, default=0)
    a.set_defaults(func=cmd_assign)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
