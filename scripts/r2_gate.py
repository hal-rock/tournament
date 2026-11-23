"""Replication round 2's gate (README, "Replication round 2", Gate): each first-round final's plain Assistant, coin and tool
coin, re-scored on the round-2 node (outputs/replication/r2-<key>/), against its round-1 scoring (outputs/replication/<key>/,
tool-<key>/). Passes at utilities r >= 0.999 with SD within 2% (scripts/compare_engines.py's statistics).

    uv run python scripts/r2_gate.py [keys ...]   # default: every first-round model with round-2 results on disk

Prints a table and writes outputs/replication/round2_gate.csv. A model passes only if all its conditions pass.
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_engines import compare  # noqa: E402

ROOT = Path("outputs/replication")
KEYS = ["deepseek-v4-flash", "llama-405b", "deepseek-v3", "kimi-k2", "trinity-large", "nemotron4-340b"]
R_BAR, SD_BAR = 0.999, 0.02


def round1(key, condition):
    return ROOT / ("tool-" + key if condition == "coinflip_tool" else key) / "results" / f"instruct__{condition}.csv.gz"


def main(argv=None):
    keys = (argv if argv is not None else sys.argv[1:]) or [k for k in KEYS if (ROOT / f"r2-{k}" / "results").exists()]
    rows = []
    for key in keys:
        for condition in ("assistant", "coinflip", "coinflip_tool"):
            new = ROOT / f"r2-{key}" / "results" / f"instruct__{condition}.csv.gz"
            if not new.exists() or not round1(key, condition).exists():
                continue
            c = compare(new, round1(key, condition))
            ok = c["r_utilities"] >= R_BAR and abs(c["sd_ratio"] - 1) <= SD_BAR
            rows.append(dict(model=key, condition=condition, prompts=c["prompts"], r_prompt=round(c["r_prompt"], 4),
                             median_abs_diff=round(c["median_abs_diff"], 3), r_util=round(c["r_utilities"], 5),
                             sd_ratio=round(c["sd_ratio"], 4), same_tokens=c["same_prompt_tokens"], passes=ok))
    for r in rows:
        print(f"{r['model']:18} {r['condition']:14} prompts {r['prompts']:6}  per-prompt r {r['r_prompt']:.4f} "
              f"(median |diff| {r['median_abs_diff']:.3f})  utilities r {r['r_util']:.5f}  SD ratio {r['sd_ratio']:.4f}  "
              f"{'pass' if r['passes'] else 'FAIL'}")
    for key in sorted({r["model"] for r in rows}):
        mine = [r for r in rows if r["model"] == key]
        print(f"{key}: {'passes' if all(r['passes'] for r in mine) else 'fails'} the gate ({len(mine)} conditions)")
    if rows:
        out = ROOT / "round2_gate.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return rows


if __name__ == "__main__":
    main()
