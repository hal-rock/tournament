"""Compare two scorings of the same condition: vLLM (score_replication.py) against an earlier HF run, or any two results
files. The gate is the repo's usual one for re-scored conditions: fitted utilities at r >= 0.999 with SD within 2% on the
same GPU type, r >= 0.99 across engines or GPU types (PLAN_replication.md, "Engine").

    uv run python scripts/compare_engines.py outputs/replication/validate-gemma4-31b/results/instruct__coinflip.csv.gz \\
        outputs/mgldkcxm/results/instruct__coinflip.csv.gz [more pairs of files ...] [--bar 0.99]

Reports, per pair of files: prompts matched; per-prompt log-odds r, median and max |difference|; outcome mass r and the two
medians; then the per-task utilities (analyze.fit_utilities on both orders): r, SD ratio, max |difference| in SDs.
"""

import argparse
import csv
import gzip
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze import fit_utilities, pair_effects  # noqa: E402


def read(path):
    with gzip.open(path, "rt") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("log_odds", "outcome_mass"):
            r[k] = float(r[k])
    return {(r["first"], r["second"]): r for r in rows}


def compare(path_a, path_b):
    a, b = read(path_a), read(path_b)
    keys = sorted(set(a) & set(b))
    assert keys, f"no prompts in common between {path_a} and {path_b}"
    la, lb = np.array([a[k]["log_odds"] for k in keys]), np.array([b[k]["log_odds"] for k in keys])
    ma, mb = np.array([a[k]["outcome_mass"] for k in keys]), np.array([b[k]["outcome_mass"] for k in keys])
    out = dict(prompts=len(keys), only_a=len(set(a) - set(b)), only_b=len(set(b) - set(a)),
               r_prompt=float(np.corrcoef(la, lb)[0, 1]), median_abs_diff=float(np.median(np.abs(la - lb))),
               max_abs_diff=float(np.max(np.abs(la - lb))), r_mass=float(np.corrcoef(ma, mb)[0, 1]),
               median_mass_a=float(np.median(ma)), median_mass_b=float(np.median(mb)),
               # Both scorings should have read the same token ids; equal lengths are a cheap check of that.
               same_prompt_tokens=float(np.mean([a[k]["prompt_tokens"] == b[k]["prompt_tokens"] for k in keys])))
    rows = [dict(stage="x", condition="c", first=k[0], second=k[1], log_odds=v) for k, v in zip(keys, la)]
    rows_b = [dict(stage="x", condition="c", first=k[0], second=k[1], log_odds=v) for k, v in zip(keys, lb)]
    ea, eb = pair_effects(rows)[("x", "c")], pair_effects(rows_b)[("x", "c")]
    ids = sorted({t for p in ea["pairs"] for t in p})
    ua, ub = fit_utilities(ids, ea), fit_utilities(ids, eb)
    out.update(r_utilities=float(np.corrcoef(ua, ub)[0, 1]), sd_ratio=float(ua.std() / ub.std()),
               max_abs_diff_sd=float(np.max(np.abs((ua - ua.mean()) - (ub - ub.mean()))) / ub.std()))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="pairs of results files: new reference [new reference ...]")
    ap.add_argument("--bar", type=float, default=0.99, help="r of utilities required (0.999 on the same engine and GPU)")
    a = ap.parse_args(argv)
    assert len(a.files) % 2 == 0, "give files in pairs"
    ok = True
    for new, ref in zip(a.files[::2], a.files[1::2]):
        s = compare(new, ref)
        passed = s["r_utilities"] >= a.bar and abs(s["sd_ratio"] - 1) <= 0.02 and s["same_prompt_tokens"] == 1
        ok &= passed
        print(f"{'PASS' if passed else 'FAIL'} {Path(new).name} vs {ref}")
        print("   " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in s.items()))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
