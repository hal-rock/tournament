"""The transplant tables for "Pushing the heads/tails lean" (README): per final, its lean slope (slope of the logit lens's
heads - tails pair effect on the Assistant's A_i - A_j, preference prompt, 1,000 pilot pairs; scripts/lens_slopes.py's
measure) at each pushed token for every block from the onset, and the final's Assistant utility per task (its sweep).
The base of each family is pushed by its final's table. Onset: the shallowest block from which the lens lean at " up"
correlates >= 0.5 with the output coin (every block after it too), from the same run.

Usage: uv run python scripts/inject_tables.py  -> configs/up_inject/transplant_<family>.json
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coin_report import load  # noqa: E402
from up_followups import corr_cols, lens_arrays, runs, sweep_utilities  # noqa: E402

TOKENS = [",", " the", " ...", " it", " came", " up", "."]


def slopes_cols(X, a):
    ac = a - a.mean()
    return np.tensordot(ac, X - X.mean(0), axes=(0, 0)) / (ac ** 2).sum()


def main():
    all_runs = runs()
    for fam in ("gemma4-31b", "llama-70b", "glm-4.5-air"):
        run = next(r for r in all_runs if r["family"] == fam and r["report"] == "coin_report_bold" and "line3" in r["arm_sets"])
        D = load(run["path"] / "rows.jsonl")
        util = sweep_utilities(fam)
        ids, lens, out, mass = lens_arrays(run["path"])
        sel = np.array([{p: k for k, p in enumerate(D["pair"])}[p] for p in ids])
        a = np.array([util[x][0] - util[y][0] for x, y in zip(D["a"], D["b"])])[sel]
        toks = run["config"]["suffix_tokens"]
        n_line = toks.index(" up") + 1
        ks = {t: (n_line - 1 if t == " up" else n_line if t == "." else toks[:n_line].index(t)) for t in TOKENS}
        r_up = corr_cols(lens[:, :, ks[" up"], 0], D["c"][sel])
        onset = int(np.argmax([np.all(r_up[l:] >= 0.5) for l in range(len(r_up))]))
        b = slopes_cols(lens[..., 0], a)  # [L, P]
        table = {"family": fam, "run": run["id"], "onset": onset, "n_layers": lens.shape[1],
                 "slope": {t: [float(x) for x in b[onset:, k]] for t, k in ks.items()},
                 "A": {tid: v[0] for tid, v in util.items()}}
        path = Path(f"configs/up_inject/transplant_{fam}.json")
        path.write_text(json.dumps(table, indent=1))
        print(f"{fam}: onset {onset} of {lens.shape[1]}; mean slope from the onset: "
              + ", ".join(f"{t!r} {np.mean(table['slope'][t]):+.3f}" for t in TOKENS) + f" -> {path}")


if __name__ == "__main__":
    main()
