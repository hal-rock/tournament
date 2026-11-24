"""Replication round 2, exploratory (Hal, 2026-11-24): do the large models' length-rule choices follow length in characters
(what the analysis uses), words, or the model's own tokens?

Per model, each task's description (as build_prompt puts it in the prompt, stripped) is measured in characters, whitespace
words and tokens of the model's scoring tokenizer. For the instructed Assistant (and the default, as a reference):
- agreement with each measure's ranking (ifasked.agreement: share of task pairs the measure orders whose utilities order them
  the same way), over all 300 tasks and the 257 non-harmful ones;
- on the scored pairs (raw order-averaged log-odds, no fit) where characters and tokens (or words and tokens) disagree about
  which description is shorter: the share of picks that follow tokens;
- the mirrored contrast on those pairs, (P(pick the token-longer task | "longer") - P(same | "shorter")) / 2, which cancels
  any content preference that happens to correlate with one measure (> 0: the instructions follow tokens; < 0: characters).
  Likewise words against tokens and characters against words.

    uv run python scripts/length_measures.py
Writes outputs/replication/round2/length_measures.csv.
"""

import csv
import sys
from pathlib import Path

import numpy as np
import yaml
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import round2_analysis as r2  # noqa: E402
from ifasked import agreement  # noqa: E402
from round2_length_figures import pair_picks, raw_path  # noqa: E402


def tokenizer(key):
    """The model's scoring tokenizer, as score_replication loads it (Kimi K2's remote code rebuilt by the tests)."""
    path = Path("configs/replication") / (f"{key}.yaml" if key in r2.NEW else f"round2/{key}.yaml")
    (m,) = [x for x in yaml.safe_load(path.read_text())["models"] if x["stage"] == "instruct"]
    if key == "kimi-k2":
        from test_replication_kimi_k2 import stand_in
        return stand_in(m["name"], m["revision"])
    return AutoTokenizer.from_pretrained(m.get("tokenizer_from", m["name"]), revision=m.get("revision"),
                                         trust_remote_code=m.get("trust_remote_code", False))


def main():
    tasks = sorted(r2.ra.read_jsonl(r2.ra.DEFAULT_TASKS), key=lambda t: t["id"])  # Model's task order
    texts = [t["prompt"].strip() for t in tasks]
    ids = [t["id"] for t in tasks]
    chars = np.array([len(x) for x in texts], float)
    words = np.array([len(x.split()) for x in texts], float)
    rows = []
    for key in r2.FIRST_ROUND + r2.NEW:
        tok = tokenizer(key)
        toks = np.array([len(tok(x, add_special_tokens=False).input_ids) for x in texts], float)
        U, info, origins, nonharm, _, _ = r2.readouts(key)
        measures = {"chars": chars, "words": words, "tokens": toks}
        row = dict(model=key, r_chars_tokens=float(np.corrcoef(np.log(chars), np.log(toks))[0, 1]),
                   r_words_tokens=float(np.corrcoef(np.log(words), np.log(toks))[0, 1]),
                   chars_per_token=float(np.median(chars / toks)))
        for rule in ("shorter", "longer"):
            sign = 1 if rule == "longer" else -1
            for name in (f"A_{rule}", "A0"):
                u = U[name]
                for mname, mv in measures.items():
                    row[f"{name}|{rule}|{mname}|all"] = agreement(u, sign * mv)
                    row[f"{name}|{rule}|{mname}|nonharm"] = agreement(u[nonharm], sign * mv[nonharm])
            # Scored pairs where the measures disagree about which description is shorter.
            picks = pair_picks(raw_path(key, info[f"A_{rule}"]["source"]))
            pos = {t: i for i, t in enumerate(ids)}
            for m1, v1, m2, v2 in (("chars", chars, "tokens", toks), ("words", words, "tokens", toks),
                                   ("chars", chars, "words", words)):
                q = []  # 1 if the pick is the task longer by m2 (on pairs where m1 and m2 disagree)
                for (a, b), d in picks.items():
                    i, j = pos[a], pos[b]
                    s1, s2 = np.sign(v1[i] - v1[j]), np.sign(v2[i] - v2[j])
                    if s1 == 0 or s2 == 0 or s1 == s2:
                        continue
                    q.append(float((d > 0) == (s2 > 0)))
                row[f"q_{m2}_longer_vs_{m1}|{rule}"] = float(np.mean(q))
            for other, ov in (("chars", chars), ("words", words)):
                follow_tok, n = [], 0
                for (a, b), d in picks.items():
                    i, j = pos[a], pos[b]
                    o, t = np.sign(ov[i] - ov[j]), np.sign(toks[i] - toks[j])
                    if o == 0 or t == 0 or o == t:
                        continue
                    n += 1
                    a_longer_by_tokens = t > 0
                    a_wanted = a_longer_by_tokens if rule == "longer" else not a_longer_by_tokens
                    follow_tok.append((d > 0) == a_wanted)
                row[f"disagree_{other}|{rule}|n"] = n
                row[f"disagree_{other}|{rule}|follows_tokens"] = float(np.mean(follow_tok)) if n else float("nan")
        # The mirrored contrast: (P(pick the m2-longer task | "longer") - P(same | "shorter")) / 2 on the pairs where m1 and
        # m2 disagree. A content preference that correlates with one measure moves both rules' picks the same way and
        # cancels; > 0 means the instructions follow m2, < 0 means m1.
        for m1, m2 in (("chars", "tokens"), ("words", "tokens"), ("chars", "words")):
            row[f"mirror_{m2}_vs_{m1}"] = (row[f"q_{m2}_longer_vs_{m1}|longer"] - row[f"q_{m2}_longer_vs_{m1}|shorter"]) / 2
        rows.append(row)
        print(f"{key:18} r(log chars, log tokens) {row['r_chars_tokens']:.3f}  r(log words, log tokens) "
              f"{row['r_words_tokens']:.3f}  chars/token {row['chars_per_token']:.2f}")
        for rule in ("shorter", "longer"):
            a = lambda m, s: row[f"A_{rule}|{rule}|{m}|{s}"]
            print(f"   {rule:8} A_k agreement  all: chars {a('chars', 'all'):.3f} words {a('words', 'all'):.3f} "
                  f"tokens {a('tokens', 'all'):.3f} | non-harmful: chars {a('chars', 'nonharm'):.3f} "
                  f"words {a('words', 'nonharm'):.3f} tokens {a('tokens', 'nonharm'):.3f} | pairs where chars and tokens "
                  f"disagree: n {row[f'disagree_chars|{rule}|n']}, follows tokens "
                  f"{row[f'disagree_chars|{rule}|follows_tokens']:.2f}; words vs tokens: n {row[f'disagree_words|{rule}|n']}, "
                  f"follows tokens {row[f'disagree_words|{rule}|follows_tokens']:.2f}")
        print(f"   mirrored (> 0 follows the second): tokens vs chars {row['mirror_tokens_vs_chars']:+.3f}, tokens vs words "
              f"{row['mirror_tokens_vs_words']:+.3f}, words vs chars {row['mirror_words_vs_chars']:+.3f}")
    out = r2.OUT / "round2" / "length_measures.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
