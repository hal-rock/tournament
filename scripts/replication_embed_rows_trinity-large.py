"""Did Trinity Large's base train the chat markers our prompts use? Reads input-embedding rows of a base checkpoint and
of Trinity-Large-Thinking straight from the HF checkpoints with HTTP range requests (~50 MB in all, no weights), as in
../self-modeling/scripts/embed_rows.py.

    .venv/bin/python scripts/replication_embed_rows_trinity-large.py                 # TrueBase (the config's base)
    .venv/bin/python scripts/replication_embed_rows_trinity-large.py --base arcee-ai/Trinity-Large-Base
    # -> replication/onboarding/trinity-large_embed_rows[_<base>].csv

The ids are the ones the replication prompts contain, tokenized by Thinking's tokenizer (which the base borrows).
Thinking's <think> / </think> are ids 13 / 14; TrueBase's own tokenizer calls them <|reserved_special_0|> /
<|reserved_special_1|>, Trinity-Large-Base's calls them <think> / </think>. For each row:
- rel_norm: L2 norm over the median norm of 2,000 ordinary BPE rows (ids 1000-1999 and 100000-100999), same model;
- cos_mean: cosine with the mean ordinary row. Trained rows share the embedding's common direction (ordinary rows:
  median ~0.2); a row still at its random initialization has cosine ~0 with it;
- cos_base_final: cosine between the base's and Thinking's row (1.0: training after the base never touched it).
Reference rows no prompt uses: ids 21-31, <|reserved_special_8..18|> in both bases, never in their data.
"""

import argparse
import csv
import json
import os
import struct
import urllib.request
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download, hf_hub_url
from transformers import AutoTokenizer

TRUEBASE, FINAL = "arcee-ai/Trinity-Large-TrueBase", "arcee-ai/Trinity-Large-Thinking"
EMBED = "model.embed_tokens.weight"
MARKERS = ["<|begin_of_text|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]  # every marker the prompts use
CONTEXT = ["<|end_of_text|>", "<|pad|>"]  # not in the prompts; for comparison
ORDINARY = list(range(1000, 2000)) + list(range(100000, 101000))
RESERVED = list(range(21, 32))


def _get(url, start, n):
    headers = {"Range": f"bytes={start}-{start + n - 1}"}
    if os.environ.get("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r:
        data = r.read()
    assert len(data) == n, (len(data), n)  # a server ignoring Range would send the whole file
    return data


def embedding_rows(repo, ids, name=EMBED):
    """{id: row} of the input embedding as float32 (BF16 checkpoint), one range read per run of nearby ids."""
    index = json.loads(Path(hf_hub_download(repo, "model.safetensors.index.json")).read_text())["weight_map"]
    url = hf_hub_url(repo, index[name])
    n_header = struct.unpack("<Q", _get(url, 0, 8))[0]
    meta = json.loads(_get(url, 8, n_header))[name]
    assert meta["dtype"] == "BF16" and meta["shape"] == [200192, 3072], meta
    dim = meta["shape"][1]
    start = 8 + n_header + meta["data_offsets"][0]
    ids = sorted(set(ids))
    runs = [[ids[0]]]
    for i in ids[1:]:
        if i - runs[-1][-1] <= 64:
            runs[-1].append(i)
        else:
            runs.append([i])
    out = {}
    for run in runs:
        lo, hi = run[0], run[-1]
        raw = _get(url, start + lo * dim * 2, (hi - lo + 1) * dim * 2)
        block = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(-1, dim)
        out.update({i: block[i - lo] for i in run})
    return out


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=TRUEBASE)
    base = ap.parse_args().base
    out_path = Path("replication/onboarding/trinity-large_embed_rows"
                    + ("" if base == TRUEBASE else "_" + base.split("/")[1]) + ".csv")
    tok = AutoTokenizer.from_pretrained(FINAL)
    ids = {s: tok.encode(s, add_special_tokens=False) for s in MARKERS + CONTEXT}
    assert all(len(v) == 1 for v in ids.values()), ids
    ids = {s: v[0] for s, v in ids.items()}
    base_names = {i: t for t, i in AutoTokenizer.from_pretrained(base).get_added_vocab().items()}
    want = list(ids.values()) + RESERVED + ORDINARY
    rows, ref, mean = {}, {}, {}
    for repo in (base, FINAL):
        rows[repo] = embedding_rows(repo, want)
        ordinary = np.stack([rows[repo][i] for i in ORDINARY])
        ref[repo] = float(np.median(np.linalg.norm(ordinary, axis=1)))
        mean[repo] = ordinary.mean(0)
        to_mean = [cos(r, mean[repo]) for r in ordinary]
        print(f"{repo}: median ordinary norm {ref[repo]:.4f}; ordinary rows' cosine with their mean: "
              f"median {np.median(to_mean):.3f}, 5th pct {np.percentile(to_mean, 5):.3f}")
    moved = [cos(rows[base][i], rows[FINAL][i]) for i in ORDINARY]
    print(f"ordinary rows' cosine base vs final: median {np.median(moved):.4f}, 5th-95th pct "
          f"{np.percentile(moved, 5):.4f}-{np.percentile(moved, 95):.4f}")
    recs = []
    for kind, group in (("marker", {s: ids[s] for s in MARKERS}), ("context", {s: ids[s] for s in CONTEXT}),
                        ("reserved", {base_names[i]: i for i in RESERVED})):
        for token, i in group.items():
            rec = dict(base=base, kind=kind, id=i, token=tok.convert_ids_to_tokens(i), base_token=base_names.get(i, ""))
            for repo, tag in ((base, "base"), (FINAL, "final")):
                rec[f"{tag}_rel_norm"] = float(np.linalg.norm(rows[repo][i])) / ref[repo]
                rec[f"{tag}_cos_mean"] = cos(rows[repo][i], mean[repo])
            rec["cos_base_final"] = cos(rows[base][i], rows[FINAL][i])
            recs.append(rec)
    cols = list(recs[0])[1:]
    print("  ".join(cols))
    for rec in recs:
        print("  ".join(f"{rec[c]:.3f}" if isinstance(rec[c], float) else str(rec[c]) for c in cols))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, list(recs[0]))
        w.writeheader()
        w.writerows({c: f"{v:.4f}" if isinstance(v, float) else v for c, v in rec.items()} for rec in recs)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
