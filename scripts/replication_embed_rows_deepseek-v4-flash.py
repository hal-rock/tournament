"""Did DeepSeek-V4-Flash-Base train the chat markers our prompts use? Reads input-embedding rows of the base and of
DeepSeek-V4-Flash straight from the HF checkpoints with HTTP range requests (~50 MB in all, no weights), as in
../self-modeling/scripts/embed_rows.py.

    .venv/bin/python scripts/replication_embed_rows_deepseek-v4-flash.py   # -> replication/onboarding/deepseek-v4-flash_embed_rows.csv

The embedding (`embed.weight`, [129280, 4096]) is BF16 in both checkpoints; only linear and expert weights are FP8/FP4.
Both stages share one tokenizer.json. The prompts use BOS, <｜User｜>, <｜Assistant｜> and </think> (chat mode). For each row:
- rel_norm: L2 norm over the median norm of 2,000 ordinary BPE rows (ids 1000-1999 and 100000-100999), same model;
- cos_mean: cosine with the mean ordinary row. Trained rows share the embedding's common direction; a row still at
  its random initialization has cosine ~0 with it;
- cos_base_final: cosine between the base's and the final's row (1.0: post-training never touched it);
- cos_ph_mean: cosine with the mean of the 800 <｜place▁holder▁no▁N｜> rows (ids 128000-128799, never in the data).
  These don't sit near zero as in Trinity: they share one offset (each has cosine 0.45-0.50 with their mean, ordinary
  rows 0.04-0.13), so a row still at its initialization would show up as cos_ph_mean ~0.45-0.50.
"""

import csv
import json
import os
import struct
import urllib.request
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download, hf_hub_url
from transformers import AutoTokenizer

BASE, FINAL = "deepseek-ai/DeepSeek-V4-Flash-Base", "deepseek-ai/DeepSeek-V4-Flash"
EMBED = "embed.weight"
MARKERS = ["<｜begin▁of▁sentence｜>", "<｜User｜>", "<｜Assistant｜>", "</think>"]  # every marker the prompts use
CONTEXT = ["<think>", "<｜end▁of▁sentence｜>", "<｜▁pad▁｜>"]  # not in the prompts; for comparison
ORDINARY = list(range(1000, 2000)) + list(range(100000, 101000))
PLACEHOLDERS = list(range(128000, 128800))
OUT = Path("replication/onboarding/deepseek-v4-flash_embed_rows.csv")


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
    assert meta["dtype"] == "BF16" and meta["shape"] == [129280, 4096], meta
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
    tok = AutoTokenizer.from_pretrained(FINAL)
    ids = {s: tok.encode(s, add_special_tokens=False) for s in MARKERS + CONTEXT}
    assert all(len(v) == 1 for v in ids.values()), ids
    ids = {s: v[0] for s, v in ids.items()}
    want = list(ids.values()) + PLACEHOLDERS + ORDINARY
    rows, ref, mean, ph_mean = {}, {}, {}, {}
    for repo in (BASE, FINAL):
        rows[repo] = embedding_rows(repo, want)
        ph_mean[repo] = np.stack([rows[repo][i] for i in PLACEHOLDERS]).mean(0)
        ordinary = np.stack([rows[repo][i] for i in ORDINARY])
        ref[repo] = float(np.median(np.linalg.norm(ordinary, axis=1)))
        mean[repo] = ordinary.mean(0)
        to_mean = [cos(r, mean[repo]) for r in ordinary]
        print(f"{repo}: median ordinary norm {ref[repo]:.4f}; ordinary rows' cosine with their mean: "
              f"median {np.median(to_mean):.3f}, 5th pct {np.percentile(to_mean, 5):.3f}")
    moved = [cos(rows[BASE][i], rows[FINAL][i]) for i in ORDINARY]
    print(f"ordinary rows' cosine base vs final: median {np.median(moved):.4f}, 5th-95th pct "
          f"{np.percentile(moved, 5):.4f}-{np.percentile(moved, 95):.4f}")
    recs = []
    for kind, group in (("marker", {s: ids[s] for s in MARKERS}), ("context", {s: ids[s] for s in CONTEXT})):
        for token, i in group.items():
            rec = dict(kind=kind, id=i, token=token)
            for repo, tag in ((BASE, "base"), (FINAL, "final")):
                rec[f"{tag}_rel_norm"] = float(np.linalg.norm(rows[repo][i])) / ref[repo]
                rec[f"{tag}_cos_mean"] = cos(rows[repo][i], mean[repo])
                rec[f"{tag}_cos_ph_mean"] = cos(rows[repo][i], ph_mean[repo])
            rec["cos_base_final"] = cos(rows[BASE][i], rows[FINAL][i])
            recs.append(rec)
    # The placeholders, summarized (median and range over the 800 rows), plus how many share one vector.
    rec = dict(kind="placeholder", id=f"{PLACEHOLDERS[0]}-{PLACEHOLDERS[-1]}", token="<｜place▁holder▁no▁N｜> (median)")
    for repo, tag in ((BASE, "base"), (FINAL, "final")):
        norms = [float(np.linalg.norm(rows[repo][i])) / ref[repo] for i in PLACEHOLDERS]
        cm = [cos(rows[repo][i], mean[repo]) for i in PLACEHOLDERS]
        n_distinct = len({rows[repo][i].tobytes() for i in PLACEHOLDERS})
        print(f"{repo} placeholders: rel_norm median {np.median(norms):.3f} (range {min(norms):.3f}-{max(norms):.3f}), "
              f"cos_mean median {np.median(cm):.3f} (range {min(cm):.3f}-{max(cm):.3f}), {n_distinct} distinct rows")
        rec[f"{tag}_rel_norm"], rec[f"{tag}_cos_mean"] = float(np.median(norms)), float(np.median(cm))
        rec[f"{tag}_cos_ph_mean"] = float(np.median([cos(rows[repo][i], ph_mean[repo]) for i in PLACEHOLDERS]))
    rec["cos_base_final"] = float(np.median([cos(rows[BASE][i], rows[FINAL][i]) for i in PLACEHOLDERS]))
    recs.append(rec)
    cols = list(recs[0])
    print("  ".join(cols))
    for r in recs:
        print("  ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows({c: f"{v:.4f}" if isinstance(v, float) else v for c, v in r.items()} for r in recs)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
