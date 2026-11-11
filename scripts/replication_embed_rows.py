"""Did a base model train the chat markers our prompts use? (replication onboarding, PLAN_replication.md)

Reads rows of the base checkpoint's input embedding (and LM head) straight from Hugging Face with HTTP range requests,
megabytes rather than the whole model, and compares each marker's row norm with ordinary tokens'. A row never trained
keeps its initialization: often ~0, one vector shared by many tokens, or random at the initialization scale. Unused ids
of the same checkpoint (reserved tokens, padding rows past the tokenizer) show what "never trained" looks like there.
Adapted from ../self-modeling/scripts/embed_rows.py (Qwen3, GLM-4.5-Air, Trinity, Llama 3.1 405B).

    uv run python scripts/replication_embed_rows.py --label kimi-k2 \\
        --base moonshotai/Kimi-K2-Base --revision ce72df012259dcc55d945e890f815fe7ef69159c \\
        --markers '<|im_user|>' '<|im_middle|>' '<|im_end|>' '<|im_assistant|>' '<|im_system|>' \\
        --also '[BOS]' '[EOS]' --unused 163602:163838 \\
        --final moonshotai/Kimi-K2-Instruct --final-revision fd1984e2b7a3350dbf7305fe73a4ede25c14de50

`--final` also reads the post-trained model's rows and reports how far each moved from the base's: a token
post-training used as input (or as a target, for the head) moves like ordinary tokens; one it never used stays put.

Token ids come from the tokenizer files' added-token lists (tokenizer.json, else tokenizer_config.json), so nothing is
executed: no remote code, no tokenizer class. `--tokens-from` reads them from another repo (the base usually has the
same ones). Ordinary reference rows are blocks of 64 ids spread over the ordinary vocabulary, below the first added id
past 1000. Embeddings stored as BF16/F16/F32 are read as they are; FP8 (F8_E4M3) is decoded and multiplied by its
`weight_scale_inv` block scales (DeepSeek layout) or per-row/per-tensor `weight_scale`.

Writes one CSV row per (matrix, id): replication/onboarding/<label>_embed_rows.csv by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import struct
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download, hf_hub_url

EMBED_NAMES = ["model.embed_tokens.weight", "model.language_model.embed_tokens.weight",
               "transformer.embedding.word_embeddings.weight", "backbone.embeddings.weight", "embed.weight"]
HEAD_NAMES = ["lm_head.weight", "model.language_model.lm_head.weight", "output_layer.weight", "head.weight"]
ROW_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1}
BLOCKS = (0.05, 0.25, 0.5, 0.75)  # where the ordinary reference blocks start, as fractions of the ordinary vocabulary
BLOCK_ROWS = 64


def _get(url: str, start: int, n: int) -> bytes:
    headers = {"Range": f"bytes={start}-{start + n - 1}"}
    if os.environ.get("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=300) as r:
        data = r.read()
    assert len(data) == n, (len(data), n)  # a server ignoring Range would send the whole file
    return data


class Checkpoint:
    """A sharded safetensors checkpoint on the Hub, read by range requests."""

    def __init__(self, repo: str, revision: str | None):
        self.repo, self.revision = repo, revision
        path = hf_hub_download(repo, "model.safetensors.index.json", revision=revision)
        self.index = json.loads(Path(path).read_text())["weight_map"]
        self._headers: dict[str, tuple[int, dict]] = {}

    def header(self, filename: str) -> tuple[int, dict]:
        if filename not in self._headers:
            url = hf_hub_url(self.repo, filename, revision=self.revision)
            n = struct.unpack("<Q", _get(url, 0, 8))[0]
            self._headers[filename] = (8 + n, json.loads(_get(url, 8, n)))
        return self._headers[filename]

    def meta(self, name: str) -> dict:
        start, header = self.header(self.index[name])
        return dict(header[name], data_start=start, url=hf_hub_url(self.repo, self.index[name], revision=self.revision))

    def find(self, names: list[str]) -> str | None:
        return next((n for n in names if n in self.index), None)

    def raw_rows(self, name: str, ids: list[int]) -> np.ndarray:
        """Rows `ids` of a 2-D tensor as stored (float32 for float dtypes, uint8 codes for FP8), in the order given."""
        m = self.meta(name)
        n_rows, dim = m["shape"]
        assert all(0 <= i < n_rows for i in ids), f"ids outside {name}'s {n_rows} rows"
        size = ROW_BYTES[m["dtype"]]
        out = {}
        # One request per run of nearby ids (gaps under 64 rows are read through).
        order = sorted(set(ids))
        runs, lo, prev = [], order[0], order[0]
        for i in order[1:]:
            if i - prev > 64:
                runs.append((lo, prev))
                lo = i
            prev = i
        runs.append((lo, prev))
        for lo, hi in runs:
            raw = _get(m["url"], m["data_start"] + m["data_offsets"][0] + lo * dim * size, (hi - lo + 1) * dim * size)
            block = _decode(raw, m["dtype"]).reshape(hi - lo + 1, dim)
            for i in range(lo, hi + 1):
                out[i] = block[i - lo]
        return np.stack([out[i] for i in ids])

    def whole(self, name: str) -> np.ndarray:
        """A small tensor (a scale), all of it, as float32."""
        m = self.meta(name)
        lo, hi = m["data_offsets"]
        return _decode(_get(m["url"], m["data_start"] + lo, hi - lo), m["dtype"]).reshape(m["shape"])

    def rows(self, name: str, ids: list[int]) -> np.ndarray:
        """Rows `ids` of a weight matrix as float32, dequantized if stored as FP8."""
        m = self.meta(name)
        rows = self.raw_rows(name, ids)
        if m["dtype"] != "F8_E4M3":
            return rows
        rows = E4M3[rows]
        n_rows, dim = m["shape"]
        inv, per = name.replace(".weight", ".weight_scale_inv"), name.replace(".weight", ".weight_scale")
        if inv in self.index:  # DeepSeek layout: one float32 per (b x b) block, multiplied in
            s = self.whole(inv)
            b = math.ceil(n_rows / s.shape[0])
            assert s.shape == (math.ceil(n_rows / b), math.ceil(dim / b)), (s.shape, b)
            return np.stack([r * np.repeat(s[i // b], b)[:dim] for i, r in zip(ids, rows)])
        if per in self.index:  # compressed-tensors layout: one scale for the tensor, or one per row
            s = self.whole(per).reshape(-1)
            assert s.size in (1, n_rows), s.shape
            return rows * (s[0] if s.size == 1 else s[ids][:, None])
        raise NotImplementedError(f"{name} is FP8 with no scale tensor this script knows")


def _decode(raw: bytes, dtype: str) -> np.ndarray:
    if dtype == "BF16":
        return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
    if dtype == "F16":
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    if dtype == "F32":
        return np.frombuffer(raw, dtype=np.float32).copy()
    if dtype == "F8_E4M3":
        return np.frombuffer(raw, dtype=np.uint8).copy()  # codes; _decode_e4m3 turns them into values
    raise NotImplementedError(dtype)


def _e4m3_table() -> np.ndarray:
    """float8_e4m3fn: 1 sign, 4 exponent (bias 7), 3 mantissa bits; no infinities, 0x7F/0xFF are NaN."""
    codes = np.arange(256)
    sign = np.where(codes & 0x80, -1.0, 1.0)
    exp, man = (codes >> 3) & 0xF, codes & 0x7
    value = np.where(exp == 0, man / 8 * 2.0 ** -6, (1 + man / 8) * 2.0 ** (exp - 7))
    value = sign * value
    value[(codes & 0x7F) == 0x7F] = np.nan
    return value.astype(np.float32)


E4M3 = _e4m3_table()


def added_tokens(repo: str, revision: str | None) -> dict[int, str]:
    """id -> token string for the tokenizer's added tokens, from its JSON files (no tokenizer code is run)."""
    try:
        tj = json.loads(Path(hf_hub_download(repo, "tokenizer.json", revision=revision)).read_text())
        return {t["id"]: t["content"] for t in tj["added_tokens"]}
    except Exception:  # no tokenizer.json (tiktoken- or sentencepiece-based repos)
        cfg = json.loads(Path(hf_hub_download(repo, "tokenizer_config.json", revision=revision)).read_text())
        return {int(i): t["content"] if isinstance(t, dict) else str(t) for i, t in cfg["added_tokens_decoder"].items()}


def parse_range(s: str) -> list[int]:
    lo, hi = (int(x) for x in s.split(":"))
    return list(range(lo, hi))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True, help="family key, e.g. kimi-k2 (names the CSV)")
    ap.add_argument("--base", required=True, help="base checkpoint repo")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--tokens-from", default=None, help="repo[@revision] whose tokenizer files list the added tokens")
    ap.add_argument("--markers", nargs="+", required=True, help="the special tokens our prompts use")
    ap.add_argument("--also", nargs="*", default=[], help="other added tokens to print next to the markers (e.g. BOS)")
    ap.add_argument("--unused", nargs="*", default=[], help="id ranges lo:hi of tokens never used (untrained reference)")
    ap.add_argument("--final", default=None, help="the post-trained repo: also read its rows and how far each moved")
    ap.add_argument("--final-revision", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tok_repo, _, tok_rev = (args.tokens_from or args.base).partition("@")
    added = added_tokens(tok_repo, tok_rev or (args.revision if not args.tokens_from else None))
    ids_of = {t: i for i, t in added.items()}
    missing = [t for t in args.markers if t not in ids_of]
    assert not missing, f"not added tokens of {tok_repo}: {missing}"

    ckpt = Checkpoint(args.base, args.revision)
    final = Checkpoint(args.final, args.final_revision) if args.final else None
    matrices = {"embed": ckpt.find(EMBED_NAMES), "head": ckpt.find(HEAD_NAMES)}
    assert matrices["embed"], f"no input embedding among {EMBED_NAMES}"
    n_rows = ckpt.meta(matrices["embed"])["shape"][0]
    first_added = min(i for i in added if i >= 1000)  # the ordinary BPE vocabulary ends below it
    ordinary = [i for f in BLOCKS for i in range(int(f * first_added), int(f * first_added) + BLOCK_ROWS)
                if i not in added]
    unused = sorted({i for r in args.unused for i in parse_range(r)} - set(ids_of[t] for t in args.markers))
    unused = [i for i in unused if i < n_rows]
    specials = sorted(i for i in added if i < n_rows and i not in set(unused))  # the other added tokens, markers flagged
    print(f"{args.base}@{args.revision or 'main'}: {n_rows} embedding rows, {len(added)} added tokens, "
          f"ordinary vocabulary below {first_added}")

    recs = []
    for matrix, name in matrices.items():
        if name is None:
            print(f"\n(no {matrix} matrix among {HEAD_NAMES if matrix == 'head' else EMBED_NAMES})")
            continue
        m = ckpt.meta(name)
        print(f"\n{matrix}: {name} {m['dtype']} {m['shape']} in {ckpt.index[name]}")
        groups = {"ordinary": ordinary, "unused": unused, "added": specials}
        rows = {g: ckpt.rows(name, ids) for g, ids in groups.items() if ids}
        ref = float(np.median(np.linalg.norm(rows["ordinary"], axis=1)))
        # A row identical to other rows was never moved from a shared initialization.
        count = Counter(r.tobytes() for g in rows for r in rows[g])
        # The post-trained model's rows: their norms (against its own ordinary median) and how far each moved from the
        # base's, in units of the base's ordinary median norm. A row post-training never used stays put.
        post = {g: final.rows(name, ids) for g, ids in groups.items() if ids} if final and name in final.index else None
        post_ref = float(np.median(np.linalg.norm(post["ordinary"], axis=1))) if post else None
        mine = []
        for g, ids in groups.items():
            for k, (i, r) in enumerate(zip(ids, rows.get(g, []))):
                norm = float(np.linalg.norm(r))
                rec = dict(label=args.label, base=args.base, revision=args.revision or "", matrix=matrix, kind=g,
                           id=i, token=added.get(i, ""), marker=added.get(i) in args.markers, norm=norm,
                           rel_norm=norm / ref, n_identical=count[r.tobytes()], ref_norm=ref)
                if post:
                    p = post[g][k]
                    rec.update(final_rel_norm=float(np.linalg.norm(p)) / post_ref,
                               moved=float(np.linalg.norm(p - r)) / ref)
                mine.append(rec)
        recs += mine
        rel = lambda kind, lo=0, hi=10**9: np.array([r["rel_norm"] for r in mine
                                                       if r["kind"] == kind and lo <= r["id"] < hi])
        blocks = [np.median(rel("ordinary", int(f * first_added), int(f * first_added) + BLOCK_ROWS)) for f in BLOCKS]
        o = rel("ordinary")
        print(f"  ordinary rows ({len(o)}): median norm {ref:.4f} (= 1.0); median per block "
              + ", ".join(f"{b:.2f}" for b in blocks) + f"; 5-95% {np.quantile(o, 0.05):.2f}-{np.quantile(o, 0.95):.2f}")
        u = rel("unused")
        if len(u):
            n_dup = sum(r["n_identical"] > 1 for r in mine if r["kind"] == "unused")
            print(f"  unused ids ({len(u)}): rel. norm median {np.median(u):.4f}, range {u.min():.4f}-{u.max():.4f}; "
                  f"identical to another row: {n_dup}")
        others = np.array([r["rel_norm"] for r in mine if r["kind"] == "added" and not r["marker"]])
        if len(others):
            print(f"  other added tokens ({len(others)}): rel. norm median {np.median(others):.4f}, "
                  f"range {others.min():.4f}-{others.max():.4f} (all in the CSV)")
        if post:
            moved = lambda kind: np.array([r["moved"] for r in mine if r["kind"] == kind])
            print(f"  {args.final}: ordinary median norm {post_ref:.4f}; moved from base (in base ordinary-median units): "
                  f"ordinary median {np.median(moved('ordinary')):.4f}"
                  + (f", unused median {np.median(moved('unused')):.4f}" if len(u) else ""))
        cols = f"  {'id':>7}  {'token':24} {'rel_norm':>9} {'n_identical':>11}" + (
            f" {'final_rel_norm':>14} {'moved':>8}" if post else "")
        print("  markers (* = also shown):\n" + cols)
        for r in mine:
            if r["marker"] or r["token"] in args.also:
                print(f"  {r['id']:>7}  {(r['token'] + (' *' if not r['marker'] else '')):24} {r['rel_norm']:9.4f} "
                      f"{r['n_identical']:11d}" + (f" {r['final_rel_norm']:14.4f} {r['moved']:8.4f}" if post else ""))

    out = Path(args.out or f"replication/onboarding/{args.label}_embed_rows.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(max(recs, key=len)), restval="")
        w.writeheader()
        w.writerows(recs)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
