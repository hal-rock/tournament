"""Did the base model ever train the chat markers our replication prompts use? Reads input-embedding and LM-head rows
straight from Hugging Face with HTTP range requests (tens of MB, not the model), after
../self-modeling/scripts/embed_rows.py, for the replication configs given:

    python scripts/replication_embed_rows_llama_nemotron.py llama-405b nemotron4-340b
    # -> replication/onboarding/<key>_embed_rows.csv

Rows checked, per config:
- every special (added) token in the rendered prompts of all its conditions, found by tokenizing them as the scorer
  does, plus the template words between them (EXTRA_WORDS);
- reference rows: ordinary BPE tokens 2000-2255 (whose median norm is the unit), 64 random ordinary tokens, and the
  added tokens the prompts never use (reserved / unused ids, mostly never trained);
- the same tokens' rows in the final model (without the 64 random ones, and only the first 16 unused added tokens), to
  see what post-training did to them.
A row never trained keeps its initialization: small next to ordinary rows, exactly zero, or one vector shared by many
ids (n_identical > 1).
"""

import csv
import json
import os
import random
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np
import yaml
from huggingface_hub import hf_hub_download, hf_hub_url
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prompts import build_prompt  # noqa: E402
from replication import FINAL_CONDITIONS, expand_conditions  # noqa: E402

# Template text between the markers, which the model reads as ordinary tokens (in the prompts' own tokenization).
# Nemotron's are pieces: SentencePiece's whitespace symbol and the newline byte.
EXTRA_WORDS = {
    "llama-405b": ["system", "user", "assistant", "\n\n"],
    "nemotron4-340b": ["▁", "System", "User", "Assistant", "<0x0A>"],
}
PIECES = ("▁", "<0x0A>")
EMBED = "model.embed_tokens.weight"
HEAD = "lm_head.weight"
DTYPES = {"BF16": 2, "F16": 2, "F32": 4}
ORDINARY = list(range(2000, 2256))


class Rows:
    """Rows of one weight matrix in a safetensors checkpoint on the Hub, by HTTP range requests."""

    def __init__(self, repo, name):
        index = json.loads(Path(hf_hub_download(repo, "model.safetensors.index.json")).read_text())["weight_map"]
        self.url = hf_hub_url(repo, index[name])
        n_header = struct.unpack("<Q", self._read(0, 8))[0]
        meta = json.loads(self._read(8, n_header))[name]
        self.n_rows, self.dim = meta["shape"]
        self.dtype, self.size = meta["dtype"], DTYPES[meta["dtype"]]
        self.start = 8 + n_header + meta["data_offsets"][0]

    def _read(self, start, n):
        req = urllib.request.Request(self.url, headers={"Authorization": f"Bearer {os.environ['HF_TOKEN']}",
                                                        "Range": f"bytes={start}-{start + n - 1}"})
        with urllib.request.urlopen(req, timeout=300) as r:
            data = r.read()
        assert len(data) == n, (len(data), n)  # a server ignoring Range would send the whole file
        return data

    def get(self, ids):
        """float32 rows for ids, read in contiguous runs (ids within 64 rows of each other share a read)."""
        out, ids = {}, sorted(set(ids))
        runs, run = [], [ids[0]]
        for i in ids[1:]:
            if i - run[-1] <= 64:
                run.append(i)
            else:
                runs.append(run)
                run = [i]
        runs.append(run)
        for run in runs:
            lo, hi = run[0], run[-1]
            raw = self._read(self.start + lo * self.dim * self.size, (hi - lo + 1) * self.dim * self.size)
            if self.dtype == "BF16":
                block = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
            else:
                block = np.frombuffer(raw, dtype=np.float16 if self.size == 2 else np.float32).astype(np.float32)
            block = block.reshape(hi - lo + 1, self.dim)
            out.update({i: block[i - lo] for i in run})
        return out


def prompt_ids(cfg):
    """Ids of one rendered prompt per condition the config runs (final stage: it has every condition)."""
    m = cfg["models"][-1]
    tok = AutoTokenizer.from_pretrained(m.get("tokenizer_from", m["name"]))
    template = AutoTokenizer.from_pretrained(m.get("chat_template_from", m["name"]))
    assert set(expand_conditions(m["conditions"])) == set(FINAL_CONDITIONS)
    seen = set()
    for c in FINAL_CONDITIONS:
        p = build_prompt(c, "FIRST", "SECOND", template, cfg["system_prompt"], turn_format=m.get("turn_format"))
        seen.update(tok(p, add_special_tokens=False).input_ids)
    return tok, seen


def main(keys):
    for key in keys:
        cfg = yaml.safe_load(Path(f"configs/replication/{key}.yaml").read_text())
        tok, seen = prompt_ids(cfg)
        added = {i: t.content for i, t in tok.added_tokens_decoder.items()}
        used = sorted(i for i in seen if i in added)
        words = {}
        for w in EXTRA_WORDS[key]:
            enc = [tok.convert_tokens_to_ids(w)] if w in PIECES else tok.encode(w, add_special_tokens=False)
            assert len(enc) == 1 and enc[0] in seen, (w, enc)
            words[enc[0]] = w
        unused = sorted(i for i in added if i not in seen)
        rng = random.Random(0)
        random_ids = sorted(rng.sample([i for i in range(len(tok)) if i not in added and i > 3000], 64))
        rows = []
        for stage, m in (("base", cfg["models"][0]), ("final", cfg["models"][-1])):
            groups = {"marker": used, "template word": sorted(words), "ordinary": ORDINARY,
                      "random ordinary": random_ids if stage == "base" else [],
                      "unused added": unused if stage == "base" else unused[:16]}
            for matrix in (EMBED, HEAD):
                got = Rows(m["name"], matrix).get([i for g in groups.values() for i in g])
                unit = float(np.median([np.linalg.norm(got[i]) for i in ORDINARY]))
                # Identical rows share one initialization (+ 0.0 makes -0.0 and 0.0 the same bytes).
                keys_ = {i: (v + 0.0).tobytes() for i, v in got.items()}
                count = {}
                for k in keys_.values():
                    count[k] = count.get(k, 0) + 1
                for group, ids in groups.items():
                    for i in ids:
                        norm = float(np.linalg.norm(got[i]))
                        rows.append(dict(key=key, stage=stage, model=m["name"], matrix=matrix, group=group, id=i,
                                         token=added.get(i, words.get(i, tok.convert_ids_to_tokens(i))), norm=norm,
                                         rel_norm=norm / unit, is_zero=norm == 0, n_identical=count[keys_[i]], unit=unit))
        out = Path(f"replication/onboarding/{key}_embed_rows.csv")
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\n{key} (rel_norm = norm / median norm of ordinary tokens 2000-2255 in the same matrix)")
        for stage in ("base", "final"):
            for matrix in (EMBED, HEAD):
                d = [r for r in rows if r["matrix"] == matrix and r["stage"] == stage]
                print(f"  {stage} {d[0]['model']} {matrix} (unit {d[0]['unit']:.4f})")
                for r in d:
                    if r["group"] in ("marker", "template word") or (
                            r["group"] == "unused added" and r["token"] in (tok.bos_token, tok.eos_token)):
                        print(f"    {r['group']:>13} {r['id']:>7} {r['token']!r:>24} rel_norm {r['rel_norm']:.4f}"
                              f"  n_identical {r['n_identical']}")
                for g in ("random ordinary", "unused added"):
                    sel = [r for r in d if r["group"] == g]
                    rel = np.array([r["rel_norm"] for r in sel])
                    if len(rel):
                        print(f"    {g}: n={len(rel)}, median rel_norm {np.median(rel):.4f} (5-95%: "
                              f"{np.quantile(rel, .05):.4f}-{np.quantile(rel, .95):.4f}), exactly zero: "
                              f"{sum(r['is_zero'] for r in sel)}, sharing a row: {sum(r['n_identical'] > 1 for r in sel)}")
        print(f"  wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
