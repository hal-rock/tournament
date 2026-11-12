"""Transcript page of the SFT conversations that push the coin toward the Assistant's ranking most and least (README, "OLMo 2
32B SFT data attribution"): the top and bottom stratified units by cos on the primary metric (`benign`), at each point scored.

Each conversation is shown as the scorer read it: OLMo 2 DPO's template over the whole conversation (the page notes any unit
whose ids differ from the scorer's, e.g. cut at 4,096 tokens). Header chips give the unit's category, source, rank, cos on the
main queries (in SDs over units) and its content labels. Holds WildChat and other third-party chat text, so it's written under
outputs/ (gitignored).

    uv run --with pandas python olmo2_attr_page.py [--n 25]
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

import olmo2_attr as oa
from matskit import chat, view

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "olmo2_attr"
OUT = ROOT / "outputs" / "olmo2_attr"
SHOW = ["benign", "t1", "t2", "standard", "harm_gap", "asst_benign", "perm1"]
LABELS = ["task_type", "response_stance", "voice", "own_view", "mentions_ai", "moral_language"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(oa.FRAME)
    rows = {r["idx"]: r for r in map(json.loads, open(OUT / "sft_rows.jsonl"))}
    pages = []
    for point in ("base", "sft"):
        p = DATA / f"units_{point}.csv"
        if not p.exists():
            continue
        u = pd.read_csv(p, index_col="idx")
        s = u[u["set"] == "strat"].copy()
        for m in SHOW:
            s[f"z_{m}"] = s[f"cos_{m}"] / s[f"cos_{m}"].std()
        s["rank"] = s.cos_benign.rank(ascending=False).astype(int)
        order = s.sort_values("cos_benign", ascending=False)
        for side, pick in (("top", order.head(args.n)), ("bottom", order.tail(args.n).iloc[::-1])):
            for idx, r in pick.iterrows():
                msgs = [{"role": m["role"], "content": m["content"] or ""} for m in rows[idx]["messages"]]
                meta = {"category": r.category, "source": r.source, "rank": f"{r['rank']} of {len(s)}", "assistant tokens": int(r.n),
                        **{f"{m} (SDs)": f"{r[f'z_{m}']:+.2f}" for m in SHOW}}
                for f in LABELS:
                    if f in r and pd.notna(r[f]):
                        meta[f] = str(r[f])
                t = view.from_messages(tok, msgs, title=f"{point} {side}: #{r['rank']} idx {idx}", group=f"{point} {side}", meta=meta)
                ids, _ = oa.encode_sft(tok, msgs)
                full = chat.encode(tok, chat.render(tok, msgs)).ids
                if list(full[:len(ids)]) != list(ids):
                    t.notes.append("The scorer's token ids differ from this re-render.")
                if len(full) > len(ids):
                    t.notes.append(f"The scorer read only the first {len(ids):,} of {len(full):,} tokens (max length 4,096).")
                pages.append(t)
    note = (f"The {args.n} stratified SFT conversations with the highest and lowest alignment (Adam-weighted cos) with the gradient of the "
            "coin's slope on the SFT Assistant's utilities over the 257 non-harmful tasks, at each point. Groups: point x top/bottom. "
            "Chips give cos on each query in SDs over the 3,040 stratified units, so +2 is two SDs above zero, and the content labels.")
    view.write(OUT / "top_bottom.html", pages, title="OLMo 2 SFT attribution: top and bottom conversations", note=note, view="chat")


if __name__ == "__main__":
    main()
