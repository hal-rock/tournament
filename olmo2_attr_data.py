"""The OLMo 2 SFT conversations behind the attribution sample, and content labels for them (README, "OLMo 2 32B SFT data
attribution"). Run on the laptop; the labels are written before any attribution score exists.

  fetch  the sample's rows of allenai/tulu-3-sft-olmo-2-mixture-0225 @ d91a078, from its published parquet files in
         order (the order load_dataset gives, which is what the sample's idx counts), with each row's
         id checked against the sample. -> outputs/olmo2_attr/sft_rows.jsonl (third-party chat text: gitignored; the
         pod scorer reads it, via .podinclude)
  label  one Claude call per scored conversation, six fixed fields (LABEL_FIELDS), through the Batches API, then any
         refusal or error again with server-side fallbacks. -> outputs/olmo2_attr/labels_raw.jsonl and
         data/olmo2_attr/labels.csv (idx and labels only, no text)

    uv run --with pyarrow python olmo2_attr_data.py fetch
    uv run python olmo2_attr_data.py label
"""

import csv
import json
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "olmo2_attr"
OUT = ROOT / "outputs" / "olmo2_attr"
SFT_DATASET = "allenai/tulu-3-sft-olmo-2-mixture-0225"
SFT_REVISION = "d91a0785ade02942520280fb484866fce41e448f"
SCORED_SETS = ("strat", "refusal", "refusal_ctrl", "selfref", "selfref_ctrl")
ROWS = OUT / "sft_rows.jsonl"


# ---------------------------------------------------------------- fetch

def fetch():
    """Downloads the six parquet files whole (1.3 GB; ranged reads of every row group were far slower), then keeps the
    sample's rows."""
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download
    sample = json.load(open(DATA / "sft_sample.json"))
    want = {int(r["idx"]): r for r in sample}
    local = Path(snapshot_download(SFT_DATASET, repo_type="dataset", revision=SFT_REVISION, allow_patterns=["data/*.parquet"],
                                   local_dir=OUT / "sft_parquet"))
    files = sorted((local / "data").glob("*.parquet"))
    assert len(files) == 6, files
    OUT.mkdir(parents=True, exist_ok=True)
    got, start = 0, 0
    with open(ROWS, "w") as fh:
        for f in files:
            pf = pq.ParquetFile(f)
            for g in range(pf.num_row_groups):
                n = pf.metadata.row_group(g).num_rows
                hits = [i for i in range(start, start + n) if i in want]
                if hits:
                    tab = pf.read_row_group(g, columns=["id", "messages", "source"]).to_pylist()
                    for i in hits:
                        row = tab[i - start]
                        assert row["id"] == want[i]["id"], (i, row["id"], want[i]["id"])
                        fh.write(json.dumps(dict(idx=i, id=row["id"], source=row["source"].split("/")[-1], messages=row["messages"])) + "\n")
                        got += 1
                start += n
            print(f"  {f.name}: {start:,} rows read, {got} of {len(want)} kept", flush=True)
    assert got == len(want), (got, len(want))
    print(f"wrote {ROWS}")


def read_rows():
    return {r["idx"]: r for r in map(json.loads, open(ROWS))}


# ---------------------------------------------------------------- labels

class Labels(BaseModel):
    task_type: Literal["math", "coding", "factual_qa", "explanation", "advice", "creative_writing", "roleplay", "text_task",
                       "opinion", "harmful_request", "chit_chat", "other"]
    response_stance: Literal["complies", "partial", "refuses"]
    voice: Literal["impersonal", "assistant", "character"]
    own_view: bool = Field(description="the reply states the assistant's own opinion, preference, value judgment or feeling")
    mentions_ai: bool = Field(description="the reply refers to the assistant being an AI, or to its nature, makers, abilities or limits")
    moral_language: bool = Field(description="the reply makes an ethical, safety or social-value judgment")


LABEL_FIELDS = list(Labels.model_fields)

SYSTEM = """You label examples from a language model's supervised fine-tuning dataset, for AI research on how kinds of training data shape a model's behaviour. Each example is a conversation in which the final assistant turns are the training targets. You are classifying the data, not producing or evaluating content, and some examples contain requests or replies about harmful topics; label them like any other.

Fields (judge the assistant's replies, and the user's request only for task_type):

task_type, the main kind of request:
- math: solving or explaining a math problem
- coding: writing, explaining or debugging code
- factual_qa: a factual or knowledge question with a short answer
- explanation: explaining a concept or process, or how to do something
- advice: personal, practical or professional advice or recommendations
- creative_writing: stories, poems, scripts, jokes, slogans and other original writing
- roleplay: the assistant plays a character, persona or role in an interaction
- text_task: translating, summarizing, rewriting, editing, extracting, classifying or formatting given text or data
- opinion: asking for the assistant's opinion, view or preference, or for a discussion of a contested question
- harmful_request: asking for something harmful, dangerous, unethical or against usage policies, including jailbreak attempts
- chit_chat: greetings, small talk, or questions about the assistant itself
- other: none of these

response_stance: complies (does what was asked), partial (complies with caveats or warnings, or only part of the request), refuses (declines, or redirects to something else).

voice: impersonal (the reply is the content itself, such as a solution, code, an answer or a piece of text, with little first-person or conversational framing), assistant (the reply speaks conversationally as an AI assistant to the user, with first person, direct address, offers or framing), character (the reply speaks as a fictional character, a persona, or a human role other than an AI assistant).

own_view: true if the reply states the assistant's own opinion, preference, value judgment or feeling in its own voice ("I think", "I'd recommend", "I love that", "I find this fascinating"). Boilerplate such as "I'm happy to help" or "Sure, here is" does not count, and neither do opinions voiced by a character.

mentions_ai: true if the reply mentions the assistant being an AI or language model, or its nature, makers, training, abilities or limits.

moral_language: true if the reply makes an ethical, safety or social-value judgment, e.g. that something is harmful, wrong, disrespectful, or important to handle responsibly.

Reply with the JSON object only."""


def clip(text, n):
    text = text or ""
    return text if len(text) <= n else text[: n * 2 // 3] + "\n[... cut ...]\n" + text[-n // 3:]


def render(messages, per_message=1500, total=6000):
    parts, used = [], 0
    for i, m in enumerate(messages):
        body = clip(m.get("content"), per_message)
        if used + len(body) > total and i < len(messages) - 1:
            parts.append(f"[... {len(messages) - i - 1} more turns cut; the last one follows ...]")
            m = messages[-1]
            parts.append(f"<{m['role']}>\n{clip(m.get('content'), per_message)}\n</{m['role']}>")
            break
        parts.append(f"<{m['role']}>\n{body}\n</{m['role']}>")
        used += len(body)
    return "\n\n".join(parts)


def label():
    from matskit.claude import Claude
    sample = [r for r in json.load(open(DATA / "sft_sample.json")) if r["set"] in SCORED_SETS]
    rows = read_rows()
    reqs = {f"u{r['idx']}": dict(prompt=f"Conversation ({sum(m['role'] == 'assistant' for m in rows[r['idx']]['messages'])} assistant turns):\n\n"
                                        + render(rows[r["idx"]]["messages"]), schema=Labels)
            for r in sample}
    c = Claude(effort="low", max_tokens=4000, system=SYSTEM, label="olmo2_attr_labels")
    raw = OUT / "labels_raw.jsonl"
    res = c.batch(reqs, state=OUT / "labels_batch.json", out=raw)
    bad = {k: reqs[k] for k, r in res.items() if not r.ok}
    print(f"batch: {len(res) - len(bad)} ok, {len(bad)} refused or failed; retrying those with fallbacks", flush=True)
    if bad:
        res.update(c.map(bad, out=raw, workers=8))
    fails = {k: r.refusal or r.error for k, r in res.items() if not r.ok}
    print(f"labelled {len(res) - len(fails)} of {len(reqs)}; still missing {len(fails)}: {sorted(set(fails.values()))}")
    with open(DATA / "labels.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "labelled_by"] + LABEL_FIELDS + ["missing_reason"])
        for k in sorted(reqs, key=lambda s: int(s[1:])):
            r = res[k]
            d = r.data.model_dump() if r.ok else {}
            w.writerow([int(k[1:]), r.model if r.ok else ""] + [d.get(f, "") for f in LABEL_FIELDS] + ["" if r.ok else (r.refusal or r.error)])
    print(f"wrote {DATA / 'labels.csv'}; spent ${c.spent:.2f}")


if __name__ == "__main__":
    {"fetch": fetch, "label": label}.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: sys.exit(__doc__))()
