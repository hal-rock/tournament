"""Outcome mass and top next tokens of report conditions (prompts.py), on the first pilot pairs, to choose which wordings to
run in full (report_patch.py). The prompts are built exactly as report_patch.py builds them, from the same config (a
`model`, or a replication model through `model_from`). Reports only mass and top tokens, nothing about how the answers
relate to the coin or the tasks. (The replication's models get this check in vLLM instead, during their pilot:
replication.WORDING_CHECK.)

Usage: uv run python scripts/report_wording_check.py configs/coin_report/<model>.yaml [--n 24] [--conditions a b ...]
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze import read_jsonl  # noqa: E402
from prompts import CONDITIONS, build_prompt, outcome_token_ids  # noqa: E402
from report_patch import load_tokenizers, pilot_pairs, prompt_format, resolve_model  # noqa: E402
from tournament import load_model, select_pairs  # noqa: E402

DEFAULT = ["coin_report_bold", "coin_report_short", "coin_report_short_bold",
           "coin_report_short_land", "coin_report_short_land_bold", "coin_predict", "coin_predict_bold",
           "coin_predict_think", "coin_predict_think_bold", "coin_predict_my", "coin_predict_my_bold"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--conditions", nargs="+", help=f"default: {' '.join(DEFAULT)} (none with --try-prefill)")
    ap.add_argument("--try-prefill", nargs="+", default=[], metavar="BASE=PREFILL",
                    help="ad hoc variants: BASE's prompt with another prefill, plain and bold (e.g. coin_report_short='I would')")
    a = ap.parse_args()
    a.conditions = a.conditions or ([] if a.try_prefill else DEFAULT)
    for k, spec in enumerate(a.try_prefill):  # registered under try<k> / try<k>_bold, for this check only
        base, prefill = spec.split("=", 1)
        CONDITIONS[f"try{k}"] = {**CONDITIONS[base], "prefill": prefill, "outcomes": (" heads", " tails")}
        CONDITIONS[f"try{k}_bold"] = {**CONDITIONS[base], "prefill": prefill + " **", "outcomes": ("heads", "tails")}
        print(f"try{k}: {base} with prefill {prefill!r}")
        a.conditions += [f"try{k}", f"try{k}_bold"]
    cfg = yaml.safe_load(open(a.config))
    m = resolve_model(cfg)  # `model`, or a replication config's entry (`model_from`), as report_patch.py reads it
    tok, template_tok = load_tokenizers(m)
    fmt = prompt_format(m)
    model = load_model(m["name"], "cuda", torch.bfloat16, revision=m.get("revision"), max_memory=m.get("max_memory"),
                       loader=m.get("loader"), attn_implementation=m.get("attn_implementation"))
    tasks = read_jsonl(cfg["tasks_file"])
    pairs = pilot_pairs(len(tasks), cfg["n_pairs"], select_pairs(len(tasks), 15000, 0), cfg.get("seed", 0))[:a.n]
    print(f"{cfg['family']}: {len(pairs)} pilot pairs, order 0", flush=True)
    for name in a.conditions:
        out_ids = outcome_token_ids(name, tok)
        masses, tops = [], {}
        for i, j in pairs:
            text = build_prompt(name, tasks[i]["prompt"], tasks[j]["prompt"], template_tok, **fmt)
            ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
            with torch.no_grad():
                lp = torch.log_softmax(model(ids, use_cache=False).logits[0, -1].float(), -1)
            masses.append(lp[out_ids].exp().sum().item())
            for t in lp.topk(5).indices.tolist():
                tops[t] = tops.get(t, 0) + lp[t].exp().item() / len(pairs)
        top = sorted(tops.items(), key=lambda kv: -kv[1])[:8]
        masses.sort()
        print(f"{name:24} median mass {masses[len(masses) // 2]:.2f} (min {masses[0]:.2f})  top: "
              + "  ".join(f"{tok.decode([t])!r} {p:.2f}" for t, p in top), flush=True)
    for name in a.conditions:  # placeholder tasks: the log stays free of task text
        print(f"\n{name}:\n" + build_prompt(name, "<task A>", "<task B>", template_tok, **fmt)[-160:])


if __name__ == "__main__":
    main()
