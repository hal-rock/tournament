"""Score a replication config's conditions (PLAN_replication.md), with vLLM by default.

    uv run python score_replication.py configs/replication/<key>.yaml --stage instruct          # every condition of a stage
    uv run python score_replication.py configs/replication/<key>.yaml --stage base --pilot 24   # 24 pairs per condition
    uv run python score_replication.py configs/replication/<key>.yaml --stage instruct --conditions assistant coinflip
    uv run python score_replication.py <config> --stage instruct --engine hf                   # transformers, as tournament.py

Prompts are built exactly as tournament.py builds them (prompts.build_prompt, tokenized with add_special_tokens=False), on
the sweep's pairs (tournament.select_pairs), each in both orders. Each prompt is one request: vLLM generates one greedy token
and returns, through `logprob_token_ids`, the raw log-softmax over the full vocabulary at the two outcome tokens, the same
quantity as tournament.score's. Prefix caching is on: a condition's prompts share the template, any persona text and the
opening sentence, which vLLM then computes once. The top tokens come from a second pass with `logprobs=top_k` over the first
`top_sample` prompts of each condition (the main pass returns only the greedy token with the two outcomes).

Outputs, under outputs/replication/<run_name>/:
- results/<stage>__<condition>.csv.gz: rows as tournament.py writes them (model, stage, condition, first, second, logp_first,
  logp_second, log_odds, outcome_mass, prompt_tokens, top_tokens), so analyze.py reads this directory like a run's;
- progress.jsonl: one line per finished condition (prompts, seconds, prompt tokens per second, median outcome mass);
- <stage>.wandb_id: the stage's W&B run (group `replication`), resumed after a restart. Each results file is uploaded to it as
  soon as its condition finishes;
- pilot/<stage>__<condition>.csv.gz: --pilot runs (no W&B), with the top tokens of every prompt.

Conditions whose results file exists are skipped, so a crashed stage resumes where it stopped.
"""

import argparse
import gzip
import csv
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

import yaml

from prompts import build_prompt, outcome_token_ids
from replication import expand_conditions

OUT_ROOT = Path("outputs/replication")
SANITY = ("The capital of France is Paris, and the capital of Germany is", "Berlin")


def select_pairs(n_tasks, max_pairs=None, seed=0):
    """tournament.select_pairs, copied so this script doesn't import torch's HF stack (tests/test_score_replication.py
    checks the two agree)."""
    import itertools
    pairs = list(itertools.combinations(range(n_tasks), 2))
    if max_pairs is None or max_pairs >= len(pairs):
        return pairs
    return sorted(random.Random(f"pairs/{seed}").sample(pairs, max_pairs))


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_csv_gz(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tmp.rename(path)  # a crash mid-write never leaves a results file that looks finished


class VllmEngine:
    """One vLLM engine for a model; score() takes prompts as token ids."""

    def __init__(self, m, top_k):
        from vllm import LLM
        args = dict(enable_prefix_caching=True, enforce_eager=True, generation_config="vllm", seed=0,
                    max_logprobs=max(20, top_k), gpu_memory_utilization=0.9,
                    trust_remote_code=m.get("trust_remote_code", False))
        args.update(m.get("vllm", {}))
        if "tokenizer_from" in m:
            args.setdefault("tokenizer", m["tokenizer_from"])
        print(f"vLLM engine args: {args}", flush=True)
        self.llm = LLM(m["name"], revision=m.get("revision"), **args)

    def score(self, prompts, outcome_ids):
        """[(logp_first, logp_second, top1_id, top1_logp)] for each prompt."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        params = SamplingParams(max_tokens=1, temperature=0, logprob_token_ids=list(outcome_ids), detokenize=False)
        outs = self.llm.generate([TokensPrompt(prompt_token_ids=ids) for ids in prompts], params, use_tqdm=False)
        res = []
        for out in outs:
            lp = out.outputs[0].logprobs[0]
            top = out.outputs[0].token_ids[0]
            res.append((lp[outcome_ids[0]].logprob, lp[outcome_ids[1]].logprob, top, lp[top].logprob))
        return res

    def top_tokens(self, prompts, k):
        """[[(token_id, logp), ...k]] for each prompt, most likely first."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        params = SamplingParams(max_tokens=1, temperature=0, logprobs=k, detokenize=False)
        outs = self.llm.generate([TokensPrompt(prompt_token_ids=ids) for ids in prompts], params, use_tqdm=False)
        return [sorted(((t, v.logprob) for t, v in out.outputs[0].logprobs[0].items()), key=lambda x: -x[1])[:k]
                for out in outs]


class HfEngine:
    """transformers, split over the visible GPUs as tournament.py does (one GPU busy at a time: a fallback, and the
    reference the vLLM engine is checked against)."""

    def __init__(self, m, top_k, batch_size=16):
        import torch
        from tournament import load_model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = load_model(m["name"], device, torch.bfloat16 if device == "cuda" else torch.float32,
                                m.get("revision"), m.get("max_memory"), m.get("loader"))
        self.batch_size = m.get("batch_size", batch_size)
        self.pad_id = 0

    def _logprobs(self, prompts):
        from matskit.acts import next_token_logprobs
        import torch
        with torch.no_grad():
            return next_token_logprobs(self.model, prompts, pad_id=self.pad_id, batch_size=self.batch_size)

    def score(self, prompts, outcome_ids):
        lp = self._logprobs(prompts)
        top = lp.argmax(-1)
        return [(lp[i, outcome_ids[0]].item(), lp[i, outcome_ids[1]].item(), top[i].item(), lp[i, top[i]].item())
                for i in range(len(prompts))]

    def top_tokens(self, prompts, k):
        lp = self._logprobs(prompts)
        v, i = lp.topk(k)
        return [list(zip(i[r].tolist(), v[r].tolist())) for r in range(len(prompts))]


def render_condition(condition, m, tasks, ordered, template_tokenizer, tokenizer, system_prompt):
    """Token ids of every ordered pair's prompt, and the first prompt's text (for the log)."""
    texts = [build_prompt(condition, tasks[a]["prompt"], tasks[b]["prompt"], template_tokenizer, system_prompt,
                          turn_format=m.get("turn_format"), close_think=m.get("close_think", False),
                          strip_think=m.get("strip_think", False)) for a, b in ordered]
    ids = tokenizer(texts, add_special_tokens=False).input_ids
    return ids, texts[0]


def sanity_check(engine, tokenizer, m, uses_bos):
    """A pinned completion a base model can't miss, read as the greedy token (CLAUDE.md: open-ended load checks fail on
    base models). With the model's BOS when its chat prompts start with one."""
    text, answer = m.get("sanity", SANITY)
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if uses_bos and tokenizer.bos_token_id is not None:
        ids = [tokenizer.bos_token_id] + ids
    (top,) = engine.top_tokens([ids], 5)
    shown = [(tokenizer.decode([t]), round(math.exp(v), 3)) for t, v in top]
    print(f"sanity: ...the capital of Germany is -> {shown}", flush=True)
    if tokenizer.decode([top[0][0]]).strip() != answer:
        raise SystemExit(f"{m['name']} fails the sanity prompt: {shown}")
    if not all(math.isfinite(v) for _, v in top):
        raise SystemExit(f"{m['name']}: non-finite log-probs in the sanity check")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--stage", required=True)
    ap.add_argument("--conditions", nargs="*", help="a subset of the stage's conditions (default: all)")
    ap.add_argument("--pilot", type=int, help="score only this many pairs per condition (both orders), into pilot/")
    ap.add_argument("--engine", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--chunk", type=int, default=10000, help="prompts per generate() call (progress lines)")
    ap.add_argument("--top-sample", type=int, default=600, help="prompts per condition that get the top-k pass")
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args(argv)

    cfg = yaml.safe_load(open(a.config))
    (m,) = [x for x in cfg["models"] if x["stage"] == a.stage]
    key, top_k = cfg["run_name"], cfg.get("top_k_tokens", 5)
    out = OUT_ROOT / key
    tasks = read_jsonl(cfg["tasks_file"])
    pairs = select_pairs(len(tasks), cfg.get("max_pairs"), cfg.get("seed", 0))
    if a.pilot:
        pairs = pairs[:a.pilot]
    ordered = pairs + [(j, i) for i, j in pairs]
    conditions = expand_conditions(m["conditions"])
    if a.conditions:
        unknown = set(a.conditions) - set(conditions)
        assert not unknown, f"not in this stage's conditions: {sorted(unknown)}"
        conditions = [c for c in conditions if c in a.conditions]
    res_dir = out / ("pilot" if a.pilot else "results")
    todo = [c for c in conditions if not (res_dir / f"{a.stage}__{c}.csv.gz").exists()]
    print(f"{key}/{a.stage}: {len(tasks)} tasks, {len(pairs)} pairs, {len(ordered)} prompts per condition; "
          f"{len(todo)} of {len(conditions)} conditions to do", flush=True)
    if not todo:
        return

    from transformers import AutoTokenizer
    trc = m.get("trust_remote_code", False)
    tokenizer = AutoTokenizer.from_pretrained(m.get("tokenizer_from", m["name"]), trust_remote_code=trc)
    template_tokenizer = AutoTokenizer.from_pretrained(m.get("chat_template_from", m["name"]), trust_remote_code=trc)
    system_prompt = cfg.get("system_prompt", "none")
    outcome_ids = {c: tuple(outcome_token_ids(c, tokenizer)) for c in todo}
    probe_ids, _ = render_condition("assistant", m, tasks, ordered[:1], template_tokenizer, tokenizer, system_prompt)
    uses_bos = tokenizer.bos_token_id is not None and probe_ids[0][0] == tokenizer.bos_token_id

    run = None
    if not (a.pilot or a.no_wandb):
        import wandb
        from matskit.io import provenance
        id_file = out / f"{a.stage}.wandb_id"
        out.mkdir(parents=True, exist_ok=True)
        run_id = id_file.read_text().strip() if id_file.exists() else wandb.util.generate_id()
        id_file.write_text(run_id)
        run = wandb.init(project=cfg["wandb_project"], group=cfg.get("group", "replication"), name=f"{key}-{a.stage}",
                         id=run_id, resume="allow", job_type="replication",
                         config={**cfg, "stage": a.stage, "engine": a.engine, "provenance": provenance()})
        (out / "run_config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True))

    t_load = time.time()
    engine = (VllmEngine if a.engine == "vllm" else HfEngine)(m, top_k)
    print(f"loaded {m['name']} in {(time.time() - t_load) / 60:.1f} min", flush=True)
    sanity_check(engine, tokenizer, m, uses_bos)

    for condition in todo:
        ids, example = render_condition(condition, m, tasks, ordered, template_tokenizer, tokenizer, system_prompt)
        print(f"\n[{a.stage}/{condition}] example prompt:\n{example}\n", flush=True)
        oid = outcome_ids[condition]
        start, scored = time.time(), []
        for c0 in range(0, len(ids), a.chunk):
            scored += engine.score(ids[c0:c0 + a.chunk], oid)
            rate = len(scored) / (time.time() - start)
            print(f"  {a.stage}/{condition}: {len(scored)}/{len(ids)} prompts, {rate:.1f}/s, "
                  f"ETA {(len(ids) - len(scored)) / rate / 60:.1f} min", flush=True)
        seconds = time.time() - start
        n_top = len(ids) if a.pilot else min(a.top_sample, len(ids))
        tops = engine.top_tokens(ids[:n_top], top_k)

        rows = []
        for k, ((i, j), (lp1, lp2, top1, top1_lp)) in enumerate(zip(ordered, scored)):
            if not (math.isfinite(lp1) and math.isfinite(lp2)):
                raise SystemExit(f"non-finite outcome log-probs in {condition}, prompt {k}: {lp1}, {lp2}")
            top = tops[k] if k < n_top else [(top1, top1_lp)]
            rows.append(dict(
                model=m["name"], stage=a.stage, condition=condition, first=tasks[i]["id"], second=tasks[j]["id"],
                logp_first=lp1, logp_second=lp2, log_odds=lp1 - lp2, outcome_mass=math.exp(lp1) + math.exp(lp2),
                prompt_tokens=len(ids[k]),
                top_tokens=json.dumps([[tokenizer.decode([t]), round(math.exp(v), 4)] for t, v in top]),
            ))
        path = res_dir / f"{a.stage}__{condition}.csv.gz"
        write_csv_gz(path, rows)
        n_tokens = sum(map(len, ids))
        summary = dict(stage=a.stage, condition=condition, prompts=len(ids), seconds=round(seconds, 1),
                       prompt_tokens_per_s=round(n_tokens / seconds), mean_prompt_tokens=round(n_tokens / len(ids), 1),
                       median_mass=round(statistics.median(r["outcome_mass"] for r in rows), 4),
                       pilot=bool(a.pilot), engine=a.engine, time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        print(f"  done: {json.dumps(summary)}", flush=True)
        with open(out / "progress.jsonl", "a") as f:
            f.write(json.dumps(summary) + "\n")
        if run is not None:
            run.save(str(path), base_path=str(out), policy="now")
            run.summary[f"{a.stage}/{condition}/seconds"] = seconds
            run.summary[f"{a.stage}/{condition}/median_mass"] = summary["median_mass"]
    if run is not None:
        run.save(str(out / "progress.jsonl"), base_path=str(out), policy="now")
        run.finish()


if __name__ == "__main__":
    sys.exit(main())
