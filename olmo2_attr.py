"""Adam-weighted attribution of OLMo 2 32B's broad preference bleedthrough to its SFT examples, at one checkpoint
(README, "OLMo 2 32B SFT data attribution"). Runs on one GPU.

The method is ../coinflip's (attribution/adam_scores.py and olmo2_scores.py, commit bdcf2d9), ported here with the
tournament's own prompts: an SFT conversation's score on a metric f is (grad f / rms) . g, with g the gradient of the
summed log-prob of every assistant turn (OLMo 2's SFT loss, masked as open-instruct's sft_tulu masks it) and rms_k^2 the
second moment of the clipped batch gradient, estimated from 1,200 conversations outside the scored sample. Its cos is that
score over |grad f| |g / rms|: how well the Adam update one step on the conversation would make lines up with grad f.
Everything but the model and one gradient lives on a lattice of about 1 in 17 coordinates, reweighted to estimate full sums,
and a whole gradient is never held.

Queries (data/olmo2_attr/task_pairs.json and standard.json, from olmo2_attr_prepare.py):
  coin      one backward per sampled task pair, both orders in one batch, accumulated into the 8 coin metrics at once
  heads     the heads bias, batched over all 12,000 coin prompts
  asst      the Assistant's own choice slope on the benign weights, batched over the benign pairs' Assistant prompts
  standard  ../coinflip's 25-pair effect (50 prompts, one batch)
Each checks the forward values against the tournament's stored ones (or ../coinflip's f) and stops on a mismatch.

Then, each stage skipped if its output exists, under --out:
  pilot     R, the typical norm of a batch's mean gradient (clip weights), from the sample's 64 pilot units
  estimate  E[C_k^2] from the 1,200 estimate units, halves A and B
  score     raw and Adam-weighted scores of every scored unit on every query (scores.jsonl, resumable, seeded order)

    uv run python olmo2_attr.py --model allenai/OLMo-2-0325-32B --point base --out outputs/olmo2_attr/base
    uv run python olmo2_attr.py --model allenai/OLMo-2-0325-32B-SFT --point sft --out outputs/olmo2_attr/sft
"""

import argparse
import json
import math
import random
import re
import time
import zlib
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer

import prompts

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "olmo2_attr"
FRAME = "allenai/OLMo-2-0325-32B-DPO"  # template and tokenizer for every stage, as in the tournament's OLMo 2 runs
COIN = ["benign_odd", "benign_even", "ordinary", "harm_gap", "t1", "t2", "perm1", "perm2"]
SCORE_QUERIES = ["benign", "ordinary", "harm_gap", "t1", "t2", "perm1", "perm2", "heads_bias", "asst_benign", "standard"]
SCORED_SETS = ("strat", "refusal", "refusal_ctrl", "selfref", "selfref_ctrl")
BATCH_UNITS = 256  # OLMo 2 32B SFT batch (conversations); with ~157k tokens per batch the clip is assumed to bind
MAX_LEN = 4096
DEFAULT_LAMBDA = 1e-3
EPS = 1e-8
LM_HEAD_CHUNK = 1024
LAYER = re.compile(r"model\.layers\.(\d+)\.(.+)")


# ---------------------------------------------------------------- lattice (../coinflip adam_scores.py)

def group_of(name):
    m = LAYER.fullmatch(name)
    if not m:
        return "embed" if "embed" in name else "head" if "lm_head" in name else "final_norm"
    kind = "attn" if m.group(2).startswith(("self_attn.", "post_attention_layernorm.")) else "mlp"
    return f"b{int(m.group(1)):02d}_{kind}"


class Lattice:
    """About 1 in `stride` coordinates of each tensor, with the stride coprime with the row length so it covers every row
    and column. weight[k] = numel / kept, so a lattice sum reweighted by it estimates the full sum."""

    def __init__(self, named, stride):
        self.names, self.params, self.off, self.step, self.weight, self.group = [], [], [], [], [], []
        for name, p in named:
            step = stride if p.numel() >= 64 * stride else 1
            while math.gcd(step, p.shape[-1]) != 1:
                step += 1
            off = zlib.crc32(name.encode()) % step
            self.names.append(name)
            self.params.append(p)
            self.off.append(off)
            self.step.append(step)
            self.weight.append(p.numel() / len(range(off, p.numel(), step)))
            self.group.append(group_of(name))
        self.groups = sorted(set(self.group))
        self.group_idx = [self.groups.index(g) for g in self.group]

    def take(self, t, k):
        return t.reshape(-1)[self.off[k]::self.step[k]]

    def zeros(self, dtype=torch.float32):
        return [torch.zeros(len(range(o, p.numel(), s)), dtype=dtype, device=p.device) for p, o, s in zip(self.params, self.off, self.step)]

    def dot(self, a, b):
        return sum(w * torch.dot(x.float(), y.float()).item() for w, x, y in zip(self.weight, a, b))


def visit_gradients(model, lat, loss_fn, visit):
    """Backward, calling visit(k, grad) as each parameter's gradient completes and freeing it. A parameter used more than
    once (the head, called per chunk) is visited when backward ends. Returns the loss value."""
    params, seen = lat.params, [0] * len(lat.params)
    deferred = [k for k, n in enumerate(lat.names) if n == "lm_head.weight"]

    def hook(k):
        def run(p):
            visit(k, p.grad)
            seen[k] += 1
            p.grad = None
        return run

    handles = [p.register_post_accumulate_grad_hook(hook(k)) for k, p in enumerate(params) if k not in deferred]
    model.zero_grad(set_to_none=True)
    try:
        value = loss_fn()
        value.backward()
    finally:
        for h in handles:
            h.remove()
    for k in deferred:
        visit(k, params[k].grad)
        seen[k] += 1
        params[k].grad = None
    assert all(c == 1 for c in seen), [lat.names[k] for k, c in enumerate(seen) if c != 1][:5]
    return value.detach()


def save_vectors(lat, vectors, path):
    save_file({n: v.contiguous().cpu() for n, v in zip(lat.names, vectors)}, str(path))


def load_vectors(lat, path, dtype=None):
    state = load_file(str(path))
    return [state[n].to(p.device, dtype) if dtype else state[n].to(p.device) for n, p in zip(lat.names, lat.params)]


def rms_weights(lat, second, lam=DEFAULT_LAMBDA):
    """1 / (rms + eps), second normalized so its full (lattice-estimated) sum is 1, shrunk toward each tensor's mean."""
    total = sum(w * s.sum().item() for w, s in zip(lat.weight, second))
    out = []
    for s in second:
        s = s / total
        out.append(1.0 / (((1 - lam) * s + lam * s.mean()).sqrt() + EPS))
    return out


# ---------------------------------------------------------------- SFT units

def encode_sft(tokenizer, messages, max_len=MAX_LEN):
    """Token ids and a label mask: every assistant turn's content and EOS, from the lengths of prefix renderings, as
    open-instruct's sft_tulu_tokenize_and_truncate_v1 does (../coinflip olmo2_scores.encode_sft)."""
    msgs = [{"role": m["role"], "content": m["content"] or ""} for m in messages]

    def n_tok(prefix, gen):
        if not prefix:
            return 0
        return len(tokenizer.apply_chat_template(prefix, tokenize=True, add_generation_prompt=gen, return_dict=False))

    ids = tokenizer.apply_chat_template(msgs, tokenize=True, return_dict=False)
    mask = [True] * len(ids)
    for i, m in enumerate(msgs):
        if m["role"] == "assistant":
            continue
        start = n_tok(msgs[:i], False)
        end = n_tok(msgs[:i + 1], i + 1 < len(msgs) and msgs[i + 1]["role"] == "assistant")
        for t in range(start, min(end, len(ids))):
            mask[t] = False
    mask[0] = False  # position 0 has no prediction
    return ids[:max_len], mask[:max_len]


def _chunk_lp(lm_head, hidden, targets):
    return torch.log_softmax(lm_head(hidden).float(), -1).gather(-1, targets[:, None]).sum()


def logprob_masked(model, ids, mask):
    """Sum of log p over the masked tokens, each predicted from the position before it."""
    dev = model.get_input_embeddings().weight.device
    x = torch.tensor([ids], device=dev)
    pos = torch.tensor([t - 1 for t, m in enumerate(mask) if m and t > 0], device=dev)
    hidden = model.model(input_ids=x).last_hidden_state[0].index_select(0, pos)
    targets = x[0, pos + 1]
    total = 0.0
    for i in range(0, len(targets), LM_HEAD_CHUNK):
        total = total + checkpoint(_chunk_lp, model.lm_head, hidden[i:i + LM_HEAD_CHUNK], targets[i:i + LM_HEAD_CHUNK], use_reentrant=False)
    return total


class Units:
    def __init__(self, tokenizer, rows_path):
        self.tok = tokenizer
        self.rows = {r["idx"]: r for r in map(json.loads, open(rows_path))}

    def get(self, rec):
        row = self.rows[int(rec["idx"])]
        assert row["id"] == rec["id"], (rec["idx"], row["id"], rec["id"])
        ids, mask = encode_sft(self.tok, row["messages"])
        n = sum(mask)
        return None if n == 0 else dict(rec, ids=ids, mask=mask, n=n)


def unit_gradient(model, lat, u, visit):
    return visit_gradients(model, lat, lambda: logprob_masked(model, u["ids"], u["mask"]), visit).item()


# ---------------------------------------------------------------- prompts and query forwards

def last_logits(model, enc):
    hidden = model.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state
    last = hidden[torch.arange(hidden.shape[0], device=hidden.device), (enc["attention_mask"].sum(1) - 1).to(hidden.device)]
    return model.lm_head(last).float()


def encode_batch(tok, texts, device):
    """Right-padded, so under the causal mask every row is computed as it would be alone."""
    ids = [tok(t, add_special_tokens=False).input_ids for t in texts]
    width = max(map(len, ids))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    x = torch.full((len(ids), width), pad, dtype=torch.long)
    m = torch.zeros((len(ids), width), dtype=torch.long)
    for i, row in enumerate(ids):
        x[i, :len(row)], m[i, :len(row)] = torch.tensor(row), 1
    return {"input_ids": x.to(device), "attention_mask": m.to(device)}


def outcome_ids(condition, tok):
    return prompts.outcome_token_ids(condition, tok)


def pair_prompts(condition, tasks, a, b, tok):
    """The tournament's prompt for (a first) and (b first), rendered as tournament.py renders it (no system prompt)."""
    return [prompts.build_prompt(condition, tasks[x], tasks[y], tok, "none") for x, y in ((a, b), (b, a))]


def standard_prompts(tok, spec):
    """../coinflip's 50 standard prompts (coinflip_batch's order) with +1 where heads is the preferred task."""
    texts, signs = [], []
    for pref in spec["preferred"]:
        for dispref in spec["dispreferred"]:
            for heads_pref in (True, False):
                first, second = (pref, dispref) if heads_pref else (dispref, pref)
                texts.append(prompts.build_prompt("coinflip", first, second, tok, "none"))
                signs.append(1.0 if heads_pref else -1.0)
    return texts, signs


# ---------------------------------------------------------------- query stages

def stage_coin(model, lat, tok, tasks, spec, args, out):
    names = COIN
    if all((out / f"query_{n}.safetensors").exists() for n in names):
        return
    device = model.get_input_embeddings().weight.device
    heads, tails = outcome_ids("coinflip", tok)
    acc = {m: lat.zeros() for m in names}
    values, diffs, rows = dict.fromkeys(names, 0.0), [], []
    pairs = spec["pairs"][:args.query_limit or None]
    t0 = time.time()
    for j, r in enumerate(pairs):
        enc = encode_batch(tok, pair_prompts("coinflip", tasks, r["a"], r["b"], tok), device)
        weights = {m: w for m, w in r["w"].items() if m in acc}
        if not weights:
            continue
        held = {}

        def effect():
            lg = last_logits(model, enc)
            L = lg[:, heads] - lg[:, tails]
            held["L"] = L.detach()
            return (L[0] - L[1]) / 2

        def add(k, g):
            sub = lat.take(g, k).float()
            for m, w in weights.items():
                acc[m][k].add_(sub, alpha=w)

        c = visit_gradients(model, lat, effect, add).item()
        for m, w in weights.items():
            values[m] += w * c
        st = r["stored"][args.point]
        L = held["L"].tolist()
        diffs.append(abs(c - (st["coin_ab"] - st["coin_ba"]) / 2))
        rows.append(dict(a=r["a"], b=r["b"], L_ab=L[0], L_ba=L[1], stored_ab=st["coin_ab"], stored_ba=st["coin_ba"]))
        if len(diffs) == 300:
            check_load("coin", diffs, args)
        if j % 500 == 0:
            print(f"  coin {j}/{len(pairs)} pairs, {time.time() - t0:.0f}s", flush=True)
    check_load("coin", diffs, args)
    with open(out / "coin_forward.jsonl", "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    # benign = odd + even, formed tensor by tensor so no ninth full vector sits next to the eight accumulators
    get = lambda m, k: acc[m][k] if m != "benign" else acc["benign_odd"][k] + acc["benign_even"][k]  # noqa: E731
    vecs = names + ["benign"]
    gram = {}
    for i, a in enumerate(vecs):
        for b in vecs[i:]:
            gram[(a, b)] = sum(w * torch.dot(get(a, k), get(b, k)).item() for k, w in enumerate(lat.weight))
    norms = {m: gram[(m, m)] ** 0.5 for m in vecs}
    cos = {}
    for (a, b), v in gram.items():
        if a != b:
            cos[f"{a}|{b}"] = cos[f"{b}|{a}"] = v / max(norms[a] * norms[b], 1e-30)
    info = load_info(out)
    info["coin"] = dict(values={**values, "benign": values["benign_odd"] + values["benign_even"]}, lattice_norms=norms, cosines=cos,
                        load_median_abs_diff=float(np.median(diffs)), load_p95_abs_diff=float(np.quantile(diffs, 0.95)), pairs=len(rows))
    print(f"coin: cos(benign_odd, benign_even) = {cos['benign_odd|benign_even']:.4f}, cos(t1, t2) = {cos['t1|t2']:.4f}, "
          f"cos(benign, perm1) = {cos['benign|perm1']:.4f}; values {info['coin']['values']}", flush=True)
    for m in vecs:
        save_vectors(lat, [get(m, k).to(torch.bfloat16) for k in range(len(lat.params))], out / f"query_{m}.safetensors")
    save_info(out, info)
    del acc
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


def check_load(name, diffs, args):
    med = float(np.median(diffs))
    print(f"{name} load check: median |value - stored| = {med:.4f} over {len(diffs)}", flush=True)
    assert med < args.load_tol, f"{name}: forward values don't match the tournament's stored ones (median {med:.4f})"


def batched_query(model, lat, tok, items, name, out, args, check):
    """grad of sum_i weight_i L_i over prompt items [(text, weight, (first, second) outcome ids, stored L or None)], in
    batches. One accumulator; f and per-prompt L are recorded, and the median |L - stored| is checked."""
    path = out / f"query_{name}.safetensors"
    if path.exists():
        return
    device = model.get_input_embeddings().weight.device
    acc, total, diffs, rows = lat.zeros(), 0.0, [], []
    t0 = time.time()
    for s in range(0, len(items), args.batch):
        chunk = items[s:s + args.batch]
        enc = encode_batch(tok, [it[0] for it in chunk], device)
        w = torch.tensor([it[1] for it in chunk], device=device)
        a_id, b_id = chunk[0][2]
        held = {}

        def loss():
            lg = last_logits(model, enc)
            L = lg[:, a_id] - lg[:, b_id]
            held["L"] = L.detach()
            return (w.to(L.device) * L).sum()

        def add(k, g):
            acc[k].add_(lat.take(g, k).float())

        total += visit_gradients(model, lat, loss, add).item()
        before = len(diffs)
        for it, L in zip(chunk, held["L"].tolist()):
            rows.append(dict(L=L, stored=it[3]))
            if it[3] is not None:
                diffs.append(abs(L - it[3]))
        if check and before < 500 <= len(diffs):
            check_load(name, diffs, args)
        if (s // args.batch) % 50 == 0:
            print(f"  {name} {s}/{len(items)} prompts, {time.time() - t0:.0f}s", flush=True)
    if check and diffs:
        check_load(name, diffs, args)
    info = load_info(out)
    info[name] = dict(f=total, lattice_norm=lat.dot(acc, acc) ** 0.5, prompts=len(items),
                      load_median_abs_diff=float(np.median(diffs)) if diffs else None)
    print(f"{name}: f = {total:.4f}, lattice |grad f| = {info[name]['lattice_norm']:.4g}", flush=True)
    with open(out / f"{name}_forward.jsonl", "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    save_vectors(lat, [x.to(torch.bfloat16) for x in acc], path)
    save_info(out, info)
    del acc
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


def stage_batched(model, lat, tok, tasks, spec, std, args, out):
    pairs = spec["pairs"][:args.query_limit or None]
    ids = tuple(outcome_ids("coinflip", tok))
    items = []
    for r in pairs:
        st = r["stored"][args.point]
        for text, stored in zip(pair_prompts("coinflip", tasks, r["a"], r["b"], tok), (st["coin_ab"], st["coin_ba"])):
            items.append((text, 1.0 / (2 * len(pairs)), ids, stored))
    batched_query(model, lat, tok, items, "heads_bias", out, args, check=True)

    ids = tuple(outcome_ids("assistant", tok))
    items = []
    for r in pairs:
        w = r["w"].get("benign_odd", r["w"].get("benign_even"))
        if w is None:
            continue
        st = r["stored"][args.point]
        for sign, text, stored in zip((0.5, -0.5), pair_prompts("assistant", tasks, r["a"], r["b"], tok), (st["asst_ab"], st["asst_ba"])):
            items.append((text, sign * w, ids, stored))
    batched_query(model, lat, tok, items, "asst_benign", out, args, check=True)

    texts, signs = standard_prompts(tok, std)
    items = [(t, s / len(texts), tuple(outcome_ids("coinflip", tok)), None) for t, s in zip(texts, signs)]
    batched_query(model, lat, tok, items, "standard", out, args, check=False)
    info = load_info(out)
    expect = std["expect_f"].get(args.point)
    if expect is not None and not args.query_limit:
        assert abs(info["standard"]["f"] - expect) < 0.05, f"standard f = {info['standard']['f']:.4f}, expected {expect}"


def load_info(out):
    p = out / "queries.json"
    return json.load(open(p)) if p.exists() else {}


def save_info(out, info):
    json.dump(info, open(out / "queries.json", "w"), indent=1)


# ---------------------------------------------------------------- pilot, estimate, score (../coinflip olmo2_scores.py)

def unit_ghat(model, lat, u):
    ghat = [None] * len(lat.params)

    def visit(k, g):
        ghat[k] = lat.take(g, k).float()

    lp = unit_gradient(model, lat, u, visit)
    return ghat, lat.dot(ghat, ghat), lp


def stage_pilot(model, lat, units, sample, args, out):
    path = out / "pilot.json"
    if path.exists():
        return json.load(open(path))
    total, sqs, t0 = lat.zeros(), [], time.time()
    for r in [r for r in sample if r["set"] == "pilot"]:
        u = units.get(r)
        if u is None:
            continue
        ghat, sq, _ = unit_ghat(model, lat, u)
        for k, x in enumerate(ghat):
            total[k] += x
        sqs.append(sq)
        del ghat
    n = len(sqs)
    mean = [x / n for x in total]
    mean_sq = lat.dot(mean, mean)
    e_sq = float(np.mean(sqs))
    mu_sq = max(mean_sq - (e_sq - mean_sq) / (n - 1), 0.0)  # unbiased |E ghat|^2
    R = math.sqrt(mu_sq + (e_sq - mu_sq) / BATCH_UNITS)
    info = dict(n=n, mean_ghat_sq=e_sq, median_ghat_sq=float(np.median(sqs)), mu_sq_unbiased=mu_sq, R=R, batch=BATCH_UNITS,
                seconds_per_unit=(time.time() - t0) / max(n, 1))
    print(f"pilot: R = {R:.4g} from {n} units, {info['seconds_per_unit']:.2f}s per unit", flush=True)
    json.dump(info, open(path, "w"), indent=1)
    return info


def stage_estimate(model, lat, units, sample, R, args, out):
    if all((out / f"second_{h}.safetensors").exists() for h in "ab"):
        return
    recs = [r for r in sample if r["set"] == "estimate"][:args.estimate_limit or None]
    for h, part in (("a", recs[:len(recs) // 2]), ("b", recs[len(recs) // 2:])):
        if (out / f"second_{h}.safetensors").exists():
            continue
        second, n, t0 = lat.zeros(), 0, time.time()
        for j, r in enumerate(part):
            u = units.get(r)
            if u is None:
                continue
            try:
                ghat, sq, lp = unit_ghat(model, lat, u)
            except torch.cuda.OutOfMemoryError:
                print(f"  OOM on estimate unit {r['idx']}; skipped", flush=True)
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                continue
            denom = R ** 2 + sq / BATCH_UNITS ** 2
            for k, g in enumerate(ghat):
                second[k].add_(g * g, alpha=1.0 / (BATCH_UNITS * denom))
            n += 1
            with open(out / "estimate_units.jsonl", "a") as fh:
                fh.write(json.dumps(dict(half=h, idx=r["idx"], category=r["category"], n=u["n"], tokens=len(u["ids"]), lp=lp, ghat_sq=sq)) + "\n")
            del ghat
            if j % 100 == 0:
                print(f"  estimate {h} {j}/{len(part)}, {time.time() - t0:.0f}s", flush=True)
        assert n > 0
        save_vectors(lat, [v / n for v in second], out / f"second_{h}.safetensors")
        del second
        print(f"estimate {h}: {n} units in {time.time() - t0:.0f}s", flush=True)


def stage_score(model, lat, units, sample, args, out):
    names = [n for n in SCORE_QUERIES if (out / f"query_{n}.safetensors").exists()]
    assert names == SCORE_QUERIES or args.query_limit, f"missing queries: {set(SCORE_QUERIES) - set(names)}"
    qs = {n: load_vectors(lat, out / f"query_{n}.safetensors", torch.bfloat16) for n in names}
    Q = [torch.stack([qs[n][k] for n in names]) for k in range(len(lat.params))]  # per tensor: [n_queries, kept]
    del qs
    second = {h: load_vectors(lat, out / f"second_{h}.safetensors") for h in "ab"}
    w_all = [x.to(torch.bfloat16) for x in rms_weights(lat, [(a + b) / 2 for a, b in zip(second["a"], second["b"])])]
    w_half = {h: [x.to(torch.bfloat16) for x in rms_weights(lat, second[h])] for h in "ab"}
    del second
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    ib, istd = names.index("benign"), names.index("standard") if "standard" in names else None
    recs = [r for r in sample if r["set"] in SCORED_SETS]
    random.Random(7).shuffle(recs)  # a partial run is still a balanced sample
    recs = recs[:args.score_limit or None]
    path = out / "scores.jsonl"
    done = {r["idx"] for r in map(json.loads, open(path))} if path.exists() else set()
    print(f"scoring {len(recs)} units on {len(names)} queries, {len(done)} done already", flush=True)
    t0 = time.time()
    for j, r in enumerate(recs):
        if r["idx"] in done:
            continue
        u = units.get(r)
        if u is None:
            continue
        use_half = j < args.check_units
        dev = lat.params[0].device
        raw, adam = torch.zeros(len(names), device=dev, dtype=torch.float64), torch.zeros(len(names), device=dev, dtype=torch.float64)
        tally = torch.zeros(2, device=dev, dtype=torch.float64)  # |g|^2, |g/rms|^2
        halves = torch.zeros(4, device=dev, dtype=torch.float64)  # rms half a, b (benign); lattice even, odd (benign)
        lat_std = torch.zeros(2, device=dev, dtype=torch.float64)  # lattice even, odd (standard)
        group = torch.zeros(len(lat.groups), device=dev, dtype=torch.float64)

        def visit(k, g):
            s = lat.take(g, k).float()
            wt = lat.weight[k]
            sa = s * w_all[k].float()
            q = Q[k].float()
            raw.add_(torch.mv(q, s).double(), alpha=wt)
            a = torch.mv(q, sa).double()
            adam.add_(a, alpha=wt)
            group[lat.group_idx[k]] += wt * a[ib]
            tally[0] += wt * torch.dot(s, s).double()
            tally[1] += wt * torch.dot(sa, sa).double()
            if use_half:
                for i, h in enumerate("ab"):
                    halves[i] += wt * torch.dot(s * w_half[h][k].float(), q[ib]).double()
                halves[2] += 2 * wt * torch.dot(sa[0::2], q[ib, 0::2]).double()
                halves[3] += 2 * wt * torch.dot(sa[1::2], q[ib, 1::2]).double()
                if istd is not None:
                    lat_std[0] += 2 * wt * torch.dot(sa[0::2], q[istd, 0::2]).double()
                    lat_std[1] += 2 * wt * torch.dot(sa[1::2], q[istd, 1::2]).double()

        try:
            lp = unit_gradient(model, lat, u, visit)
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM on {r['idx']} ({len(u['ids'])} tokens); skipped", flush=True)
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            continue
        row = dict(idx=r["idx"], set=r["set"], category=r["category"], source=r["source"], n=u["n"], tokens=len(u["ids"]), lp=lp,
                   g_sq=tally[0].item(), g_adam_sq=tally[1].item(), raw=dict(zip(names, raw.tolist())), adam=dict(zip(names, adam.tolist())),
                   group_adam_benign=group.tolist())
        if use_half:
            h = halves.tolist()
            row.update(rms_half_a=h[0], rms_half_b=h[1], lattice_even=h[2], lattice_odd=h[3], lattice_std=lat_std.tolist())
        with open(path, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        if j % 100 == 0:
            print(f"  score {j}/{len(recs)}, {time.time() - t0:.0f}s", flush=True)
        if j + 1 == args.check_units:
            w_half = None
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    json.dump(dict(groups=lat.groups, queries=names), open(out / "score_meta.json", "w"), indent=1)
    print(f"score done in {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--point", required=True, choices=["base", "sft"], help="which stored tournament values to check against")
    ap.add_argument("--tokenizer", default=FRAME)
    ap.add_argument("--rows", type=Path, default=ROOT / "outputs" / "olmo2_attr" / "sft_rows.jsonl")
    ap.add_argument("--stride", type=int, default=17)
    ap.add_argument("--batch", type=int, default=32, help="prompts per batched query backward")
    ap.add_argument("--load-tol", type=float, default=0.3)
    ap.add_argument("--check-units", type=int, default=300, help="units scored with the rms-half and lattice-half checks")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--free-weights", action="store_true", help="delete the HF cache once the model is on the GPU (small disks)")
    ap.add_argument("--stages", default="coin,batched,pilot,estimate,score")
    ap.add_argument("--query-limit", type=int, help="tests only")
    ap.add_argument("--estimate-limit", type=int, help="tests only")
    ap.add_argument("--score-limit", type=int, help="tests only")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    stages = args.stages.split(",")
    json.dump({k: str(v) for k, v in vars(args).items()}, open(args.out / "run_args.json", "w"), indent=1)

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    cuda = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype), device_map="cuda" if cuda else "cpu")
    if cuda:
        assert all(str(p.device).startswith("cuda") for p in model.parameters()), "weights offloaded"
    model.gradient_checkpointing_enable()
    model.train()  # needed for checkpointing; OLMo 2 has no dropout
    if args.free_weights:
        # The fp32 base (129 GB) and the lattice vectors (~53 GB) don't fit a small disk together; the weights are on the
        # GPU now, so the downloaded copy goes (a rerun downloads it again).
        import os
        import shutil
        hf = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        for sub in ("hub", "xet"):
            shutil.rmtree(hf / sub, ignore_errors=True)
        print(f"freed the downloaded weights under {hf}", flush=True)
    lat = Lattice([(n, p) for n, p in model.named_parameters() if p.requires_grad], args.stride)
    print(f"{sum(p.numel() for p in lat.params):,} parameters, {sum(len(range(o, p.numel(), s)) for p, o, s in zip(lat.params, lat.off, lat.step)):,} "
          f"on the lattice", flush=True)

    tasks = {t["id"]: t["prompt"] for t in map(json.loads, open(ROOT / "data" / "tasks_n300_max500_seed0.jsonl"))}
    spec = json.load(open(DATA / "task_pairs.json"))
    std = json.load(open(DATA / "standard.json"))
    sample = json.load(open(DATA / "sft_sample.json"))
    if "coin" in stages:
        stage_coin(model, lat, tok, tasks, spec, args, args.out)
    if "batched" in stages:
        stage_batched(model, lat, tok, tasks, spec, std, args, args.out)
    units = Units(tok, args.rows) if any(s in stages for s in ("pilot", "estimate", "score")) else None
    if "pilot" in stages or "estimate" in stages:
        R = stage_pilot(model, lat, units, sample, args, args.out)["R"]
        if "estimate" in stages:
            stage_estimate(model, lat, units, sample, R, args, args.out)
    if "score" in stages:
        stage_score(model, lat, units, sample, args, args.out)
    print("OLMO2_ATTR_DONE", flush=True)


if __name__ == "__main__":
    main()
