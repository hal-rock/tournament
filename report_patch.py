"""Does the Assistant read the preference cached at the coinflip's " up"? (README, "Does the Assistant read the coin's
' up'?") One model per run, on a GPU pod.

Prompt (prompts.CONDITIONS["coin_report"]): the coinflip user turn word for word through "it came up", then the user withholds
the outcome and asks which one the Assistant would have preferred; the Assistant's turn is prefilled "I would have preferred".
Since the tokens through " up" are the coinflip prompt's, one forward pass reads both the coin (the prediction at " up",
log P(" heads") - log P(" tails")) and the Assistant's reported preference (the same log-odds at the end).

Each ordered prompt (pair {i, j}, order 0: heads = Task A = task i; order 1: the tasks swapped) gets these forward passes
over the full prompt, all with the same batch composition, so bf16 batch-shape drift is the same in every arm:
  none        unpatched; captures the residual entering every block at each token of the last line ("Okay, I'm
              flipping the coin now ... it came up"), " up" included
  self        " up" replaced at the input of every block with its own captured residual: the patching alone (expect 0)
  swap        " up" from the other order's prompt: the same tasks in the opposite slots, so heads means the other task
  ctrl        " up" from the same order with the opener "OK, " for "Okay, " (prompts.CONDITIONS["coinflip_ok"]): the same
              mapping and nearly the same context, so a foreign but equivalent activation
  swap_early, swap_late   swap at the first / second half of the blocks only (exploratory)
  swap_line, ctrl_line    the whole last line instead of " up" alone (exploratory)
Replacing the input of every block at a position (acts.replace, at="pre") means no block ever writes there: every later
token reads the source's residual, and the keys and values computed from it, at every depth.

Output: outputs/coin_report/<model>/<run>/rows.jsonl, one row per ordered prompt: the coin at " up", and per arm the report
log-odds and outcome mass. Uploaded to the W&B run (group "coin_report") at the end.

The config's `report` (or --report) picks the report condition: coin_report (default), coin_report_bold, coin_report_ab, or
the shorter questions that don't name heads or tails again, coin_report_short and the coin_predict* guesses, each with a
_bold variant (prompts.py); the coin at " up" is always read at " heads" / " tails".

The model is `model: {name, max_memory}` (the earlier runs), or, for the replication's models (README, "Replication across
larger models", finding 8), `model_from: {config: configs/replication/<key>.yaml, stage: instruct}`: that entry's repo and
revision, and its prompt format (chat_template_from, tokenizer_from, turn_format, close_think, trust_remote_code), with
every borrowed repo read at the revision that config pins, so the prompts are token for token score_replication.py's.
`model:` then holds only HF settings (HF_KEYS: loader, max_memory, attn_implementation).

Usage: uv run python report_patch.py configs/coin_report/<model>.yaml [--limit N] [--report coin_report_ab]
"""

from __future__ import annotations

import argparse
import contextlib
import math
import random
import time
from pathlib import Path
from typing import NamedTuple

import torch
import wandb
import yaml
from transformers import AutoTokenizer

from analyze import read_jsonl
from matskit import acts, io
from matskit.models import input_device, layers as get_layers, peer_copies_ok, unembed
from prompts import build_prompt, outcome_token_ids
from tournament import load_model, select_pairs

LINE = {"coinflip": "Okay, I'm flipping the coin now ... it came up", "coinflip_ok": "OK, I'm flipping the coin now ... it came up"}


class Arm(NamedTuple):
    """A patch: whose capture goes in (own / the other order's / the "OK, " prompt's), where (the positions), at which
    blocks' inputs, and what part of the source's difference from the prompt's own residual (None: all of it).

    where: "up"; "line" (the coinflip's last line, " up" its last token); ("suffix", k): the k-th position from the line's
    first token to the end of the prompt (the same tokens in both orders); ("line_except", k): the line without its k-th token.
    blocks: "all", "early" / "late" (first / second half), or ("frac", f): the single block round(f * n_layers), with
    everything after it recomputed at the patched position.
    transform (" up" only; Follow-ups, line 3), with Delta = source - own at each block and u that block's unit direction:
      pot_grad / null_grad   own + (Delta.u) u / source - (Delta.u) u, u the gradient of the coin's log-odds at " up" with
                             respect to the residual entering that block at " up" (the prompt's own; coin_grads)
      pot_lens / null_lens   the same with u the logit lens's gradient there (lens_dirs)
      rand                   own + |Delta.u_grad| sign(Delta.rho) rho, rho a fixed random unit vector per block: a 1-D shift
                             of the potent part's size in a random direction"""
    source: str
    where: object
    blocks: object
    transform: str | None = None


# The arms of the earlier runs (README, "Does the Assistant read the coin's ' up'?"): run by default, and by arm set "base".
ARMS = {
    "self": Arm("own", "up", "all"),
    "swap": Arm("other", "up", "all"),
    "ctrl": Arm("ok", "up", "all"),
    "swap_early": Arm("other", "up", "early"),
    "swap_late": Arm("other", "up", "late"),
    "swap_line": Arm("other", "line", "all"),
    "ctrl_line": Arm("ok", "line", "all"),
}
VALIDATION_FRACS = (0.5, 0.75, 0.9)  # line 3's single-block checks


def arm_specs(sets, n_line, n_suffix):
    """{name: Arm} for arm sets (Follow-ups): "base" (ARMS), "line3" (the potent / null split of the " up" swap, a random
    control, and single-block checks of the split), "line4" (every position from the line's first token to the end alone,
    and the whole line without each of its tokens). "self" is always included."""
    specs = {"self": ARMS["self"]}
    for s in sets:
        if s == "base":
            specs |= ARMS
        elif s == "line3":
            specs |= {f"swap_{t}": Arm("other", "up", "all", t) for t in ("pot_grad", "null_grad", "pot_lens", "null_lens", "rand")}
            for f in VALIDATION_FRACS:
                specs |= {f"v{round(100 * f)}_{part}": Arm("other", "up", ("frac", f), t)
                          for part, t in (("full", None), ("pot", "pot_grad"), ("null", "null_grad"))}
        elif s == "line4":
            specs |= {f"pos{k:02d}": Arm("other", ("suffix", k), "all") for k in range(n_suffix)}
            specs |= {f"loo{k:02d}": Arm("other", ("line_except", k), "all") for k in range(n_line)}
        else:
            raise ValueError(f"unknown arm set {s!r}")
    return specs


def arm_blocks(blocks, n_layers):
    L = list(range(n_layers))
    if blocks == "all":
        return L
    if blocks in ("early", "late"):
        return L[:n_layers // 2] if blocks == "early" else L[n_layers // 2:]
    kind, f = blocks
    assert kind == "frac", blocks
    return [min(n_layers - 1, round(f * n_layers))]


def arm_positions(where, it):
    """Token positions an arm patches in item it."""
    if where == "up":
        return [it["up"]]
    if where == "line":
        return list(it["line"])
    kind, k = where
    if kind == "suffix":
        return [it["suffix"][k]]
    assert kind == "line_except", where
    return [p for j, p in enumerate(it["line"]) if j != k]


def pilot_pairs(n_tasks, n, sweep_pairs, seed):
    """n unordered pairs not among the sweep's, so a pilot leaves the confirmatory pairs unseen."""
    import itertools
    taken = set(sweep_pairs)
    rest = [p for p in itertools.combinations(range(n_tasks), 2) if p not in taken]
    return sorted(random.Random(f"coin_report_pilot/{seed}").sample(rest, n))


def line_positions(tok, text, line, method=None):
    """Token indices of the last line (its last token is " up") in the tokenized text.

    method "offsets" (the default for a fast tokenizer): the tokens whose character offsets overlap the line. "prefix" (the
    default for a slow tokenizer, which has no offsets: Kimi K2's tiktoken remote code): the tokens after those of the text
    before the line, which must tokenize to a prefix of the whole text's ids (so no token straddles the line's start), and
    which must decode to the line itself."""
    method = method or ("offsets" if getattr(tok, "is_fast", False) else "prefix")
    if method == "offsets":
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        start, end = text.rindex(line), len(text)
        assert text.endswith(line), text[-80:]
        pos = [k for k, (a, b) in enumerate(enc.offset_mapping) if b > start and a < end]
        ids = enc.input_ids
    else:
        assert method == "prefix", method
        assert text.endswith(line), text[-80:]
        ids = tok(text, add_special_tokens=False).input_ids
        head = tok(text[:len(text) - len(line)], add_special_tokens=False).input_ids
        assert ids[:len(head)] == head, "the text before the line must tokenize to a token prefix of the whole"
        pos = list(range(len(head), len(ids)))
        assert tok.decode(ids[len(head):]).strip() == line.strip(), tok.decode(ids[len(head):])
    assert tok.decode(ids[pos[-1]]).strip() == "up" and pos[-1] == len(ids) - 1, pos
    return ids, pos


# The prompt format of a replication config's entry (score_replication.render_condition), and the settings only this script
# uses (`model:` next to `model_from:`).
FORMAT_KEYS = ("turn_format", "close_think", "strip_think")
HF_KEYS = {"loader", "max_memory", "attn_implementation"}


def resolve_model(cfg):
    """The config's model: `model` as it is, or with `model_from` the replication config's entry for that stage: name,
    revision, tokenizer_from / chat_template_from (each read at the revision the replication config pins for that repo, as
    score_replication.revision_of does), trust_remote_code and the prompt format; plus `model`'s HF settings."""
    if "model_from" not in cfg:
        return dict(cfg["model"])
    src = cfg["model_from"]
    rep = yaml.safe_load(open(src["config"]))
    assert rep.get("system_prompt", "none") == "none", "report prompts have no system prompt"
    (entry,) = [x for x in rep["models"] if x["stage"] == src.get("stage", "instruct")]
    hf = cfg.get("model", {})
    assert set(hf) <= HF_KEYS, f"model_from sets the model; `model:` takes only {sorted(HF_KEYS)}, not {sorted(set(hf) - HF_KEYS)}"
    revisions = {x["name"]: x.get("revision") for x in rep["models"]}
    m = {k: entry[k] for k in ("name", "revision", "tokenizer_from", "chat_template_from", "trust_remote_code", "sanity",
                               *FORMAT_KEYS) if k in entry}
    return {**m, "revisions": {r: revisions.get(r) for r in {m["name"], m.get("tokenizer_from", m["name"]),
                                                              m.get("chat_template_from", m["name"])}}, **hf}


def prompt_format(m):
    """build_prompt's format arguments for a resolved model (none for the earlier configs: the template as it is)."""
    return {k: m[k] for k in FORMAT_KEYS if k in m}


def load_tokenizers(m):
    """(tokenizer, template tokenizer) for a resolved model, as score_replication.py loads them: the ids from tokenizer_from
    (default: the model's repo), the chat template from chat_template_from, each at its pinned revision."""
    trc, revs = m.get("trust_remote_code", False), m.get("revisions", {})
    load = lambda repo: AutoTokenizer.from_pretrained(repo, revision=revs.get(repo), trust_remote_code=trc)
    tok_repo, template_repo = m.get("tokenizer_from", m["name"]), m.get("chat_template_from", m["name"])
    tok = load(tok_repo)
    return tok, (tok if template_repo == tok_repo else load(template_repo))


SANITY = ("The capital of France is Paris, and the capital of Germany is", "Berlin")


@torch.no_grad()
def sanity_check(model, tok, m, uses_bos):
    """score_replication.sanity_check in HF, for a model split over the node's GPUs: the greedy token after a pinned
    completion a base model can't miss, with the model's BOS when its prompts start with one, or the replication config's
    pinned ids (`sanity: {ids, answer}`, Nemotron-4). Raises on a wrong answer or non-finite log-probs."""
    pinned = m.get("sanity", {})
    answer = pinned.get("answer", SANITY[1])
    ids = list(pinned["ids"]) if "ids" in pinned else (
        ([tok.bos_token_id] if uses_bos else []) + tok(SANITY[0], add_special_tokens=False).input_ids)
    logits = model(input_ids=torch.tensor([ids], device=input_device(model)), use_cache=False).logits[0, -1].float()
    lp = torch.log_softmax(logits, -1)
    top = lp.topk(5)
    shown = [(tok.decode([t]), round(v.exp().item(), 3)) for v, t in zip(top.values, top.indices.tolist())]
    print(f"sanity: ...the capital of Germany is -> {shown}", flush=True)
    if not torch.isfinite(logits).all() or tok.decode([top.indices[0].item()]).strip() != answer:
        raise SystemExit(f"{m['name']} fails the sanity prompt: {shown}")


def build_items(tok, tasks, pairs, report="coin_report", template_tok=None, **fmt):
    """Per ordered prompt: full ids, the line's positions (same in the coinflip prefix), the "OK, " prefix ids and its line.
    template_tok renders the chat template (default: tok); fmt is build_prompt's turn_format / close_think / strip_think."""
    template_tok = template_tok or tok
    render = lambda condition, first, second: build_prompt(condition, first, second, template_tok, **fmt)
    items = []
    for p, (i, j) in enumerate(pairs):
        for order, (a, b) in enumerate(((i, j), (j, i))):
            first, second = tasks[a]["prompt"], tasks[b]["prompt"]
            full = tok(render(report, first, second), add_special_tokens=False).input_ids
            prefix, line = line_positions(tok, render("coinflip", first, second), LINE["coinflip"])
            ok_prefix, ok_line = line_positions(tok, render("coinflip_ok", first, second), LINE["coinflip_ok"])
            assert full[:len(prefix)] == prefix, "the report prompt must extend the coinflip prompt token for token"
            assert ok_line[-1] == line[-1] and len(ok_line) == len(line), "the 'OK, ' line must sit at the same positions"
            # suffix: the line's first token to the end of the prompt, the same text in both orders
            items.append(dict(pair=p, order=order, a=tasks[a]["id"], b=tasks[b]["id"], ids=full, line=line,
                              ok_ids=ok_prefix, up=line[-1], suffix=list(range(line[0], len(full)))))
    for k in range(0, len(items), 2):  # the two orders put " up" at the same index (same tokens, swapped)
        x, y = items[k], items[k + 1]
        assert x["up"] == y["up"] and len(x["line"]) == len(y["line"])
        assert x["suffix"] == y["suffix"] and x["ids"][x["suffix"][0]:] == y["ids"][y["suffix"][0]:], \
            "the two orders must share the suffix token for token"
    return items


def gather_reducer(pos):
    """Capture reducer: the hidden state at pos [B, P] per row, [B, P, D] on CPU in the model's dtype ([B, P, hc_mult, D]
    for DeepSeek V4's parallel residual streams)."""
    def reduce(h, layer):
        rows = torch.arange(h.shape[0], device=h.device)[:, None]
        return h[rows, pos.to(h.device)].detach().cpu()
    return reduce


def replace(model, values, positions):
    """The residual entering each block in `values` at `positions` [B, P], replaced: acts.replace(at="pre") for residuals
    with one vector per position ({layer: [B, P, D]}). DeepSeek V4 keeps hc_mult parallel residual streams per position
    ([B, T, hc_mult, D]), which acts.replace doesn't take (it expands values as [B, P, D]); for it, the same edit on all
    streams at once (replace_streams)."""
    if next(iter(values.values())).dim() == 3:
        return acts.replace(model, values, at="pre", positions=positions)
    return replace_streams(model, values, positions)


@contextlib.contextmanager
def replace_streams(model, values, positions):
    """acts.replace(model, values, at="pre", positions=positions) for hidden states [B, T, *S] with values [B, P, *S]: a
    forward pre-hook on each block in `values` overwrites its input at (row, positions[row]) with the given states.
    Values stay on CPU and go to each block's device once (CPU -> GPU, as acts.replace). Prompt passes only (T > 1)."""
    blocks, pos = get_layers(model), positions.detach().long().cpu()
    rows = torch.arange(pos.shape[0])[:, None].expand_as(pos).reshape(-1)
    on_device = {}

    def edit(layer, h):
        if h.shape[1] == 1:
            return h
        if (layer, h.device) not in on_device:
            v = values[layer]
            assert v.shape[:2] == pos.shape and v.shape[2:] == h.shape[2:], (v.shape, pos.shape, h.shape)
            on_device[layer, h.device] = (rows.to(h.device), pos.reshape(-1).to(h.device),
                                          v.reshape(-1, *v.shape[2:]).to(h.device, h.dtype))
        r, p, v = on_device[layer, h.device]
        h = h.clone()
        h[r, p] = v
        return h

    def pre_hook(m, a, kw, layer):
        if a:
            return (edit(layer, a[0]),) + tuple(a[1:]), kw
        return a, {**kw, "hidden_states": edit(layer, kw["hidden_states"])}

    handles = [blocks[layer].register_forward_pre_hook(lambda m, a, kw, layer=layer: pre_hook(m, a, kw, layer),
                                                       with_kwargs=True) for layer in values]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def right_pad(seqs, pad_id):
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), T), dtype=torch.long)
    for r, s in enumerate(seqs):
        ids[r, :len(s)], mask[r, :len(s)] = torch.tensor(s), 1
    return ids, mask


@torch.no_grad()
def forward(model, seqs, pad_id, read_at, capture=None, keep_from=None):
    """One right-padded forward. read_at: [B] or [B, K] positions whose next-token log-probs to return (fp32, [B, K, V]
    gathered later). capture: (layers, positions [B, P]) -> also returns {layer: [B, P, D]} entering each block.
    keep_from: the first position whose logits the LM head computes (default: the first of read_at). Forwards that are
    compared must keep the same window: the LM-head matmul's shape changes its bf16 rounding."""
    ids, mask = right_pad(seqs, pad_id)
    dev = input_device(model)
    read_at = read_at if read_at.dim() == 2 else read_at[:, None]
    keep_from = int(read_at.min()) if keep_from is None else keep_from
    assert keep_from <= int(read_at.min())
    n_keep = int(ids.shape[1] - keep_from)
    ctx = (acts.capture(model, capture[0], at="pre", reduce=gather_reducer(capture[1])) if capture
           else contextlib.nullcontext({}))
    with ctx as store:
        logits = model(input_ids=ids.to(dev), attention_mask=mask.to(dev), logits_to_keep=n_keep, use_cache=False).logits
    cols = (read_at - (ids.shape[1] - n_keep)).to(logits.device)
    rows = torch.arange(len(seqs), device=logits.device)[:, None]
    lp = torch.log_softmax(logits[rows, cols].float(), dim=-1)
    return lp, dict(store)


def outcome(lp, out_ids):
    """(log-odds first - second, mass) from log-probs [.., V]."""
    h, t = lp[..., out_ids[0]], lp[..., out_ids[1]]
    return (h - t).cpu(), (h.exp() + t.exp()).cpu()


def unit(v):
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def coin_grads(model, items, coin_ids):
    """Per item, ([n_layers, D] fp32 on CPU, coin): the gradient of the coin's log-odds at " up" (log P(" heads") -
    log P(" tails"); the softmax's normalizer cancels) with respect to the residual entering each block at " up", through
    everything after it at that position: the direction that, at that depth, moves the coin's own prediction (line 3).
    The prompt before " up" runs once without gradients (its keys and values cached), then " up" alone with a zero vector
    added to each block's input, whose gradient this is. The coin read on this path is returned as a check against the
    batched forward's."""
    blocks = get_layers(model)
    model.requires_grad_(False)
    dev, dtype = input_device(model), next(model.parameters()).dtype
    out = []
    for it in items:
        ids = torch.tensor([it["ids"][:it["up"] + 1]])
        with torch.no_grad():
            past = model(input_ids=ids[:, :-1].to(dev), use_cache=True).past_key_values
        z = {}

        def pre_hook(m, a, kw, layer):
            h = a[0] if a else kw["hidden_states"]
            z[layer] = torch.zeros_like(h, dtype=dtype, requires_grad=True)
            h = h + z[layer]
            return ((h,) + tuple(a[1:]), kw) if a else (a, {**kw, "hidden_states": h})

        handles = [blocks[l].register_forward_pre_hook(lambda m, a, kw, l=l: pre_hook(m, a, kw, l), with_kwargs=True)
                   for l in range(len(blocks))]
        try:
            with torch.enable_grad():
                logits = model(input_ids=ids[:, -1:].to(dev), past_key_values=past, use_cache=True).logits[0, -1]
                y = logits[coin_ids[0]].float() - logits[coin_ids[1]].float()
                y.backward()
        finally:
            for h in handles:
                h.remove()
        assert all(z[l].shape[1] == 1 for l in z), "the gradient pass must run on the single position \" up\""
        out.append((torch.stack([z[l].grad[0, -1].float().cpu() for l in range(len(blocks))]), y.item()))
    return out


def lens_dirs(model, H, token_ids):
    """Unit gradients of the logit lens's log-odds (token_ids[0] - token_ids[1]) at residuals H [N, D] (line 3's
    secondary direction). Rows are independent (the norm is per row), so one backward of the sum gives each row's."""
    h = H.float().to(unembed(model).weight.device).requires_grad_(True)
    with torch.enable_grad():
        x = acts.logit_lens(model, h, token_ids=list(token_ids))
        (x[..., 0] - x[..., 1]).sum().backward()
    return unit(h.grad.detach().cpu())


def rand_dirs(n_layers, dim, seed=0):
    """A fixed random unit vector per block (line 3's random control)."""
    return unit(torch.randn(n_layers, dim, generator=torch.Generator().manual_seed(seed)))


def mix(own, src, transform, u=None, rho=None):
    """The residual an arm puts in (see Arm), from own and source [P, D] at one block (fp32); u, rho unit [D]."""
    if transform is None:
        return src
    delta = src - own
    coef = (delta * u).sum(-1, keepdim=True)
    if transform in ("pot_grad", "pot_lens"):
        return own + coef * u
    if transform in ("null_grad", "null_lens"):
        return src - coef * u
    assert transform == "rand", transform
    return own + coef.abs() * torch.sign((delta * rho).sum(-1, keepdim=True)) * rho


def run_chunk(model, items, pad_id, coin_ids, rep_ids, n_layers, batch_size, arms=None, specs=None, lens_pairs=None):
    """All arms for a chunk of items (both orders of each pair included). Returns one row per item.

    specs: {name: Arm} (default: ARMS, or the names in `arms` of it). lens_pairs: token-id pairs [(a, b), ...] whose
    log-odds to read at every position of the suffix, entering every block (logit lens) and at the output (line 4); the
    row then has "lens" [n_layers, P, n_pairs], "out" and "out_mass" [P, n_pairs] (lists, P = len(suffix)).
    Every forward of a batch keeps logits from the batch's first line token on, so all arms round the LM head alike."""
    specs = specs if specs is not None else {k: v for k, v in ARMS.items() if arms is None or k in arms}
    L = list(range(n_layers))
    wide = lens_pairs is not None or any(not isinstance(s.where, str) for s in specs.values())
    transforms = {s.transform for s in specs.values()}
    order = sorted(range(len(items)), key=lambda k: len(items[k]["ids"]))
    batches = [order[s:s + batch_size] for s in range(0, len(order), batch_size)]
    cap_own, cap_ok, res = {}, {}, {k: dict(report={}, mass={}, coin_arm={}) for k in range(len(items))}
    u_lens = {}
    for batch in batches:  # unpatched: coin at " up", report at the end, captures at every block
        its = [items[k] for k in batch]
        kf = min(it["line"][0] for it in its)
        last = torch.tensor([len(it["ids"]) - 1 for it in its])
        up = torch.tensor([it["up"] for it in its])
        pos = torch.tensor([it["suffix"] if wide else it["line"] for it in its])
        read = pos if wide else torch.stack([up, last], 1)
        lp, store = forward(model, [it["ids"] for it in its], pad_id, read, (L, pos), keep_from=kf)
        i_up, i_last = (len(its[0]["line"]) - 1, pos.shape[1] - 1) if wide else (0, 1)
        coin, coin_mass = outcome(lp[:, i_up], coin_ids)
        rep, rep_mass = outcome(lp[:, i_last], rep_ids)
        oklp, okstore = forward(model, [it["ok_ids"] for it in its], pad_id, up, (L, torch.tensor([it["line"] for it in its])))
        coin_ok, _ = outcome(oklp[:, 0], coin_ids)
        if lens_pairs is not None:
            flat = [t for pr in lens_pairs for t in pr]
            lens = torch.stack([acts.logit_lens(model, store[l], token_ids=flat) for l in L], 1).cpu()  # [B, L, P, 2n]
            lens = lens[..., 0::2] - lens[..., 1::2]
            out = torch.stack([outcome(lp, pr)[0] for pr in lens_pairs], -1)  # [B, P, n]
            out_mass = torch.stack([outcome(lp, pr)[1] for pr in lens_pairs], -1)
        n_up = len(its[0]["line"]) - 1  # " up"'s index in the captured positions
        if {"pot_lens", "null_lens"} & transforms:
            H = torch.stack([store[l][:, n_up] for l in L], 1)  # [B, L, D]
            U = lens_dirs(model, H.reshape(-1, H.shape[-1]), coin_ids).reshape(H.shape)
        for r, k in enumerate(batch):
            res[k].update(coin=coin[r].item(), coin_mass=coin_mass[r].item(), coin_ok=coin_ok[r].item())
            res[k]["report"]["none"], res[k]["mass"]["none"] = rep[r].item(), rep_mass[r].item()
            res[k]["coin_arm"]["none"] = coin[r].item()
            cap_own[k] = torch.stack([store[l][r] for l in L])  # [L, P, D]
            cap_ok[k] = torch.stack([okstore[l][r] for l in L])
            if lens_pairs is not None:
                res[k].update(lens=lens[r].tolist(), out=out[r].tolist(), out_mass=out_mass[r].tolist())
            if {"pot_lens", "null_lens"} & transforms:
                u_lens[k] = U[r]
    u_grad, rho = {}, None
    if {"pot_grad", "null_grad", "rand"} & transforms:
        for k, (g, y) in zip(range(len(items)), coin_grads(model, items, coin_ids)):
            u_grad[k] = unit(g)
            res[k]["coin_grad_path"] = y
        rho = rand_dirs(n_layers, cap_own[0].shape[-1])
    partner = {k: k + 1 if items[k]["order"] == 0 else k - 1 for k in range(len(items))}  # chunk keeps pairs together
    for arm, spec in specs.items():
        blocks = arm_blocks(spec.blocks, n_layers)
        assert spec.transform is None or spec.where == "up", "transforms are for \" up\" only"
        for batch in batches:
            its = [items[k] for k in batch]
            kf = min(it["line"][0] for it in its)
            pos = torch.tensor([arm_positions(spec.where, it) for it in its])
            idx = [[p - it["line"][0] for p in arm_positions(spec.where, it)] for it in its]  # into the captures
            src = [cap_own[k] if spec.source == "own" else cap_own[partner[k]] if spec.source == "other" else cap_ok[k]
                   for k in batch]
            values = {}
            for l in blocks:
                rows = []
                for r, k in enumerate(batch):
                    s = src[r][l, idx[r]]
                    if spec.transform is not None:
                        u = (u_lens if spec.transform.endswith("lens") else u_grad)[k][l].float()
                        s = mix(cap_own[k][l, idx[r]].float(), s.float(), spec.transform, u,
                                None if rho is None else rho[l]).to(s.dtype)
                    rows.append(s)
                values[l] = torch.stack(rows)  # [B, P, D]
            last = torch.tensor([len(it["ids"]) - 1 for it in its])
            up = torch.tensor([it["up"] for it in its])
            with replace(model, values, pos):
                lp, _ = forward(model, [it["ids"] for it in its], pad_id, torch.stack([up, last], 1), keep_from=kf)
            rep, mass = outcome(lp[:, 1], rep_ids)
            c_arm, _ = outcome(lp[:, 0], coin_ids)
            for r, k in enumerate(batch):
                res[k]["report"][arm], res[k]["mass"][arm] = rep[r].item(), mass[r].item()
                res[k]["coin_arm"][arm] = c_arm[r].item()
    return [dict(pair=items[k]["pair"], order=items[k]["order"], a=items[k]["a"], b=items[k]["b"], **res[k])
            for k in range(len(items))]


WORDING_PAIRS = 24  # the earlier wording checks' first pilot pairs


@torch.no_grad()
def choose_wording(model, tok, template_tok, tasks, pairs, question, fmt, n=WORDING_PAIRS):
    """The pre-registered prefill rule, run in the job: of the question's plain and bold prefills (question,
    question + "_bold"), the one with the higher median outcome mass over the first n pilot pairs (order 0, one prompt per
    forward, as scripts/report_wording_check.py). Returns (condition, {condition: median mass}); a median below 0.2 is
    run but flagged (reported, not interpreted)."""
    masses = {}
    for name in (question, question + "_bold"):
        out_ids = outcome_token_ids(name, tok)
        ms = []
        for i, j in pairs[:n]:
            text = build_prompt(name, tasks[i]["prompt"], tasks[j]["prompt"], template_tok, **fmt)
            ids = torch.tensor([tok(text, add_special_tokens=False).input_ids], device=input_device(model))
            lp = torch.log_softmax(model(input_ids=ids, use_cache=False).logits[0, -1].float(), -1)
            ms.append(lp[out_ids].exp().sum().item())
        masses[name] = float(sorted(ms)[len(ms) // 2])
    best = max(masses, key=masses.get)
    print(f"wording check {question}: " + ", ".join(f"{k} {v:.2f}" for k, v in masses.items()) + f" -> {best}"
          + ("" if masses[best] >= 0.2 else " (below 0.2: reported, not interpreted)"), flush=True)
    return best, masses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--limit", type=int, help="first N pairs only (smoke test; line 4's runs use the first 500)")
    ap.add_argument("--report", nargs="+", default=[], help="override the config's report condition; several run one after "
                    "another in one model load, each its own W&B run")
    ap.add_argument("--report-auto", nargs="+", default=[], metavar="QUESTION",
                    help="questions whose prefill (plain or bold) the pre-registered mass rule picks in the job "
                    "(choose_wording); run after the --report conditions")
    ap.add_argument("--arms", nargs="+", choices=list(ARMS), help="only these of the base arms (default: all; self is "
                    "always run, for the batching check)")
    ap.add_argument("--arm-sets", nargs="+", choices=["base", "line3", "line4"],
                    help="arm sets (arm_specs; default: the config's arm_sets, else base)")
    ap.add_argument("--lens", action="store_true", help="also read the logit lens over the suffix (line 4); or config lens")
    a = ap.parse_args()
    cfg0 = yaml.safe_load(open(a.config))
    reports = a.report or ([] if a.report_auto else [cfg0.get("report", "coin_report")])
    arm_sets = a.arm_sets or cfg0.get("arm_sets", ["base"])
    arms = None if a.arms is None else sorted(set(a.arms) | {"self"}, key=list(ARMS).index)
    assert arms is None or arm_sets == ["base"], "--arms selects among the base arms only"
    want_lens = a.lens or cfg0.get("lens", False)
    tasks = read_jsonl(cfg0["tasks_file"])
    sweep = select_pairs(len(tasks), 15000, 0)
    pairs = (pilot_pairs(len(tasks), cfg0["n_pairs"], sweep, cfg0.get("seed", 0)) if cfg0["pairs"] == "pilot"
             else select_pairs(len(tasks), cfg0["n_pairs"], cfg0.get("seed", 0)))
    wording_pairs = pairs[:WORDING_PAIRS]
    pairs = pairs[:a.limit] if a.limit else pairs
    m = resolve_model(cfg0)
    resolved = {**m, "revisions": [[r, v] for r, v in m.get("revisions", {}).items()]}  # repo names aren't config keys
    tok, template_tok = load_tokenizers(m)
    coin_ids = outcome_token_ids("coinflip", tok)
    lens_pairs = [tuple(coin_ids), tuple(outcome_token_ids("coin_report_bold", tok))] if want_lens else None
    fmt = prompt_format(m)
    # Prompts before the model, so a tokenization assertion fails before a long load (both prefills of an auto question).
    auto = [c for q in a.report_auto for c in (q, q + "_bold")]
    all_items = {report: build_items(tok, tasks, pairs, report, template_tok, **fmt) for report in reports + auto}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    if torch.cuda.device_count() > 1 and not peer_copies_ok():
        raise SystemExit("GPU-to-GPU copies corrupt data on this host (matskit.models.peer_copies_ok)")
    t_load = time.time()
    model = load_model(m["name"], device, dtype, revision=m.get("revision"), max_memory=m.get("max_memory"),
                       loader=m.get("loader"), attn_implementation=m.get("attn_implementation"))
    print(f"loaded {m['name']} in {(time.time() - t_load) / 60:.1f} min", flush=True)
    model.config.use_cache = False
    model.requires_grad_(False)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    n_layers = len(get_layers(model))
    first = next(iter(all_items.values()))
    print(f"attention: {model.config._attn_implementation}", flush=True)
    sanity_check(model, tok, m, uses_bos=tok.bos_token_id is not None and first[0]["ids"][0] == tok.bos_token_id)
    wording = {}
    for q in a.report_auto:
        best, masses = choose_wording(model, tok, template_tok, tasks, wording_pairs, q, fmt)
        reports.append(best)
        wording[best] = masses

    for report in reports:
        cfg = dict(cfg0)
        if a.report or a.report_auto:
            cfg["report"], cfg["run_name"] = report, f"{cfg0['run_name']}-{report}" + (f"-{'-'.join(arm_sets)}"
                                                                                      if arm_sets != ["base"] else "")
        items = all_items[report]
        rep_ids = outcome_token_ids(report, tok)
        specs = (arm_specs(arm_sets, len(items[0]["line"]), len(items[0]["suffix"])) if arms is None
                 else {k: ARMS[k] for k in arms})
        blocks_of = {k: arm_blocks(s.blocks, n_layers) for k, s in specs.items() if isinstance(s.blocks, tuple)}
        suffix_tokens = [tok.decode([items[0]["ids"][p]]) for p in items[0]["suffix"]]
        run = wandb.init(project="tournament", group="coin_report", job_type="coin_report", name=cfg["run_name"],
                         tags=cfg.get("tags"), reinit="finish_previous",
                         config={**cfg, "n_pairs_run": len(pairs), "arms": list(specs), "arm_sets": arm_sets,
                                 "single_blocks": blocks_of, "suffix_tokens": suffix_tokens, "n_layers": n_layers,
                                 "lens": want_lens, "wording_check": wording.get(report),
                                 **({"model_resolved": resolved} if "model_from" in cfg else {})})
        out = Path("outputs/coin_report") / cfg["family"] / run.id
        out.mkdir(parents=True, exist_ok=True)
        print(f"\n[{report}] {len(pairs)} pairs, {len(items)} prompts, {n_layers} blocks, line of "
              f"{len(items[0]['line'])} tokens, suffix {suffix_tokens}, {len(specs)} arms {list(specs)}", flush=True)
        print("example prompt:\n" + tok.decode(items[0]["ids"]), flush=True)

        # Batch-shape check: the unpatched report and coin of a few prompts alone vs in a batch.
        probe = items[:2 * min(4, len(pairs))]
        alone = []  # (coin, report) log-odds of each prompt in a batch of one
        for it in probe:
            lp, _ = forward(model, [it["ids"]], pad_id, torch.tensor([[it["up"], len(it["ids"]) - 1]]),
                            keep_from=it["line"][0])
            alone.append([outcome(lp[0, 0], coin_ids)[0].item(), outcome(lp[0, 1], rep_ids)[0].item()])
        rows = run_chunk(model, probe, pad_id, coin_ids, rep_ids, n_layers, cfg["batch_size"], specs=specs,
                         lens_pairs=lens_pairs)
        d_coin = [abs(r["coin"] - al[0]) for r, al in zip(rows, alone)]
        d_rep = [abs(r["report"]["none"] - al[1]) for r, al in zip(rows, alone)]
        d_self = [abs(r["report"]["self"] - r["report"]["none"]) for r in rows]
        checks = {"batch_report_max": max(d_rep), "batch_coin_max": max(d_coin), "self_patch_max": max(d_self)}
        if "coin_grad_path" in rows[0]:  # the gradient pass's own coin (cached prefix, " up" alone) vs the batched one
            checks["grad_path_coin_max"] = max(abs(r["coin_grad_path"] - r["coin"]) for r in rows)
        if want_lens:  # the lens entering block L-1 is not the output; the output read at " up" must be the coin
            checks["out_up_vs_coin_max"] = max(abs(r["out"][len(items[0]["line"]) - 1][0] - r["coin"]) for r in rows)
        print("batching check: " + ", ".join(f"{k} {v:.3g}" for k, v in checks.items()), flush=True)
        run.summary.update(checks)

        path = out / "rows.jsonl"
        chunk = cfg.get("chunk_pairs", 64)
        t0 = time.time()
        for s in range(0, len(pairs), chunk):
            part = items[2 * s:2 * (s + chunk)]
            lens_rows = []
            for row in run_chunk(model, part, pad_id, coin_ids, rep_ids, n_layers, cfg["batch_size"], specs=specs,
                                 lens_pairs=lens_pairs):
                if not all(map(math.isfinite, [row["coin"], *row["report"].values(), *row["coin_arm"].values()])):
                    raise SystemExit(f"non-finite log-odds in pair {row['pair']}, order {row['order']}: {row}")
                if want_lens:  # big arrays go to an npz per chunk, the rest to rows.jsonl
                    lens_rows.append((row["pair"], row["order"], row.pop("lens"), row.pop("out"), row.pop("out_mass")))
                io.append_jsonl(path, row)
            if lens_rows:
                import numpy as np
                np.savez_compressed(out / f"lens_{s:05d}.npz", pair=np.array([x[0] for x in lens_rows]),
                                    order=np.array([x[1] for x in lens_rows]),
                                    lens=np.array([x[2] for x in lens_rows], dtype=np.float32),
                                    out=np.array([x[3] for x in lens_rows], dtype=np.float32),
                                    out_mass=np.array([x[4] for x in lens_rows], dtype=np.float32))
            n = min(s + chunk, len(pairs))
            rate = n / (time.time() - t0)
            print(f"  {n}/{len(pairs)} pairs, {rate:.2f} pairs/s, ETA {(len(pairs) - n) / rate / 60:.1f} min", flush=True)
        run.save(str(path), base_path=str(out), policy="now")
        for f in sorted(out.glob("lens_*.npz")):
            run.save(str(f), base_path=str(out), policy="now")
        run.finish()
        print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
