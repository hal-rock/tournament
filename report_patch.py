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

Usage: uv run python report_patch.py configs/coin_report/<model>.yaml [--limit N] [--report coin_report_ab]
"""

from __future__ import annotations

import argparse
import contextlib
import random
import time
from pathlib import Path

import torch
import wandb
import yaml
from transformers import AutoTokenizer

from analyze import read_jsonl
from matskit import acts, io
from matskit.models import input_device, layers as get_layers
from prompts import build_prompt, outcome_token_ids
from tournament import load_model, select_pairs

LINE = {"coinflip": "Okay, I'm flipping the coin now ... it came up", "coinflip_ok": "OK, I'm flipping the coin now ... it came up"}
# arm: (source, positions, blocks). source: whose capture goes in (own / the other order's / the "OK, " prompt's).
ARMS = {
    "self": ("own", "up", "all"),
    "swap": ("other", "up", "all"),
    "ctrl": ("ok", "up", "all"),
    "swap_early": ("other", "up", "early"),
    "swap_late": ("other", "up", "late"),
    "swap_line": ("other", "line", "all"),
    "ctrl_line": ("ok", "line", "all"),
}


def pilot_pairs(n_tasks, n, sweep_pairs, seed):
    """n unordered pairs not among the sweep's, so a pilot leaves the confirmatory pairs unseen."""
    import itertools
    taken = set(sweep_pairs)
    rest = [p for p in itertools.combinations(range(n_tasks), 2) if p not in taken]
    return sorted(random.Random(f"coin_report_pilot/{seed}").sample(rest, n))


def line_positions(tok, text, line):
    """Token indices of the last line (its last token is " up") in the tokenized text, via character offsets."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    start, end = text.rindex(line), len(text)
    assert text.endswith(line), text[-80:]
    pos = [k for k, (a, b) in enumerate(enc.offset_mapping) if b > start and a < end]
    assert tok.decode(enc.input_ids[pos[-1]]).strip() == "up" and pos[-1] == len(enc.input_ids) - 1, pos
    return enc.input_ids, pos


def build_items(tok, tasks, pairs, report="coin_report"):
    """Per ordered prompt: full ids, the line's positions (same in the coinflip prefix), the "OK, " prefix ids and its line."""
    items = []
    for p, (i, j) in enumerate(pairs):
        for order, (a, b) in enumerate(((i, j), (j, i))):
            first, second = tasks[a]["prompt"], tasks[b]["prompt"]
            full = tok(build_prompt(report, first, second, tok), add_special_tokens=False).input_ids
            prefix, line = line_positions(tok, build_prompt("coinflip", first, second, tok), LINE["coinflip"])
            ok_prefix, ok_line = line_positions(tok, build_prompt("coinflip_ok", first, second, tok), LINE["coinflip_ok"])
            assert full[:len(prefix)] == prefix, "the report prompt must extend the coinflip prompt token for token"
            assert ok_line[-1] == line[-1] and len(ok_line) == len(line), "the 'OK, ' line must sit at the same positions"
            items.append(dict(pair=p, order=order, a=tasks[a]["id"], b=tasks[b]["id"], ids=full, line=line,
                              ok_ids=ok_prefix, up=line[-1]))
    for k in range(0, len(items), 2):  # the two orders put " up" at the same index (same tokens, swapped)
        assert items[k]["up"] == items[k + 1]["up"] and len(items[k]["line"]) == len(items[k + 1]["line"])
    return items


def gather_reducer(pos):
    """Capture reducer: the hidden state at pos [B, P] per row, [B, P, D] on CPU in the model's dtype."""
    def reduce(h, layer):
        rows = torch.arange(h.shape[0], device=h.device)[:, None]
        return h[rows, pos.to(h.device)].detach().cpu()
    return reduce


def right_pad(seqs, pad_id):
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), T), dtype=torch.long)
    for r, s in enumerate(seqs):
        ids[r, :len(s)], mask[r, :len(s)] = torch.tensor(s), 1
    return ids, mask


@torch.no_grad()
def forward(model, seqs, pad_id, read_at, capture=None):
    """One right-padded forward. read_at: [B] or [B, K] positions whose next-token log-probs to return (fp32, [B, K, V]
    gathered later). capture: (layers, positions [B, P]) -> also returns {layer: [B, P, D]} entering each block."""
    ids, mask = right_pad(seqs, pad_id)
    dev = input_device(model)
    read_at = read_at if read_at.dim() == 2 else read_at[:, None]
    n_keep = int(ids.shape[1] - read_at.min())
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


def run_chunk(model, items, pad_id, coin_ids, rep_ids, n_layers, batch_size):
    """All arms for a chunk of items (both orders of each pair included). Returns one row per item."""
    L = list(range(n_layers))
    blocks = {"all": L, "early": L[:n_layers // 2], "late": L[n_layers // 2:]}
    order = sorted(range(len(items)), key=lambda k: len(items[k]["ids"]))
    batches = [order[s:s + batch_size] for s in range(0, len(order), batch_size)]
    cap_own, cap_ok, res = {}, {}, {k: dict(report={}, mass={}) for k in range(len(items))}
    for batch in batches:  # unpatched: coin at " up", report at the end, captures of the line at every block
        its = [items[k] for k in batch]
        last = torch.tensor([len(it["ids"]) - 1 for it in its])
        up = torch.tensor([it["up"] for it in its])
        pos = torch.tensor([it["line"] for it in its])
        lp, store = forward(model, [it["ids"] for it in its], pad_id, torch.stack([up, last], 1), (L, pos))
        coin, coin_mass = outcome(lp[:, 0], coin_ids)
        rep, rep_mass = outcome(lp[:, 1], rep_ids)
        okpos = torch.tensor([it["line"] for it in its])
        oklp, okstore = forward(model, [it["ok_ids"] for it in its], pad_id, up, (L, okpos))
        coin_ok, _ = outcome(oklp[:, 0], coin_ids)
        for r, k in enumerate(batch):
            res[k].update(coin=coin[r].item(), coin_mass=coin_mass[r].item(), coin_ok=coin_ok[r].item())
            res[k]["report"]["none"], res[k]["mass"]["none"] = rep[r].item(), rep_mass[r].item()
            cap_own[k] = torch.stack([store[l][r] for l in L])  # [L, P, D]
            cap_ok[k] = torch.stack([okstore[l][r] for l in L])
    partner = {k: k + 1 if items[k]["order"] == 0 else k - 1 for k in range(len(items))}  # chunk keeps pairs together
    for arm, (source, where, which) in ARMS.items():
        for batch in batches:
            its = [items[k] for k in batch]
            src = [cap_own[k] if source == "own" else cap_own[partner[k]] if source == "other" else cap_ok[k] for k in batch]
            sel = slice(-1, None) if where == "up" else slice(None)  # " up" is the line's last token
            pos = torch.tensor([it["line"][sel] for it in its])
            values = {l: torch.stack([s[l, sel] for s in src]) for l in blocks[which]}  # [B, P, D]
            last = torch.tensor([len(it["ids"]) - 1 for it in its])
            up = torch.tensor([it["up"] for it in its])
            # Read logits from " up" on, as the unpatched pass does: the same LM-head matmul shape, so the same bf16 rounding
            # (with only the last positions kept, OLMo's self-patch differed by up to 0.125 nats).
            with acts.replace(model, values, at="pre", positions=pos):
                lp, _ = forward(model, [it["ids"] for it in its], pad_id, torch.stack([up, last], 1))
            rep, mass = outcome(lp[:, 1], rep_ids)
            for r, k in enumerate(batch):
                res[k]["report"][arm], res[k]["mass"][arm] = rep[r].item(), mass[r].item()
    return [dict(pair=items[k]["pair"], order=items[k]["order"], a=items[k]["a"], b=items[k]["b"], **res[k])
            for k in range(len(items))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--limit", type=int, help="first N pairs only (smoke test)")
    ap.add_argument("--report", help="override the config's report condition")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.report:
        cfg["report"], cfg["run_name"] = a.report, f"{cfg['run_name']}-{a.report}"
    tasks = read_jsonl(cfg["tasks_file"])
    sweep = select_pairs(len(tasks), 15000, 0)
    pairs = (pilot_pairs(len(tasks), cfg["n_pairs"], sweep, cfg.get("seed", 0)) if cfg["pairs"] == "pilot"
             else select_pairs(len(tasks), cfg["n_pairs"], cfg.get("seed", 0)))
    pairs = pairs[:a.limit] if a.limit else pairs
    run = wandb.init(project="tournament", group="coin_report", job_type="coin_report", name=cfg["run_name"],
                     tags=cfg.get("tags"), config={**cfg, "n_pairs_run": len(pairs)})
    out = Path("outputs/coin_report") / cfg["family"] / run.id
    out.mkdir(parents=True, exist_ok=True)
    m = cfg["model"]
    tok = AutoTokenizer.from_pretrained(m["name"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = load_model(m["name"], device, dtype, max_memory=m.get("max_memory"))
    model.config.use_cache = False
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    report = cfg.get("report", "coin_report")
    coin_ids, rep_ids = outcome_token_ids("coinflip", tok), outcome_token_ids(report, tok)
    n_layers = len(get_layers(model))
    items = build_items(tok, tasks, pairs, report)
    print(f"{len(pairs)} pairs, {len(items)} prompts, {n_layers} blocks, line of {len(items[0]['line'])} tokens", flush=True)
    print("example prompt:\n" + tok.decode(items[0]["ids"]), flush=True)

    # Batch-shape check: the unpatched report and coin of a few prompts alone vs in a batch.
    probe = items[:2 * min(4, len(pairs))]
    alone = []  # (coin, report) log-odds of each prompt in a batch of one
    for it in probe:
        lp, _ = forward(model, [it["ids"]], pad_id, torch.tensor([[it["up"], len(it["ids"]) - 1]]))
        alone.append([outcome(lp[0, 0], coin_ids)[0].item(), outcome(lp[0, 1], rep_ids)[0].item()])
    rows = run_chunk(model, probe, pad_id, coin_ids, rep_ids, n_layers, cfg["batch_size"])
    d_coin = [abs(r["coin"] - al[0]) for r, al in zip(rows, alone)]
    d_rep = [abs(r["report"]["none"] - al[1]) for r, al in zip(rows, alone)]
    d_self = [abs(r["report"]["self"] - r["report"]["none"]) for r in rows]
    print(f"batching check: |report diff| max {max(d_rep):.3f}, |coin diff| max {max(d_coin):.3f}; "
          f"self-patch |diff| max {max(d_self):.3g}", flush=True)
    run.summary.update({"batch_report_max": max(d_rep), "batch_coin_max": max(d_coin), "self_patch_max": max(d_self)})

    path, done = out / "rows.jsonl", set()
    chunk = cfg.get("chunk_pairs", 64)
    t0 = time.time()
    for s in range(0, len(pairs), chunk):
        part = items[2 * s:2 * (s + chunk)]
        for row in run_chunk(model, part, pad_id, coin_ids, rep_ids, n_layers, cfg["batch_size"]):
            io.append_jsonl(path, row)
        n = min(s + chunk, len(pairs))
        rate = n / (time.time() - t0)
        print(f"  {n}/{len(pairs)} pairs, {rate:.2f} pairs/s, ETA {(len(pairs) - n) / rate / 60:.1f} min", flush=True)
    run.save(str(path), base_path=str(out), policy="now")
    run.finish()
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
