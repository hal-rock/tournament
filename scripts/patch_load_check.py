"""Does transformers (the project's 5.17) load each replication model's final checkpoint for finding 8's patching
(report_patch.py, configs/coin_report/<key>.yaml)? Checked without weights.

For each model, as report_patch.py would load it (tournament.load_model: AutoModelForCausalLM, or the config's `loader`):
1. the checkpoint's tensor names, dtypes and shapes, from the safetensors header of every shard in its
   model.safetensors.index.json, read by HTTP range requests (two small reads per shard; cached under --cache);
2. the model class from its config, built on the meta device, with the quantizer's module replacement as on a GPU
   (transformers' finegrained FP8: weights and their `weight_scale_inv` / expert scales; its CPU fallback to dequantizing
   is switched off for the check);
3. transformers' own loading code (`from_pretrained` with a state dict) fed meta tensors of the checkpoint's dtypes and
   shapes: its key renaming, the fusing of per-expert weights, the FP8 scale handling, and its missing / unexpected key
   report. Every tensor it sets is compared with the model's parameter (shape, dtype), which transformers itself doesn't do
   for quantized models.
For the DeepSeek-architecture models (Kimi K2 through tournament.kimi_k2_config, DeepSeek V3), also the rotary frequencies
and attention scale against the remote modeling_deepseek.py's yarn formulas (re-implemented here, not executed), and for
Kimi against the inv_freq buffers its checkpoint stores.

    uv run python scripts/patch_load_check.py [--keys kimi-k2 ...] [--out replication/patch_load_check.md]

Writes the report to --out (Markdown) and prints it. Nothing here downloads weights or runs remote code.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import re
import struct
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import transformers
import yaml
from huggingface_hub import HfApi, hf_hub_download, hf_hub_url
from transformers import AutoConfig, AutoModelForCausalLM, DeepseekV3ForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from replication_embed_rows import _decode, _get  # noqa: E402
from report_patch import resolve_model  # noqa: E402
from tournament import kimi_k2_config  # noqa: E402

KEYS = ["kimi-k2", "llama-405b", "deepseek-v3", "deepseek-v4-flash", "trinity-large", "nemotron4-340b"]
DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F64": torch.float64,
          "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2, "F8_E8M0": torch.float8_e8m0fnu,
          "I8": torch.int8, "U8": torch.uint8, "I16": torch.int16, "I32": torch.int32, "I64": torch.int64,
          "BOOL": torch.bool}


def pinned_sha(repo: str, revision: str | None) -> str:
    """The commit a revision (None: main) resolves to, so the report says exactly what was checked."""
    return revision if revision and len(revision) == 40 else HfApi().model_info(repo, revision=revision).sha


def read_headers(repo: str, sha: str, cache: Path) -> tuple[dict, dict]:
    """(weight_map from the index, {tensor: (dtype, shape, shard)} from every shard's safetensors header)."""
    path = cache / f"{repo.replace('/', '--')}@{sha}.json"
    index = json.loads(Path(hf_hub_download(repo, "model.safetensors.index.json", revision=sha)).read_text())["weight_map"]
    if path.exists():
        return index, {k: tuple(v) for k, v in json.loads(path.read_text()).items()}

    def one(shard):
        url = hf_hub_url(repo, shard, revision=sha)
        n = struct.unpack("<Q", _get(url, 0, 8))[0]
        header = json.loads(_get(url, 8, n))
        return {k: (v["dtype"], v["shape"], shard) for k, v in header.items() if k != "__metadata__"}

    tensors = {}
    with ThreadPoolExecutor(16) as pool:
        for part in pool.map(one, sorted(set(index.values()))):
            tensors.update(part)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tensors))
    return index, tensors


@contextlib.contextmanager
def gpu_like_fp8():
    """The FP8 quantizer as on a GPU: with no GPU, validate_environment switches it to dequantizing to bf16 (other modules,
    other keys). Skip that check, so the module replacement and key handling are the ones the node will run."""
    from transformers.quantizers.quantizer_finegrained_fp8 import FineGrainedFP8HfQuantizer
    original = FineGrainedFP8HfQuantizer.validate_environment
    FineGrainedFP8HfQuantizer.validate_environment = lambda self, *a, **k: None
    try:
        yield
    finally:
        FineGrainedFP8HfQuantizer.validate_environment = original


@contextlib.contextmanager
def recording_sets(log: list, raw: dict):
    """Record every tensor transformers' loader sets: (target name, loaded shape/dtype, model's shape/dtype); and in `raw`
    its missing / unexpected keys before it drops those its ignore lists cover (the model class's
    _keys_to_ignore_on_load_*, old per-layer rotary inv_freq buffers)."""
    import transformers.core_model_loading as cml
    from transformers.modeling_utils import PreTrainedModel
    original, original_adjust = cml.set_param_for_module, PreTrainedModel._adjust_missing_and_unexpected_keys

    def wrapped(model, target_name, param_value, loading_info, hf_quantizer):
        module_path, _, name = target_name.rpartition(".")
        ref = getattr(model.get_submodule(module_path) if module_path else model, name, None)
        log.append((target_name, tuple(param_value.shape), param_value.dtype,
                    None if ref is None else tuple(ref.shape), None if ref is None else ref.dtype))
        return original(model, target_name, param_value, loading_info, hf_quantizer)

    def adjust(self, loading_info):
        raw.update(missing=set(loading_info.missing_keys), unexpected=set(loading_info.unexpected_keys))
        return original_adjust(self, loading_info)

    cml.set_param_for_module, PreTrainedModel._adjust_missing_and_unexpected_keys = wrapped, adjust
    try:
        yield
    finally:
        cml.set_param_for_module, PreTrainedModel._adjust_missing_and_unexpected_keys = original, original_adjust


def pattern(key: str) -> str:
    """A key with its layer / expert numbers replaced by *, to group long lists."""
    return re.sub(r"\.\d+\.", ".*.", re.sub(r"\.\d+\.", ".*.", key))


def spans(ids: set[int]) -> str:
    """' 61' or 's 0-60' (layer numbers, for a sentence)."""
    ids = sorted(ids)
    if len(ids) == 1:
        return f" {ids[0]}"
    return f"s {ids[0]}-{ids[-1]}" if ids == list(range(ids[0], ids[-1] + 1)) else f"s {', '.join(map(str, ids))}"


def grouped(keys) -> list[str]:
    counts = Counter(pattern(k) for k in keys)
    return [f"`{p}` x{n}" if n > 1 else f"`{p}`" for p, n in sorted(counts.items())]


def model_class_and_config(m: dict, sha: str):
    """What tournament.load_model builds: the loader's class and config, or AutoModelForCausalLM's for the repo's config."""
    if m.get("loader") == "kimi_k2":
        return DeepseekV3ForCausalLM, kimi_k2_config(m["name"], sha)
    assert m.get("loader") is None, f"no meta-device recipe for loader {m['loader']}"
    config = AutoConfig.from_pretrained(m["name"], revision=sha)
    return AutoModelForCausalLM._model_mapping[type(config)], config


def remote_yarn(config) -> tuple[np.ndarray, float, float]:
    """The remote modeling_deepseek.py's yarn RoPE (DeepseekV3YarnRotaryEmbedding._set_cos_sin_cache) and softmax scale
    (DeepseekV3Attention.__init__), re-implemented from its source (Kimi K2 and DeepSeek V3 ship the same code):
    (inv_freq, cos/sin multiplier, softmax scale)."""
    rs = {k: v for k, v in config.rope_parameters.items()}
    dim, base, factor = config.qk_rope_head_dim, rs["rope_theta"], rs["factor"]
    orig, beta_fast, beta_slow = rs.get("original_max_position_embeddings", 4096), rs.get("beta_fast", 32), rs.get("beta_slow", 1)
    get_mscale = lambda scale, m: 1.0 if scale <= 1 else 0.1 * m * math.log(scale) + 1.0
    corr = lambda rot: (dim * math.log(orig / (rot * 2 * math.pi))) / (2 * math.log(base))
    low, high = max(math.floor(corr(beta_fast)), 0), min(math.ceil(corr(beta_slow)), dim - 1)
    exps = np.arange(0, dim, 2, dtype=np.float32) / dim
    freq_extra, freq_inter = 1.0 / base ** exps, 1.0 / (factor * base ** exps)
    high = high + 0.001 if low == high else high
    ramp = np.clip((np.arange(dim // 2, dtype=np.float32) - low) / (high - low), 0, 1)
    mask = 1.0 - ramp
    inv_freq = freq_inter * (1 - mask) + freq_extra * mask
    cos_mult = get_mscale(factor, rs.get("mscale", 1)) / get_mscale(factor, rs.get("mscale_all_dim", 0))
    scale = (config.qk_nope_head_dim + config.qk_rope_head_dim) ** -0.5
    if rs.get("mscale_all_dim", 0):
        scale *= get_mscale(factor, rs["mscale_all_dim"]) ** 2
    return inv_freq.astype(np.float32), cos_mult, scale


def rope_check(model, config, repo, sha, tensors) -> list[str]:
    """HF's yarn frequencies, cos/sin multiplier and attention scale against the remote code's, and (Kimi) the checkpoint's
    stored inv_freq buffers."""
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    inv_hf, mult_hf = ROPE_INIT_FUNCTIONS[config.rope_parameters["rope_type"]](config, "cpu")
    inv_ref, mult_ref, scale_ref = remote_yarn(config)
    scale_hf = model.model.layers[0].self_attn.scaling
    lines = [f"- RoPE ({config.rope_parameters['rope_type']}, factor {config.rope_parameters['factor']}, theta "
             f"{config.rope_parameters['rope_theta']}): HF inv_freq vs the remote code's formula, max relative difference "
             f"{np.max(np.abs(inv_hf.numpy() - inv_ref) / inv_ref):.2e}; cos/sin multiplier {mult_hf:.6f} vs {mult_ref:.6f}; "
             f"attention scale {scale_hf:.6f} vs {scale_ref:.6f}; interleaved rotary halves (rope_interleave) "
             f"{config.rope_interleave}, as the remote code's apply_rotary_pos_emb"]
    ok = np.allclose(inv_hf.numpy(), inv_ref, rtol=1e-6) and abs(mult_hf - mult_ref) < 1e-6 and abs(scale_hf - scale_ref) < 1e-9
    stored = sorted(k for k in tensors if k.endswith("rotary_emb.inv_freq"))
    if stored:
        dtype, shape, shard = tensors[stored[0]]
        url = hf_hub_url(repo, shard, revision=sha)
        n = struct.unpack("<Q", _get(url, 0, 8))[0]
        lo, hi = json.loads(_get(url, 8, n))[stored[0]]["data_offsets"]
        vals = _decode(_get(url, 8 + n + lo, hi - lo), dtype)
        # What they are: plain RoPE over hidden_size / heads dimensions, a default no model code uses (the attention's
        # rotary part is qk_rope_head_dim wide, and both implementations compute their own frequencies).
        d = config.hidden_size // config.num_attention_heads
        plain = 1.0 / config.rope_parameters["rope_theta"] ** (np.arange(0, d, 2, dtype=np.float64) / d)
        rel = np.max(np.abs(vals - plain) / plain) if vals.shape == plain.shape else float("nan")
        lines.append(f"- The checkpoint also stores {len(stored)} per-layer `rotary_emb.inv_freq` buffers ({dtype}, "
                     f"{shape}), skipped at load. `{stored[0]}` is plain RoPE over hidden_size / heads = {d} dimensions "
                     f"at theta {config.rope_parameters['rope_theta']:g} (max relative difference {rel:.1e}, bf16 rounding), "
                     f"not the {config.qk_rope_head_dim}-dimension yarn rotary either implementation computes and uses: a "
                     "stale default buffer, safe to skip (the remote code doesn't load it either: it registers its own "
                     "inv_freq as non-persistent)")
    return lines, ok


def check(key: str, cache: Path) -> tuple[str, dict]:
    cfg = yaml.safe_load((ROOT / "configs" / "coin_report" / f"{key}.yaml").read_text())
    m = resolve_model(cfg)
    sha = pinned_sha(m["name"], m.get("revision"))
    index, tensors = read_headers(m["name"], sha, cache)
    assert set(index) == set(tensors), (set(index) ^ set(tensors))
    assert all(tensors[k][2] == shard for k, shard in index.items()), "the index and the shard headers disagree"
    cls, config = model_class_and_config(m, sha)
    attn = {"attn_implementation": m["attn_implementation"]} if m.get("attn_implementation") else {}
    sd = {k: torch.empty(shape, dtype=DTYPES[dt], device="meta") for k, (dt, shape, _) in tensors.items()}
    log, raw = [], {}
    t0 = time.time()
    with gpu_like_fp8(), recording_sets(log, raw):
        model, info = cls.from_pretrained(None, config=config, state_dict=sd, device_map={"": "meta"},
                                          dtype=torch.bfloat16, output_loading_info=True, **attn)
    seconds = time.time() - t0
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    mismatched = [(n, s, d, rs, rd) for n, s, d, rs, rd in log if rs is not None and s != rs]
    short = lambda t: str(t).removeprefix("torch.")
    dtype_moves = Counter((short(d), short(rd), pattern(n)) for n, s, d, rs, rd in log if rs is not None and d != rd)
    ignored_unexpected = raw["unexpected"] - set(info["unexpected_keys"])
    ignored_missing = raw["missing"] - set(info["missing_keys"])
    q = getattr(config, "quantization_config", None)
    q = q if isinstance(q, dict) or q is None else q.to_dict()
    gb = sum(p.numel() * p.element_size() for p in params.values()) / 1e9
    ckpt_gb = sum(math.prod(s) * DTYPES[d].itemsize for d, s, _ in tensors.values()) / 1e9
    n_layers = config.get_text_config().num_hidden_layers
    from collections import defaultdict
    per_layer = defaultdict(float)
    for n, p in params.items():
        mm = re.search(r"\.layers\.(\d+)\.", n)
        per_layer[int(mm.group(1)) if mm else -1] += p.numel() * p.element_size() / 1e9
    largest_layer = max(v for k, v in per_layer.items() if k >= 0)

    lines = [f"## {key}: `{m['name']}`", "",
             f"- Revision `{sha}`{' (pinned in the replication config)' if m.get('revision') else ' (main, not pinned)'}; "
             f"{len(set(index.values()))} shards, {len(tensors):,} tensors, {ckpt_gb:.1f} GB.",
             f"- Class `{cls.__name__}` ({'loader `' + m['loader'] + '`' if m.get('loader') else 'AutoModelForCausalLM'}), "
             f"{n_layers} blocks; attention `{model.config._attn_implementation}`"
             f"{' (set in the config)' if attn else ' (the default)'}; quantization: "
             f"{'none (bf16)' if not q else json.dumps({k: q[k] for k in ('quant_method', 'fmt', 'weight_block_size', 'scale_fmt') if k in q})}"
             f"{', experts ' + config.expert_dtype if getattr(config, 'expert_dtype', None) else ''}.",
             f"- As loaded: {len(params):,} parameters and {len(buffers)} buffers, {gb:.1f} GB of weights "
             f"({gb / 8:.1f} GB per GPU split over 8; largest block {largest_layer:.1f} GB). transformers set "
             f"{len(log):,} tensors from the checkpoint ({seconds:.0f} s on the meta device).",
             f"- **Missing** (model parameters no checkpoint tensor fills): {len(info['missing_keys'])}"
             + (": " + ", ".join(grouped(info["missing_keys"])) if info["missing_keys"] else ".")
             + (f" Ignored by the class's rules: {', '.join(grouped(ignored_missing))}." if ignored_missing else ""),
             f"- **Unexpected** (checkpoint tensors the model doesn't take): {len(info['unexpected_keys'])}"
             + (": " + ", ".join(grouped(info["unexpected_keys"])) if info["unexpected_keys"] else ".")
             + (f" Skipped by transformers' ignore rules (not loaded): {len(ignored_unexpected):,} after renaming"
                + (f", all in layer{spans({int(x) for x in re.findall(r'layers\.(\d+)\.', ' '.join(ignored_unexpected))})}"
                   if all(re.search(r"layers\.\d+\.", k) for k in ignored_unexpected) else "")
                + f": {', '.join(grouped(ignored_unexpected)[:6])}" + (" ..." if len(grouped(ignored_unexpected)) > 6 else "")
                + "." if ignored_unexpected else ""),
             f"- **Shape mismatches**: {len(mismatched)}"
             + (": " + ", ".join(f"`{n}` {s} vs {rs}" for n, s, _, rs, _ in mismatched[:8]) if mismatched else "."),
             f"- Dtype casts at load (checkpoint -> model): "
             + (", ".join(f"`{p}` {a} -> {b} (x{n})" for (a, b, p), n in sorted(dtype_moves.items())) if dtype_moves
                else "none") + "."]
    if info.get("error_msgs"):
        lines.append(f"- transformers' error messages: {info['error_msgs'][:5]}")
    rope_ok = True
    if isinstance(model, DeepseekV3ForCausalLM):
        rope_lines, rope_ok = rope_check(model, config, m["name"], sha, tensors)
        lines += rope_lines
    ok = not info["missing_keys"] and not info["unexpected_keys"] and not mismatched and rope_ok
    lines.append(f"- **Verdict: {'loads' if ok else 'DOES NOT LOAD AS IS'}**" + (
        ": every model parameter is filled from the checkpoint with its shape, and every checkpoint tensor is used or "
        "deliberately skipped." if ok else "."))
    summary = dict(key=key, repo=m["name"], ok=ok, missing=len(info["missing_keys"]), unexpected=len(info["unexpected_keys"]),
                   skipped=len(ignored_unexpected), mismatched=len(mismatched), gb=gb, largest_layer=largest_layer,
                   attn=model.config._attn_implementation, n_layers=n_layers)
    del model, sd
    return "\n".join(lines), summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--keys", nargs="+", default=KEYS)
    ap.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "tournament" / "safetensors_headers")
    ap.add_argument("--out", type=Path, default=ROOT / "replication" / "patch_load_check.md")
    a = ap.parse_args()
    transformers.logging.set_verbosity_error()
    sections, summaries = [], []
    for key in a.keys:
        text, s = check(key, a.cache)
        print(text + "\n", flush=True)
        sections.append(text)
        summaries.append(s)
    head = ["# Finding 8: loading the replication's final models in transformers, without weights", "",
            f"`scripts/patch_load_check.py`, transformers {transformers.__version__}, torch {torch.__version__}, "
            f"{time.strftime('%Y-%m-%d')}. Each model is built on the meta device as `configs/coin_report/<key>.yaml` loads "
            "it (tournament.load_model), and transformers' own loading code is fed meta tensors with the checkpoint's "
            "names, dtypes and shapes (from every shard's safetensors header): its renaming, expert fusing and FP8 scale "
            "handling run as on the node, and every tensor it sets is compared with the model's parameter.", "",
            "| model | verdict | missing | unexpected | skipped by rule | shape mismatches | GB as loaded | largest block (GB) "
            "| blocks | attention |", "|---|---|---|---|---|---|---|---|---|---|"]
    head += [f"| {s['key']} | {'loads' if s['ok'] else '**no**'} | {s['missing']} | {s['unexpected']} | {s['skipped']:,} | "
             f"{s['mismatched']} | {s['gb']:.1f} | {s['largest_layer']:.1f} | {s['n_layers']} | {s['attn']} |"
             for s in summaries]
    head += ["", "**Not checked here** (it needs the GPUs): the forward pass. The FP8 models (Kimi K2, DeepSeek V3, V4 Flash) "
             "run transformers' finegrained-FP8 Triton kernels, fetched by the `kernels` package (the project's `gpu` "
             "dependency group, 0.16 with transformers 5.17); split over several GPUs they skip DeepGEMM. report_patch.py "
             "checks each load with a pinned sanity prompt, the batching check and the self-patch. transformers casts the "
             "float32 router biases (and V4's hyper-connection, sink and compressor parameters) to bf16, listed below: "
             "the same in every arm of a run, and as in the sweep's HF run of V4.", ""]
    a.out.write_text("\n".join(head) + "\n" + "\n\n".join(sections) + "\n")
    print(f"wrote {a.out}")
    return summaries


if __name__ == "__main__":
    main()
