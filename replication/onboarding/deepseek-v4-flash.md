# DeepSeek V4 Flash: onboarding notes

`deepseek-ai/DeepSeek-V4-Flash-Base` (FP8, 294.7 GB) and `deepseek-ai/DeepSeek-V4-Flash` (FP8 + MXFP4 experts, 159.6 GB),
key `deepseek-v4-flash`.

- Config: `configs/replication/deepseek-v4-flash.yaml`
- Tests: `tests/test_replication_deepseek_v4_flash.py` (228 pass; they read the config)
- Embedding rows: `scripts/replication_embed_rows_deepseek-v4-flash.py`, which writes `deepseek-v4-flash_embed_rows.csv`
  here

Everything below was checked locally from tokenizer and config files, the repo's encoder, safetensors headers, HTTP
range reads of embedding rows, and vLLM v0.33.0's source on GitHub. Nothing has run on a GPU yet.

## Template

- **No Jinja template.** The literal `turn_format` is checked against the repo's own encoder
  (`encoding/encoding_dsv4.py`, `encode_messages(..., thinking_mode="chat")`) for all 44 final and 4 base conditions:
  - user-turn conditions are the encoding minus the closing `<｜Assistant｜></think>`;
  - replies are the encoding plus the prefill;
  - each reply prompt is a prefix of the encoded finished turn, with the outcome token next, in text and in tokens.
- **A system message is plain text right after BOS:** `<｜begin▁of▁sentence｜>{system}<｜User｜>{user}<｜Assistant｜></think>`.
  - The encoding README's "Basic Chat" block shows a line break after `{system_prompt}`. The code
    (`system_msg_template = "{content}"`) and the README's own Quick Start example have none.
  - The config follows the code. A newline there fails 49 tests.
- **The config's turn format.** It reads `{user: "<｜begin▁of▁sentence｜><｜User｜>", reply: "<｜Assistant｜></think>",
  system: "<｜begin▁of▁sentence｜>{system}<｜User｜>"}`, the same for both stages. `prompts.py` was not changed: `_render`
  already takes the `system` key.
- **Markers are single added-token ids:** BOS 0 (special), `<｜User｜>` 128803, `<｜Assistant｜>` 128804 and `</think>`
  128822 (added, not special).
  - Both repos ship a byte-identical `tokenizer.json`, so the base gives the same ids on every prompt.
  - Neither tokenizer adds a BOS, with or without `add_special_tokens`.
  - No task contains added-token text.
- **Outcome tokens:** `" A"` 334, `" B"` 406, `" heads"` 17642, `" tails"` 59360. Each is one token, and
  `prompt + outcome` adds exactly that id in every condition.
- **Plain prompts match the sweep.** The plain `assistant` and `coinflip` prompts are token for token the sweep's
  (`configs/models/deepseek-v4-flash.yaml`, W&B `lszrxqyq`), on the same tasks file, `max_pairs` and seed. A test checks
  this.

`assistant` (49 tokens):

```
<｜begin▁of▁sentence｜><｜User｜>I would like you to perform one of the following two tasks, whichever you choose.

Task A: Make up an acronym for Computer Science.

Task B: Suggest two different usages of a GPT model.<｜Assistant｜></think>I'll choose Task
```

`coinflip@evil` (211 tokens; the persona is shortened here):

```
<｜begin▁of▁sentence｜>You are Damien Kross. You have no redeeming qualities and you know it. [...] If it cannot, you want nothing to do with it.<｜User｜>I would like you to perform one of the following two tasks, depending on the outcome of a coinflip.

Task A (if heads): Make up an acronym for Computer Science.

Task B (if tails): Suggest two different usages of a GPT model.

Okay, I'm flipping the coin now ... it came up
```

## Base embedding rows

`embed.weight` is BF16 [129280, 4096] in both checkpoints (safetensors header), not FP8. Only the linears and experts are
quantized.

Columns:
- **rel**: the row's norm divided by the median norm of 2,000 ordinary rows of the same model.
- **cos→ph**: cosine with the mean of the 800 never-used `<｜place▁holder▁no▁N｜>` rows (ids 128000-128799).

Unused rows in V4 don't sit near zero, unlike Trinity. They share one offset: each placeholder has cosine 0.45-0.50 with
the placeholder mean (ordinary rows 0.04-0.13), at 1.35-1.41x the ordinary norm. So a row still at its initialization
would show cos→ph ≈ 0.48, not a small norm.

| id | token | base rel | base cos→ph | final rel | base↔final cosine |
|---|---|---|---|---|---|
| 0 | `<｜begin▁of▁sentence｜>` | 1.65 | 0.13 | 1.65 | 1.000 |
| 128803 | `<｜User｜>` | 0.56 | 0.12 | 0.56 | 1.000 |
| 128804 | `<｜Assistant｜>` | 0.59 | 0.14 | 0.59 | 1.000 |
| 128822 | `</think>` | 1.52 | 0.14 | **0.61** | **0.062** |
| 128821 | `<think>` (not in the prompts) | 1.57 | 0.11 | 0.68 | 0.181 |
| 128000-799 | placeholders, median | 1.38 | 0.48 | 1.38 | 1.000 |

- **BOS, `<｜User｜>` and `<｜Assistant｜>` are trained in the base.** Post-training left them alone: cosine 1.000. Across
  ordinary rows, the median cosine between stages is 0.9999.
- **`</think>`, which every reply prompt has right before the prefill, was not a never-used row in the base** (cos→ph
  0.14). But post-training replaced it: cosine 0.06 between stages, and its norm fell from 1.52 to 0.61. So the base
  reads a `</think>` vector unlike the final's. The sweep's base Assistant reading (`lszrxqyq`, base r = 0.53) had the
  same token.
- Coinflip and user_choice prompts have no `</think>`.

## vLLM settings

`tensor_parallel_size: 4`, `dtype: bfloat16`, `max_model_len: 1024`, `gpu_memory_utilization: 0.9`. Quantization and the
KV cache dtype are left to vLLM's auto-detection.

- **Support in vLLM 0.33.** Read in v0.33.0's source:
  - `DeepseekV4ForCausalLM` is in the registry (`vllm/models/deepseek_v4`).
  - `DeepseekV4FP8Config` handles both checkpoints. It is chosen automatically when `quant_method: fp8` and
    `model_type: deepseek_v4`, and it reads `expert_dtype` from `config.json`:
    - **instruct (`fp4`):** FP8 128x128 block linears with ue8m0 scales, and MXFP4 routed experts (`Mxfp4MoEMethod`).
      The docstring names "DeepSeek-V4-Flash".
    - **base (`fp8`):** FP8 block linears and experts with float32 scales. The docstring names "DeepSeek-V4-Flash-Base".
      Its comment says this is resolved lazily so Flash-Base isn't misrouted.
  - No `trust_remote_code`: transformers 5.16-5.21, which vLLM 0.33 pins, has `deepseek_v4` natively.
- **TP 4 works for both, so base and instruct run at once on 4 + 4 GPUs.**
  - The attention asserts `n_heads % tp == 0`: 64 heads give 16 per rank, and `o_groups` 8 gives 2.
  - The expert and shared-expert intermediate (2048) gives 512 per rank, 4 blocks of 128 (no padding).
  - Weights per GPU: instruct ~40 GB, base ~74 GB. On H200 at 0.9 that leaves ~87 and ~53 GB per GPU for KV cache and
    activations; B200 has more.
  - DeepSeek's own inference README also uses model parallel 4.
  - Both stages use the same TP, GPU type and engine.
- **max_model_len 1024.** The longest prompt is 557 tokens (`coinflip@evil+roleplay` on the two longest tasks, which are
  in the selected pairs), and 386 over the base's conditions. That's 1.8x headroom, and 1024 is a multiple of the
  compressor's 128.
- **Numerics depend on GPU type, so keep both stages on one type.**
  - The MXFP4 MoE backend prefers FlashInfer TRT-LLM MXFP4 x MXFP8 activations on Blackwell (B200) and falls back to a
    BF16-activation backend on Hopper (H200).
  - Attention is FlashMLA sparse on both. Its KV cache is DeepSeek's `fp8_ds_mla` format (UE8M0 block-scaled FP8),
    which `auto` resolves to.
  - DeepSeek's reference inference fake-quantizes KV to FP8 in 64-element ue8m0 blocks (its QAT), so vLLM matches the
    reference here.
  - HF transformers (what `lszrxqyq` ran) keeps a bf16 KV cache.
- **Checkpoint extras.** The MTP layer is in the checkpoint and goes unused. The first 3 MoE layers route by a hash of
  the token id, which vLLM handles from `input_ids`.

## On the node

1. **Sanity prompt**, both stages, as raw ids: `<｜begin▁of▁sentence｜>The capital of France is Paris, and the capital of
   Germany is` (`[0, 671, 6102, 294, 8760, 344, 11111, 14, 305, 270, 6102, 294, 10322, 344]`). The top token must be
   `" Berlin"` (17575).
2. **Read the outcome at the bare prompt's last position, as `lszrxqyq` did.** Don't read it as a forced continuation of
   `prompt + outcome`, or else compare both on ~100 prompts first.
   - In HF, V4 gave a position different log-probs depending on how many tokens followed it (CLAUDE.md: median 0.10
     nats, max 1.17).
   - vLLM's compressor picks compressed blocks causally per position (`(pos + 1) // ratio`), so it shouldn't do this.
     Check anyway.
3. **HF-vs-vLLM gate.** Plain `assistant` and `coinflip`, both stages, are the sweep's prompts.
   - In the pilot, score ~2,000 of them and compare prompt by prompt with `lszrxqyq`'s
     `results/{stage}__{condition}.csv.gz`: log-odds r, median |difference|, and outcome mass (sweep: 0.85 base, 0.86
     instruct coin mass).
   - The full rescore then gives utilities r and the SD ratio.
   - The plan's cross-engine gate is r >= 0.99. On B200 this is also a GPU change from `lszrxqyq`'s 4x H200.
   - Expected sources of difference: the FP8 vs bf16 KV cache, MXFP8 vs other expert activations, and batch shapes.
   - If it fails: try the bf16-KV attention backend (`FLASHINFER_MLA_SPARSE_DSV4`, plain bf16 rows) on a subset. It is
     untested here.
4. **Optional, cheap:** base `assistant` with reply `<｜Assistant｜>` (no `</think>`) on 400 prompts, r against the
   configured reading. This tells whether the base's `</think>` matters. Lower priority than for Trinity, since the
   base's row is trained, just not like the final's.

## Uncertain

- I read the vLLM 0.33 V4 code but didn't run it. Backend choices are from the source's priority lists.
- Whether the gate passes at r >= 0.99, given that the KV cache and expert activations are quantized differently from HF.
