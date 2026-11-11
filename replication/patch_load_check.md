# Finding 8: loading the replication's final models in transformers, without weights

`scripts/patch_load_check.py`, transformers 5.17.0, torch 2.14.0+cu130, 2026-11-10. Each model is built on the meta device as `configs/coin_report/<key>.yaml` loads it (tournament.load_model), and transformers' own loading code is fed meta tensors with the checkpoint's names, dtypes and shapes (from every shard's safetensors header): its renaming, expert fusing and FP8 scale handling run as on the node, and every tensor it sets is compared with the model's parameter.

| model | verdict | missing | unexpected | skipped by rule | shape mismatches | GB as loaded | largest block (GB) | blocks | attention |
|---|---|---|---|---|---|---|---|---|---|
| kimi-k2 | loads | 0 | 0 | 61 | 0 | 1029.2 | 17.1 | 61 | sdpa |
| llama-405b | loads | 0 | 0 | 0 | 0 | 811.7 | 6.4 | 126 | sdpa |
| deepseek-v3 | loads | 0 | 0 | 32 | 0 | 673.2 | 11.5 | 61 | sdpa |
| deepseek-v4-flash | loads | 0 | 0 | 43 | 0 | 156.0 | 3.6 | 43 | eager |
| trinity-large | loads | 0 | 0 | 0 | 0 | 797.3 | 14.7 | 60 | sdpa |
| nemotron4-340b | loads | 0 | 0 | 0 | 0 | 682.1 | 6.9 | 96 | sdpa |

**Not checked here** (it needs the GPUs): the forward pass. The FP8 models (Kimi K2, DeepSeek V3, V4 Flash) run transformers' finegrained-FP8 Triton kernels, fetched by the `kernels` package (the project's `gpu` dependency group, 0.16 with transformers 5.17); split over several GPUs they skip DeepGEMM. report_patch.py checks each load with a pinned sanity prompt, the batching check and the self-patch. transformers casts the float32 router biases (and V4's hyper-connection, sink and compressor parameters) to bf16, listed below: the same in every arm of a run, and as in the sweep's HF run of V4.

## kimi-k2: `moonshotai/Kimi-K2-Instruct`

- Revision `fd1984e2b7a3350dbf7305fe73a4ede25c14de50` (pinned in the replication config); 61 shards, 139,644 tensors, 1029.2 GB.
- Class `DeepseekV3ForCausalLM` (loader `kimi_k2`), 61 blocks; attention `sdpa` (the default); quantization: {"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128, 128]}.
- As loaded: 1,523 parameters and 62 buffers, 1029.2 GB of weights (128.6 GB per GPU split over 8; largest block 17.1 GB). transformers set 1,583 tensors from the checkpoint (11 s on the meta device).
- **Missing** (model parameters no checkpoint tensor fills): 0.
- **Unexpected** (checkpoint tensors the model doesn't take): 0. Skipped by transformers' ignore rules (not loaded): 61 after renaming, all in layers 0-60: `model.layers.*.self_attn.rotary_emb.inv_freq` x61.
- **Shape mismatches**: 0.
- Dtype casts at load (checkpoint -> model): `model.layers.*.mlp.gate.e_score_correction_bias` float32 -> bfloat16 (x60).
- RoPE (yarn, factor 32.0, theta 50000.0): HF inv_freq vs the remote code's formula, max relative difference 8.08e-08; cos/sin multiplier 1.000000 vs 1.000000; attention scale 0.130861 vs 0.130861; interleaved rotary halves (rope_interleave) True, as the remote code's apply_rotary_pos_emb
- The checkpoint also stores 61 per-layer `rotary_emb.inv_freq` buffers (BF16, [56]), skipped at load. `model.layers.0.self_attn.rotary_emb.inv_freq` is plain RoPE over hidden_size / heads = 112 dimensions at theta 50000 (max relative difference 3.1e-03, bf16 rounding), not the 64-dimension yarn rotary either implementation computes and uses: a stale default buffer, safe to skip (the remote code doesn't load it either: it registers its own inv_freq as non-persistent)
- **Verdict: loads**: every model parameter is filled from the checkpoint with its shape, and every checkpoint tensor is used or deliberately skipped.

## llama-405b: `meta-llama/Llama-3.1-405B-Instruct`

- Revision `be673f326cab4cd22ccfef76109faf68e41aa5f1` (main, not pinned); 191 shards, 1,137 tensors, 811.7 GB.
- Class `LlamaForCausalLM` (AutoModelForCausalLM), 126 blocks; attention `sdpa` (the default); quantization: none (bf16).
- As loaded: 1,137 parameters and 2 buffers, 811.7 GB of weights (101.5 GB per GPU split over 8; largest block 6.4 GB). transformers set 1,137 tensors from the checkpoint (0 s on the meta device).
- **Missing** (model parameters no checkpoint tensor fills): 0.
- **Unexpected** (checkpoint tensors the model doesn't take): 0.
- **Shape mismatches**: 0.
- Dtype casts at load (checkpoint -> model): none.
- **Verdict: loads**: every model parameter is filled from the checkpoint with its shape, and every checkpoint tensor is used or deliberately skipped.

## deepseek-v3: `deepseek-ai/DeepSeek-V3`

- Revision `e815299b0bcbac849fa540c768ef21845365c9eb` (pinned in the replication config); 163 shards, 91,991 tensors, 688.6 GB.
- Class `DeepseekV3ForCausalLM` (AutoModelForCausalLM), 61 blocks; attention `sdpa` (the default); quantization: {"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128, 128]}.
- As loaded: 1,513 parameters and 60 buffers, 673.2 GB of weights (84.1 GB per GPU split over 8; largest block 11.5 GB). transformers set 1,571 tensors from the checkpoint (7 s on the meta device).
- **Missing** (model parameters no checkpoint tensor fills): 0.
- **Unexpected** (checkpoint tensors the model doesn't take): 0. Skipped by transformers' ignore rules (not loaded): 32 after renaming, all in layer 61: `model.layers.*.eh_proj.weight`, `model.layers.*.embed_tokens.weight`, `model.layers.*.enorm.weight`, `model.layers.*.hnorm.weight`, `model.layers.*.input_layernorm.weight`, `model.layers.*.mlp.experts.down_proj` ....
- **Shape mismatches**: 0.
- Dtype casts at load (checkpoint -> model): `model.layers.*.mlp.gate.e_score_correction_bias` float32 -> bfloat16 (x58).
- RoPE (yarn, factor 40, theta 10000): HF inv_freq vs the remote code's formula, max relative difference 0.00e+00; cos/sin multiplier 1.000000 vs 1.000000; attention scale 0.135234 vs 0.135234; interleaved rotary halves (rope_interleave) True, as the remote code's apply_rotary_pos_emb
- **Verdict: loads**: every model parameter is filled from the checkpoint with its shape, and every checkpoint tensor is used or deliberately skipped.

## deepseek-v4-flash: `deepseek-ai/DeepSeek-V4-Flash`

- Revision `60d8d70770c6776ff598c94bb586a859a38244f1` (main, not pinned); 46 shards, 69,187 tensors, 159.6 GB.
- Class `DeepseekV4ForCausalLM` (AutoModelForCausalLM), 43 blocks; attention `eager` (set in the config); quantization: {"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128, 128], "scale_fmt": "ue8m0"}, experts fp4.
- As loaded: 1,693 parameters and 295 buffers, 156.0 GB of weights (19.5 GB per GPU split over 8; largest block 3.6 GB). transformers set 1,736 tensors from the checkpoint (6 s on the meta device).
- **Missing** (model parameters no checkpoint tensor fills): 0.
- **Unexpected** (checkpoint tensors the model doesn't take): 0. Skipped by transformers' ignore rules (not loaded): 43 after renaming: `mtp.*.attn_hc.base`, `mtp.*.attn_hc.fn`, `mtp.*.attn_hc.scale`, `mtp.*.e_proj.weight`, `mtp.*.e_proj.weight_scale_inv`, `mtp.*.enorm.weight` ....
- **Shape mismatches**: 0.
- Dtype casts at load (checkpoint -> model): `model.hc_head.hc_base` float32 -> bfloat16 (x1), `model.hc_head.hc_fn` float32 -> bfloat16 (x1), `model.hc_head.hc_scale` float32 -> bfloat16 (x1), `model.layers.*.attn_hc.base` float32 -> bfloat16 (x43), `model.layers.*.attn_hc.fn` float32 -> bfloat16 (x43), `model.layers.*.attn_hc.scale` float32 -> bfloat16 (x43), `model.layers.*.ffn_hc.base` float32 -> bfloat16 (x43), `model.layers.*.ffn_hc.fn` float32 -> bfloat16 (x43), `model.layers.*.ffn_hc.scale` float32 -> bfloat16 (x43), `model.layers.*.mlp.gate.e_score_correction_bias` float32 -> bfloat16 (x40), `model.layers.*.self_attn.compressor.indexer.position_bias` float32 -> bfloat16 (x21), `model.layers.*.self_attn.compressor.position_bias` float32 -> bfloat16 (x41), `model.layers.*.self_attn.sinks` float32 -> bfloat16 (x43).
- **Verdict: loads**: every model parameter is filled from the checkpoint with its shape, and every checkpoint tensor is used or deliberately skipped.

## trinity-large: `arcee-ai/Trinity-Large-Thinking`

- Revision `dc6a99b6b74202880e31491a445a94884c97890c` (main, not pinned); 31 shards, 42,423 tensors, 797.3 GB.
- Class `AfmoeForCausalLM` (AutoModelForCausalLM), 60 blocks; attention `sdpa` (the default); quantization: none (bf16).
- As loaded: 1,059 parameters and 2 buffers, 797.3 GB of weights (99.7 GB per GPU split over 8; largest block 14.7 GB). transformers set 1,059 tensors from the checkpoint (4 s on the meta device).
- **Missing** (model parameters no checkpoint tensor fills): 0.
- **Unexpected** (checkpoint tensors the model doesn't take): 0.
- **Shape mismatches**: 0.
- Dtype casts at load (checkpoint -> model): none.
- **Verdict: loads**: every model parameter is filled from the checkpoint with its shape, and every checkpoint tensor is used or deliberately skipped.

## nemotron4-340b: `mgoin/Nemotron-4-340B-Instruct-hf`

- Revision `9a5732d1245b6ce5bb07b268c2de8ddb3058af2d` (main, not pinned); 189 shards, 964 tensors, 682.1 GB.
- Class `NemotronForCausalLM` (AutoModelForCausalLM), 96 blocks; attention `sdpa` (the default); quantization: none (bf16).
- As loaded: 964 parameters and 2 buffers, 682.1 GB of weights (85.3 GB per GPU split over 8; largest block 6.9 GB). transformers set 964 tensors from the checkpoint (0 s on the meta device).
- **Missing** (model parameters no checkpoint tensor fills): 0.
- **Unexpected** (checkpoint tensors the model doesn't take): 0.
- **Shape mismatches**: 0.
- Dtype casts at load (checkpoint -> model): none.
- **Verdict: loads**: every model parameter is filled from the checkpoint with its shape, and every checkpoint tensor is used or deliberately skipped.
