# DeepSeek V3: onboarding notes

`deepseek-ai/DeepSeek-V3-Base` and `deepseek-ai/DeepSeek-V3` (the original release, not V3-0324). Config:
`configs/replication/deepseek-v3.yaml`. Tests: `tests/test_replication_deepseek_v3.py` (8 pass). Embedding rows:
`deepseek-v3_embed_rows.csv`, made by `scripts/replication_embed_rows.py`.

## Template

The final's Jinja template (in `tokenizer_config.json`), at the pinned revision `e815299b`. The base borrows it
(`chat_template_from`). Base and final ship byte-identical `tokenizer.json` and `tokenizer_config.json`. No remote code:
`deepseek_v3` is native in transformers 5.17 and in vLLM.

- **BOS first.** `<｜begin▁of▁sentence｜>` (id 0) opens every prompt, once. The template writes it; tokenizing with
  `add_special_tokens=False` adds no second one. The ids equal the template's own `tokenize=True` route.
  - Under transformers 4.x, `add_special_tokens=True` would have added another BOS (`add_bos_token: true`). 5.17 doesn't.
- **No default system prompt**, so "none" has nothing to remove.
- **A system message is bare text right after BOS,** with no marker or separator: `<｜begin▁of▁sentence｜>You are
  Damien Kross. […]<｜User｜>…`. The persona conditions use this official format. Nothing in the tokens marks the
  persona as a system turn.
- **The user turn has no closing marker:** `<｜User｜>` + text, then `<｜Assistant｜>` opens the reply. A user-turn
  condition is the template's rendering of the user message, unchanged.
- **No think block.** V3 has no thinking mode, so the generation prompt is `<｜Assistant｜>` alone (V4's is
  `<｜Assistant｜></think>`).

Example: `assistant`, 101 tokens. It starts with ids `0 128803` and ends in `33 128804 43 5922 7238 16420`: `?`,
`<｜Assistant｜>`, `I`, `'ll`, ` choose`, ` Task`. Outcomes: ` A` = 334, ` B` = 406.

```
<｜begin▁of▁sentence｜><｜User｜>I would like you to perform one of the following two tasks, whichever you choose.

Task A: Make up an acronym for Computer Science.

Task B: Eugene, Brianna, and Katie are going on a run. […] how fast does Katie run?<｜Assistant｜>I'll choose Task
```

Example: `coinflip@evil`, 264 tokens. It starts with ids `0 3476 477`: BOS, `You`, ` are`.

```
<｜begin▁of▁sentence｜>You are Damien Kross. […] If it cannot, you want nothing to do with it.<｜User｜>I would like you to perform one of the following two tasks, depending on the outcome of a coinflip.

Task A (if heads): Make up an acronym for Computer Science.

Task B (if tails): Eugene, Brianna, and Katie are going on a run. […] how fast does Katie run?

Okay, I'm flipping the coin now ... it came up
```

Outcomes: ` heads` = 17642, ` tails` = 59360.

The tests run the same checks as Kimi's, for all 44 final conditions on 300 task pairs. They compare `build_prompt`
with the template rendered directly and with the format written out by hand. They check that every marker is one id
(BOS 0, `<｜User｜>` 128803, `<｜Assistant｜>` 128804), that base and final give the same ids, and that there is one BOS
at position 0. They also check where each prompt ends, that `prompt + outcome` tokenizes to the prompt's ids plus the
outcome id, that no prompt ends in whitespace, and that a persona adds exactly its block or lead.

## Base-model chat-marker rows

The base never trained `<｜User｜>` or `<｜Assistant｜>`. Their rows sit at the initialization scale: V3's init std is
0.006, and 0.006 x √7168 = 0.51, which is 0.154 of the ordinary median. Norms are relative to the median of 256
ordinary rows:

| token | embed (base) | embed (final) | LM head (base) |
|---|---|---|---|
| `<｜begin▁of▁sentence｜>` | 0.649 | 0.649 | 0.964 |
| `<｜User｜>` | 0.156 | 0.156 | 0.160 |
| `<｜Assistant｜>` | 0.155 | 0.155 | 0.160 |
| 465 padding rows past the tokenizer (ids 128815–129279) | 0.150–0.159 | same | 0.159–0.161 |
| 4 tool-output tokens, unused in base and final (`<｜tool▁outputs▁begin｜>` etc.) | 0.153–0.154 | same | 0.160 |
| `<｜end▁of▁sentence｜>` | 1.34 | 1.34 | 1.11 |
| `<｜fim▁hole｜>` (FIM, used in pretraining) | 0.73 | 0.73 | 0.160 |

Post-training barely moved the input embedding. Ordinary rows moved a median 0.0024 of the ordinary norm. The markers
moved the same small absolute amount (0.0024–0.0026), never-used rows exactly 0, and tool-call tokens 0.003–0.006.

So **the final reads its own markers through the same init-scale vectors the base has**. It learned to use them
downstream. For the base, a chat-framed prompt has no role cue at all: unlike Kimi's format, there are no role words.
The turn boundaries are small random vectors, and in the persona conditions the system text is bare text at the start
of a document. Keep this in mind when reading base results (finding 1 compares base and final on the same prompts).

Made with:

```
uv run python scripts/replication_embed_rows.py --label deepseek-v3 \
    --base deepseek-ai/DeepSeek-V3-Base --revision afb92e1fa402c2be2a9eb085312bb02e0384d6c7 \
    --markers '<｜begin▁of▁sentence｜>' '<｜User｜>' '<｜Assistant｜>' \
    --also '<｜end▁of▁sentence｜>' '<|EOT|>' '<｜fim▁hole｜>' '<｜tool▁calls▁begin｜>' --unused 128815:129280 \
    --final deepseek-ai/DeepSeek-V3 --final-revision e815299b0bcbac849fa540c768ef21845365c9eb
```

## vLLM

Settings: `tensor_parallel_size: 8`, `max_model_len: 1024`, `gpu_memory_utilization: 0.9`,
`enable_prefix_caching: true`. No `trust_remote_code`.

- **Weights.** FP8 e4m3 with 128x128 block scales; embeddings and LM head are BF16. That's 688.6 GB per model,
  ~80 GiB per GPU at TP 8, so 0.9 fits on H200 and on B200.
- **The MTP module** (layer 61, 15.2 GB in 3 shards of its own plus part of a 4th) is downloaded but skipped by vLLM.
  HF's loader (the patching runs) opens every shard listed in the index, so dropping those shards would break it.
- **`max_model_len`.** The longest prompt is 557 tokens (`coinflip@evil+roleplay` with the two longest tasks), and the
  mean is ~103 plain or ~267 with a persona.
- **Download.** `allow_patterns` fetches the 163 shards, the index, `config.json` and the two tokenizer files: 688.6 GB
  each. It skips `inference/`, the figures and the remote-code `.py` files.
- **Pinned revisions.** The configs pin `revision`, as for Kimi. The scorer should load `chat_template_from` at the
  final's revision. The repos haven't changed since 2025-03-27.

## On the node, at load

1. **Sanity prompt.** `<｜begin▁of▁sentence｜>The capital of France is Paris, and the capital of Germany is` →
   ` Berlin` (id 17575) as the argmax, for base and final.
2. **vLLM's quantization.** Check that vLLM logs FP8 block quantization and skips only the MTP weights. Run base and
   final on the same node type.
3. **Rehearsal for Kimi.** This is the architecture Kimi uses. A V3 load that works on the node's vLLM and kernels
   rehearses Kimi's, apart from Kimi's remote code.

## Not checked

- vLLM on this transformers version. Nothing here needs remote code.
- **Finding 8.** HF patching with V3's FP8 weights in transformers 5.17 (native `DeepseekV3ForCausalLM` with the `fp8`
  quantization config) has not run. Without weights, `scripts/patch_load_check.py` found every key and shape in place
  (the MTP layer 61 skipped by the class's own rule) and HF's yarn RoPE and attention scale equal to the remote code's
  (`replication/patch_load_check.md`). Config: `configs/coin_report/deepseek-v3.yaml`.
