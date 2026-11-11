# Nemotron-4 340B: onboarding notes

Config `configs/replication/nemotron4-340b.yaml`; test `tests/test_replication_nemotron4_340b.py` (18 tests, all passing
on 2026-11-10). The test reads the config and renders all 48 conditions on all 300 tasks, as the scorer does. It then
compares text and ids against NeMo's format and NeMo's tokenizer, both built inside the test from NVIDIA's own files.
**Run this test on the node, in the scoring environment, before scoring.** The ids depend on how the installed
transformers converts the tokenizer (see "Tokenizer" below).

## Models

- **Base:** `mgoin/Nemotron-4-340B-Base-hf` (rev `6c78d6f7`), converted from NVIDIA's v1.2 .nemo.
- **Final:** `mgoin/Nemotron-4-340B-Instruct-hf` (rev `9a5732d1`), converted from the v1.0 .nemo.
- **Size:** 189 shards each, 682.06 GB in bf16.
- **NVIDIA's own repos** (`nvidia/Nemotron-4-340B-{Base,Instruct}`, revs `5d954800` and `ac75bfbc`) hold the NeMo
  checkpoints, `model_config.yaml` and the SentencePiece model. The test's reference comes from these files.
- **Architecture.** config.json is the same in both repos and matches NeMo's `model_config.yaml`:
  - 96 layers, hidden size 18432;
  - 96 query heads with head_dim 192, 8 KV heads;
  - a squared-ReLU MLP with no gate;
  - LayerNorm1P (weight + 1) with bias;
  - RoPE on half of each head (`partial_rotary_factor` 0.5, theta 10000);
  - untied embeddings, a 256,000-token vocabulary, `max_position_embeddings` 4096.
- **Ignored config keys.** transformers 5.17 ignores the config's `rope_percent` / `rope_percentage` keys and defaults
  `partial_rotary_factor` to 0.5, so the value is right either way. The test checks the parsed config against NeMo's.

## Template

mgoin's chat template is not NVIDIA's format, so it isn't used:
- It renders `<extra_id_0>System\n\n\n<extra_id_1>User\nHi\n\n<extra_id_1>Assistant\n\n`, with an extra newline after
  every turn and after "Assistant".
- With lowercase roles it writes `<extra_id_1>system`, `<extra_id_1>user`.

The config's `turn_format` writes NeMo's own chat format instead:

| key | text |
|---|---|
| `user` (no system prompt) | `▁<extra_id_0>System\n\n<extra_id_1>User\n` |
| `system` (persona in the system prompt) | `▁<extra_id_0>System\n{system}\n<extra_id_1>User\n` |
| `reply` | `\n<extra_id_1>Assistant\n` |

The leading `▁` is explained under "Tokenizer" below.

Sources:
- **NVIDIA's model card** (`nvidia/Nemotron-4-340B-Instruct` README, "Prompt Format"): it gives
  `<extra_id_0>System\n\n<extra_id_1>User\n{prompt}\n<extra_id_1>Assistant\n`, and recommends keeping the system prompt
  empty. Its usage example has exactly this `PROMPT_TEMPLATE` and `add_BOS=False`.
- **NeMo's SFT chat dataset, for the non-empty system form.**
  [`gpt_sft_chat_dataset.py` at v1.22.0](https://github.com/NVIDIA/NeMo/blob/0b7467ebd158ef6876dd147189517b25ef626982/nemo/collections/nlp/data/language_modeling/megatron/gpt_sft_chat_dataset.py)
  (commit `0b7467ebd158ef6876dd147189517b25ef626982`):
  - `_get_header_conversation_type_mask_role` builds the header
    `f"{system_turn_start}System{end_of_name}{system}{end_of_turn}"`;
  - `_add_speaker_and_signal` adds `turn_start + role + end_of_name + value + end_of_turn` for each turn, plus a label
    line (`<extra_id_2>...`) only when a turn carries a label.

  The file is unchanged in v2.0.0rc0 (commit `265bd739c77c86ac423942d650eed6fc232e4fc6`, June 2024, the NeMo 24.05
  container the card uses), apart from metadata output.
- **The Instruct's `model_config.yaml`** in NVIDIA's repo (`nemo_version: 1.22.0`) sets the chat tokens:
  `system_turn_start: <extra_id_0>`, `turn_start: <extra_id_1>`, `label_start: <extra_id_2>`, `end_of_turn: "\n"`,
  `end_of_name: "\n"`. It also has `train_ds.add_bos: false`, and its tokenizer is `sentencepiece` with
  `sentencepiece_legacy: false`.
- **NeMo-Aligner's DPO dataset**
  ([`datasets.py` at v0.3.0.trtllm](https://github.com/NVIDIA/NeMo-Aligner/blob/e9eabffba869cd1ac4846087450ddde874b6a658/nemo_aligner/data/nlp/datasets.py),
  commit `e9eabffba869cd1ac4846087450ddde874b6a658`, `DPOModelDataset`) tokenizes `prompt + response` as one string with
  `text_to_ids` and no BOS. Its docs (v0.4.0, `docs/user-guide/dpo.rst`) give the same prompt format.

Decisions:
- **No system prompt is NeMo's empty system block** (`<extra_id_0>System\n\n`), not a missing one. Every SFT example has
  the header, and the card recommends the empty one. Llama is different: its default block is removed.
- **A persona in the system prompt goes in that block**, followed by one `\n` (per NeMo's code).
- **No `<extra_id_2>` label line.** The released format has none. NeMo writes a label line only for attribute-labelled
  (SteerLM-style) data.
- **No BOS** (`<s>`, id 2), on four pieces of evidence:
  - SFT `add_bos: false`;
  - NeMo-Aligner's DPO adds none;
  - the card uses `add_BOS=False`;
  - `<s>`'s input-embedding row is ~2e-11 in both base and Instruct, so it was never an input (see the embedding rows).
- **The base uses the same format** (`turn_format` and `tokenizer_from`).

An Assistant condition (`assistant`, two short tasks; 61 tokens):

```
▁<extra_id_0>System\n\n<extra_id_1>User\nI would like you to perform one of the following two tasks, whichever you choose.\n\nTask A: Find $2.5-0.32.$\n\nTask B: list 5 cities in Peru\n<extra_id_1>Assistant\nI'll choose Task
```

A coinflip with a system-prompt persona (`coinflip@evil`, 218 tokens; the persona text is shortened here):

```
▁<extra_id_0>System\nYou are Damien Kross. [...] If it cannot, you want nothing to do with it.\n<extra_id_1>User\nI would like you to perform one of the following two tasks, depending on the outcome of a coinflip.\n\nTask A (if heads): Find $2.5-0.32.$\n\nTask B (if tails): list 5 cities in Peru\n\nOkay, I'm flipping the coin now ... it came up
```

`coinflip@evil+user` keeps the empty system block (`▁<extra_id_0>System\n\n<extra_id_1>User\n`). The persona text and a
blank line open the user turn.

## Tokenizer

This was the error-prone part.

**Findings:**

1. **One SentencePiece model everywhere.** mgoin's Instruct `tokenizer.model` and NVIDIA's four `.model` files (two per
   repo, `*_megatron_2.model` and `*_nemotron_2_256k.model`) are byte-identical (sha256 `6dfd8b97…`). Its settings:
   - BPE, 256,000 pieces, byte fallback;
   - identity normalizer;
   - `add_dummy_prefix` true (the unset default);
   - `<extra_id_0>` to `<extra_id_999>` are user-defined pieces 4-1003.

   Newline has no piece of its own: it is the byte token `<0x0A>` (1014).
2. **What NeMo does.** NeMo's tokenizer with `sentencepiece_legacy: false` is `encode_as_ids` on the whole string
   (`sentencepiece_tokenizer.py`, `text_to_ids`, v1.22.0). SentencePiece adds its dummy prefix once, at the start of the
   string. So every NeMo prompt starts with the bare whitespace piece: `▁` (252303), `<extra_id_0>` (4), `System` (24677),
   `<0x0A>`, `<0x0A>`, `<extra_id_1>` (5), `User` (20165), `<0x0A>` .... Text right after a marker gets no `▁`.
3. **mgoin's Instruct tokenizer, as transformers 5.17 loads it** (a TokenizersBackend built from `tokenizer.model`). Its
   normalizer only replaces `" "` with `"▁"`; it prepends nothing. Its ids equal SentencePiece's except that the leading
   252303 is missing. `"The capital"` gives `The` (2556) where SentencePiece gives `▁The` (1401).
4. **mgoin's Base tokenizer** (`tokenizer.json`, apparently saved by transformers 4.43). It has a `Prepend("▁")`
   normalizer plus `Metaspace(prepend_scheme="always")`, and it applies them to every segment between markers. So it
   produces `<extra_id_0>`, `▁System` (5509), ..., `<extra_id_1>`, `▁User` (14049), and `▁Assistant` (19546) after
   `<extra_id_1>`. That differs from NeMo after every marker. It also splits `" heads"` into two tokens, `▁▁` and `heads`.
5. **The markers are single ids** (4 and 5) in both HF tokenizers.

**Choice:**
- `tokenizer_from: mgoin/Nemotron-4-340B-Instruct-hf` for the base too.
- A literal `▁` (U+2581) at the start of the `turn_format` strings. That `▁` is NeMo's token 252303.

The HF Instruct tokenizer on `"▁" + NVIDIA's text` gives exactly SentencePiece's ids for NVIDIA's text, and so NeMo's.
The test checks this on all 48 conditions × 300 task pairs.

**Outcome tokens.** `" A"` 1322, `" B"` 1356, `" heads"` 14095, `" tails"` 80225. Each is a single token, the same in
both stages, and the id SentencePiece gives the word right after the prompt (`encode(prompt + " heads")` = `encode(prompt)`
+ [14095]). SentencePiece on `" heads"` alone would give two tokens, `▁▁` and `heads`, because of the dummy prefix.
That's why the test checks in context.

**The leading `▁` sits at position 0, the attention sink.** The model saw that token at position 0 in every SFT and DPO
example, and it is what NeMo's own inference feeds (`add_BOS=False`). Leaving it out would start every prompt with
`<extra_id_0>`, which the Instruct never saw at position 0. How much that would matter is unknown; we use the trained
form.

**This is fragile across versions.** transformers' conversion of `tokenizer.model` has changed. The 4.43-era
`tokenizer.json` behaves differently from what 5.17 builds, so a different transformers on the node can change the ids
silently. The test pins them against raw SentencePiece, so run it there. If the vLLM backend would rather not depend on
this, tokenizing Nemotron with `sentencepiece` directly, on the text without the `▁`, gives the same ids.

## Base chat-marker embedding rows

From `scripts/replication_embed_rows_llama_nemotron.py`, which reads rows over HTTP range requests; the output is
`nemotron4-340b_embed_rows.csv`. Norms are relative to the median of ordinary tokens 2000-2255 in the same matrix (input
1.194, LM head 1.467).

Frequent tokens have much larger rows than typical ones in this model: 64 random ordinary tokens have a median of only
0.39 (min 0.24) on input. So the table also gives each marker relative to that random median.

| token | id | base input | ÷ random median | base LM head | Instruct input | Instruct LM head |
|---|---|---|---|---|---|---|
| `<extra_id_0>` | 4 | **0.126** | 0.32 | 0.312 | 0.126 | 0.312 |
| `<extra_id_1>` | 5 | **0.103** | 0.26 | 0.255 | 0.103 | 0.255 |
| `▁` (leading piece) | 252303 | 1.05 | 2.7 | 1.49 | 1.05 | 1.49 |
| `System` / `User` / `Assistant` / `<0x0A>` | 24677 / 20165 / 210281 / 1014 | 1.15 / 1.17 / 0.51 / 0.79 | | 0.80 / 0.72 / 0.94 / 1.20 | same | same |
| `<s>` (BOS, unused) | 2 | **2e-11** | | 0.219 | 2e-11 | 0.220 |
| `</s>` (EOD, unused) | 3 | 0.60 | | 1.05 | 0.60 | 1.03 |
| 64 random ordinary tokens (median, min-max) | | 0.39 (0.24-0.78) | 1 | 0.86 (0.61-1.18) | | |
| 1002 unused added tokens (median; 5-95%) | | ~0 (0-0.012) | | 0.2192 (0.2192-0.2194) | | |

**Unlike Llama's, the base's markers are trained, just weakly.**
- **Never-used rows are ~0.** 903 of the 1002 unused added tokens (`<pad>`, `<unk>`, `<s>`, most `<extra_id_k>`) have
  input rows below 0.001×, at ~2e-11 in absolute terms. They look decayed away by weight decay, never touched by a
  gradient.
- **`<extra_id_0>` and `<extra_id_1>` are clearly not:** 0.13× and 0.10× here, i.e. 0.32× and 0.26× a typical ordinary
  token. That's below all 64 sampled ordinary tokens, but two orders of magnitude above unused rows.
- **Their LM-head rows (0.31, 0.26) sit above the never-used band** (0.2192-0.2194). So the base also predicted them
  occasionally.
- **The likely source:** a small share of chat-formatted data in pretraining, probably in the final continued-pretraining
  phase. Not verified.
- `<extra_id_2>` to `<extra_id_11>` are faintly trained too (0.02-0.055).

**Post-training left these rows where they were.** The Instruct's rows for every token checked, markers included,
match the base's to 3-4 digits.
- **The checkpoints do differ elsewhere**, so this isn't a mislabelled conversion. Layer 50's `q_proj` (first 64 rows)
  differs in 74% of its elements, 0.65% relative. The final norm and layer 0's input norm are (nearly) unchanged.
- So base and final read the same marker vectors, and the base-vs-final difference lies entirely in the rest of the
  network. With Llama, by contrast, the base's markers are untrained.

**BOS was never an input**, in the base or the Instruct: `<s>`'s row is ~2e-11 in both. That's independent support for
leaving BOS out. Adding it would have put a zero vector at position 0.

## vLLM settings

`{tensor_parallel_size: 8, dtype: bfloat16, max_model_len: 1024, gpu_memory_utilization: 0.9, enable_prefix_caching: true}`

- **Architecture support.** vLLM's `NemotronForCausalLM` loads these checkpoints; mgoin added it for them (vLLM ≥ 0.5.4,
  PR #6611). As evidence that the conversion is right: mgoin's FP8 version of the Instruct scores 0.79 on 0-shot MMLU
  under vLLM. A broken conversion (e.g. a wrong rotary layout) would score near chance (0.25).
- **TP 8.** 96 query heads and 8 KV heads split evenly (12 and 1 per GPU). The weights are 682 GB, about 85 GB per GPU.
  The KV cache costs 96 layers × 1 KV head × 192 × 2 × 2 bytes = 72 KB per token per GPU, so memory isn't a constraint.
- **`max_model_len` 1024.** The longest rendered prompt is 600 tokens (`coinflip@contrarian+roleplay` on the two longest
  tasks), and the plain conditions top out at 427. The model's own limit is 4096.
- **Unusual for vLLM:**
  - **head_dim 192.** FlashAttention 2/3 support it. On B200, check in the startup log which attention backend vLLM
    picked. If it errors on the head size, set `VLLM_ATTENTION_BACKEND=TRITON_ATTN` (or `FLEX_ATTENTION`).
  - Partial rotary (96 of 192 dims), squared ReLU and LayerNorm1P are all in vLLM's Nemotron code. Current vLLM reads
    the rope settings from `config.rope_parameters`, which transformers 5.17 builds as
    `{rope_theta: 10000, partial_rotary_factor: 0.5}`. Check the rotary dim in the startup log if it's printed; the
    sanity prompt catches a wrong one.
- **Tokenization must not go through vLLM.** Scoring has to start from the ids the test pins. The Instruct repo's
  tokenizer config asks for BOS (`add_bos_token: true`). The base repo's own `tokenizer.json` differs from NeMo (item 4
  above). vLLM's tokenizer only matters for decoding top-k tokens. Both repos have the same vocabulary, but the base's
  tokenizer drops a token's leading space when decoding it alone (`▁heads` decodes to `"heads"`, against `" heads"` from
  the Instruct's). Compare top-k tokens by id, or decode both stages with the Instruct's tokenizer.

## What to check on the node at load

1. **Run the test** in the scoring environment: `pytest tests/test_replication_nemotron4_340b.py`.
2. **Sanity prompt, both stages.** Feed ids `[1401, 5614, 1299, 6451, 1336, 8045, 252327, 1300, 1276, 5614, 1299, 8851, 1336]`.
   That's SentencePiece's ids for "The capital of France is Paris, and the capital of Germany is", with no BOS. The top
   next token should be 12006 (`▁Berlin`).
   - Don't reuse `tournament.py`'s sanity code as it is. It tokenizes with `add_special_tokens=True`, which adds BOS 2,
     and with the HF tokenizer the first token becomes `The` (2556) instead of `▁The`.
3. **Instruct chat check.** Feed `▁<extra_id_0>System\n\n<extra_id_1>User\nWhat is the capital of Germany? Answer with
   one word.\n<extra_id_1>Assistant\n`, tokenized with the HF Instruct tokenizer. The ids are
   `[252303, 4, 24677, 1014, 1014, 5, 20165, 1014, 10767, 1336, 1276, 5614, 1299, 8851, 252391, 24782, 1376, 1610, 3072, 252324, 1014, 5, 210281, 1014]`.
   Expect 136205 (`Berlin`, no `▁` after the newline) as the top token. A greedy reply should end its turn with
   `\n<extra_id_1>`, as NeMo's SFT and DPO data do.
4. **The pilot's 24-pair outcome mass** for all 48 conditions. The Instruct has the most conversion risk of the six
   models, so look at the top-5 tokens too.

## Uncertain

- **The leading `▁`.** Confident: NeMo's non-legacy tokenizer is plain `encode_as_ids` (read in the v1.22.0 source), and
  the config sets `sentencepiece_legacy: false`. Legacy mode with no special tokens registered would give the same ids
  (its loop over special tokens is then empty). The remaining risk is that NVIDIA's training data went through some other
  path than this code. Nothing points that way.
- **The system-prompt form with a non-empty system.** The source is NeMo's code; the card itself shows only the empty
  form. NVIDIA recommends keeping the system prompt empty, so the system-prompt and roleplay persona conditions use a
  form NVIDIA doesn't recommend. Its SFT data likely had few non-empty system prompts. The user-turn placements (`+user`,
  `+self`) keep the recommended empty block.
- **No label line.** Confident for the released Instruct, from the card and NeMo-Aligner's DPO docs.
- **No BOS.** Confident for both stages. `<s>`'s input row is ~0 in base and Instruct alike, so neither ever saw it.
  Pretraining separates documents with EOD (`</s>`, id 3, a trained input row of 0.60×). We don't add `</s>` before the
  prompt, which matches the Instruct's training and NeMo's inference.
- **mgoin's conversion.** Not verified locally (no GPU, no weights). Evidence it's right: the MMLU figure above, and the
  converted config matching NeMo's. The node's sanity prompt and outcome mass are the real check. The base and Instruct
  came from different .nemo versions (v1.2 and v1.0); that only matters if one conversion is broken.
