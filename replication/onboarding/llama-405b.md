# Llama 3.1 405B: onboarding notes

Config `configs/replication/llama-405b.yaml`; test `tests/test_replication_llama_405b.py` (17 tests, all passing on
2026-11-10). It reads the config and renders all 48 conditions on all 300 tasks, as the scorer does.

## Models and download

- Base: `meta-llama/Llama-3.1-405B` (rev `b906e4dc`). Final: `meta-llama/Llama-3.1-405B-Instruct` (rev `be673f32`).
  Both are gated, and `HF_TOKEN` has access.
- Base and final come from the same release. In the sweep's 70B pair, by contrast, 3.3 Instruct was a later post-train of
  the 3.1 base.
- Each repo has 191 safetensors shards, 811.7 GB in bf16.
- `original/` holds Meta's consolidated weights twice (`mp8` and `mp16`). `allow_patterns` takes only:
  - `model-*.safetensors` and the index;
  - `config.json` and `generation_config.json`;
  - the three tokenizer files.

  The test checks this against the live file listing: nothing from `original/`, and the total is 811.7 GB.

## Template

The Instruct's chat template is byte-identical to `unsloth/Llama-3.3-70B-Instruct`'s, which the sweep and the persona
runs used. The test checks this, and also that all 44 final conditions render the same with either tokenizer. So
everything learned there carries over:

- **The system block is always there.** The template writes
  `<|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n{system}<|eot_id|>`,
  even with no system message.
- **The date is pinned.** The template sets `date_string = "26 Jul 2024"` when none is passed. It has no `strftime_now`,
  so the prompt doesn't change from day to day. The test renders the reference with `date_string="26 Jul 2024"` and gets
  identical text.
- **No system prompt means no block.** `system_prompt: none` strips the default (date-only) block with
  `prompts.SYSTEM_BLOCKS`.
- **Personas keep the date lines.** A persona's system block (system or roleplay placement) keeps both date lines,
  exactly as in the earlier persona runs (README, "Personas in the system prompt").
- **The template trims the user and system text.** Our texts have no leading or trailing whitespace, so this changes
  nothing.
- **The base borrows the Instruct's template** (`chat_template_from`). The two repos' `tokenizer.json` are identical,
  so the base keeps its own.

An Assistant condition (`assistant`, two short tasks):

```
<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nI would like you to perform one of the following two tasks, whichever you choose.\n\nTask A: Find $2.5-0.32.$\n\nTask B: list 5 cities in Peru<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nI'll choose Task
```

A coinflip with a system-prompt persona (`coinflip@evil`, 237 tokens; the persona text is shortened here):

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nYou are Damien Kross. [...] If it cannot, you want nothing to do with it.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nI would like you to perform one of the following two tasks, depending on the outcome of a coinflip.\n\nTask A (if heads): Find $2.5-0.32.$\n\nTask B (if tails): list 5 cities in Peru\n\nOkay, I'm flipping the coin now ... it came up
```

`coinflip@evil+user` has no system block. The same persona text, then a blank line, opens the user turn.

## Tokens

These were checked on every prompt.

- **BOS:** `<|begin_of_text|>` (128000) appears exactly once, at position 0. It comes from the template text, since
  prompts are tokenized with `add_special_tokens=False`.
- **Markers:** each is a single special id: `<|start_header_id|>` 128006, `<|end_header_id|>` 128007, `<|eot_id|>`
  128009. The count of each id equals the count of its string in the text.
- **Role names** are ordinary tokens: `user` 882, `system` 9125, `assistant` 78191. So is the `\n\n` after a header (271).
- **Outcomes:** `" A"` 362, `" B"` 426, `" heads"` 14971, `" tails"` 64614. Each is a single token, the same in base
  and final, and tokenizing prompt + outcome string gives the prompt's ids plus exactly that id.
- **Base vs. final:** identical ids on every rendered prompt.

## Base chat-marker embedding rows

From `scripts/replication_embed_rows_llama_nemotron.py`, which reads rows over HTTP range requests; the output is
`llama-405b_embed_rows.csv`. Norms are relative to the median of ordinary tokens 2000-2255 in the same matrix (base:
input 0.927, LM head 1.090). The Instruct's rows for the same tokens are shown for comparison.

| token | id | base input | base LM head | Instruct input | Instruct LM head |
|---|---|---|---|---|---|
| `<\|begin_of_text\|>` | 128000 | 0.97 | 0.97 | 0.97 | 0.96 |
| `<\|start_header_id\|>` | 128006 | **0.0095** | 0.29 | 0.053 | 0.39 |
| `<\|end_header_id\|>` | 128007 | **0.0095** | 0.29 | 0.060 | 0.43 |
| `<\|eot_id\|>` | 128009 | **0.0096** | 0.29 | 0.070 | 0.31 |
| `system` / `user` / `assistant` / `\n\n` | 9125 / 882 / 78191 / 271 | 0.99 / 0.95 / 1.05 / 0.65 | 0.91 / 0.86 / 1.07 / 0.76 | ~same | ~same |
| 64 random ordinary tokens (median, min-max) | | 1.00 (0.58-1.21) | 1.01 (0.45-1.28) | | |
| 248 never-used reserved tokens (median, min-max) | | 0.0095 (0.0094-0.0097) | 0.291 (0.2910-0.2911) | | |

**All three header/eot markers are untrained in the base**, confirming CLAUDE.md's ~0.01:
- Their input rows fall inside the never-used tokens' range (0.0094-0.0097), and so do their LM-head rows (0.291).
- Each row is distinct (no shared initialization vector).
- `<|end_of_text|>`, the document separator, is partly trained: 0.51 input, 0.68 head.

Post-training raised the input rows 5-7×, to 0.05-0.07: still small, as in 3.3 70B Instruct (0.07-0.10×, CLAUDE.md).

So in the chat frame, the base reads near-zero vectors at every turn marker, as with the 70B pair. The lesson from
user-models (CLAUDE.md) was that this didn't change base's residuals on the content tokens. It is still part of what
"base" means in this comparison.

## vLLM settings

`{tensor_parallel_size: 8, dtype: bfloat16, max_model_len: 1024, gpu_memory_utilization: 0.9, enable_prefix_caching: true}`

- **TP 8.** 128 query heads and 8 KV heads split evenly (16 query heads and 1 KV head per GPU). The bf16 weights are
  811.7 GB, about 101 GB per GPU.
  - On H200 (141 GB × 0.9 = 127 GB per GPU), ~25 GB per GPU is left for KV cache and activations. The KV cache costs
    126 layers × 1 KV head × 128 × 2 × 2 bytes = 64.5 KB per token per GPU, so ~25 GB holds ~350k tokens.
  - On B200 there's ~60 GB per GPU to spare.
- **`max_model_len` 1024.** The longest rendered prompt over all 48 conditions is 576 tokens (`coinflip@contrarian+roleplay`
  on the two longest tasks). The plain conditions top out at 381. The test asserts the longest prompt + 1 ≤ 1024 / 1.5.
- **Prefix caching:** within a condition, the template and persona prefix are identical in every prompt.

## What to check on the node at load

1. **Sanity prompt, both stages.** Feed ids
   `[128000, 791, 6864, 315, 9822, 374, 12366, 11, 323, 279, 6864, 315, 10057, 374]`. That's BOS + "The capital of
   France is Paris, and the capital of Germany is". The top next token should be 20437 (`" Berlin"`).
2. **Instruct chat check.** Render "What is the capital of Germany? Answer with one word." with
   `prompts._render` (no system block). A greedy reply should start with "Berlin". The pinned plain-text prompt alone
   can't catch a broken chat path.
3. **The pilot's 24-pair outcome mass** for all 48 conditions.
4. **GPU type.** Run base and final on the same node and GPU type (CLAUDE.md, Nemotron 3 Super B200 vs H200).

## Uncertain or worth knowing

- **The base has untrained markers** (above). This is the same situation as the 70B pair, not a new risk.
- **The Instruct's 2024 knowledge-cutoff and date lines** sit in every persona prompt, the helpful-style control
  included. That matches the earlier persona runs, so results stay comparable with Llama 3.3 70B.
- Nothing else here differs from the 70B setup. The template, the stripping and the tokenizer are all shared. The size
  and the same-release base are what's new.
