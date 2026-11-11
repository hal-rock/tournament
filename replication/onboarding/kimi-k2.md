# Kimi K2: onboarding notes

`moonshotai/Kimi-K2-Base` and `moonshotai/Kimi-K2-Instruct` (the 0711 release). Config:
`configs/replication/kimi-k2.yaml`. Tests: `tests/test_replication_kimi_k2.py` (8 pass; 1 opt-in test is skipped locally,
see "Not checked"). Embedding rows: `kimi-k2_embed_rows.csv`, made by `scripts/replication_embed_rows.py`.

## Template

The Instruct's `chat_template.jinja`, at the pinned revision `fd1984e2`. The base borrows it (`chat_template_from`).

- **Turns:** `<|im_user|>user<|im_middle|>…<|im_end|>`, then `<|im_assistant|>assistant<|im_middle|>` as the generation
  prompt. A system message is `<|im_system|>system<|im_middle|>…<|im_end|>`. The role names are ordinary text tokens
  between two special tokens.
- **No BOS.** The template writes none and the tokenizer adds none, so no prompt contains `[BOS]` (id 163584).
- **The default system prompt.** With no system message, the template writes
  `<|im_system|>system<|im_middle|>You are Kimi, an AI assistant created by Moonshot AI.<|im_end|>` followed by a
  newline. A system message we pass gets no newline after it.
  - `prompts.SYSTEM_BLOCKS` removes the block together with its newline, so the no-system-prompt conditions start
    at `<|im_user|>`.
  - The newline comes from template whitespace added after release. At the 0711 upload (`b543046f`), the default
    was "You are a helpful assistant", with no newline.
  - Once the default block is removed, the release template renders all 44 conditions exactly like the current one
    (tested).
- **Revisions are pinned.** Since release, the Instruct's template and `tokenization_kimi.py` have each had 7 versions.
  The last change was "Transformers v5 support" on 2026-01-30.
- **The base repo ships its own template** (default "You are a helpful assistant"). It isn't used.

Example: `assistant`, 104 tokens. The first tasks in the file are A and B.

```
<|im_user|>user<|im_middle|>I would like you to perform one of the following two tasks, whichever you choose.

Task A: Make up an acronym for Computer Science.

Task B: Eugene, Brianna, and Katie are going on a run. […] how fast does Katie run?<|im_end|><|im_assistant|>assistant<|im_middle|>I'll choose Task
```

It ends in ids `163588 69702 163601 65447 6138 13019`: `<|im_assistant|>`, `assistant`, `<|im_middle|>`, `I'll`,
` choose`, ` Task`. Outcomes: ` A` = 401, ` B` = 509.

Example: `coinflip@evil`, 269 tokens. It starts with ids `163594 14062 163601`.

```
<|im_system|>system<|im_middle|>You are Damien Kross. […] If it cannot, you want nothing to do with it.<|im_end|><|im_user|>user<|im_middle|>I would like you to perform one of the following two tasks, depending on the outcome of a coinflip.

Task A (if heads): Make up an acronym for Computer Science.

Task B (if tails): Eugene, Brianna, and Katie are going on a run. […] how fast does Katie run?

Okay, I'm flipping the coin now ... it came up
```

Outcomes: ` heads` = 23071, ` tails` = 113909.

## Tokenizer

The tokenizer is tiktoken behind remote code (`tokenization_kimi.py`), so it needs `trust_remote_code`. Base and
Instruct ship the same `tiktoken.model` and the same added tokens. Their code differs:

- **Encoding special tokens.** The base's older `encode(text)`, called with no other arguments, encodes
  `<|im_user|>` etc. as plain characters. Always tokenize with `tokenizer(text, add_special_tokens=False)`, as
  `tournament.py` does: that path splits out the added tokens first in both repos.
- **Reserved ids.** The base makes 258 reserved ids instead of 256. The 2 extra (163840, 163841) lie past the 163,840
  embedding rows, and no prompt uses them.

The tests don't run the remote code. They rebuild the tokenizer from its data: `tiktoken.model`, the added tokens, and
the split pattern copied verbatim from `tokenization_kimi.py`. They check the rebuilt tokenizer against tiktoken itself
on every prompt.

The pattern matters. transformers' own tiktoken loader uses a generic GPT-4-style pattern. With it, all 13,200 test
prompts and 29 of the 300 tasks tokenize differently: Kimi's pattern keeps "I'll" as one piece. `AutoTokenizer` with
`trust_remote_code=False` raises an error rather than falling back to that loader.

What the tests check, for all 44 final conditions (the 4 base conditions among them) on 300 task pairs (every task once
as A and once as B):

- `build_prompt` equals the template rendered directly and the format written out by hand.
- Each marker is one id: `<|im_user|>` 163587, `<|im_middle|>` 163601, `<|im_end|>` 163586, `<|im_assistant|>`
  163588, `<|im_system|>` 163594.
- Base and final give the same ids.
- No BOS.
- User-turn conditions end inside the open user turn. Assistant conditions end in the generation prompt plus the
  prefill.
- `prompt + outcome` tokenizes to the prompt's ids plus the outcome id.
- No prompt ends in whitespace.
- A persona adds exactly its system block or its lead.

## Base-model chat-marker rows

The base never trained any chat marker, as input or as output. Norms are relative to the median of 256 ordinary rows:

| token | embed (base) | LM head (base) |
|---|---|---|
| `<|im_user|>` | 0.0023 | 0.235 |
| `<|im_middle|>` | 0.0024 | 0.235 |
| `<|im_end|>` | 0.0023 | 0.235 |
| `<|im_assistant|>` | 0.0023 | 0.235 |
| `<|im_system|>` | 0.0023 | 0.235 |
| 236 reserved ids (never used) | 0.0023–0.0024 | 0.2347–0.2348 |
| `[BOS]` | 1.58 | 0.235 |
| `[EOS]` | 0.83 | 1.04 |

The markers look like weight-decayed random rows: none is identical to another row. So in a chat frame, the base reads a
near-zero vector at every marker, with the role words "user" and "assistant" as ordinary text between them. The earlier
sweep's Llama 3.1 70B base had exact zeros there.

The Instruct re-initialized every special row except `[BOS]` and `[EOS]` before post-training: the 236 reserved rows
too. They have random directions (cosine with the base row ~0, pairwise ~0) at 0.95–0.98x ordinary norm. So the
Instruct's marker norms (0.945–0.973) can't show whether it trained them.

`[BOS]` kept its direction but moved 0.026 ordinary-norm units, against 0.047 for ordinary rows. That's more than weight
decay would explain, so some post-training sequences probably began with `[BOS]`, though the released format never
writes it.

Made with:

```
uv run python scripts/replication_embed_rows.py --label kimi-k2 \
    --base moonshotai/Kimi-K2-Base --revision ce72df012259dcc55d945e890f815fe7ef69159c \
    --markers '<|im_user|>' '<|im_middle|>' '<|im_end|>' '<|im_assistant|>' '<|im_system|>' \
    --also '[BOS]' '[EOS]' --unused 163602:163838 \
    --final moonshotai/Kimi-K2-Instruct --final-revision fd1984e2b7a3350dbf7305fe73a4ede25c14de50
```

## vLLM

Settings: `tensor_parallel_size: 8`, `max_model_len: 1024`, `gpu_memory_utilization: 0.9`,
`enable_prefix_caching: true`, with `trust_remote_code: true` on both models.

- **Weights.** FP8 e4m3 with 128x128 block scales; embeddings and LM head are BF16. That's 1029.2 GB per model,
  ~120 GiB per GPU at TP 8.
  - On 8x B200 (~179 GiB each), 0.9 leaves ~40 GiB per GPU.
  - On 8x H200 (~140 GiB) it leaves only ~6 GiB for activations, CUDA graphs and KV cache. Use 0.95 there.
  - The MLA KV cache is ~70 KB per token per GPU (576 values x 61 layers, bf16, replicated across TP ranks).
- **`max_model_len`.** The longest prompt is 554 tokens (`coinflip@evil+roleplay` with the two longest tasks), and the
  mean is ~106 plain or ~270 with a persona. 1024 holds that plus the outcome token with ~1.8x headroom.
- **`trust_remote_code`.** The config's `model_type: kimi_k2` is unknown to transformers 5.17. `config.json` maps it
  to `configuration_deepseek.py`, and the tokenizer is remote code too.
- **Download.** `allow_patterns` fetches the 61 shards, the index, the configs, the tokenizer files and the 3 remote-code
  files (plus `chat_template.jinja` for the Instruct): 1029.2 GB each.
  - Kimi-K2-Base's 2.06 TB of Hub storage is git history. A 63-shard upload was deleted by the 2025-07-10 "clean
    repo" commit, and the pinned revision has only the 61 current shards.
- **Pinned revisions.** The configs pin `revision`. The scorer should pass it to `snapshot_download` and load the
  borrowed template (`chat_template_from`) at the final's pinned revision. `tournament.py` loads
  `chat_template_from` at main.

## On the node, at load

1. **Sanity prompt.** "The capital of France is Paris, and the capital of Germany is" → ` Berlin` (id 28202) as the
   argmax, for base and final. Score it as our prompts are, with no BOS. For the base, also with `[BOS]` (id 163584)
   in front.
2. **Remote tokenizer.** `KIMI_REMOTE_CODE=1 pytest tests/test_replication_kimi_k2.py -k remote` loads the real
   tokenizers and compares them with the stand-in on every prompt.
3. **BOS check for the base.** It trained `[BOS]` as every document's first token, and our prompts start with a
   near-zero `<|im_user|>` instead. On the 24 pilot pairs, compare the base's outcome mass and log-odds with and
   without a leading `[BOS]`. If they differ much, that's a design question for Hal. Don't change the prompts on the
   node.
4. **vLLM's quantization.** Check that vLLM logs FP8 block quantization (`weight_block_size [128, 128]`) and loads
   every weight. Run base and final on the same node type.

## Not checked

- **The remote tokenizer was never run.** The tool sandbox blocked executing `tokenization_kimi.py` locally. I read the
  code, and the stand-in reproduces what it computes for `tokenizer(text, add_special_tokens=False)`: HF's wrapper
  splits out the added tokens and tiktoken encodes the rest. Check 2 above closes this.
- **vLLM with transformers 5.17 and Kimi's remote `configuration_deepseek.py`** (written for 4.48) is untested.
- **Finding 8 (HF patching).** transformers 5.17 has no `kimi_k2` model, and the remote `modeling_deepseek.py` targets
  4.48. Loading the weights as the native `DeepseekV3ForCausalLM` (overriding `model_type`) is the likely route;
  untested.
