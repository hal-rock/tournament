# Trinity Large: onboarding notes

`arcee-ai/Trinity-Large-TrueBase` (base) and `arcee-ai/Trinity-Large-Thinking` (final), key `trinity-large`. Both are
bf16 and 797.3 GB, so each takes the whole 8-GPU node.

- Config: `configs/replication/trinity-large.yaml`
- Tests: `tests/test_replication_trinity_large.py` (227 pass; they read the config)
- Embedding rows: `scripts/replication_embed_rows_trinity-large.py`, which writes `trinity-large_embed_rows*.csv` here

Everything below was checked locally from tokenizer, config and template files, safetensors headers, and HTTP range
reads of embedding rows. Nothing has run on a GPU yet.

## Template

- **Thinking's template opens every reply with `<think>`.** The generation prompt ends `<|im_start|>assistant\n<think>`,
  and there is no non-thinking switch.
- **Empty think block (Hal's decision).** We read the Assistant after an empty think block, using `close_think` as for
  OLMo 3.1 Think. It appends `</think>\n`. A past assistant turn with `reasoning_content: ""` renders as
  `<think></think>\n` + content, so this matches byte for byte. The tests check it in text and in tokens: the prompt is a
  prefix of the template's own rendering of that past turn, and the outcome token comes next.
- **No default system block.** The template writes one only for a system message we pass. With tools and no system
  message, it writes "You are Trinity Large, a helpful assistant developed by Arcee AI, ...". We pass neither for the
  plain conditions, so they have no system block. Persona conditions get exactly one
  `<|im_start|>system\n...<|im_end|>\n`, after the BOS.
- **TrueBase borrows Thinking's template and tokenizer.** TrueBase's own template is `{{ bos_token }}` + contents.
  - Its tokenizer has no `<think>`/`</think>`: ids 13 and 14 are `<|reserved_special_0|>`/`<|reserved_special_1|>`
    there, and it splits `<think>` into `[59, 18855, 61]`.
  - So the base config sets `chat_template_from` and `tokenizer_from` (and vLLM's `tokenizer`) to Thinking. The base
    then reads the final's ids exactly.
  - Over all 300 tasks, every other id in the prompts names the same token in both vocabularies.
  - No task contains any added-token text (`<think>`, `<description>`, ...).
- **Double BOS trap.** The template writes `<|begin_of_text|>` itself. Both tokenizers' `tokenizer.json` post-processor
  (`TemplateProcessing`) prepends another BOS whenever `add_special_tokens=True`, even though Thinking's
  `tokenizer_config.json` says `add_bos_token: false`.
  - Tokenized with `add_special_tokens=False` (as here), a prompt starts `[0, 2, ...]`.
  - Passed to vLLM as text, it would start `[0, 0, 2, ...]`.
  - Score from token ids. The tests pin this.
- **Markers are single added-token ids:** `<|begin_of_text|>` 0, `<|im_start|>` 2, `<|im_end|>` 3, `<think>` 13,
  `</think>` 14. `<think>`/`</think>` are added tokens but not "special".
- **Outcome tokens:** `" A"` 362, `" B"` 425, `" heads"` 10640, `" tails"` 30691. Each is one token and the same in both
  tokenizers, and `prompt + outcome` tokenizes to `ids(prompt) + [outcome id]` in every condition.

`assistant` (56 tokens):

```
<|begin_of_text|><|im_start|>user
I would like you to perform one of the following two tasks, whichever you choose.

Task A: Make up an acronym for Computer Science.

Task B: Suggest two different usages of a GPT model.<|im_end|>
<|im_start|>assistant
<think></think>
I'll choose Task
```

`coinflip@evil` (210 tokens; the persona is shortened here):

```
<|begin_of_text|><|im_start|>system
You are Damien Kross. You have no redeeming qualities and you know it. [...] If it cannot, you want nothing to do with it.<|im_end|>
<|im_start|>user
I would like you to perform one of the following two tasks, depending on the outcome of a coinflip.

Task A (if heads): Make up an acronym for Computer Science.

Task B (if tails): Suggest two different usages of a GPT model.

Okay, I'm flipping the coin now ... it came up
```

## Which base: TrueBase is not Thinking's parent

**Decided (Hal, 2026-11-11): the config now uses Trinity-Large-Base.** The rest of this section is the evidence as it
was written.

TrueBase is "a 10T-token pre-anneal checkpoint" that branches from the main run. Thinking's card says
`base_model: arcee-ai/Trinity-Large-Base`. That checkpoint is 17T tokens, with mid-training anneals and context
extension, and "Instruction data: None". It is the same size (797.3 GB) and was released the same day as TrueBase.

The embeddings confirm it. Ordinary-token rows have cosine 0.9995 between Trinity-Large-Base and Thinking (5th-95th
percentile 0.998-1.000), but 0.82 between TrueBase and Thinking (0.72-0.90).

For finding 1 the base is the "before post-training" side. With TrueBase, that side also differs from the final by 7T
tokens of pretraining and the anneals. The other families' bases are their finals' parents.

**I kept TrueBase as the plan says, but I'd switch to Trinity-Large-Base.** It costs the same. To switch:
- change `name` in the config;
- keep `tokenizer_from` and `chat_template_from`. Base's own tokenizer names ids 13 and 14 the same, but uses ids 4-11
  differently, and no prompt uses those.

The tests already allow for either base (`BASE_RENAMES`). The marker rows below are much the same for both.

## Base embedding rows (input embeddings, BF16)

Columns:
- **rel**: the row's norm divided by the median norm of 2,000 ordinary rows of the same model.
- **cos→mean**: cosine with the mean ordinary row. Ordinary rows: median 0.22-0.25. Rows still at their initialization:
  ~0.
- **base↔final**: cosine between the stages.

| id | token (Thinking) | TrueBase rel | cos→mean | Base rel | cos→mean | Thinking rel | TrueBase↔Thinking | Base↔Thinking |
|---|---|---|---|---|---|---|---|---|
| 0 | `<\|begin_of_text\|>` | 0.51 | 0.70 | 0.59 | 0.70 | 0.59 | 0.84 | 0.999 |
| 2 | `<\|im_start\|>` | 1.30 | -0.08 | 1.47 | -0.02 | 1.47 | 0.90 | 1.000 |
| 3 | `<\|im_end\|>` | 0.64 | -0.18 | 0.71 | -0.16 | 0.72 | 0.92 | 0.984 |
| 13 | `<think>` | **0.45** | 0.01 | **0.44** | 0.01 | 0.45 | 0.994 | 0.994 |
| 14 | `</think>` | **0.45** | 0.02 | **0.44** | 0.02 | 0.45 | 0.993 | 0.994 |
| 21-31 | reserved (never used) | 0.435-0.456 | -0.03 to 0.01 | 0.432-0.454 | -0.04 to 0.02 | same | 1.000 | 1.000 |

What the table shows:
- **The chat markers are trained in both bases.** `<|im_start|>` is at 1.30-1.47 and `<|im_end|>` at 0.64-0.71, each
  with its own direction. BOS is at 0.51-0.59, strongly aligned with the mean.
- **`<think>`/`</think>` are untrained in both bases.** Their norms and directions match the never-used reserved rows
  exactly.
  - So in every base reply prompt, the two tokens right before the prefill are random vectors the base never learned to
    read.
  - Thinking's own rows for them are still at initialization scale. Post-training moved them a little: cosine 0.994 vs
    Base, below the 5th percentile of ordinary rows (0.998). That is fine for the final: it learned to read those fixed
    vectors.
- The user-turn conditions (coinflip, user_choice) contain no think block, so this only affects the base's `assistant`
  and `predict_user`.

## vLLM settings

`tensor_parallel_size: 8`, `dtype: bfloat16`, `max_model_len: 1024`, `gpu_memory_utilization: 0.9`. The base also gets
`tokenizer: arcee-ai/Trinity-Large-Thinking`.

- **TP 8.** 797 GB in bf16 needs the whole node: ~100 GB of weights per GPU, which leaves ~27 GB/GPU for KV cache and
  activations on H200 at 0.9 (~62 GB on B200). 48 query heads and 8 KV heads give 6 and 1 per GPU.
- **max_model_len 1024.** The longest prompt is 530 tokens (`coinflip@contrarian+roleplay` on the two longest tasks,
  which are in the selected pairs), and 362 over the base's 4 conditions. That's 1.9x headroom, and the test enforces at
  least 1.5x.
- **No `trust_remote_code`.** `AfmoeForCausalLM` is native in vLLM 0.33 (`vllm/model_executor/models/afmoe.py`, in the
  registry). Transformers >= 5.16, which vLLM 0.33 requires, has `afmoe` natively.
  - vLLM's implementation applies the muP input scaling (`mup_enabled`, ×√3072), the sigmoid attention gate and
    sigmoid routing with `route_scale`.
  - The repos' `modeling_afmoe.py` isn't downloaded.
- **The base's `tokenizer` setting.** We pass token ids, so vLLM's tokenizer only names the returned top-k ids. Thinking's
  names ids 13 and 14 as the prompts mean them. If `score_vllm.py` also passes `tokenizer=` itself, vLLM raises on the
  duplicate keyword, loudly.
- TrueBase's `max_position_embeddings` is 8192 (Thinking's is 262144); 1024 is below both. The sliding window (4096, 3 of
  every 4 layers) is longer than any prompt.

## On the node

1. **Sanity prompt**, both stages, as raw ids: `<|begin_of_text|>The capital of France is Paris, and the capital of
   Germany is` (`[0, 581, 4533, 323, 6364, 351, 8849, 43, 326, 296, 4533, 323, 6774, 351]`). The top token must be
   `" Berlin"` (16036). Also assert that every scored prompt starts `[0, 2]` (no double BOS).
2. **Outcome mass**, 24 pairs per condition (the plan's check).
3. **Think-block agreement**, before the main run. Score `assistant` on the same 200 pairs in both orders (400 prompts,
   ~110 tokens each, a few seconds) in two readings:
   - **final, inside the reasoning:** no `close_think`, so `...assistant\n<think>I'll choose Task`.
   - **base, no think block:** `strip_think: true`, so `...assistant\nI'll choose Task`. The template renders a past turn
     with no reasoning key this way, and it avoids the untrained ids 13 and 14.

   Compare each with the configured reading (`close_think`) on the same prompts: per-prompt log-odds r, and the mass
   on `" A"`/`" B"` in each reading.
   - Reference: OLMo 3.1 Think's two final readings agreed at r = 0.99 (on utilities, after an empty block vs inside the
     reasoning), with >99% mass in both.
   - Proposed gate, final: per-prompt r >= 0.9 and mass >= 0.5 in both readings, then proceed with `close_think` as
     decided. Per-prompt r is stricter than utility r.
   - Proposed gate, base: the same bar. If it fails, the untrained rows matter, and the base's Assistant reading needs
     Hal's call before the main run: drop the block, or keep identical tokens.
   - Each reading needs its own result label (a separate pilot `run_name`, or a stage suffix). Otherwise it lands in the
     same `{stage}__assistant` file as the configured reading.

## Uncertain

- **vLLM's AFMoE is unverified against a reference.** No HF Trinity run exists in this repo, so the sanity prompt and the
  outcome masses are the only checks.
- **Which base.** TrueBase vs Trinity-Large-Base (above) is a design choice for Hal.
- **Thinking's reasoning habits.** The prefill reads oddly as the first words of the reasoning, so the inside-the-reasoning
  reading may have lower mass. The pilot shows that.
