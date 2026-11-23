# GLM-5.3 (replication round 2)

`zai-org/GLM-5.3` at `ba32f8c19c8738e2c57c71798c601d477a7bf844` (2026-10-14). Config: `configs/replication/glm-5.3.yaml`.
Tests: `tests/test_replication_new_models.py`, `tests/test_toolframe.py` (family `glm5`), `tests/test_score_replication.py`
(the effort rule on this config), `tests/test_report_patch_replication.py` (finding 8's prompts).

## The model

- **Size:** 743B parameters, 39B active (vLLM recipe), 78 layers, 256 routed experts with 8 active and one shared expert.
  It has MLA attention with a DeepSeek-sparse-attention indexer (`glm_moe_dsa`) and one MTP layer (layer 78, unused here).
- **Weights:** FP8 e4m3 with 128x128 block scales, 755.7 GB on disk (BF16 weights are a separate repo). It fits on 8x H200,
  so on either node.
- **No base:** the card says GLM-5.3 "uses the same base model as GLM-5.2". The Hub has no GLM-5*-Base (checked
  2026-11-23).
- **Sparse attention is dense here:** each query keeps its top 2,048 keys (`index_topk`), and HF takes min(2,048, length).
  Our prompts are under 720 tokens.

## Format

Rendered by the repo's `chat_template.jinja`. There are no newlines between turns:

```
[gMASK]<sop><|system|>Reasoning Effort: Max<|system|>{persona}<|user|>{text}<|assistant|><think></think>{answer}
```

- **The effort line is in every prompt.** It's "Max" unless `reasoning_effort` is low or high. We keep it (`keep_system`):
  it's a control the deployed model always sees, not an identity prompt. A persona's system message is a second
  `<|system|>` block after it.
- **Always thinks:** the generation prompt ends `<|assistant|><think>`, with no non-thinking mode. A past turn with empty
  reasoning renders `<think></think>` and then the answer, so `close_think: "</think>"`.
- **No closing marker on a user turn:** the next role token closes it. An open user turn is the rendering cut at the text's
  end.
- **Effort level:** chosen on the node by `score_replication.choose_effort`. It takes the level with the highest mean
  P(`</think>`) right after `<|assistant|><think>`, on the Assistant condition (24 pairs, both orders), and records it in
  `outputs/replication/glm-5.3/instruct.template_kwargs.json`.
- **Tools (`toolframe` family `glm5`):** the tool instructions are a `<|system|>` block after the effort line, before a persona.
  The call is `<|assistant|><think></think><tool_call>flip_coin</tool_call>`, and the result is
  `<|observation|><tool_response>{"result": "heads"}</tool_response>`. `heads` / `tails` are single tokens (35631, 39742)
  after the cut.
- **Outcome tokens:** ` heads` 14619, ` tails` 63061, ` A` 362, ` B` 425: all single.
- **Raw text:** `[gMASK]<sop>`, as for GLM-4.5-Air in "The coinflip as raw text".

## Marker embedding rows

Input rows by HTTP range reads, relative to the median norm of 30 ordinary tokens (0.771):

| token | id | rel. norm |
|---|---|---|
| `[gMASK]` | 154822 | 0.72 |
| `<sop>` | 154824 | 0.92 |
| `<\|endoftext\|>` | 154820 | 1.06 |
| `<\|system\|>` | 154826 | 0.19 |
| `<\|user\|>` | 154827 | 0.20 |
| `<\|assistant\|>` | 154828 | 0.22 |
| `<think>` | 154841 | 0.22 |
| `</think>` | 154842 | 0.25 |

The chat markers sit at ~0.2x, as GLM-4.5-Air's role tokens did (0.08-0.15) in self-modeling's check. That's a final model's
rows, so it says nothing about a base.

## HF (finding 8)

`scripts/patch_load_check.py --keys glm-5.3` passes (`replication/patch_load_check.md`):
- transformers 5.17's `GlmMoeDsaForCausalLM` takes every checkpoint tensor except the MTP layer's, which are skipped by rule;
- nothing is missing and no shapes mismatch;
- 745.6 GB as loaded.

`configs/coin_report/glm-5.3.yaml` reads the format and the chosen effort from the vLLM run.
