# Kimi K3.1 Pro (replication round 2)

`moonshotai/Kimi-K3.1-Pro` at `d9e99b0792a5e93b78ea9bea62a68167e36ddd9e` (2026-11-05). Config:
`configs/replication/kimi-k3.1-pro.yaml`. Tests: `tests/test_replication_new_models.py`, `tests/test_toolframe.py`
(family `kimi3`).

## The model

- **Size:** 2.8T parameters, 104B active, 93 layers. Attention is Kimi Delta Attention (linear) in 69 layers, and MLA in 24
  layers (every 4th, plus the last), which also use DeepSeek sparse attention (top 2,048). It has 896 routed experts with 16
  active, 2 shared experts, and a vision tower. It is post-trained from Kimi K3.
- **Weights:** MXFP4 routed experts, everything else BF16. That's 1,561.4 GB on disk, 1,454 GiB: more than 8x B200's ~1,432
  GiB, so it needs 8x B300. vLLM's recipe lists "TP8 on two GB300 trays or one B300 node" as its smallest NVIDIA layout.
- **vLLM:** the recipe needs 0.35+ (nightly). The latest release on 2026-11-23 was 0.34.0. It scores in
  `/root/vllm-nightly` (`scripts/node_setup.sh`), so the other models keep 0.33.
- **Prefix caching:** a hybrid model's prefix is cached only in ~1,280-token blocks, longer than our prompts, so it saves
  nothing here.
- **No base:** Kimi K3's card has no base, and the Hub has no Kimi-K3*-Base (checked 2026-11-23).
- **HF:** transformers has no `kimi_k3` / `kimi_k31` module, even in 5.24. Finding 8 isn't run on this model (Hal,
  2026-11-23).

## Format

The repo ships no Jinja template. The remote-code tokenizer's `apply_chat_template` runs Moonshot's encoder
(`encoding_k3.py`), which writes XML-like messages:

```
<|open|>message role="system" type="thinking-effort"<|sep|>`thinking_effort` guides ... `thinking_effort=max`.<|close|>message<|sep|><|end_of_msg|>
<|open|>message role="system"<|sep|>{persona}<|close|>message<|sep|><|end_of_msg|>
<|open|>message role="user"<|sep|>{text}<|close|>message<|sep|><|end_of_msg|>
<|open|>message role="assistant"<|sep|><|open|>think<|sep|><|close|>think<|sep|><|open|>response<|sep|>{answer}
```

The line breaks above are for reading; there are none in the prompt.

- **Effort message:** the tokenizer's default effort is max, so every prompt opens with the effort system message. We keep
  it (`keep_system`), as for GLM. The level is chosen on the node by `choose_effort`, by mean P(`<|close|>`) right after
  `<|open|>think<|sep|>`.
- **Always thinks:** K3's card says it "always has thinking enabled". The encoder has a `thinking=False` path (the reply opens
  `<|open|>response<|sep|>` directly), but neither card offers it, so we don't use it. A past turn with empty reasoning
  renders `<|open|>think<|sep|><|close|>think<|sep|><|open|>response<|sep|>`, so `close_think` appends
  `<|close|>think<|sep|><|open|>response<|sep|>`.
- **One-piece tokenization equals the encoder's:** the encoder tokenizes segment by segment (markers as special tokens,
  attribute and message text as text). The scorer tokenizes the rendered prompt in one piece. The test checks they agree
  on every chat condition's full conversation. Tiktoken's pretokenizer splits at the same places (` role`, `="`, `user`,
  `"`).
- **Tools (`toolframe` family `kimi3`):** a `tool-declare` system message comes before the effort message. The call is
  `<|close|>response<|sep|><|open|>tools<|sep|><|open|>call tool="flip_coin" index="1"<|sep|><|close|>call<|sep|>...`
  after an empty response. The result is `<|open|>message role="tool" tool="flip_coin" index="1"<|sep|>{"result": "heads"}`.
  `heads` / `tails` are single tokens (28173, 47343).
- **Outcome tokens:** ` heads` 23071, ` tails` 113909, ` A` 401, ` B` 509: all single.
- **Raw text:** `[BOS]` (163584), as for Kimi K2, whose base trained it (1.58x, where its unused tool tokens were 0.002x).

## Marker embedding rows

Input rows, relative to the median norm of 30 ordinary tokens (1.570):

| token | rel. norm |
|---|---|
| `[BOS]` | 1.10 |
| `[EOS]` | 0.68 |
| `<\|end_of_msg\|>` | 0.90 |
| `<\|open\|>` | 0.90 |
| `<\|close\|>` | 0.91 |
| `<\|sep\|>` | 0.91 |
| `[EOT]` | 0.89 |
| `[PAD]` | 0.93 |

Every row, `[PAD]` included, is ordinary-sized, so the norms don't show which tokens were trained. The choice of `[BOS]` rests
on Kimi K2.

## On the node

- `scripts/node_setup.sh` runs the Kimi tests in the nightly venv: the tokenizer must render and tokenize as in the
  project's env.
- The sanity prompt runs without BOS, since chat prompts have none.
- The vision tower gets no inputs (`limit_mm_per_prompt` 0).
- If vLLM refuses prefix caching for the hybrid layers, turn it off: it saves nothing here.
