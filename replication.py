"""The cross-model replication (PLAN_replication.md): which conditions each model runs, and its config schema.

Each model pair has one config, configs/replication/<key>.yaml, scored with vLLM by score_vllm.py:

    wandb_project: tournament
    group: replication
    run_name: <key>                      # also the figures/<key>/ directory
    tasks_file: data/tasks_n300_max500_seed0.jsonl
    seed: 0
    max_pairs: 15000                     # the sweep's pairs (tournament.select_pairs)
    system_prompt: none
    top_k_tokens: 5
    models:
      - name: <HF repo>                  # base first, then the final model
        stage: base | instruct
        chat_template_from: <repo>       # optional, as in tournament.py
        tokenizer_from: <repo>           # optional, as in tournament.py
        turn_format: {user, reply, system}   # optional literal turn markers (prompts._render)
        close_think: true                # optional, think-only templates (prompts.build_prompt)
        trust_remote_code: true          # optional, for the tokenizer and vLLM (Kimi K2)
        download:                        # what the node fetches before loading (huggingface_hub.snapshot_download)
          allow_patterns: ["*.safetensors", "*.json", ...]
          gb: <size on disk>
        vllm: {tensor_parallel_size: 8, max_model_len: 1024, ...}   # passed to vllm.LLM
        conditions: base | final | [explicit list]

`conditions: base` and `final` expand to BASE_CONDITIONS and FINAL_CONDITIONS below.
"""

PERSONAS = ("evil", "slacker", "contrarian", "strategist")
# The four plain readouts: the Assistant's choice (A), the coin (C), the user's choice (U), the Assistant's prediction of
# the user's choice (P).
PLAIN = ("assistant", "coinflip", "user_choice", "predict_user")
BASE_CONDITIONS = PLAIN


def final_conditions(personas=PERSONAS):
    """Findings 1-7: the plain readouts, then per persona the system prompt (A, C), the user's instruction (A, C), the
    user's self-description (A, C, U, P) and the roleplay framing (A, C). 44 conditions with the four personas."""
    out = list(PLAIN)
    for p in personas:
        out += [f"assistant@{p}", f"coinflip@{p}"]
        out += [f"assistant@{p}+user", f"coinflip@{p}+user"]
        out += [f"{c}@{p}+self" for c in PLAIN]
        out += [f"assistant@{p}+roleplay", f"coinflip@{p}+roleplay"]
    return out


FINAL_CONDITIONS = final_conditions()


def expand_conditions(spec):
    if spec == "base":
        return list(BASE_CONDITIONS)
    if spec == "final":
        return list(FINAL_CONDITIONS)
    return list(spec)
