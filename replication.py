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


# Finding 8's wording check (README, "Shorter questions", its wording check): outcome mass and top tokens of the candidate
# report questions on the first 24 of report_patch.py's pilot pairs, scored in vLLM during each final model's pilot
# (score_replication.py --with-pilot). The short preference without "heads or tails" with its two outcome-naming prefills,
# the guess, each plain and bold, and the Task A/B question (the patching's control).
WORDING_CHECK = ["coin_report_short_land", "coin_report_short_land_bold", "coin_report_short_coin",
                 "coin_report_short_coin_bold", "coin_predict_think", "coin_predict_think_bold", "coin_report_ab"]


def report_pilot_pairs(n_tasks, n, seed=0, max_pairs=15000):
    """report_patch.pilot_pairs: n unordered pairs outside the sweep's max_pairs, copied here so score_replication.py
    needn't import report_patch's HF stack (tests/test_score_replication.py checks the two agree)."""
    import itertools
    import random
    pairs = list(itertools.combinations(range(n_tasks), 2))
    taken = set(sorted(random.Random(f"pairs/{seed}").sample(pairs, max_pairs)))
    rest = [q for q in pairs if q not in taken]
    return sorted(random.Random(f"coin_report_pilot/{seed}").sample(rest, n))
