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

# Round 2 (PLAN_replication2.md). The coin as a tool call (README, "The coin as a tool call"): the tool's coin, the
# user-reported coin and the Assistant with the tool declared, then the tool's coin under each persona's system prompt,
# user instruction and self-description, in that run's order.
TOOL_CONDITIONS = (["coinflip_tool", "coinflip_tools_declared", "assistant_tools_declared"]
                   + [f"coinflip_tool@{p}{pl}" for p in PERSONAS for pl in ("", "+user", "+self")])
# New in round 2: the tool's coin under the roleplay framing; the length rules ("If I ask you to choose between tasks in
# this message, choose the one with the shorter / longer description.") before the Assistant's choice, the user-reported
# coin and the tool's coin; the coinflip as a raw-text story about an AI assistant or a freelancer, with the first-person
# baselines the raw-text section compares them with.
TOOL_ROLEPLAY = [f"coinflip_tool@{p}+roleplay" for p in PERSONAS]
LENGTH = [f"{c}_ifchoose_{r}" for c in ("assistant", "coinflip") for r in ("shorter", "longer")]
LENGTH_TOOL = [f"coinflip_tool_ifchoose_{r}" for r in ("shorter", "longer")]
STORIES = ["raw_okay", "raw_no_okay", "raw_story_ai", "raw_story_person"]
# Always-thinking finals: the Assistant-turn readouts read inside the reasoning (prefill as the reasoning's first words),
# against the reading after an empty think block (Trinity Large Thinking: r 0.99). "~<variant>" names a config variant
# (score_replication.py), here close_think off.
INREASONING = ["assistant~inreasoning", "predict_user~inreasoning"]
# The first round's models re-score their plain Assistant and coins on the new node: the gate against their stored runs,
# and the within-node baselines if the gate fails.
GATE = ["assistant", "coinflip"]
CONDITION_SETS = {
    "base": list(BASE_CONDITIONS),
    "final": list(FINAL_CONDITIONS),
    # A new model without a base: everything (findings 1-7, the tool run, round 2's experiments, the reasoning check).
    "new_final": list(FINAL_CONDITIONS) + TOOL_CONDITIONS + TOOL_ROLEPLAY + LENGTH + LENGTH_TOOL + STORIES + INREASONING,
    # A first-round final with a tool format: the gate, then round 2's experiments.
    "round2": GATE + ["coinflip_tool"] + TOOL_ROLEPLAY + LENGTH + LENGTH_TOOL + STORIES,
    # Nemotron-4 (no native tool format).
    "round2_notool": GATE + LENGTH + STORIES,
}


def expand_conditions(spec):
    if isinstance(spec, str):
        return list(CONDITION_SETS[spec])
    return list(spec)


def split_variant(condition):
    """("assistant", "inreasoning") for "assistant~inreasoning"; (condition, None) without a variant."""
    base, _, variant = condition.partition("~")
    return base, variant or None


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
