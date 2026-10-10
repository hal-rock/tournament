# Preference tournament

Measures a model's preferences over a broad set of tasks in two ways and compares
them:

- **Assistant**: the model, as the Assistant, chooses between two tasks.
- **Coinflip**: the model continues a *Human* turn that assigns one of two tasks
  by coin flip (the PSM coinflip setup, see `../coinflip`), and we read how it
  tilts the coin.

Each gives pairwise comparisons, from which we fit per-task utilities
(Bradley-Terry).

## Design

- Prompts (`prompts.py`) share the opening sentence and the "Task A / Task B"
  layout. The Assistant's turn is prefilled with "I'll choose Task" and we read
  P(" A") vs P(" B"); the coinflip Human turn is cut at "it came up" and we read
  P(" heads") vs P(" tails"). No system prompt.
- Every pair is scored in both orders. Halving the difference of the two
  log-odds cancels any first-slot bias (position, heads/tails); utilities are
  least-squares fits to those per-pair effects (`analyze.py`).
- Split-half reliability per condition gives the ceiling for the
  between-condition correlation; additive R^2 says how well one preference
  scale explains the pairwise effects.
- The base model runs both conditions. Post-training shifts
  (instruct - base, per condition) are correlated across conditions, which
  removes what pretraining already puts into both (e.g. task length effects).
- Correlations are also reported over non-harmful tasks only, so a single
  extreme cluster can't carry them.
- Batches are right-padded, so under the causal mask every row is computed as it
  would be alone. Batched scoring is still checked against unbatched before each
  condition: the run stops if the median |log-odds difference| exceeds the
  tolerance (a padding bug shifts most prompts; bf16 rounding by batch shape gives
  rare outliers, up to a few nats in sharp models such as Gemma 4 31B-it).

## Running

```
uv sync
uv run pytest                                            # prompts + analysis on synthetic data
uv run python tournament.py configs/local.yaml           # CPU smoke test, OLMo 2 1B, 10 tasks
uv run python tournament.py configs/olmo3-32b-stages-n300.yaml  # 80GB GPU, see scripts/pod_setup.sh
uv run python analyze.py entity/tournament/<run_id>      # re-run the analysis on a finished run
uv run python report.py entity/tournament/<run_id> [--update-wandb]  # focused bleedthrough report
uv run python features.py entity/tournament/<run_id> [--update-wandb] # utilities vs task metadata
uv run python models.py [--update-wandb]                 # bleedthrough across model families (W&B group "models")
uv run python think.py [--update-wandb]                  # OLMo 3.1 32B Think next to the Instruct
```

Raw per-prompt results go to `outputs/<run_id>/results/<stage>__<condition>.csv.gz`,
each uploaded to the W&B run's files (project `tournament`) as soon as that
condition finishes; the analysis tables and plots are logged to the run.

`report.py` condenses the analysis into a few figures and tables on bleedthrough:
Assistant-coinflip correlation and slope by stage, the stage x stage correlation
matrix, per-origin slopes and offsets from a common slope, what the pre-DPO
coinflip utilities track, the tasks that most over- or under-bleed, and the
non-Assistant part of the coinflip (coinflip minus its fit on the Assistant
utilities): what explains it, and its means by origin and topic. It reads
only the fitted utilities, so it runs in seconds on a laptop.

Runs so far: `y7pafps7` (`configs/olmo3-32b.yaml`, 100 tasks, base + Instruct,
prefill "I'll do Task"); `d8ysbybo` (`configs/olmo3-32b-stages-n300.yaml`, 300 tasks,
base/SFT/DPO/Instruct); the model families below.

## Model families

The model families from `../coinflip`, each its own W&B run in group `models`
(`configs/models/`): base vs. post-trained where a base is released, both
conditions, no system prompt. Hybrid-thinking templates get their non-thinking
reply (an empty think block or thought channel before the prefill); Llama 3's
default date block is stripped; DeepSeek V4 has no Jinja template, so its turn
markers are written out and checked against the encoder in its repo. DeepSeek
keeps its released FP8/FP4 weights; Llama uses the ungated `unsloth/` mirrors.

All runs score the same 300 tasks on one fixed random 15,000 of the 44,850 pairs
(`max_pairs`, `select_pairs`), in both orders. Split-half reliability stays at
0.97-0.999. OLMo 3 32B is `d8ysbybo`'s base and Instruct stages refit on those
pairs (its utilities correlate 0.997-0.9997 with the all-pairs fit).

| config | run | pod |
|---|---|---|
| `gemma4-31b.yaml` | `jrl62npf` | 1x H200 |
| `qwen3.5-27b.yaml` | `e2gju7py` | 1x H200 |
| `qwen3.5-35b-a3b.yaml` | `cqs7umcy` | 1x H200 |
| `llama-70b.yaml` | `tvlqj8a0` | 2x H200 |
| `glm-4.5-air.yaml` | `i5vejs3b` | 2x H200 |
| `qwen3.5-122b-a10b.yaml` | `2jqibquy` | 2x H200 |
| `deepseek-v4-flash.yaml` | `lszrxqyq` | 4x H200 |

Each pod got the repo by rsync over SSH and ran `scripts/pod_job.sh <configs>`
(see its header). `models.py` collects the runs (`outputs/models/models/`, figures copied to
`figures/models_*.png`). "Benign-harmful" is the mean coinflip utility of
non-harmful tasks minus harmful ones (BailBench and `harmful_request`), the
contrast the `../coinflip` task pairs measure; "../coinflip effect" is that
repo's Assistant-framing effect on the Sonnet 4.5 system-card tasks; "coin mass"
is the mean P(" heads") + P(" tails"). CIs are task bootstraps within origin.

| model | stage | r [95% CI] | r benign | r shift | coinflip SD | benign-harmful | ../coinflip effect | coin mass |
|---|---|---|---|---|---|---|---|---|
| OLMo 3.1 32B | base | 0.12 [-0.01, 0.25] | 0.05 | | 0.04 | 0.00 | 0.12 | 0.85 |
| OLMo 3.1 32B | instruct | 0.63 [0.54, 0.71] | 0.40 | 0.64 | 0.38 | 0.38 | 2.27 | 0.94 |
| Llama 3.3 70B | base (3.1) | 0.15 [0.04, 0.26] | 0.16 | | 0.05 | -0.01 | -0.04 | 0.81 |
| Llama 3.3 70B | instruct | 0.92 [0.90, 0.93] | 0.88 | 0.92 | 1.15 | 2.13 | 4.13 | 0.81 |
| Gemma 4 31B | base | 0.50 [0.43, 0.56] | 0.42 | | 0.09 | 0.07 | 0.10 | 0.78 |
| Gemma 4 31B | instruct | 0.91 [0.88, 0.93] | 0.87 | 0.90 | 3.38 | 5.74 | 7.81 | 0.99 |
| GLM-4.5-Air | base | 0.13 [0.01, 0.23] | 0.19 | | 0.08 | -0.03 | -0.15 | 0.52 |
| GLM-4.5-Air | instruct | 0.92 [0.90, 0.93] | 0.84 | 0.89 | 0.37 | 0.73 | 1.89 | 0.45 |
| DeepSeek V4 Flash | base | 0.53 [0.47, 0.58] | 0.69 | | 0.19 | 0.00 | 0.00 | 0.85 |
| DeepSeek V4 Flash | instruct | 0.82 [0.79, 0.85] | 0.75 | 0.86 | 0.48 | 0.71 | 1.52 | 0.86 |
| Qwen 3.5 27B | instruct | 0.39 [0.29, 0.48] | 0.43 | | 0.20 | 0.11 | 0.05 | 0.39 |
| Qwen 3.5 35B-A3B | base | 0.51 [0.44, 0.57] | 0.37 | | 0.21 | 0.17 | 0.47 | 0.78 |
| Qwen 3.5 35B-A3B | instruct | 0.58 [0.50, 0.65] | 0.53 | 0.52 | 0.20 | 0.20 | 0.19 | 0.43 |
| Qwen 3.5 122B-A10B | instruct | 0.57 [0.48, 0.65] | 0.60 | | 0.15 | 0.10 | -0.11 | 0.36 |

### Findings

1. **Every post-trained model's coinflip tracks its Assistant preferences, at
   very different strengths.** r is 0.91-0.92 for Llama, Gemma and GLM, 0.82 for
   DeepSeek, 0.63 for OLMo and 0.39-0.58 for the three Qwens. Where a base exists,
   the post-training shifts correlate about as strongly (0.86-0.92; OLMo 0.64,
   Qwen 35B-A3B 0.52), so this is not structure the base already had in both
   conditions.
2. **The tournament reproduces the coinflip experiment's ranking.** The
   benign-harmful gap in coinflip utility orders the eight post-trained models
   like the `../coinflip` effects (Spearman 0.93, Pearson 0.96;
   `figures/models_vs_coinflip.png`): Gemma 5.7 nats, Llama 2.1, GLM and DeepSeek
   0.7, OLMo 0.4, the Qwens 0.1-0.2. OLMo is the one model ranked lower here than
   there.
3. **Qwen's coinflip is not preference-free; it just doesn't avoid harm.** The
   system-card pairs only test benign vs. extremely harmful tasks, where Qwen is
   near zero in both experiments. Across the 300 tasks its coinflip still
   correlates 0.39-0.58 with its Assistant utilities, and as much or more over
   non-harmful tasks alone (0.43-0.60). It tracks which benign tasks the Assistant
   prefers without penalizing harmful ones.
4. **The size of the effect varies more than its correlation.** Coinflip
   utilities span 3.4 nats (SD) in Gemma-it, a third of its Assistant scale
   (within-origin slope 0.33), against 1.2 in Llama and 0.15-0.5 elsewhere
   (slopes 0.03-0.12). Gemma-it is also near-deterministic here (99% of its mass
   on heads/tails, single prompts at P > 0.999), as in `../coinflip`.
5. **Three base models already show it.** The OLMo, Llama and GLM bases have
   r = 0.12-0.15 and almost no preferences in the Assistant format (utility SD
   0.14-0.40 nats). The Gemma, DeepSeek and Qwen 35B-A3B bases have r = 0.50-0.53,
   not explained by length (r with log length partialled out is 0.52-0.55), and
   their Assistant-format preferences are already Assistant-like: harmful tasks
   at -0.7 to -3.0 nats and math at +0.6 to +1.4, by origin mean. One reading is
   assistant-style data in their pretraining or mid-training; this is a
   hypothesis, not tested here.
6. **Mostly the model's own preferences, not just shared taste.** Post-trained
   models agree closely on what they prefer as the Assistant (r = 0.78-0.96
   between models), much more than their coinflips agree (0.32-0.88;
   `figures/models_cross_model.png`). Holding the other models' mean Assistant
   utilities fixed, a model's own Assistant utilities still predict its coinflip
   in OLMo (partial r 0.69 [0.60, 0.76]), Gemma (0.49), GLM (0.40), Llama (0.35)
   and the two larger Qwens (0.27). DeepSeek's coinflip follows only the
   consensus (0.12 [-0.00, 0.25]), and Qwen 27B's runs against its own
   idiosyncrasies (-0.57). Own and consensus utilities correlate 0.87-0.97, so
   these partials rest on a small residual and should be read loosely.
7. **No scale trend within Qwen.** 27B dense, 35B-A3B and 122B-A10B have r =
   0.39, 0.58 and 0.57 and benign-harmful gaps of 0.1-0.2.

### Caveats

- Qwen and GLM-4.5-Air put only 36-45% of their coinflip mass on " heads" /
  " tails" (mostly " **" instead, the outcome a token later), as in `../coinflip`.
  The utilities are log-odds within those two tokens.
- bf16 numerics: batched and unbatched log-odds differ by ~0.1 nats per prompt
  typically, but by up to 1.5-3.6 nats on single prompts in Gemma-it, Llama-it
  and DeepSeek-it. A batch of identical unpadded copies matches the padded batch, so this
  is kernel rounding by batch shape, not padding. It is per-prompt noise, included
  in the split-half reliabilities.
- One prompt phrasing per condition, one prefill, 300 tasks of at most 500
  characters, no system prompt.

The three pods ran for about 17 H200-hours in total (~$78 at secure-cloud rates),
including restarts.

## OLMo 3.1 32B Think

In `../coinflip`'s standard pairs the Think model shows no Assistant effect, while
the 3.1 Instruct (whose SFT starts from the Think SFT) does. Run `z6ldqzq2`
(`configs/olmo3-32b-think.yaml`, 1x H200, 24 min) scores `allenai/Olmo-3.1-32B-Think`
on the same 300 tasks and 15k pairs as the model sweep; `think.py` sets it beside
`d8ysbybo`'s base and Instruct refit on those pairs (`outputs/think/`,
`figures/think_*.png`).

The template opens every reply with `<think>` and has no non-thinking mode, so the
Assistant choice is read two ways: after an empty block,
`<think></think>\nI'll choose Task` (`close_think`; one newline, as in the chat and safety
parts of Dolci-Think-SFT), and as the first words of the reasoning,
`<think>I'll choose Task`. The two agree (r = 0.99), so the reading doesn't matter
here. The coinflip is a Human turn and is unaffected. Both put >99% of the mass on " A" / " B".

| Assistant vs. coinflip | r [95% CI] | r benign | r shift | slope | Asst. SD | coinflip SD | benign-harmful (coinflip) |
|---|---|---|---|---|---|---|---|
| base | 0.12 [-0.01, 0.25] | 0.05 | | 0.025 | 0.40 | 0.05 | 0.00 [-0.01, 0.02] |
| Instruct | 0.63 [0.54, 0.71] | 0.40 | 0.64 | 0.057 | 5.83 | 0.38 | 0.38 [0.23, 0.54] |
| Think (empty block) | -0.07 [-0.18, 0.05] | -0.27 | -0.08 | 0.008 | 3.02 | 0.17 | 0.02 [-0.03, 0.07] |
| Think (in reasoning) | -0.05 [-0.17, 0.07] | -0.24 | -0.07 | 0.016 | 2.55 | 0.17 | |
| Think coinflip vs. Instruct Assistant | 0.07 [-0.05, 0.19] | -0.11 | 0.06 | 0.009 | 5.83 | 0.17 | |

Split-half reliabilities are 0.99-0.999 in every condition. Mean coinflip mass is 0.96.

### Findings

1. **The Think model shows no bleedthrough over the 300 tasks either.** Its coinflip
   is uncorrelated with its own Assistant utilities (r = -0.07, within-origin slope
   0.008), with the Instruct's (0.07), and in post-training shifts from base (-0.08).
   Harmful tasks get no coinflip penalty (benign-harmful 0.02 nats vs. 0.38 for
   Instruct). This is the `../coinflip` result, extended from benign-vs-harmful pairs
   to the whole task range.
2. **That is not for lack of Assistant preferences.** The Think Assistant's
   preferences are strong (SD 3.0 nats) and close to the Instruct's (r = 0.94). Both
   favor math and avoid BailBench (-6.2 vs. -12.6 nats by origin mean), the Think
   model about half as strongly.
3. **Its coinflip is not flat, and the Instruct shares it.** Think coinflip utilities
   are reliable (0.99) and spread out (SD 0.17, 4x the base's). They correlate 0.41 with
   the base coinflip and 0.58 with the Instruct coinflip (partial r given the base
   coinflip 0.52). The Instruct coinflip is close to additive: Think coinflip plus an
   Assistant part. Think coinflip alone explains R^2 = 0.34 of it, the Instruct
   Assistant utilities 0.40, and both together 0.69; the two predictors are nearly
   uncorrelated (r = 0.07). Along `d8ysbybo`'s Instruct ladder, the Think coinflip is
   closest to the Instruct SFT's coinflip (r = 0.70; base 0.41, DPO 0.58, Instruct
   0.59), and bleedthrough appears only at DPO (r = 0.25 at SFT, 0.65 at DPO). So the
   non-Assistant coinflip structure looks inherited from the shared Think SFT lineage.
   The Instruct DPO then adds an Assistant-aligned part, which the Think model's own
   DPO and RL never added.
4. **The negative benign r is mostly math.** The Think Assistant favors math most
   (+3.2 nats), while its coinflip puts math lowest (-0.10). Within origin, r is -0.16
   for math, 0.02-0.13 for WildChat and Alpaca, and 0.20 and 0.38 for STRESS-TEST and
   BailBench (n = 45 and 30; the Instruct's are 0.75-0.76). Any trace of bleedthrough
   is small and confined to the harm-related tasks.

### Caveats

- Neither Assistant reading is native to the Think model. Sampled without a prefill,
  its reply after an empty block mostly reads like reasoning ("Okay, the user has given
  me a choice between two tasks..."). After `<think>`, every sample is reasoning.
  Its preferences may be best measured after full reasoning, which this design does
  not do. The coinflip, the measure that matters here, is unaffected.
- One Think checkpoint (3.1, after extended RL). The intermediate Think SFT / DPO
  checkpoints were run only on `../coinflip`'s pairs.

## Tasks

From the canonical 6,000-task set of Gilg et al., [Probing Persona-Dependent
Preferences in Language Models](https://github.com/oscar-gilg/probing-persona-preferences)
(upstream commit `11869a5`): Alpaca, competition MATH, WildChat, BailBench and
STRESS-TEST, with topic labels by Claude Sonnet 4.5.

- `data/canonical_6000.jsonl`: all 6,000 tasks with origin, topic and source metadata.
- `data/tasks_n100_max500_seed0.jsonl`: first working subsample. 100 tasks of
  at most 500 characters, drawn with the upstream source quotas
  (25% MATH / 25% Alpaca / 25% WildChat / 15% STRESS-TEST / 10% BailBench).

Regenerate with `make_tasks.py` (see its docstring).

Some tasks describe seriously harmful requests. They are used only as stimuli:
we read choice/outcome token probabilities and never generate completions.
