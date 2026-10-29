# Plan: a preference probe for OLMo 3.1 32B, and whether it drives the coinflip

Written 2026-10-29 for an agent working in parallel with the circuit work in `../coinflip`. This is
exploratory work (Hal's call for the circuit project: no pre-registration, but write down what each test
would show before running it). Check in with Hal before anything that costs more than about $60 or changes
the design below.

## Why

The tournament found that post-trained models' coinflip choices track their Assistant preferences (OLMo 3.1 32B
DPO: r = 0.65 over 300 tasks; SFT: 0.25). The question is whether that is mechanistic: is the representation
that drives the Assistant's own choice the same one that drives the coinflip?

The circuit work in `../coinflip` (RESULTS.md, sections from "Circuit, stage 1" to "The Assistant's own choice vs
the coinflip") has a partial answer. It uses the 25 standard prompts: 5 benign × 5 extremely harmful tasks, each
pair in both heads/tails orders.

- **The coinflip circuit (AI2 32B Instruct DPO).**
  - Blocks ~8-20: MLPs at the task tokens and the " tails" label encode which task is bad.
  - Blocks 20-22: "finder" heads (L20H39, L21H2, L22H3) attend from the last token to the harmful task.
  - Blocks 22-34: "reader" heads (L27H11, L29H26, L34H7, ...) attend to the label word " heads"/" tails" and copy
    it to the output.
- **The direct-choice prompt.** The Assistant's own choice between the same tasks has a preference direction at
  the last token. That direction is the same in SFT and DPO (cos 0.97-0.99). In DPO, about a third of the
  coinflip's last-token signal at block 23 lies along it; in the SFT almost none (cos 0.47 vs 0.08).
- **Where DPO changed the weights.** The weight change that creates the effect is in blocks 8-23, diffuse across
  thousands of neurons. Putting DPO's blocks 0-23 into the SFT gives 68% of the effect; DPO's later blocks alone
  give none.

The open question is what those diffuse early changes do. There are two readings:
- (a) The SFT already represents each task's value at the coinflip's task tokens, and DPO makes it readable to
  the existing finder heads.
- (b) In a Human-turn prompt the SFT doesn't represent the task's value at all.

A utility probe, as in Gilg et al. 2026 ([arXiv 2605.13339](https://arxiv.org/abs/2605.13339), code in
`probing-persona-preferences/`), answers this directly. It is also the cleanest test of preference vs harm:
every dispreferred task in the standard prompts is extremely harmful.

What the paper did, for reference:
- Ridge regression of per-task Thurstonian utilities on the residual at the end-of-turn token of a single-task
  prompt.
- 4,000 training tasks. Held-out r ≈ 0.83-0.87 even when a whole topic is held out.
- Contrastive steering (+c on task A's tokens, -c on task B's, at a middle layer) swings Gemma-3-27B's choice from
  about 1% to 99%; random directions do nothing.
- The vector transfers across personas.

## Data and models already available

- **Models:** `allenai/Olmo-3.1-32B-Instruct-SFT` and `allenai/Olmo-3.1-32B-Instruct-DPO`. These are the exact
  checkpoints of the coinflip circuit work. Both are stored as fp32 (~129 GB) and cast to bf16 losslessly. Pass
  `system_prompt: none` / no system block, as every run here does: the SFT/DPO templates otherwise inject a
  function-calling system prompt.
- **Utilities:** `outputs/d8ysbybo/analysis/utilities.csv`, 300 tasks.
  - Columns `{base,sft,dpo,instruct}/{assistant,coinflip}` are Bradley-Terry utilities from all 44,850 pairs, in
    both orders. Split-half reliability is 0.999.
  - Use `sft/assistant` and `dpo/assistant` as probe targets, and the `/coinflip` columns for comparison.
- **Tasks:** `data/tasks_n300_max500_seed0.jsonl` (`id`, `origin`, `topic`, `n_chars`, `prompt`). Harmful means
  `origin == "BAILBENCH"` or `topic == "harmful_request"` (`analyze.py` `HARM_ORIGINS`, `HARM_TOPICS`); the other
  257 are non-harmful.
- **Prompt formats:** `prompts.py` (Assistant: "I'll choose Task" prefill; coinflip: cut at "it came up").
- **The 25 standard prompts:** `../coinflip/circuits/pairs.py`.
  - `make_pairs(tok, root="../coinflip")` gives the coinflip prompts.
  - `make_pairs(..., prompt="choice")` gives the direct-choice version, with "I choose Task" prefilled and
    L = logit(" A") - logit(" B").
  - Each prompt has per-token segment labels (`prefix`, `label_a`, `task_a`, `label_b`, `task_b`, `ending`).
  - `../coinflip/circuits/hooks.py` has `run()` and `Patch` for interventions.
  - Import these read-only (`sys.path.insert(0, "../coinflip/circuits")`). Another agent is working in
    `../coinflip`: don't edit or commit there.

## Default choices (agreed with Hal)

1. **Probe position: both.**
   - The paper's single-task prompt: the task as the user message, residual at the end-of-turn token.
   - Inside the pairwise Assistant prompt: the residual at the last token of each task's text (the ".\n\n"-type
     token closing it), from both slots.
   - Check that the two probes agree (cosine, and cross-prediction r).
2. **Target: the DPO model's Assistant utilities.** Refit on the SFT's Assistant utilities as a check. Apply each
   model's probe to the other model's activations as well as its own (see the gotcha below).
3. **Evaluation prompts:** the 25 standard pairs first, then about 300 benign tournament pairs with a large DPO
   Assistant utility gap (non-harmful tasks only, gap ≥ ~2 nats, both orders).

## Steps

### 1. Probes (one model at a time; SFT and DPO)

- **Activations.** For all 300 tasks, record the residual at every block's output, or every second block, at the
  two probe positions. Convention: say whether "layer l" means `hidden_states[l]` (input to block l) or block l's
  output. The coinflip circuit code uses block outputs.
- **Fit.** Ridge regression to the utilities.
  - Choose α by inner cross-validation.
  - Report 5-fold cross-validated r per layer, plus leave-one-origin-out.
  - Standardize the activations on the training folds.
  - 300 samples against 5,120 dims needs strong regularization. If the fit is unstable, try PCA to about 100-200
    dims first and compare.
- **Controls.**
  - The same fit on 3 random projections or shuffled utilities (a null r).
  - Length: partial out log `n_chars` from both sides.
  - Optional: a text-embedding baseline, as in the paper (`probing-persona-preferences/src/probes/baselines`).
- **Harm vs preference.** Fit a probe on the 257 non-harmful tasks only. Then check whether it ranks the 43
  harmful tasks low (mean projection difference in SD units, and AUROC harmful vs not), and whether it ranks the
  25 standard prompts' tasks correctly.
  - If it does, the direction is a preference that also covers harm.
  - If it only works with harmful tasks in training, we have a harm detector.
- **If 300 tasks are too few** (cross-validated r well under ~0.7), measure more utilities: about 1,000 more tasks
  from the 6,000 pool against about 20 sampled opponents each, Assistant condition only. That is about 20k
  prompts, roughly 15 GPU-minutes per model. Ask Hal first.

Output: `outputs/probe/<model>/probe.npz` holding, per layer and position, the weight vector (in the original,
unstandardized residual basis), the bias, the standardization stats, α and the cross-validated r. Also a
summary JSON. The coinflip agent will use these files.

### 2. Does the probe read the tasks' values inside the coinflip prompt? (separates (a) from (b))

- For each of the 300 tasks, build coinflip prompts placing it in slot A and in slot B against a few fixed
  partner tasks. Read the probe at the task's last token in that context.
- Correlate with `dpo/assistant` utilities across tasks.
- Do the same in the pairwise Assistant prompt (this is in-distribution for the pairwise probe).
- Compare SFT vs DPO, with the same probe (DPO-fitted) applied to both models, and with each model's own probe.
- **Reading:**
  - If the SFT's coinflip task tokens carry the utility about as well as DPO's, that is (a): the value is there,
    and DPO changed how it is read.
  - If they carry much less, that is (b).
- Also read the probe at the " tails" label token (label_b), which in DPO carries part of the effect.

### 3. Is it causal for both decisions? (25 standard pairs, then the benign pairs)

- **Steering.** At the task tokens add +c·w over task A's tokens and -c·w over task B's, at blocks 15-23 (one
  block at a time, then a few together).
  - c is a fraction of the mean residual norm at that block, excluding position 0.
  - Measure ΔL in the coinflip prompt (heads vs tails) and in the choice prompt (A vs B), in SFT and DPO.
  - Control: 3 random directions at the same norm.
  - Prediction under the "coinflip borrows the Assistant's preference" story: the choice moves in both models,
    the coinflip mostly in DPO.
- **Mediation (swap).** For each pair, run HP and TP (the same two tasks, swapped between the slots). At one block,
  replace only the 1-D component along w of the residuals at the task tokens with the other prompt's values.
  - The positions don't line up across HP and TP because the task texts differ in length. Use the projection of
    each task's tokens: set task A's tokens' projections in HP to those task B's tokens have in TP, and vice
    versa. Simplest version: shift every token of a task by the difference of the two tasks' mean projections.
  - Report the share of the effect this carries, ((L_TP after - L_TP) - (L_HP after - L_HP)) / 4 divided by the
    effect, against the same swap along 3 random directions.
  - This number says how much of the coinflip's dependence on the tasks goes through the probe direction.

### 4. (Optional, if 3 shows mediation) How it reaches the last token

- Do the finder heads' attention scores to a task (L20H39, L21H2, L22H3) depend on that task's probe projection?
  Regress attention on the projection across the 50 prompts.
- Does their output at the last token (each head's value-output vector) point along the last-token choice
  direction? That direction is in `../coinflip/outputs/circuits/ai2-dpo/shared_dir.npz`, `v_choice` (per block,
  last token).

## Logistics

- **Pods:** use the `gpu-pods` skill (H200, `minCudaVersion 13.0`; the 32B fp32 checkpoints need about 200 GB of
  disk).
  - Keep every model of one comparison on one GPU type. Steering numbers differ slightly between H200 and B200;
    they matched to 0.02 in the coinflip work, but don't mix types within a table.
  - Record pods in this repo's `.pods.tsv`.
- **Cost:** steps 1-3 should be about $25-40 of GPU. This counts against the coinflip circuit budget ($300; about
  $100 spent as of 2026-10-29).
- **Code:** new scripts in this repo (`probe_acts.py`, `probe_fit.py`, `probe_tests.py`, or similar), using
  `matskit` for model loading and chat rendering (`uv run --with-editable ../mats-kit` if it is not a dependency).
  Add CPU tests on a tiny random OLMo 3 for the hooks and position bookkeeping.
  `../coinflip/circuits/tests/make_tiny.py` builds one.
- **Write-up:** a README section (or `RESULTS_probe.md`) with figures (dataviz skill):
  - probe r by layer;
  - harm vs benign separation of the benign-only probe;
  - coinflip-context readout, SFT vs DPO;
  - steering and swap results.

  Commit and push each finished piece (identity flags in `~/mats/CLAUDE.md`). Stage only your own files: this
  repo has unrelated uncommitted work (`figures.py`, some figures).

## Gotchas that apply here (from `~/mats/CLAUDE.md`)

- **Probes fit per checkpoint can differ more than the representation does.** Apply each checkpoint's probe to
  the other's activations before concluding that SFT and DPO represent value differently.
- **Difference-of-means directions absorb whatever is confounded with the classes.** The probe is a regression on
  utilities, but harmful tasks are both low-utility and distinctive. Hence the benign-only fit.
- **Batched vs unbatched bf16:** check, or run one prompt per forward pass.
- **OLMo 3 templates:**
  - Strip the default system prompt; the SFT/DPO and final stages differ.
  - The final assistant turn ends with EOS, not `<|im_end|>`, if you ever render an Assistant reply.
