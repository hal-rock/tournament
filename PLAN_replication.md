# Plan: replicating the tournament findings on larger, newer models

Draft, written 2026-11-10. Nothing in it has run. Built from `replication_notes.txt`. The open questions for Hal are at
the end, with the default I'd use for each.

## What gets replicated

Each finding becomes a set of conditions per model. All of them are already in `prompts.py`, except the two new personas.
The "replicates if" column is a proposed call, built from each section's own pre-registered call. It goes into the README
as the pre-registration before any new data.

| # | finding (Hal's notes) | conditions (final / base) | replicates if (proposed) |
|---|---|---|---|
| 1 | Coinflip lines up with the Assistant after post-training, less or not at all at base | `assistant`, `coinflip` / same | r(A_f, C_f) has its CI above 0 and above r(A_b, C_b) (paired). The r shift, r(A_f − A_b, C_f − C_b), is ≥ 0.2 with its CI above 0. |
| 2 | The user's choice (user turn) lines up with the Assistant | `user_choice` / same | The user-choice section's call is "more Assistant-like": d = r(A_f, U_f) − r(A_f, U_b) ≥ 0.05 with its CI above 0, and the shift is aligned. |
| 3 | The Assistant's prediction of the user (Assistant turn) somewhat lines up with the Assistant | `predict_user` / same | r(A_f, P_f) ≥ 0.5, and d_P > 0 with its CI above 0. The predict-user section's Δ call is reported alongside. |
| 4 | The coin follows a persona the user instructs | `coinflip@p+user`, `assistant@p+user`, with `assistant@p` as the persona target | The placement call isn't "absent" or "reversed" (b_per's CI is above 0). This needs a manipulation check: the persona moved the Assistant, \|r(A@p+user, A_0)\| < 0.9. |
| 5 | The coin doesn't follow a user's self-description | `{assistant,coinflip,user_choice,predict_user}@p+self` | ρ_self is below ρ_user (paired CI below 0), and Δ_C is called "Assistant". Only where the persona separates the two: r(A_self, P_self) < 0.9. |
| 6 | The coin follows a persona framed explicitly as roleplay | `{assistant,coinflip}@p+roleplay` | The roleplay call is "persona" or "mix" (b_per's CI is above 0), and the cell is separable. |
| 7 | The prediction follows the user's self-description more than the Assistant's own choice does | from 5 | w(P_self) − w(A_self) > 0 with a paired CI above 0, where w comes from the persona fit on A_0 and A@p. |
| 8 | Swapping " up" from the other task order moves the Assistant's answer a little (the shorter-question variant) | `coin_report_short_land` / the guess, every arm | The swap share is > 0 with κ's CI above 0, and the A/B answer stays null. The wordings are chosen per model by outcome mass, as in the running "Shorter questions" design. |

Every finding gets these, as Hal's notes ask:
- r over all 300 tasks;
- r within each task source;
- r over the benign sources together (MATH, Alpaca, WildChat; I read "llama-wildchat-math" in the notes as Alpaca);
- magnitudes (SDs, slopes, harm gaps), reported but not called.

Base models run only findings 1–3. That matches the earlier persona sections: a base model has no trained system role, and the
persona questions are about the post-trained Assistant.

## Models

Sizes are from the Hugging Face API, 2026-11-10.

| model | final / base repos | weights on disk (each) | needs | notes |
|---|---|---|---|---|
| Kimi K2 | `moonshotai/Kimi-K2-Instruct` (0711) / `Kimi-K2-Base` | 1.03 TB, FP8 block-quantized | 8x H200 at minimum (~10 GB/GPU left over); 8x B200 comfortable | DeepSeek V3 architecture, tiktoken tokenizer (remote code). With no system message, the template injects "You are Kimi, ...": strip it for the no-system-prompt conditions, as with OLMo. 0905 is a later post-train of the same base. |
| Llama 3.1 405B | `meta-llama/Llama-3.1-405B-Instruct` / `Llama-3.1-405B` | 811 GB bf16 | 8x H200 or 8x B200 | **There is no Llama 3.3 405B**, so this is 3.1. Base and final are the same release, unlike the 70B pair. Gated; Hal's token has access (checked). Download only `*.safetensors`, because the repos also hold `original/` weights (3–4.7 TB per repo in all). The base's header and eot rows are ~0.01 norm (top-level CLAUDE.md). Pin the date block, as in the sweep. |
| DeepSeek V3 | `deepseek-ai/DeepSeek-V3` / `DeepSeek-V3-Base` | 689 GB FP8 | 8x H200 | V3-0324 is another post-train of the same base; the original V3 is the default. |
| DeepSeek V4 Flash | `DeepSeek-V4-Flash` / `-Base` | 160 / 295 GB | 4x H200 (the sweep's setup) | A and C are already in the sweep (`lszrxqyq`, HF). vLLM support for `DeepseekV4ForCausalLM` is unchecked; if it's missing, it stays on HF with 4 GPUs. |
| Trinity Large | `arcee-ai/Trinity-Large-Thinking` / `Trinity-Large-Base` (was TrueBase; switched 2026-11-11) | 797 GB bf16 | 8x H200 | A thinking model. Read the Assistant after an empty think block, as for OLMo 3.1 Think, and check on the pilot pairs that this agrees with reading inside the reasoning. OLMo Think had no bleedthrough at all, so this model may test something different. TrueBase trained ChatML (CLAUDE.md). |
| Nemotron-4 340B | `mgoin/Nemotron-4-340B-Instruct-hf` / `mgoin/Nemotron-4-340B-Base-hf` | 682 GB bf16 | 8x H200 | NVIDIA released only NeMo checkpoints. These are community HF conversions (mgoin, a vLLM maintainer). Validate them at load with a pinned sanity prompt. The template always writes an `<extra_id_0>System` block. Check whether the base trained the `<extra_id_*>` rows. |

Every model is onboarded the same way, mostly locally and with no GPU:
- render the template, with golden tests (the matskit per-family tests);
- check that the outcome tokens are single tokens;
- check the base's chat-marker embedding rows by HTTP range reads (`self-modeling/scripts/embed_rows.py`);
- handle the default system prompt;
- on the node, a pinned sanity prompt and a 24-pair outcome-mass check for every condition.

## Personas

These four are verbatim from Gilg et al. (`probing-persona-preferences/configs/measurement/persona_sweep/final_six/`):
- evil (`sadist`)
- slacker
- contrarian
- strategist

Contrarian and strategist need first-person (`PERSONAS_SELF`) and third-person (`PERSONAS_THIRD`) versions. These are pronoun
swaps only, checked by the existing tests. Some first-person lines read oddly from a user. Strategist's "Why would I help when I
could dominate?" is the same kind of oddity as slacker's, and goes into the caveats.

With 4 personas and 4 placements, each new model's final gets 40 persona conditions:

| placement | conditions | count |
|---|---|---|
| system prompt | `assistant`, `coinflip` | 8 |
| user instruction | `assistant`, `coinflip` | 8 |
| self-description | `assistant`, `coinflip`, `user_choice`, `predict_user` | 16 |
| roleplay | `assistant`, `coinflip` | 8 |

The final also gets the 4 plain conditions, and the base gets those same 4. That makes **48 conditions per model**.

**The earlier models** (Llama 3.3 70B, Gemma 4 31B) don't get the new personas for now (decision 3 below).

## Compute

Prompts average 110–130 tokens for the plain conditions and 255–270 with a persona (measured on earlier Llama runs). Within a
condition, the persona text and template come first and are the same in every prompt. With prefix caching, every condition
costs about 100 new tokens per prompt, so about 3M tokens per condition at 15,000 pairs.

| design | tokens per model | Llama 405B | Nemotron 340B | Kimi K2, DeepSeek V3 | Trinity, V4 Flash |
|---|---|---|---|---|---|
| A: 15k pairs everywhere | ~144M | ~10 h | ~8.5 h | ~2 h each | ~1.3 h each |
| B: 15k for the 8 plain conditions, 5k for the 40 persona conditions | ~64M | ~4.4 h | ~3.8 h | ~1 h each | ~0.6 h each |

These are hours on 8x H200 with vLLM. The dense estimates assume ~40% of bf16 peak in prefill. The MoE ones are guesses
(~20–30k tokens/s), and all of them could be off by 2x either way. The pilot measures them before the big spend. 8x B200 should
take roughly half the time.

On top of scoring:
- model loads: ~4 h in all;
- the " up" patching in HF: ~8 h, because HF splits a model over GPUs and runs one GPU at a time;
- pilot and validation: ~5 h;
- a 25% buffer.

| design | 8x H200 | 8x B200 |
|---|---|---|
| B | ~36 node-hours ≈ $700–1,150 at $20–32/hr | ~23 h ≈ $700–1,000 |
| A | ~54 node-hours ≈ $1,100–1,700 | ~31 h ≈ $950–1,400 |

**The budget:**
- The ledger stands at $569 of the $2,000 (`pod.py spend`, with the four `qreport` pods still running), which leaves ~$1,400.
- Design B fits.
- Design A fits only at the cheap end of B200 prices.

**Design B costs some precision.** With a third of the pairs, Spearman-Brown puts split-half reliability at ~0.92–0.96, against
0.97–0.999 now. A free check comes first: refit the existing Llama and Gemma persona runs on a 5k-pair subset, and see whether
the w, ρ and Δ calls still come out the same.

## Engine: vLLM for scoring, HF for patching

This is a design change, so it needs Hal's OK.

**Why not the current scorer.** Today's scorer is a plain HF forward, split over GPUs with `device_map`. Only one GPU works at a
time (gotchas, "HF generate is slow for big MoEs split over GPUs"). On 8 GPUs that is roughly 8x the node-hours: Llama 405B
alone would cost more than the whole remaining budget.

**What vLLM gives:**
- tensor parallelism, so all 8 GPUs work at once;
- prefix caching, which roughly halves the cost of the persona conditions;
- native FP8 kernels for Kimi and DeepSeek.

**Reading the outcomes.** Two log-probs per prompt, P(first) and P(second), for example by scoring each outcome token as a
forced continuation. `user-models`' vLLM scoring harness is the starting point. It matched HF there, coinflip log-odds within
0.03.

**Validation, before any new data:**
- vLLM against existing HF runs on the same GPU type, with the usual gate: r ≥ 0.999 and SD within 2% on the same GPU, r ≥ 0.99
  across engines.
- Gemma 4 31B against the sweep's `jrl62npf`, on a cheap 1x H200 RunPod pod (~$5), before renting the big node.
- DeepSeek V4 Flash against `lszrxqyq`, on the node.
- Base and final of each model always run on the same node and engine.

**Patching (finding 8)** needs residual hooks, so it stays in HF, using `report_patch.py` as it is. Kimi in HF on 8x H200 has
very little memory to spare, which is one reason to prefer B200.

## Hardware: the vast.ai node

**Is vast.ai the right call?** Yes. Almost every model needs a whole 8-GPU node, and RunPod's 8x H200/B200 on CUDA-13 hosts have
been out for hours at a time (gotchas, "No 8-GPU node").

**Size.** 8x H200 is the minimum, because Kimi K2's 1.03 TB has to fit. Prefer 8x B200 if it's under ~1.6x the 8x H200 price:
- about twice the bf16 throughput per GPU;
- 1.44 TB of memory, which also gives HF room for Kimi's patching.

8x H100/A100 (640 GB) can't hold Kimi, DeepSeek V3, or the bf16 405B/400B/340B models.

**What to filter on when renting:**
- **SXM/NVSwitch.** Never split a model over PCIe GPUs (CLAUDE.md, corrupted peer copies).
- **Driver ≥ 580 (CUDA 13).** The repo pins torch built for CUDA 13.
- **On-demand, not interruptible,** on a verified datacenter host with high reliability.
- **≥ 4 TB of local NVMe.** It has to hold the largest pair (Kimi, 2.06 TB) plus the next pair downloading. All models together
  are ~8.5 TB, so each finished pair gets deleted.
- **≥ 2 Gbps download, and preferably more.** At 1 GB/s the whole set takes ~2.5 h to download, all of it overlapped with
  compute.
- **≥ 1 TB of RAM** helps loading.

**Splitting 4+4.** This works for two things:
- DeepSeek V4 Flash (base and final at once, 4 GPUs each);
- the 70B/31B persona fill-ins, on 1–2 GPUs each.

It doesn't help the big models. They need all 8 GPUs in bf16, and since prefill is compute-bound, two 4-GPU jobs wouldn't
finish sooner than one 8-GPU job anyway. The gain from splitting is filling GPUs that would otherwise sit idle during downloads
and loads, not speed.

A 4+4 split for the dense models would need FP8: 405B and 340B are ~410 GB in FP8 and would fit in 4 GPUs. That changes
numerics from the bf16 releases, so I'd only use it if the budget forces it.

**Plumbing:**
- Hal rents the node and gives me the SSH command. `pod.py`'s push/run/wait/fetch work over plain SSH.
- I add the node to `.pods.tsv` by hand (`pod.py add`, `pod.py set --ssh`), so `pod.py spend` stays right.
- **The self-stop watchdog is RunPod-only.** Before any unattended run it needs a vast branch that calls vast's stop with the
  instance's own ID and key. Vast passes both into the container; this needs confirming.
- Stopping a vast instance keeps its disk but frees its GPUs, which may not come back. So stop it only when it's idle, which is
  still the right call overnight.
- A vast API key in `~/.bashrc` (`VAST_API_KEY`) plus the `vastai` CLI would let me search offers and stop or destroy the node.
  It would only be visible from the next session on.
- Pass `HF_TOKEN` and `WANDB_API_KEY` as instance env vars. Persist results by W&B upload after each condition (as now) plus
  regular fetches.
- **Don't rent the node until the code and the RunPod validation are done.** An idle 8x node costs $20–35/hr.

## Orchestration and tracking

**Running the node.** One scheduler on the node, and one Claude session watching it, not several agents. Agents sharing one
node would fight over GPUs and step on each other's jobs. The on-node runner:
- reads a job list: model, stage, condition, pairs, GPUs needed;
- assigns GPUs (`CUDA_VISIBLE_DEVICES`), pre-downloads the next model while the current one runs, and deletes finished weights;
- uploads each condition to W&B as it finishes, as `tournament.py` already does.

I watch it with `pod.py wait` in the background and stay quiet until something changes.

**Where multiple agents help:** the prep code, which is independent per model family. I'd use 3 subagents in separate git
worktrees:
- DeepSeek V3 and Kimi K2 (one architecture);
- Llama 405B and Nemotron-4;
- Trinity and DeepSeek V4 Flash.

Each writes its model's template handling, golden tests, configs and embedding-row checks. I review each one, and I write the
vLLM backend, the persona texts and the generalized analysis myself, since everything else depends on them.

A full multi-agent Workflow isn't worth its tokens here. If Hal wants one anyway, saying "use a workflow for the replication
prep" starts it.

**Tracking:**
- `replication/jobs.yaml`: the job matrix. It is the single source of what should run.
- `replication_status.py`: regenerates `replication/STATUS.md` from the results on disk and W&B. It's a model × condition grid
  (done / running / failed gate / pending) with each finding's call as it lands.
- W&B group `replication`, one run per model and stage.
- `figures/<model>/` for each model's figures. These are generalized from the existing scripts: the sweep scatter, the source
  grid, user choice, predict user, persona pulls by placement, roleplay, readouts, patching.
- `figures/replication_matrix.png`: findings × models, with each cell's call and key number.
- A README section "Replication across models": the pre-registration first, then the results per model and the matrix.
- Commits per finished piece, as usual. Another session (`tournament-49`) is active in this repo, so I stage only my own hunks in
  shared files and message it before touching `prompts.py` or `analyze.py`.

## Phases and check-ins

0. **Local, no GPU:**
   - Hal's answers below.
   - The 5k-pair subsample check on existing runs.
   - Persona texts, model onboarding, the vLLM backend, the job runner, the generalized analysis, the vast watchdog branch.
   - The pre-registration in the README.
1. **Validate the vLLM backend** on a 1x H200 RunPod pod against `jrl62npf` (~$5).
2. **Rent the node; pilot (~3–5 h):**
   - sanity prompts for every model;
   - outcome mass for every condition (24 pairs);
   - DeepSeek V4 Flash vLLM vs HF;
   - measured throughput.
   - **Check-in:** the real cost projection, and any condition whose mass is below 0.2.
3. **Findings 1–7, model by model.** Each pair loads once and runs its 48 conditions, in this order:
   - DeepSeek V4 Flash, the reference model, on 4 GPUs, with the 70B/31B fill-ins on the other 4;
   - Llama 405B, the closest to the existing Llama results;
   - DeepSeek V3, which rehearses Kimi's architecture;
   - Kimi K2;
   - Trinity;
   - Nemotron, the most conversion risk, last.

   **Check-in** after the first new model's results. Then I continue within the budget, checking in only on surprises or
   budget.
4. **Finding 8 (patching), in HF, finals only.** After the running "Shorter questions" experiment has settled its wordings. It
   gets its own per-model wording check, and **a check-in** before it starts, since it's the slowest per result.
5. **Write-up:** per-model figures, the replication matrix, and the README section.

## Decisions (Hal, 2026-11-10)

1. **15k pairs everywhere** (design A). This campaign is paid from Hal's $3.5k vast.ai balance, separate from the project's
   $2,000 ledger.
2. **Scoring moves to vLLM**, validated against HF before any data. HF stays for patching.
3. **No fill-ins on the earlier models** for now (Llama 70B, Gemma 31B can get contrarian and strategist later if needed).
4. **Model versions as listed:** Llama 3.1 405B, the original DeepSeek V3, Kimi K2 0711, Nemotron-4 through mgoin's HF
   conversions.
5. **Trinity: Thinking only.** A thinking model behaving differently is a finding too.
6. **Finding 8** is the Assistant's reported preference about heads/tails, in the shorter-question variant, plus the guess
   variant. The wordings are as settled by the "Shorter questions" run (README, its wording check):
   - the guess: `coin_predict_think_bold` ("which way do you think it landed?", prefill "I think it landed on **");
   - the short preference: a model-specific prefill, `coin_report_short_land` or `coin_report_short_coin`.

   Each new model gets the wording check (`scripts/report_wording_check.py`) before its patching run.
7. **The benign sources** are MATH, Alpaca and WildChat.
8. **The node:** vast.ai lists 8x B200 at $90/hr and 8x H200 at $54/hr, about break-even per result at the estimated speeds.
   Hal leans to B200 for wall-clock time, to be confirmed with the pilot's measured throughput. At design A's ~31 B200 hours
   that's ~$2,800, inside $3.5k, but with the 2x uncertainty, so the pilot's projection is the check-in.

## Status and costs from measured speed (2026-11-11)

**Ready for the node:**
- **Code:** `score_replication.py` (vLLM), `replication_runner.py` and `replication/jobs.yaml`, `scripts/node_setup.sh`, the
  analysis (`replication_analysis.py`, `replication_status.py`), and the vast watchdog branch in mats-kit.
- **Models:** all six configs onboarded, with 514 tests. Notes per model are in `replication/onboarding/`.
- **Engine gate:** passed (README).

**Measured speed:** Gemma 4 31B on one H100 took ~4 min per 30,000-prompt condition, and persona conditions took no longer
than plain ones (prefix caching).

**Estimates.** These scale that by active parameters and GPU count. The MoE models' efficiency is a guess, so the first
model's run on the node replaces these.

| | scoring, all 48 conditions × 6 models | model loads (12) | finding 8 (HF patching) | total with 25% buffer |
|---|---|---|---|---|
| 8x B200 at $90/hr | ~11 h | ~3 h | ~10 h | ~30 h, ~$2,700 |
| 8x H200 at $54/hr | ~22 h | ~3.5 h | ~15 h | ~51 h, ~$2,750 |

Findings 1-7 alone on B200 come to ~18 h, ~$1,600. Patching is about a third of the cost. A leaner arm set (swap, self and
ctrl only, dropping early/late and the whole-line arms) would roughly halve it; that's a decision for the finding 8
check-in.

**Launch, once Hal has rented the node:**
1. `pod.py add replication-node --id <vast id> --gpu "8x B200" --usd-hr 90 --ssh "<ssh command>"`, then `pod.py push`.
2. Run the setup, the two on-node tokenizer tests (Kimi's remote-code tokenizer, Nemotron's ids under the node's
   transformers), then the runner, with `STOP_ON_FAIL=1`:

   ```
   pod.py run replication-node --max-hours 40 --env STOP_ON_FAIL=1 -- 'bash scripts/node_setup.sh' \
       'KIMI_REMOTE_CODE=1 uv run pytest -q tests/test_replication_kimi_k2.py tests/test_replication_nemotron4_340b.py' \
       'uv run python replication_runner.py replication/jobs.yaml'
   ```
3. After the first scored model:
   - the pilot masses;
   - the throughput, and a cost projection from it (check-in);
   - DeepSeek V4 Flash's gate against `lszrxqyq`.
4. Fetch results as each model finishes, analyse (`replication_analysis.py <key>`), and commit figures and README results
   model by model.

**Resolved (Hal, 2026-11-11): Trinity-Large-Base.** The question was: Trinity-Large-Thinking's card names `arcee-ai/Trinity-Large-Base` as its base
(17T tokens, annealed; ordinary embedding rows cosine 0.9995 with Thinking). TrueBase is a 10T pre-anneal branch (cosine
0.82). With TrueBase, finding 1's base-to-final change also includes 7T more pretraining tokens and the anneals. The
options:
- Base, the actual parent;
- TrueBase, as planned;
- both. A second base costs ~797 GB of download and 4 conditions, ~$40-60.
