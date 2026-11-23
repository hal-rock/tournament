# Plan: replication round 2 (two newer models without bases, and three new experiments)

Written 2026-11-23, before any data. Built from the "new bits" section of `replication_notes.txt`, the first round
(`PLAN_replication.md`, README "Replication across larger models" and "The coin as a tool call") and the small-model versions
of the new experiments (README "Shallow instructions, reworded" and "The coinflip as raw text"). Hal's answers to the open
questions are at the end; the design and pre-registration are in the README, "Replication round 2".

## What runs on which models

**Models:**
- **The first round's six,** each with a base: Kimi K2, Llama 3.1 405B, DeepSeek V3, DeepSeek V4 Flash, Trinity Large
  Thinking and Nemotron-4 340B.
- **Two new finals, neither with a released base:**
  - Kimi K3.1 Pro (`moonshotai/Kimi-K3.1-Pro`);
  - GLM-5.3 (`zai-org/GLM-5.3`, the 743B model, not GLM-5.3-Flash).

  The Hub has no GLM-5*-Base and no Kimi-K3*-Base (checked 2026-11-23).

**Experiments** (conditions are `prompts.CONDITIONS` names; p is each of the four personas):

| | experiment | conditions | old six | new two |
|---|---|---|---|---|
| R | the first round's findings 1-7 | the 44 final conditions of `replication.final_conditions()` | done | **run** |
| R8 | finding 8, the " up" patching (HF) | wording check, then the short preference, the guess and Task A/B | done (not Trinity) | **run** (see Q4 for Kimi) |
| T | the coin as a tool call | `coinflip_tool`, `coinflip_tools_declared`, `assistant_tools_declared`, `coinflip_tool@p` / `@p+user` / `@p+self` (15) | done (not Nemotron) | **run** |
| T-rp | **new:** the tool's coin under roleplay | `coinflip_tool@p+roleplay` (4) | **run** (not Nemotron) | **run** |
| L | **new:** length choice rules | `{assistant,coinflip}_ifchoose_{shorter,longer}`, plus the new `coinflip_tool_ifchoose_{shorter,longer}` (6) | **run** (Nemotron: 4, no tool) | **run** |
| S | **new:** raw-text stories | `raw_story_ai`, `raw_story_person`, with their baselines `raw_no_okay` and `raw_okay` (4) | **run** (bases: Q2) | **run** |
| G | gate re-scores on the new node | `assistant`, `coinflip`, `coinflip_tool` (3) | **run** | (part of R and T) |

**Notes on reading the notes:**
- **"System-prompt and user-instructed persona preferences with the tool-call report" already ran on five of the old six,** in
  the tool run (`coinflip_tool@p` and `@p+user`, results e81c6e9). So for them only the roleplay placement is new. Nemotron-4 has
  no native tool format, so it gets no tool conditions.
- **The extra experiments run on the first set after `replication_notes.txt`** were the coin as a tool call, and Trinity's
  check that reading inside the reasoning agrees with reading after an empty think block. Both run on the new two here. The
  guess's outcome lean and guess utilities were analyses of data on disk, and they'll be re-run on the new data.
- **"Choice instruction based on task length"** is the reworded rule ("If I ask you to choose between tasks in this message,
  choose the one with the shorter description."), `shorter` and `longer` only. With the tool, the user turn opens with that
  rule and ends "Okay, please flip the coin now.", as in `coinflip_tool`.
- **"3rd person plaintext framings"** are `raw_story_ai` ("A user asked an AI assistant ...") and `raw_story_person` ("A
  client asked a freelancer ..."), word for word as in the raw-text section. `raw_no_okay` and `raw_okay` come along because
  that section's contrasts compare each story with them, and they cost only minutes. Raw text needs no chat template, so the
  old six's bases could run it too (Q2).

**Conditions per model:**
- each new model: 44 + 15 + 4 + 6 + 4 = 73, plus the wording check and the reasoning check on 24 pilot pairs;
- each old model with a tool format: 4 + 6 + 4 + 3 = 17;
- Nemotron-4: 4 + 4 + 2 = 10.

Every condition uses the sweep's 300 tasks and 15,000 pairs in both orders, as before.

## The two new models

| | Kimi K3.1 Pro | GLM-5.3 |
|---|---|---|
| released | 2026-11-02 | 2026-08-25 (same base as GLM-5.2) |
| parameters | 2.8T total, 104B active | 743B total, 39B active |
| on disk | 1,561 GB: MXFP4 routed experts, the rest BF16 | 756 GB: FP8 e4m3, 128x128 blocks |
| architecture | 93 layers: Kimi Delta Attention (linear) with 24 MLA layers using DeepSeek sparse attention; 896 experts, 16 active; a vision tower | 78 layers: MLA with a DSA indexer; 256 experts, 8 active; one MTP layer |
| thinking | always on (K3's card: "always has thinking enabled"; effort low / high / max) | always on: the generation prompt ends `<\|assistant\|><think>` |
| prompt format | Moonshot's own encoder (`encoding_k3.py`), XML-like: `<\|open\|>message role="user"<\|sep\|>...` | `[gMASK]<sop>`, then `<\|system\|>Reasoning Effort: Max` in every prompt, then the turns |
| tools | `tool-declare` system message; calls as `<\|open\|>call tool="flip_coin" ...`; results as `role="tool"` messages | GLM's `<tool_call>` format; results in `<tool_response>` |
| vLLM | needs 0.35 nightly (vLLM recipe; 0.34.0, 2026-11-16, is the latest release) | supported (recipe: 0.29+) |
| HF transformers | no `kimi_k3` / `kimi_k31` module, even in 5.24. Kimi K3's repo ships remote modeling code, K3.1 Pro's doesn't | native `glm_moe_dsa` (5.15+; we have 5.17) |

**Onboarding**, as for the first six:
- template rendering with golden tests;
- single-token outcomes;
- default system blocks;
- the tool format in `toolframe.py`;
- the raw-text document start;
- a pinned sanity prompt at load;
- a 24-pair pilot of every condition.

Plus, for these two:
- **Both always think,** like Trinity Large Thinking.
  - Assistant-turn readouts go after an empty think block.
  - On the pilot pairs, they're checked against reading inside the reasoning (prefill "<think>I'll choose Task"). Trinity gave
    r 0.99.
  - User-turn readouts (the coin, the user's choice) have no think block, as before.
  - Kimi's encoder still has a no-thinking path (`thinking=False` opens the response channel directly), but the cards don't
    offer it. I'd use the empty think block, as for Trinity, unless the pilot shows the no-thinking path is the model's own:
    sensible greedy replies, and a check on tiny prompts.
- **GLM's effort line.** The template writes "Reasoning Effort: Max" into every prompt unless low or high is passed. I'd pick the
  level by a rule fixed before any data: the one where the model itself is likeliest to close the think block at once, P(`</think>`
  right after `<think>`) over the pilot prompts. That makes the empty block least off-policy.
- **Sparse attention is dense here.** Both DSA indexers keep the top 2,048 tokens, and our prompts are under 600, so every token
  is kept.
- **Kimi's linear-attention layers lose prefix caching.** vLLM caches a hybrid model's prefix only at ~1,280-token block
  boundaries, longer than our prompts. So persona conditions cost full price there. That's in the estimates below.
- **Local tests without weights:** tiny random Kimi K3 checkpoints are on the Hub (`nikita-savelyev-cerebras/tiny-random-kimi-k3-mxfp4`,
  `eatang/Kimi-K3-4layer`).
- **Kimi K3.1 Pro's vision tower** is skipped at load (text only).

## Hardware: 8x B200 isn't enough for Kimi K3.1 Pro

- **8x B200 holds ~1,432 GiB** (179 GiB usable per GPU). **Kimi K3.1 Pro's weights alone are 1,454 GiB.**
  - That's before the KV cache, activations or CUDA context, so the weights don't fit at all.
  - vLLM's recipe lists its smallest NVIDIA layout as "TP8 on two GB300 trays or one B300 node".
- **8x B300 (288 GB per GPU, ~2.3 TB) holds it** with ~700 GB to spare.
- GLM-5.3 (704 GiB) and all six old models fit on either node.

**Recommendation: one 8x B300 node for everything.**
- It's needed for Kimi anyway.
- One node means one setup and one runner.
- Each model loads once for all its conditions, across every experiment.

If a B300 costs much more per hour than a B200 ($90/hr last time), the alternative is to split the work over two nodes running at
the same time:
- a B200 node for the old six and GLM-5.3, ~15 h;
- a B300 node for Kimi K3.1 Pro, ~8 h.

That's cheaper and quicker in wall-clock time, but needs two setups.

**What to filter on when renting,** as last time:
- SXM;
- driver ≥ 580;
- on-demand, on a verified datacenter host;
- ≥ 4 TB of NVMe (Kimi K3.1 Pro's 1.56 TB plus the next model downloading);
- ≥ 2 TB of RAM if possible;
- fast downloads (the Illinois B200 node pulled ~7 GB/s).

**GPU type and engine for the old six.**
- Their existing conditions were scored on B200 with vLLM 0.33. The new conditions are compared with them: T-rp against
  `coinflip_tool@p` and `assistant@p+roleplay`, L against A_0, C_0 and T.
- **Gate:** re-score A_0, C_0 and T on the new node and compare with the stored runs. The bar is the standing one: r ≥ 0.999 and
  SD within 2%.
  - Where a model passes, the new conditions are compared with the stored ones.
  - Where it fails, every comparison uses conditions re-scored on the new node. That's ~1 minute per condition for most models.
- **Two venvs on the node:** vLLM 0.33 for the old six, so their engine is unchanged, and the 0.35 nightly for Kimi K3.1 Pro (and
  GLM-5.3 if 0.33 can't run it). The runner gets a per-job Python.

## Compute and cost

**Measured last time** (8x B200, vLLM 0.33, prefix caching on), minutes per 30,000-prompt condition, and load times:

| model | min / condition | final's load |
|---|---|---|
| Llama 3.1 405B | 2.9 | 6.5 min |
| Nemotron-4 340B | 2.5 | ~1-9 min |
| DeepSeek V4 Flash | ~1.0 (1.5 on 4 GPUs) | 20 min |
| Kimi K2 | 0.85 | ~60 min |
| DeepSeek V3 | 0.75 | ~35 min |
| Trinity Large Thinking | 0.3 | 22 min |
| GLM-5.3 (estimate) | ~0.9: 39B active, like DeepSeek V3's 37B | ~40 min (FP8 like DeepSeek V3; vLLM processed that at ~350 MB/s) |
| Kimi K3.1 Pro (estimate) | 2.5-5: 3x K2's active parameters, no prefix caching | 1-1.5 h |

B300 and B200 run BF16 and FP8 at about the same speed, and B300 runs FP4 faster, so I've used the B200 numbers.

**Node-hours:**

| block | estimate |
|---|---|
| Old six finals: loads ~2.5 h, 10-17 conditions each ~2.1 h, on-policy and pilots ~0.5 h | ~5 h |
| GLM-5.3: load, 73 conditions, checks | ~2 h |
| Kimi K3.1 Pro: load, 73 conditions, checks | ~5-8 h |
| Finding 8 on GLM-5.3, in HF (DeepSeek V3 took 48 min) | ~1 h |
| Setup, the nightly engine, debugging the new models | ~3 h |
| **Core, with a 25% buffer** | **~20-24 h** |
| Optional, Q2: the old six's bases for the stories (loads ~2.7 h) | +3.3 h |
| Optional, Q3: tool-declared persona controls | +2.5 h |
| Optional, Q4: finding 8 on Kimi K3.1 Pro | +3 h, plus engineering |

**Cost:**
- Core: ~22 h at the B300's hourly price. That's ~$2.0k at last time's $90/hr, or ~$2.9k at $130/hr.
- Everything optional on top: ~31 h.

**Budget.** This repo's ledger shows 17.97 h plus 3.86 h of vast at $90/hr = ~$1,965 spent, so at most ~$1,535 is left of the
$3.5k. It's less if other projects have used the vast balance since (forum and its-going-great rented vast nodes). The core plan
doesn't fit that at any likely B300 price (Q1).

**Where to cut if needed:**
- **Kimi K3.1 Pro** is about a third of the cost.
- **5,000 pairs instead of 15,000** for the new experiments' conditions would cut scoring by ~2/3. That helps most for Kimi K3.1
  Pro, Llama and Nemotron, but doesn't shrink loads. Spearman-Brown puts split-half reliability at ~0.92-0.96.
- **Trinity's user-reported coin barely moves** (SD 0.20), so L and S say little there. Its tool coin does track its Assistant
  (r 0.81).

## Pre-registration

Each experiment gets a README section with design, calls and predictions, committed before any of its data.

1. **"Replication on two newer models without released bases"** (findings 1-8 on GLM-5.3 and Kimi K3.1 Pro).
   - Findings 4-8 keep their calls.
   - Findings 1-3 lose their base comparisons. The final-only call: r(A_f, X_f) ≥ 0.5 with its CI above 0, for X = C, U, P.
     - The old six's base values are listed as context: r(A_b, C_b) from -0.07 to 0.76.
     - The write-up says plainly that without a base we can't tell post-training apart from what pretraining already had.
   - Their columns in `figures/replication_matrix.png` are marked "no base".
   - The tool run's calls 1-3 are unchanged.
2. **"The tool's coin under roleplay"** (old five and new two).
   - T_rp is fit on z(A_0) and z(A_rp), with the roleplay section's call ("persona" or "mix" if b_per's CI is above 0, with A_rp
     separable).
   - Paired against the user-reported coin's roleplay fit (Δw_rp = w_rp(T) − w_rp(C)).
   - Against the tool's system-prompt fit.
3. **"Length rules on larger models, with the user-reported and the tool's coin"** (all eight finals). The ifchoose run's
   primary measures:
   - the Assistant's following (agreement with the rule);
   - each coin's d_pr (partial r with the rule given A_0, minus the default coin's);
   - each coin's mirrored length contrast.

   Plus the paired tool-minus-user-reported difference in each.
4. **"Raw-text stories on larger models"** (all eight finals; bases per Q2). Without bases: r(A_f, X_f), the slope ratio against
   the chat coinflip, and each story against `raw_no_okay`. With bases: the raw-text section's call (tracks, absent or reversed,
   from retention and the r shift).

**Reported for every finding, as the first round's notes ask:**
- r over all 300 tasks;
- within each source;
- over the benign sources together (MATH, Alpaca, WildChat);
- magnitudes beside them, not called.

## Phases and check-ins

0. **Local, no spend:**
   - Hal's answers;
   - onboarding the new two, with tests on the tiny random checkpoints;
   - the `coinflip_tool_ifchoose_*` conditions;
   - raw-text document starts for the six families;
   - analysis extensions (`replication_analysis.py`, `tool_analysis.py`, `ifasked.py`, `raw_text.py` reading replication outputs);
   - a weightless HF load check of GLM-5.3 for finding 8;
   - the job list and the second venv in `scripts/node_setup.sh`;
   - the pre-registration commit.
1. **Rent the node.** Run setup, the sanity prompts, the re-score gate, the thinking checks and the reasoning check.
2. **Score, one load per model,** in this order:
   - DeepSeek V4 Flash: quick, and it exercises the gate;
   - GLM-5.3. **Check-in** here: the first new model's results, measured speeds and the cost projection;
   - Kimi K3.1 Pro: the largest and riskiest, early enough to drop if it fails;
   - Llama 405B, DeepSeek V3, Kimi K2, Trinity, Nemotron;
   - then the bases, if Q2 says so.

   Each model's results are fetched, analysed and committed as it finishes.
3. **Finding 8 in HF.** It runs only on a new model whose coin is sizable (not Trinity-like, SD ≥ 0.5 nats). **Check-in** before
   it starts.
4. **Write-up:** README results per section, `figures/<key>/` per model, and the matrix extended by two columns.

## Open questions (default in brackets)

1. **Budget for this round,** given ~$1,535 left of the vast budget and a core estimate of ~22 node-hours. [Run the core plan if
   you top up; otherwise cut Kimi K3.1 Pro to 5,000 pairs per condition and drop Trinity from L and S.]
2. **Bases for the raw-text stories.** Bases give the raw-text section's call (retention, r shift), but cost ~3.3 node-hours,
   mostly loads (Kimi K2 base ~1 h). [Finals only. The base question is about pretraining, and the new two have no base anyway.]
3. **Tool-declared persona controls.** These are `coinflip_tools_declared@p` in all four placements, 16 per tool model. They fix
   the tool run's main caveat: its persona comparisons crossed "who reports" with "is a tool declared". [Skip, unless the roleplay
   comparison is the one you most care about.]
4. **Finding 8 on Kimi K3.1 Pro.** HF has no K3.1 implementation, and dequantizing its MXFP4 experts to BF16 would need ~5.6 TB. The
   options:
   - try K3's remote code with compressed-tensors keeping MXFP4, on the tiny random checkpoint first;
   - build the patching inside vLLM (forward hooks on its workers), validated against HF on GLM-5.3. That's about a day of my work;
   - skip it for Kimi.

   [Try the first locally, and skip if it doesn't load. GLM-5.3 gets finding 8 either way.]
5. **Kimi K3.1 Pro vs K3, GLM-5.3 vs GLM-5.3-Flash.** [K3.1 Pro and the full GLM-5.3, as written in the notes.]

## Decisions (Hal, 2026-11-23)

1. **Budget:** the vast balance is topped up to $3.2k, all of it available for this round. The core plan runs as written.
2. **No bases for the raw-text stories.** Finals only; Hal may ask for bases while the node runs.
3. **No tool-declared persona controls.**
4. **No " up" patching on Kimi K3.1 Pro.** GLM-5.3 gets finding 8.
5. **Versions:** Kimi K3.1 Pro and the full GLM-5.3 (743B), as in the notes.
6. **The reading of the notes** (above) stands.
7. **One 8x B300 node,** rented by Hal when the code is ready.

## Status (2026-11-23)

Ready for the node:
- `prompts.py`: the tool coin's length-rule conditions; always-thinking templates' think-closing text, kept effort block and
  template arguments. `toolframe.py`: the `glm5` and `kimi3` tool families.
- `replication.py`: round 2's condition sets. `score_replication.py`: per-condition variants (the reasoning check), raw-text
  document starts, the effort rule. `replication_runner.py`: a per-job interpreter. `scripts/node_setup.sh`: the vLLM nightly
  venv for Kimi K3.1 Pro.
- Configs: the two new models, the six round-2 configs, GLM-5.3's patching config; `replication/round2_jobs.yaml`,
  `replication/round2_patch_jobs.yaml`.
- Tests: `tests/test_replication_new_models.py`, `tests/test_replication_round2.py`, additions to `tests/test_toolframe.py`
  and `tests/test_score_replication.py`. GLM-5.3's weightless HF load check passes (`replication/patch_load_check.md`).
- The README pre-registration.

Written while the node runs (before each experiment's results are read): the round-2 analysis (gate comparison, final-only
findings 1-3, T-rp, the length rules with the tool's coin, the stories without bases).

Launch, once Hal has rented the node:
1. `pod.py add replication2-node --id <vast id> --gpu "8x B300" --usd-hr <price> --ssh "<ssh command>"`, then `pod.py push`.
2. `pod.py run replication2-node --max-hours 30 --env STOP_ON_FAIL=1 -- 'bash scripts/node_setup.sh'
   'uv run python replication_runner.py replication/round2_jobs.yaml'`.
3. After DeepSeek V4 Flash: the gate. After GLM-5.3: its pilot masses, chosen effort, reasoning check, measured speed and a
   cost projection (check-in).
