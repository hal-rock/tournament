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
  layout. The Assistant's turn is prefilled with "I'll do Task" and we read
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
- Batched scoring is checked against unbatched on every run, before each condition.

## Running

```
uv sync
uv run pytest                                            # prompts + analysis on synthetic data
uv run python tournament.py configs/local.yaml           # CPU smoke test, OLMo 2 1B, 10 tasks
uv run python tournament.py configs/olmo3-32b.yaml       # 80GB GPU, see scripts/pod_setup.sh
uv run python analyze.py entity/tournament/<run_id>      # re-run the analysis on a finished run
```

Results go to W&B (project `tournament`) and to `outputs/<run_id>/`.

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
