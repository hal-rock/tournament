# Preference tournament

Measures a model's preferences over a broad set of tasks in two ways and compares
them:

- **Assistant**: the model, as the Assistant, chooses between two tasks.
- **Coinflip**: the model continues a *Human* turn that assigns one of two tasks
  by coin flip (the PSM coinflip setup, see `../coinflip`), and we read how it
  tilts the coin.

Each gives pairwise comparisons, from which we fit per-task utilities
(Bradley-Terry).

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
