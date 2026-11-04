"""Example prompts of the language conditions as each final model reads them, beside the English ones.

Two fixed task pairs (benign vs benign, harmful vs benign; no WildChat text, so the page can be committed),
each in English and every language the model runs (configs/languages/), Assistant and coinflip. The text is
build_prompt's output, the exact string the tournament scores; base models read the same string under their
own tokenizer (tests/test_prompts.py). The outcome tokens are shown at the cut.

Usage: uv run python scripts/language_examples.py   -> figures/languages_examples.html
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transformers import AutoTokenizer  # noqa: E402

from matskit import view  # noqa: E402
from prompts import CONDITIONS, build_prompt  # noqa: E402

FINALS = {  # family: (final model, turn_format, languages)
    "Llama 3.3 70B": ("unsloth/Llama-3.3-70B-Instruct", None, ("es",)),
    "Gemma 4 31B": ("google/gemma-4-31B-it", None, ("es",)),
    "GLM-4.5-Air": ("zai-org/GLM-4.5-Air", None, ("es", "zh")),
    "DeepSeek V4 Flash": ("deepseek-ai/DeepSeek-V4-Flash",
                          {"user": "<｜begin▁of▁sentence｜><｜User｜>", "reply": "<｜Assistant｜></think>"}, ("es", "zh")),
}
PAIRS = [("alpaca_4751", "competition_math_773"), ("bailbench_294", "alpaca_2030")]


def tasks(lang):
    path = "data/tasks_n300_max500_seed0.jsonl" if lang == "en" else f"data/tasks_n300_max500_seed0_{lang}.jsonl"
    return {t["id"]: t for t in map(json.loads, open(path))}


def main():
    by_lang = {lang: tasks(lang) for lang in ("en", "es", "zh")}
    pages = []
    for family, (name, turn_format, langs) in FINALS.items():
        tok = AutoTokenizer.from_pretrained(name)
        for first, second in PAIRS:
            for lang in ("en",) + langs:
                t = by_lang[lang]
                for kind in ("assistant", "coinflip"):
                    cond = kind if lang == "en" else f"{kind}_{lang}"
                    spec = CONDITIONS[cond]
                    text = build_prompt(cond, t[first]["prompt"], t[second]["prompt"], tok, turn_format=turn_format)
                    body = spec["body"].format(first=t[first]["prompt"].strip(), second=t[second]["prompt"].strip())
                    parts = [{"text": body, "role": "user", "turn": 0}]
                    if spec["prefill"]:
                        parts.append({"text": spec["prefill"], "role": "assistant", "turn": 1})
                    tr = view.from_exact(tok, text, parts, title=f"{family}, {lang}, {kind}: {first} vs {second}",
                                         group=f"{family} | {first} vs {second} | {kind}",
                                         meta={"model": name, "language": lang, "condition": cond,
                                               "outcomes": " / ".join(repr(o) for o in spec["outcomes"])})
                    pages.append(tr)
    view.write("figures/languages_examples.html", pages, title="Language conditions: example prompts",
               note="Each final model's prompts for two fixed task pairs (benign vs benign, harmful vs benign), in English "
                    "and each language it runs, as build_prompt renders them for scoring. The model's next token is read "
                    "at the end: the outcomes are in each transcript's meta. Base models read the same text.")


if __name__ == "__main__":
    main()
