"""EgoSchema sample loading, prompt construction, and answer parsing."""
import json
import os
import random
import re

from . import config

LETTERS = config.LETTERS


def load_samples() -> list:
    subset = json.load(open(os.path.join(config.DATA_DIR, "subset.json")))
    random.seed(config.SEED)
    return random.sample(subset, min(config.N_SAMPLES, len(subset)))


def build_prompt(item: dict) -> str:
    options = [item[f"option {i}"] for i in range(5)]
    choices = "\n".join(f"{LETTERS[i]}. {opt}" for i, opt in enumerate(options))
    return (
        f"Question: {item['question']}\n{choices}\n"
        "Answer with the letter of the correct option only."
    )


def parse_letter(response: str) -> int:
    m = re.search(r"[A-E]", response.strip().upper())
    return LETTERS.index(m.group()) if m else -1
