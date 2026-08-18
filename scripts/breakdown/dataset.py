"""Sample loading, prompt construction, and answer parsing for EgoSchema and
VideoMME. Both are normalized to the same shape so runner.py doesn't need to
know which dataset it's running:

    {"item_id": str, "video_path": str, "question": str,
     "options": list[str], "answer_idx": int}
"""
import json
import os
import random
import re

from . import config

LETTERS = config.LETTERS


def _load_egoschema() -> list:
    subset = json.load(open(os.path.join(config.DATA_DIR, "egoschema", "subset.json")))
    random.seed(config.SEED)
    n = len(subset) if not config.N_SAMPLES else min(config.N_SAMPLES, len(subset))
    sampled = random.sample(subset, n)
    items = []
    for it in sampled:
        items.append({
            "item_id": it["q_uid"],
            "video_path": os.path.join(config.DATA_DIR, "egoschema", "videos", f"{it['q_uid']}.mp4"),
            "question": it["question"],
            "options": [it[f"option {i}"] for i in range(5)],
            "answer_idx": it["answer"],
        })
    return items


def _load_videomme() -> list:
    """Only questions whose video was actually downloaded locally are usable
    -- data/video_mme/videos/ has 465 of the 900 referenced videos."""
    questions = json.load(open(os.path.join(config.DATA_DIR, "video_mme", "questions.json")))
    video_dir = os.path.join(config.DATA_DIR, "video_mme", "videos")
    available = {f.rsplit(".", 1)[0] for f in os.listdir(video_dir)}
    usable = [q for q in questions if q["video_id"] in available]
    random.seed(config.SEED)
    sampled = random.sample(usable, min(config.N_SAMPLES, len(usable))) if config.N_SAMPLES else usable
    items = []
    for it in sampled:
        items.append({
            "item_id": it["question_id"],
            "video_path": os.path.join(video_dir, f"{it['video_id']}.mp4"),
            "question": it["question"],
            "options": it["options"],
            "answer_idx": LETTERS.index(it["answer"]),
        })
    return items


_LOADERS = {"egoschema": _load_egoschema, "video_mme": _load_videomme}


def load_samples() -> list:
    return _LOADERS[config.DATASET]()


def build_prompt(item: dict) -> str:
    choices = "\n".join(f"{LETTERS[i]}. {opt}" for i, opt in enumerate(item["options"]))
    return (
        f"Question: {item['question']}\n{choices}\n"
        "Answer with the letter of the correct option only."
    )


def parse_letter(response: str, n_options: int) -> int:
    m = re.search(r"[A-%s]" % LETTERS[n_options - 1], response.strip().upper())
    return LETTERS.index(m.group()) if m else -1
