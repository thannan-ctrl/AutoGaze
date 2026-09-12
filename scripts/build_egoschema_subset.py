"""Merge EgoSchema's official questions.json + subset_answers.json into
data/egoschema/subset.json, the file dataset.py actually reads. See
QUICKSTART.md step 4.
"""
import json
import os

from breakdown import config

EGOSCHEMA_DIR = os.path.join(config.DATA_DIR, "egoschema")


def main():
    questions = {x["q_uid"]: x for x in json.load(open(os.path.join(EGOSCHEMA_DIR, "questions.json")))}
    answers = json.load(open(os.path.join(EGOSCHEMA_DIR, "subset_answers.json")))
    subset = [{**questions[q_uid], "answer": answer} for q_uid, answer in answers.items()]
    json.dump(subset, open(os.path.join(EGOSCHEMA_DIR, "subset.json"), "w"))
    print(f"{len(subset)} questions written")


if __name__ == "__main__":
    main()
