import json
import pickle
import sys


def JSON(file_path):
    return json.load(open(file_path))


def JSONL(file_path):
    data = []
    with open(file_path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


TASK_KEYS = [
    "mention_id",
    "topic",
    "doc_id",
    "sentence_id",
    "sentence",
    "marked_sentence",
    "marked_doc",
    "lemma",
    "gold_cluster",
]

TASK_KEYS_ANNOTATIONS = TASK_KEYS + [
    "text",
    "spans",
    "arg0",
    "arg1",
    "argL",
    "argT",
    "roleset_id",
    "lemma",
]


if __name__ == "__main__":
    in_file_path = sys.argv[1]
    mention_map_file = sys.argv[2]
    out_file = sys.argv[3]

    mention_map = pickle.load(open(mention_map_file, "rb"))

    if in_file_path.endswith("jsonl"):
        annos = JSONL(in_file_path)
    else:
        annos = JSON(in_file_path)

    print("tasks len:", len(annos))

    annos_fixed = []
    mention_ids = set()

    for task in annos:
        task["marked_sentence"] = mention_map[task["mention_id"]]["marked_sentence"]

        if "mention_id" not in task:
            mention_id = task["spans"][0]["mention_id"]
        else:
            mention_id = task["mention_id"]

        mention = mention_map[mention_id]

        marked_doc = mention_map[mention_id]["marked_doc"]
        task["marked_doc"] = marked_doc.replace("\n", " <p/>")

        if "bert_doc" in task:
            task.pop("bert_doc")

        new_task = {}
        for key in TASK_KEYS_ANNOTATIONS:
            if key in task:
                new_task[key] = task[key]
            else:
                new_task[key] = mention[key]
        if task["mention_id"] not in mention_ids:
            annos_fixed.append(new_task)
            mention_ids.add(mention_id)

    print(len(annos_fixed))

    json.dump(annos_fixed, open(out_file, "w"), indent=1)
