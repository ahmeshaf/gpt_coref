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
        marked_doc = mention_map[task["mention_id"]]["marked_doc"]
        task["marked_doc"] = marked_doc.replace("\n", " <p/>")

        if "bert_doc" in task:
            task.pop("bert_doc")

        if task["mention_id"] not in mention_ids:
            annos_fixed.append(task)
            mention_ids.add(task["mention_id"])

    print(len(annos_fixed))

    json.dump(annos_fixed, open(out_file, "w"), indent=1)
