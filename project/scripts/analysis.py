from pathlib import Path

from .utils import load_tasks

import json
import random
import re
import typer

random.seed(42)
app = typer.Typer()


@app.command()
def rs_agreement(ann1_path: Path, ann2_path: Path, output_folder: Path):
    if not output_folder.exists():
        output_folder.mkdir(parents=True)

    ann1_tasks = load_tasks(ann1_path)
    ann2_tasks = load_tasks(ann2_path)

    mid2ann1_tasks = {t["mention_id"]: t for t in ann1_tasks}
    mid2ann2_tasks = {t["mention_id"]: t for t in ann2_tasks}

    if len(set(mid2ann1_tasks.keys()).difference(set(mid2ann2_tasks.keys()))):
        raise Exception("not matching keys")

    mention_ids = sorted(mid2ann1_tasks.keys())
    disagree_tasks = []
    agree_tasks = []

    for mid in mention_ids:
        ann1_task = mid2ann1_tasks[mid]
        ann2_task = mid2ann2_tasks[mid]

        rs_ann1 = ann1_task["roleset_id"]
        rs_ann2 = ann2_task["roleset_id"]

        if rs_ann1 != rs_ann2:
            ann1_task["user_input"] = f"Ann1 chose {rs_ann1}\nAnn2 chose {rs_ann2}\n"
            disagree_tasks.append(ann1_task)
        else:
            agree_tasks.append(ann1_task)

    rs_agree_test = random.sample(agree_tasks, k=100)

    json.dump(
        disagree_tasks, open(str(output_folder) + "/rs_disagree.json", "w"), indent=1
    )
    json.dump(agree_tasks, open(str(output_folder) + "/rs_agree.json", "w"), indent=1)
    json.dump(
        rs_agree_test, open(str(output_folder) + "/rs_agree_test.json", "w"), indent=1
    )

    print("Total tasks:", len(mention_ids))
    print("RS agreement:", len(agree_tasks))
    print("RS disagreement:", len(disagree_tasks))


@app.command()
def rs_agreement_score(ann1_path: Path, ann2_path: Path):
    ann1_tasks = load_tasks(ann1_path)
    ann2_tasks = load_tasks(ann2_path)

    mid2ann1_tasks = {t["mention_id"]: t for t in ann1_tasks}
    mid2ann2_tasks = {t["mention_id"]: t for t in ann2_tasks}

    if len(set(mid2ann1_tasks.keys()).difference(set(mid2ann2_tasks.keys()))):
        raise Exception("not matching keys")

    mention_ids = sorted(mid2ann1_tasks.keys())
    disagree_tasks = []
    agree_tasks = []

    for mid in mention_ids:
        ann1_task = mid2ann1_tasks[mid]
        ann2_task = mid2ann2_tasks[mid]

        rs_ann1 = ann1_task["roleset_id"]
        rs_ann2 = ann2_task["roleset_id"]

        if rs_ann1 != rs_ann2:
            ann1_task["user_input"] = f"Ann1 chose {rs_ann1}\nAnn2 chose {rs_ann2}\n"
            disagree_tasks.append(ann1_task)
        else:
            agree_tasks.append(ann1_task)

    print("Total tasks:", len(mention_ids))
    print("RS agreement:", len(agree_tasks), len(agree_tasks) / len(mention_ids))
    print("RS disagreement:", len(disagree_tasks) / len(mention_ids))


@app.command()
def nested_arguments_count(ann1_path: Path, ann2_path: Path):
    ann1_tasks = load_tasks(ann1_path)
    ann2_tasks = load_tasks(ann2_path)

    mid2ann1_tasks = {t["mention_id"]: t for t in ann1_tasks}
    mid2ann2_tasks = {t["mention_id"]: t for t in ann2_tasks}

    # if len(set(mid2ann1_tasks.keys()).difference(set(mid2ann2_tasks.keys()))):
    #     raise Exception("not matching keys")

    mention_ids = sorted(set(mid2ann1_tasks).intersection(set(mid2ann2_tasks.keys())))

    task_nested_arg = []

    loc_men_ids = []
    time_men_ids = []

    for mid in mention_ids:
        ann1_task = mid2ann1_tasks[mid]
        ann2_task = mid2ann2_tasks[mid]

        locs = ann1_task['argL'] + ann2_task["argL"]
        locs = locs.strip()

        tims = ann1_task['argT'] + ann2_task["argT"]
        tims = tims.strip()

        if len(re.findall(r"\.\d{2}", ann1_task["arg1"] + " " + ann2_task["arg1"])):
            task_nested_arg.append(mid)

        if locs and ("NA" not in locs):
            loc_men_ids.append(mid)
        else:
            print(locs)

        if not (tims == "NA" or tims == ""):
            time_men_ids.append(mid)

    doc_ids = set([task['doc_id'] for task in ann1_tasks + ann2_tasks])
    print("Docs:", len(doc_ids))
    print("Total:", len(mention_ids))
    print("Nested ARG-1:", len(task_nested_arg))
    print("/w Locs:", len(loc_men_ids))
    print("/w Times:", len(time_men_ids))


if __name__ == "__main__":
    app()
