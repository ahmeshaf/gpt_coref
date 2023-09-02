from pathlib import Path

from .utils import load_tasks

import json
import random
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

    json.dump(disagree_tasks, open(str(output_folder) + "/rs_disagree.json", "w"), indent=1)
    json.dump(agree_tasks, open(str(output_folder) + "/rs_agree.json", "w"), indent=1)
    json.dump(rs_agree_test, open(str(output_folder) + "/rs_agree_test.json", "w"), indent=1)

    print('Total tasks:', len(mention_ids))
    print('RS agreement:', len(agree_tasks))
    print('RS disagreement:', len(disagree_tasks))


@app.command()
def rs_2(ann1_path: Path):
    pass


if __name__ == "__main__":
    app()
