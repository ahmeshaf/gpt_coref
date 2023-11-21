import json
from collections import defaultdict
from prodigy.components.loaders import JSONL, JSON
from prodigy.util import set_hashes
from scripts.parse_ecb import WhitespaceTokenizer
from recipes.constants import *
from scipy.spatial.distance import cosine
from typing import Optional
from tqdm import tqdm

import copy
import numpy as np
import pickle
import prodigy
import spacy
import sys
import typer


app = typer.Typer()
# -------------------------- HELPERS --------------------------- #


def sort_tasks(stream):
    return sorted(
        [t for t in stream],
        key=lambda x: (x["doc_id"], int(x["sentence_id"]), int(x["spans"][0]["start"])),
    )


def get_field_suggestions(dataset, propbank_dict):
    from prodigy.components.db import connect

    db = connect()
    dataset_arr = db.get_dataset(dataset)
    field_suggestions = {key: set() for key in [ROLESET_ID] + ALL_ARGS}
    field_suggestions["roleset_id"] = set(propbank_dict.keys())
    if dataset_arr is not None:
        for eg_ in dataset_arr:
            if eg_["answer"] == "accept":
                for arg in ALL_ARGS:
                    if arg in eg_:
                        arg_val = eg_[arg]
                        if (
                            arg_val.replace("NA", "").strip() != ""
                            and arg_val not in field_suggestions[arg]
                        ):
                            field_suggestions[arg].add(arg_val)
    return field_suggestions


# --------------------------  RECIPE --------------------------- #
@prodigy.recipe(
    "srl-update",
    dataset=("The dataset to use", "positional", None, str),
    spacy_model=("The base model", "positional", None, str),
    source=("The source data as a JSON or JSONL file", "positional", None, str),
    pb_dict_path=("Path to the pickle file of PropBank", "positional", None, str),
    update=("Whether to update the model during annotation", "flag", "UP", bool),
    port=("Port of the app", "option", "port", int),
    pb_link=("Url to the PropBank website", "option", "pl", str),
)
def srl_update(
    dataset: str,
    spacy_model: str,
    source: str,
    pb_dict_path: str,
    update: bool = False,
    port: Optional[int] = 8080,
    pb_link: Optional[str] = "http://127.0.0.1:8700/",
):
    nlp = spacy.load(spacy_model)
    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

    labels = ["EVT"]
    pb_dict = pickle.load(open(pb_dict_path, "rb"))

    # load the data
    if source.lower().endswith("jsonl"):
        stream = JSONL(source)
    elif source.lower().endswith("json"):
        stream = JSON(source)
    else:
        raise TypeError("Need jsonl file type for source.")

    stream = list(stream)

    total_tasks = len(stream)
    # print(stream[0]['answer'])
    batch_size = 10

    alias2roleset = defaultdict(set)

    for roleset, roledict in pb_dict.items():
        aliases = roledict["aliases"]
        for alias in aliases:
            alias2roleset[alias].add(roleset)
    trs_arg2val_vec = {}

    field_suggestions_task = field_suggestions = {
        key: set() for key in [ROLESET_ID] + ALL_ARGS
    }

    for t in stream:
        for arg in ALL_ARGS:
            if arg in t and t[arg]:
                if arg not in field_suggestions_task:
                    field_suggestions_task[arg] = set()
                field_suggestions_task[arg].add(t[arg])

    field_suggestions_ds = get_field_suggestions(dataset, pb_dict)

    field_suggestions = {
        arg: field_suggestions_ds[arg].union(field_suggestions_task[arg])
        for arg in [ROLESET_ID] + ALL_ARGS
    }

    topic_rs_arg2vals = {}

    def make_srl_tasks(stream_):
        texts = [(eg_["text"], eg_) for eg_ in stream_]
        for doc, task in nlp.pipe(texts, batch_size=batch_size, as_tuples=True):
            new_task = copy.deepcopy(task)
            topic = task["topic"]
            new_task["field_suggestions"] = field_suggestions
            best_roleset_id = new_task[ROLESET_ID]
            if best_roleset_id in pb_dict:
                curr_predicate = pb_dict[best_roleset_id]["frame"]
                prop_holder = (
                    pb_link + "/" + curr_predicate + ".html#" + best_roleset_id
                )
            else:
                prop_holder = pb_link

            new_task["prop_holder"] = prop_holder

            for arg_name in ALL_ARGS:
                trs_arg = (topic, best_roleset_id, arg_name)

                if trs_arg in topic_rs_arg2vals:
                    arg_vals = list(topic_rs_arg2vals[trs_arg])
                    possible_arg_vecs = [
                        trs_arg2val_vec[trs_arg + (val,)] for val in arg_vals
                    ]
                    cos_sims_args = [
                        1 - cosine(vec, doc.vector) for vec in possible_arg_vecs
                    ]
                    best_arg_val = arg_vals[np.argmax(cos_sims_args)]
                    if arg_name in new_task:
                        new_task[arg_name + "_pred"] = best_arg_val
                    else:
                        new_task[arg_name] = best_arg_val
                elif not arg_name in new_task:
                    new_task[arg_name] = ""
                elif arg_name in new_task:
                    new_task[arg_name] = new_task[arg_name]
                    new_task[arg_name + "_pred"] = ""

            if "answer" in new_task:
                new_task.pop("answer")
            new_task = set_hashes(
                new_task,
                input_keys=("text",),
                task_keys=("mention_id",),
                overwrite=True,
            )
            yield new_task

    stream = make_srl_tasks(stream)

    def make_updates(answers):
        for answer in answers:
            if answer["answer"] == "accept":
                best_roleset_id = answer[ROLESET_ID]
                topic = answer["topic"]
            for arg_name in ALL_ARGS:
                arg_val = answer[arg_name]
                trs_arg = (topic, best_roleset_id, arg_name)
                trs_arg_val = trs_arg + (arg_val,)
                doc_vector = nlp(answer["text"]).vector
                if trs_arg_val in trs_arg2val_vec:
                    trs_arg2val_vec[trs_arg_val] = (
                        trs_arg2val_vec[trs_arg_val] + doc_vector
                    )
                else:
                    trs_arg2val_vec[trs_arg_val] = doc_vector

                if arg_val:
                    field_suggestions[arg_name].add(arg_val)
                    if trs_arg not in topic_rs_arg2vals:
                        topic_rs_arg2vals[trs_arg] = set()
                    topic_rs_arg2vals[trs_arg].add(arg_val)
            answer.pop("field_suggestions")

    blocks = [
        {"view_id": "html", "html_template": PB_HTML},
        {"view_id": "html", "html_template": DOC_HTML2},
        {"view_id": "ner"},
        {"view_id": "html", "html_template": HTML_INPUT, "text": None},
        {
            "view_id": "text_input",
            "field_rows": 1,
            "field_autofocus": False,
            "field_label": "Reason for Flagging",
        },
    ]

    config = {
        "lang": nlp.lang,
        "labels": labels,  # Selectable label options
        "span_labels": labels,  # Selectable label options
        "auto_count_stream": not update,  # Whether to recount the stream at initialization
        "show_stats": True,
        "host": "127.0.0.1",
        "port": port,
        "blocks": blocks,
        "batch_size": batch_size,
        "history_length": batch_size,
        "instant_submit": False,
        "javascript": JAVASCRIPT_WSD + DO_THIS_JS,
    }

    return {
        "view_id": "blocks",  # Annotation interface to use
        "dataset": dataset,  # Name of dataset to save annotations
        "stream": stream,  # Incoming stream of examples
        # "update": make_updates,
        "exclude": None,
        "config": config,
        # "total_tasks": total_tasks,
    }


@app.command()
def simulate_srl_arg(
    dataset: str, spacy_model: str, source: str, pb_dict_path: str, output_path: str
):
    ctrl = srl_update(dataset, spacy_model, source, pb_dict_path, update=True)
    my_tasks = []
    index = 0
    for task in tqdm(ctrl["stream"], desc="Simulating", total=ctrl["total_tasks"]):
        task["answer"] = "accept"
        ctrl["update"]([task])
        my_tasks.append(task)
        index += 1

    json.dump(my_tasks, open(output_path, "w"), indent=1)


if __name__ == "__main__":
    app()
