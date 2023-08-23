from .utils import TASK_KEYS, WhitespaceTokenizer

from bs4 import BeautifulSoup
from collections import defaultdict, OrderedDict
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm
from typing import Optional

import glob
import json
import os
import pickle
import spacy
import typer
import zipfile


app = typer.Typer()

# ---------------------- Helpers -------------------------- #

VALIDATION = ["2", "5", "12", "18", "21", "23", "34", "35"]
TRAIN = [str(i) for i in range(1, 36) if str(i) not in VALIDATION]
TEST = [str(i) for i in range(36, 46)]


def add_lexical_features(mention_map, sent_doc_map):
    """
    Add lemma, derivational verb, etc
    Parameters
    ----------
    nlp: spacy.tokens.Language
    mention_map: dict

    Returns
    -------
    None
    """

    mentions = list(mention_map.values())

    mention_sentences = [mention["sentence"] for mention in mentions]
    # nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

    for mention in mentions:
        # mention = mentions[i]
        doc = sent_doc_map[mention["doc_id"], mention["sentence_id"]]
        # get mention span
        mention_span = doc[mention["token_start"] : mention["token_end"] + 1]

        if len(mention["mention_text"].split()) > 1:
            root_span = max(mention_span, key=lambda x: len(x))
        else:
            root_span = mention_span.root

        # add char spans of root
        # root_index = mention_span.root.i
        # root_span = doc[root_index:root_index+1]
        mention["start_char"] = mention_span.start_char
        mention["end_char"] = mention_span.end_char

        # get lemma
        mention["lemma"] = root_span.lemma_

        # lemma_start and end chars
        mention["pos"] = root_span.pos_

        # sentence tokens
        mention["sentence_tokens"] = [
            w.lemma_.lower()
            for w in doc
            if (not (w.is_stop or w.is_punct))
            or w.lemma_.lower() in {"he", "she", "his", "him", "her"}
        ]

        mention["has_pron"] = (
            len(
                set(mention["sentence_tokens"]).intersection(
                    {"he", "she", "his", "him", "her"}
                )
            )
            > 0
        )


def get_sent_map_simple(doc_bs):
    """

    Parameters
    ----------
    doc_bs: BeautifulSoup

    Returns
    -------
    dict
    """
    sent_map = OrderedDict()

    # get all tokens
    tokens = doc_bs.find_all("token")

    # all tokens map
    all_token_map = OrderedDict()

    for token in tokens:
        all_token_map[token["t_id"]] = dict(token.attrs)
        all_token_map[token["t_id"]]["text"] = token.text

        if token["sentence"] not in sent_map:
            sent_map[token["sentence"]] = {
                "sent_id": token["sentence"],
                "token_map": OrderedDict(),
            }
        sent_map[token["sentence"]]["token_map"][token["t_id"]] = all_token_map[
            token["t_id"]
        ]

    for val in sent_map.values():
        token_map = val["token_map"]
        val["sentence"] = " ".join([m["text"] for m in token_map.values()])

    return all_token_map, sent_map


def mention2task(mention, copy_keys=None):
    task = {}
    if copy_keys:
        for key in copy_keys:
            task[key] = mention[key]

    p_task = {
        'text': mention['sentence'],

        'spans': [{
            'token_start': mention['token_start'],
            'token_end': mention['token_end'],
            'start': mention['start_char'],
            'end': mention['end_char'],
            'label': mention['men_type'].upper(),
        }],

        'meta': {
            'Doc': mention['doc_id'],
            'Sentence': mention['sentence_id'],
        }
    }

    return {**task, **p_task}

# ------------------------------- Commands ----------------------------- #


@app.command()
def parse_annotations(annotation_folder: Path, output_folder: Path, spacy_model: str):
    """
    Read the annotations files from ECB+_LREC2014

    Parameters
    ----------
    annotation_folder
    output_folder
    spacy_model

    Returns
    -------

    """
    # get validated sentences as a map {topic: {doc_name: [sentences]}}
    valid_sentences_path = os.path.join(
        annotation_folder, "ECBplus_coreference_sentences.csv"
    )
    valid_topic_sentence_map = defaultdict(dict)
    with open(valid_sentences_path) as vf:
        rows = [line.strip().split(",") for line in vf.readlines()][1:]
        for topic, doc, sentence in rows:
            doc_name = topic + "_" + doc + ".xml"
            if doc_name not in valid_topic_sentence_map[topic]:
                valid_topic_sentence_map[topic][doc_name] = set()
            valid_topic_sentence_map[topic][doc_name].add(sentence)

    # unzip ECB+.zip
    with zipfile.ZipFile(os.path.join(annotation_folder, "ECB+.zip"), "r") as zip_f:
        zip_f.extractall(output_folder)

    # read annotations files at working_folder/ECB+
    ecb_plus_folder = os.path.join(output_folder, "ECB+/")
    doc_sent_map = {}
    mention_map = {}
    singleton_idx = 10000000000

    for ann_file in tqdm(
        list(glob.glob(ecb_plus_folder + "/*/*.xml")), desc="Reading ECB Corpus"
    ):
        ann_bs = BeautifulSoup(open(ann_file, "r", encoding="utf-8").read(), features="lxml")
        doc_name = ann_bs.find("document")["doc_name"]
        topic = doc_name.split("_")[0]
        # add document in doc_sent_map
        curr_tok_map, doc_sent_map[doc_name] = get_sent_map_simple(ann_bs)
        # get events and entities
        entities, events, instances = {}, {}, {}
        markables = [a for a in ann_bs.find("markables").children if a.name is not None]
        for mark in markables:
            if mark.find("token_anchor") is None:
                instances[mark["m_id"]] = mark.attrs
            elif "action" in mark.name or "neg" in mark.name:
                events[mark["m_id"]] = mark
            else:
                entities[mark["m_id"]] = mark

        # relations
        relation_map = {}
        relations = [a for a in ann_bs.find("relations").children if a.name is not None]
        for relation in relations:
            target_m_id = relation.find("target")["m_id"]
            source_m_ids = [s["m_id"] for s in relation.find_all("source")]
            for source in source_m_ids:
                relation_map[source] = target_m_id

        # create mention_map
        for m_id, mark in {**entities, **events}.items():
            if m_id in entities:
                men_type = "ent"
            else:
                men_type = "evt"

            if topic in TRAIN:
                split = "train"
            elif topic in VALIDATION:
                split = "dev"
            else:
                split = "test"

            mention_tokens = [
                curr_tok_map[m["t_id"]] for m in mark.find_all("token_anchor")
            ]
            if "36_4ecbplus.xml" in doc_name and mention_tokens[-1]["t_id"] == "127":
                mention_tokens = mention_tokens[-1:]

            sent_id = mention_tokens[0]["sentence"]
            if (
                doc_name not in valid_topic_sentence_map[topic]
                or sent_id not in valid_topic_sentence_map[topic][doc_name]
            ):
                continue
            mention = {
                "m_id": m_id,
                "sentence_id": sent_id,
                "topic": topic,
                "men_type": men_type,
                "split": split,
                "mention_text": " ".join([m["text"] for m in mention_tokens]),
                "sentence": doc_sent_map[doc_name][sent_id]["sentence"],
                "doc_id": doc_name,
                "type": mark.name,
            }

            # add marked_sentence
            sent_token_map = deepcopy(doc_sent_map[doc_name][sent_id]["token_map"])
            first_token_id = mention_tokens[0]["t_id"]
            final_token_id = mention_tokens[-1]["t_id"]
            mention["token_start"] = int(sent_token_map[first_token_id]["number"])
            mention["token_end"] = int(sent_token_map[final_token_id]["number"])
            sent_token_map[first_token_id]["text"] = (
                '<mark id="mark_id"> ' + sent_token_map[first_token_id]["text"]
            )
            if final_token_id not in sent_token_map:
                print(doc_name)
            sent_token_map[final_token_id]["text"] = (
                sent_token_map[final_token_id]["text"] + " </mark>"
            )
            marked_sentence = " ".join([s["text"] for s in sent_token_map.values()])
            mention["marked_sentence"] = marked_sentence

            # add marked_doc
            doc_sent_map_copy = deepcopy(doc_sent_map[doc_name])
            doc_sent_map_copy[sent_id]["sentence"] = marked_sentence
            marked_doc = "\n".join([s["sentence"] for s in doc_sent_map_copy.values()])
            mention["marked_doc"] = marked_doc

            # coref_id
            if m_id in relation_map:
                instance = instances[relation_map[m_id]]
                # Intra doc coref case
                if "instance_id" not in instance:
                    instance["instance_id"] = instance["m_id"]
                cluster_id = instance["instance_id"]
                tag_descriptor = instance["tag_descriptor"]
            else:
                cluster_id = singleton_idx
                singleton_idx += 1
                tag_descriptor = "singleton"
            mention["gold_cluster"] = cluster_id
            mention["tag_descriptor"] = tag_descriptor

            # add into mention map
            mention_id = doc_name + "_" + m_id
            mention['mention_id'] = mention_id
            mention_map[mention_id] = mention

    nlp = spacy.load(spacy_model)
    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

    sent_ids = [
        (doc_id, sent_id)
        for doc_id, sent_map in doc_sent_map.items()
        for sent_id, sent_val in sent_map.items()
    ]
    sentences = [
        sent_val["sentence"]
        for doc_id, sent_map in doc_sent_map.items()
        for sent_id, sent_val in sent_map.items()
    ]

    sent_tuples = list(zip(sentences, sent_ids))

    sent_doc_map = {}
    for doc, sent_id in tqdm(
        nlp.pipe(sent_tuples, as_tuples=True),
        desc="spacifying docs",
        total=len(sent_tuples),
    ):
        sent_doc_map[sent_id] = doc

    for doc_id, sent_map in doc_sent_map.items():
        for sent_id, sent_val in sent_map.items():
            sent_val["sentence_tokens"] = [
                w.lemma_.lower()
                for w in sent_doc_map[doc_id, sent_id]
                if (not (w.is_stop or w.is_punct))
                or w.lemma_.lower() in {"he", "she", "his", "him", "her"}
            ]

    # save doc_sent_map
    pickle.dump(doc_sent_map, open(output_folder / f"doc_sent_map.pkl", "wb"))

    # lexical features
    add_lexical_features(mention_map, sent_doc_map)

    # save pickle
    pickle.dump(mention_map, open(output_folder / f"mention_map.pkl", "wb"))

    return mention_map


@app.command()
def create_evt_tasks(
        mention_map_path: Path,
        output_path: Path,
        split: str
):
    if not output_path.parent.exists():
        output_path.parent.mkdir()
    mention_map = pickle.load(open(mention_map_path, 'rb'))
    for m_id, mention in mention_map.items():
        mention['mention_id'] = m_id
    evt_mentions = sorted([v for v in mention_map.values() if v['split'] == split and v['men_type'] == 'evt'],
                          key=lambda x: (x['doc_id'], int(x['sentence_id'])))
    evt_tasks = [mention2task(men, TASK_KEYS) for men in evt_mentions]

    json.dump(evt_tasks, open(output_path, 'w'), indent=1)

    print(f'Saved {len(evt_mentions)} event mentions!')


if __name__ == "__main__":
    app()
