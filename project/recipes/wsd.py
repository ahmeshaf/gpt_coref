from recipes.constants import *
from scripts.utils import WhitespaceTokenizer, newline2para

from collections import defaultdict
from prodigy.components.loaders import JSONL, JSON
from prodigy.util import set_hashes
from scipy.spatial.distance import cosine
from typing import Optional

import copy
import numpy as np
import pickle
import prodigy
import spacy


# --------------------- HELPERS --------------------- #

def sort_tasks(tasks):
    return sorted(tasks, key=lambda x: (x['doc_id'], int(x['sentence_id']), x['spans'][0]['start']))

# --------------------- RECIPES --------------------- #

@prodigy.recipe(
    "wsd-update",
    dataset=("The dataset to use", "positional", None, str),
    spacy_model=("The base model", "positional", None, str),
    source=("The source data as a JSON or JSONL file", "positional", None, str),
    pb_dict_path=("The path to pb.dict", "positional", None, str),
    update=("Whether to update the model during annotation", "flag", "UP", bool),
    port=("Port of the app", "option", 'port', int),
    pb_link=("Url to the PropBank website", 'option', 'pl', str)
)
def wsd_update(
    dataset: str,
    spacy_model: str,
    source: str,
    pb_dict_path: str,
    update: bool = False,
    port: Optional[int] = 8080,
    pb_link: Optional[str] = 'http://0.0.0.0:8700/',
):
    nlp = spacy.load(spacy_model)
    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

    labels = ['EVT']
    pb_dict = pickle.load(open(pb_dict_path, 'rb'))

    # load the data
    if source.lower().endswith('jsonl'):
        stream = JSONL(source)
    elif source.lower().endswith('json'):
        stream = JSON(source)
    else:
        raise TypeError("Need json/jsonl file type for source.")

    stream = sort_tasks(stream)
    batch_size = 10

    topic_sense2vec = {}
    alias2roleset = defaultdict(set)

    for roleset, roledict in pb_dict.items():
        aliases = roledict['aliases']
        for alias in aliases:
            alias2roleset[alias].add(roleset)

    trs_arg2val_vec = {}
    field_suggestions = {key: set() for key in [ROLESET_ID] + ALL_ARGS}
    field_suggestions[ROLESET_ID] = set(pb_dict.keys())

    def make_wsd_tasks(stream_):
        texts = [(eg_["text"], eg_) for eg_ in stream_]
        for doc, task in nlp.pipe(texts, batch_size=batch_size, as_tuples=True):
            span = task['spans'][0]
            spacy_span = doc[span['token_start']: span['token_end'] + 1]
            span_root = spacy_span.root
            root_lemma = span_root.lemma_.lower()
            new_task = copy.deepcopy(task)
            new_task['roleset_id'] = ''

            if root_lemma in alias2roleset:
                possible_rs = alias2roleset[root_lemma]
            else:
                possible_rs = []

            possible_rs_sort = sorted(possible_rs, key=lambda x: x.split('.')[-1])

            topic = task['topic']
            new_task['prop_holder'] = pb_link
            new_task['field_suggestions'] = field_suggestions
            for arg_name in ALL_ARGS:
                new_task[arg_name] = ''
            if len(possible_rs_sort):
                topic_rs = [(topic, rs) for rs in possible_rs_sort]

                for trs in topic_rs:
                    if trs not in topic_sense2vec:
                        topic_sense2vec[trs] = np.ones(doc.vector.shape)

                cos_sims = [1 - cosine(topic_sense2vec[trs], doc.vector) for trs in topic_rs]
                # print(cos_sims)
                (_, best_roleset_id) = topic_rs[np.argmax(cos_sims)]

                curr_predicate = pb_dict[best_roleset_id]['frame']
                curr_roleset = pb_dict[best_roleset_id]['sense']
                prop_holder = pb_link + '/' + curr_predicate + '.html#' + curr_roleset
                new_task[ROLESET_ID] = curr_roleset
                new_task['prop_holder'] = prop_holder
                new_task['roleset_id'] = best_roleset_id
                new_task['predicted'] = {'roleset_id': best_roleset_id}

            new_task['marked_doc'] = newline2para(new_task['marked_doc'])
            new_task = set_hashes(new_task, input_keys=('text',), task_keys=('mention_id',), overwrite=True)
            yield new_task

    stream = make_wsd_tasks(stream)

    def make_updates(answers):
        for answer in answers:
            if answer['answer'] == 'accept':
                if answer[ROLESET_ID]:
                    trs = (answer['topic'], answer[ROLESET_ID])
                    curr_roleid = trs[1]
                    if trs not in topic_sense2vec:
                        topic_sense2vec[trs] = nlp(answer['text']).vector
                    else:
                        topic_sense2vec[trs] = topic_sense2vec[trs] + nlp(answer['text']).vector
                    if answer['lemma'] not in alias2roleset:
                        alias2roleset[answer['lemma']] = {curr_roleid}
                    elif curr_roleid not in alias2roleset[answer['lemma']]:
                        alias2roleset[answer['lemma']].add(curr_roleid)

                    if curr_roleid not in field_suggestions[ROLESET_ID]:
                        field_suggestions[ROLESET_ID].add(curr_roleid)

    def before_db(examples):
        for eg in examples:
            if "field_suggestions" in eg:
                eg.pop("field_suggestions")
        return examples

    blocks = [
        {"view_id": "html", "html_template": PB_HTML},
        {"view_id": "html", "html_template": DOC_HTML2},
        {'view_id': 'ner'},
        {"view_id": "html", "html_template": HTML_INPUT, 'text': None},
        {'view_id': 'text_input', "field_rows": 1, "field_autofocus": False,
         "field_label": "Reason for Flagging"}
    ]

    config = {
        "lang": nlp.lang,
        "labels": labels,  # Selectable label options
        "span_labels": labels,  # Selectable label options
        "auto_count_stream": not update,  # Whether to recount the stream at initialization
        "show_stats": True,
        "host": '0.0.0.0',
        "port": port,
        'blocks': blocks,
        'batch_size': batch_size,
        'history_length': batch_size,
        'instant_submit': False,
        "javascript": JAVASCRIPT_WSD + DO_THIS_JS_DISABLE
    }

    return {
        "view_id": "blocks",  # Annotation interface to use
        "dataset": dataset,  # Name of dataset to save annotations
        "stream": stream,  # Incoming stream of examples
        "update": make_updates,
        "before_db": before_db,
        "exclude": None,
        "config": config,
    }
