import json

from spacy.tokens import Doc


TASK_KEYS = ['mention_id', 'topic', 'doc_id', 'sentence_id', 'sentence',
             'marked_sentence', 'marked_doc', 'lemma', 'gold_cluster']

TASK_KEYS_ANNOTATIONS = TASK_KEYS + ['text', 'spans', 'arg0', 'arg1', 'argL', 'argT', 'roleset_id', 'lemma']


class WhitespaceTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(" ")
        spaces = [True] * len(words)
        # Avoid zero-length tokens
        for i, word in enumerate(words):
            if word == "":
                words[i] = " "
                spaces[i] = False
        # Remove the final trailing space
        if words[-1] == " ":
            spaces[-1] = False

        return Doc(self.vocab, words=words, spaces=spaces)


def clean_up_tasks(tasks):
    cleaned_tasks = []
    for task in tasks:

        # take care of legacy code
        if 'bert_doc' in task:
            task['marked_doc'] = task['bert_doc']
        if 'bert_sentence' in task:
            task['marked_sentence'] = task['bert_sentence']

        clean_task = {}
        for key in TASK_KEYS_ANNOTATIONS:
            if key in task:
                clean_task[key] = task[key]
    return cleaned_tasks


def newline2para(text):
    return text.replace('\n', '<p/>')


def JSON(file_path):
    return json.load(open(file_path))


def JSONL(file_path):
    data = []
    with open(file_path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_tasks(file_path):
    if str(file_path).endswith('jsonl'):
        return JSONL(file_path)
    elif str(file_path).endswith('json'):
        return JSON(file_path)
    else:
        raise ValueError("not a valid path name (json/jsonl)")
