import json


TASK_KEYS = ['mention_id', 'topic', 'doc_id', 'sentence_id', 'sentence',
             'marked_sentence', 'marked_doc', 'lemma', 'gold_cluster']

TASK_KEYS_ANNOTATIONS = TASK_KEYS + ['text', 'spans', 'arg0', 'arg1', 'argL', 'argT', 'roleset_id', 'lemma']


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