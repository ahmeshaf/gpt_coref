from spacy.util import ensure_path
from typing import Optional
from pathlib import Path
from tqdm import tqdm
import openai
import pickle
import typer
import json
import time
import os


curr_file_dir = os.path.dirname(__file__)
PROMPT_FILE_SIMPLE = curr_file_dir + '/prompt.amr.txt'
app = typer.Typer()

COM_SENTENCE_PLACEHOLDER = "sentence: SENTENCE_VAL\nyour response:\n"
COM_SENTENCE_EVT_PLACEHOLDER = "sentence with eventive argument (ARG-1 is a roleset ID): SENTENCE_VAL\nyour response:\n"

SENTENCE_VAL = "SENTENCE_VAL"
DOC_VAL = "DOC_VAL"

# -------------------- Helpers --------------------- #


def _prepare(mention_map_path, mention_ids_path, output_path, cache_file, restart):
    mention_map_path = ensure_path(mention_map_path)
    mention_ids_path = ensure_path(mention_ids_path)

    gpt_cached_responses = {}

    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    if cache_file:
        if cache_file.exists() and restart:
            os.remove(cache_file)
        elif cache_file.exists():
            gpt_cached_responses = pickle.load(open(cache_file, 'rb'))
        if not cache_file.parent.exists():
            cache_file.parent.mkdir(parents=True)

    mention_map = pickle.load(open(mention_map_path, 'rb'))
    with open(mention_ids_path) as mff:
        mention_ids = [line.strip() for line in mff]

    return mention_map, mention_ids, gpt_cached_responses


def _gpt_response2dict(gpt_response):
    choice = gpt_response['choices'][0]
    return {
        'usage': gpt_response['usage'],
        'choices': [
            {
                'message': {
                    'content': choice['message']['content'],
                    'role': choice['message']['role']
                }
            }
        ]
    }


def _run_gpt_with_prompt(prompt, model_name):
    messages = []
    message = {"role": "user", "content": prompt}
    messages.append(message)
    chat_comp = openai.ChatCompletion.create(
        model=model_name,
        messages=messages
    )
    response = chat_comp
    response = json.dumps(response, indent=2)
    response = json.loads(response)
    return response

# -------------------- Commands -------------------- #


@app.command()
def dummy():
    pass


@app.command()
def gpt_simple(
        mention_map_path: Path,
        mention_ids_path: Path,
        output_path: Path,
        model_name: str,
        cache_file: Optional[Path] = None,
        restart: Optional[bool] = False
):
    mention_map, mention_ids, gpt_responses = _prepare(mention_map_path, mention_ids_path, output_path, cache_file, restart)

    with open(PROMPT_FILE_SIMPLE) as pff:
        prompt_ph = pff.read()

    pbar = tqdm(total=len(mention_ids), desc='running llm')

    while len(mention_ids):
        m_id = mention_ids[0]
        mention = mention_map[m_id]
        marked_doc = mention['bert_doc']
        marked_sentence = mention['bert_sentence']

        my_prompt = prompt_ph.replace(DOC_VAL, marked_doc)
        comp_prompt = COM_SENTENCE_PLACEHOLDER.replace(SENTENCE_VAL, marked_sentence)
        comp_prompt_evt = COM_SENTENCE_EVT_PLACEHOLDER.replace(SENTENCE_VAL, marked_sentence)

        my_prompt_reg = my_prompt + comp_prompt
        my_prompt_evt = my_prompt + comp_prompt_evt

        try:
            if m_id not in gpt_responses:
                response_reg = _run_gpt_with_prompt(my_prompt_reg, model_name)
                gpt_responses[m_id] = {'regular': {'prompt': my_prompt_reg, 'response': response_reg}}

                response_evt = _run_gpt_with_prompt(my_prompt_evt, model_name)
                gpt_responses[m_id]['eventive'] = {'prompt': my_prompt_evt, 'response': response_evt}

                pickle.dump(gpt_responses, open(cache_file, 'wb'))
        except openai.error.RateLimitError:
            print('Rate limit reached. sleeping')
            time.sleep(15)  # adding delay to avoid time-out

        if m_id in gpt_responses:
            pbar.update(1)
            mention_ids.pop(0)
        # gpt_response = response['content']


if __name__ == '__main__':
    app()


