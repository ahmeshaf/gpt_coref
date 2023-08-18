from spacy.util import ensure_path
from typing import Optional
from pathlib import Path
from tqdm import tqdm
import lexnlp.extract.en.dates
import openai
import pickle
import lexnlp
import typer
import json
import time
import os
import re


curr_file_dir = os.path.dirname(__file__)
PROMPT_FILE_SIMPLE = curr_file_dir + "/prompt.amr.txt"
app = typer.Typer()

COM_SENTENCE_PLACEHOLDER = "sentence: SENTENCE_VAL\nyour response:\n"
COM_SENTENCE_EVT_PLACEHOLDER = "sentence with eventive argument (ARG-1 is a roleset ID): SENTENCE_VAL\nyour response:\n"

SENTENCE_VAL = "SENTENCE_VAL"
DOC_VAL = "DOC_VAL"

# -------------------- Helpers --------------------- #


def get_dates(text):
    return list(lexnlp.extract.en.dates.get_dates(text))


class Argument:
    def __init__(self, line):
        self.arg_name = line.split(":")[0]
        arg_value = "".join([s.strip() for s in line.split(":")[1:]])
        self.parse_arg_value(arg_value)
        self.arg_text = ""
        self.arg_reference = ""
        self.wiki = None
        self.non_wiki = None
        self.rolesets = None
        self.dates = None
        self.args_multi = None

    def parse_arg_value(self, text):
        self.arg_text = text.split("(")[0].strip()
        self.arg_reference = (
            text.split("(")[-1].strip(")").replace("referencing", "").strip()
        )

        self.wiki = [
            w.split("wiki/")[-1].split("#")[0]
            for w in re.findall(r"(wiki/\S+)", self.arg_reference)
        ]
        self.non_wiki = [
            s.replace("https//", "")
            .replace("http//", "")
            .replace("www.", "")
            .strip(".com")
            for s in re.findall(r"(https?:?//\S+)", self.arg_reference)
            if "wikipedia" not in s
        ]
        self.rolesets = re.findall(r".*\.\d{1,3}", text)

        self.dates = list(lexnlp.extract.en.dates.get_dates(text))
        self.args_multi = [s.strip() for s in self.arg_text.split("and")]

    def __str__(self):
        return " ".join(
            [
                f"[{self.arg_name}]",
                f"[{self.arg_text}]",
                f"[args: {self.args_multi}]"
                f"[dates: {self.dates}]"
                f"[rolesets: {self.rolesets}]",
                f"[non-wiki: {self.non_wiki}]",
                f"[wiki: {self.wiki}]",
                f"[{self.arg_reference}]",
            ]
        )


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
            gpt_cached_responses = pickle.load(open(cache_file, "rb"))
        if not cache_file.parent.exists():
            cache_file.parent.mkdir(parents=True)

    mention_map = pickle.load(open(mention_map_path, "rb"))
    with open(mention_ids_path) as mff:
        mention_ids = [line.strip() for line in mff]

    return mention_map, mention_ids, gpt_cached_responses


def _gpt_response2dict(gpt_response):
    choice = gpt_response["choices"][0]
    return {
        "usage": gpt_response["usage"],
        "choices": [
            {
                "message": {
                    "content": choice["message"]["content"],
                    "role": choice["message"]["role"],
                }
            }
        ],
    }


def _run_gpt_with_prompt(prompt, model_name):
    messages = []
    message = {"role": "user", "content": prompt}
    messages.append(message)
    chat_comp = openai.ChatCompletion.create(model=model_name, messages=messages)
    response = chat_comp
    response = json.dumps(response, indent=2)
    response = json.loads(response)
    return response


# -------------------- Commands -------------------- #


@app.command()
def dummy():
    pass


@app.command()
def gpt_simple_ecb(
    mention_map_path: Path,
    mention_ids_path: Path,
    output_path: Path,
    split: str,
    model_name: str,
    cache_file: Optional[Path] = None,
    restart: Optional[bool] = False,
):
    mention_map, mention_ids, gpt_responses = _prepare(
        mention_map_path, mention_ids_path, output_path, cache_file, restart
    )

    mention_ids = [
        key
        for key, val in mention_map.items()
        if val["split"] == split and val["men_type"] == "evt"
    ]

    print(len(mention_ids))

    with open(PROMPT_FILE_SIMPLE) as pff:
        prompt_ph = pff.read()

    pbar = tqdm(total=len(mention_ids), desc="running llm")

    while len(mention_ids):
        m_id = mention_ids[0]
        mention = mention_map[m_id]
        marked_doc = mention["bert_doc"]
        marked_sentence = mention["bert_sentence"]

        my_prompt = prompt_ph.replace(DOC_VAL, marked_doc)
        comp_prompt = COM_SENTENCE_PLACEHOLDER.replace(SENTENCE_VAL, marked_sentence)
        comp_prompt_evt = COM_SENTENCE_EVT_PLACEHOLDER.replace(
            SENTENCE_VAL, marked_sentence
        )

        my_prompt = prompt_ph + "\nSentence: " + marked_sentence
        # print(my_prompt)
        try:
            if m_id not in gpt_responses:
                response_evt = _run_gpt_with_prompt(my_prompt, model_name)
                gpt_responses[m_id] = {"prompt": my_prompt, "response": response_evt}

                pickle.dump(gpt_responses, open(cache_file, "wb"))
        except openai.error.RateLimitError:
            print("Rate limit reached. sleeping")
            time.sleep(15)  # adding delay to avoid time-out
        except openai.error.APIError:
            print("APIError. sleeping")
            time.sleep(15)  # adding delay to avoid time-out
        except openai.error.APIConnectionError:
            print("APIConnectionError. sleeping")
            time.sleep(15)
        except openai.error.ServiceUnavailableError:
            print("ServiceUnavailableError. sleeping")
            time.sleep(15)

        if m_id in gpt_responses:
            # print(gpt_responses[m_id])
            response = gpt_responses[m_id]["response"]["choices"][0]["message"][
                "content"
            ]
            print(response)
            pbar.update(1)
            mention_ids.pop(0)
    json.dump(gpt_responses, open(output_path, "w"))


@app.command()
def parse_gpt_responses(input_path: Path, output_path: Path, mention_map_path: Path):
    mention_map = pickle.load(open(mention_map_path, "rb"))
    gpt_responses = json.load(open(input_path))

    arg_map = {"ARG-0": "arg0", "ARG-1": "arg1", "ARG-Loc": "argL", "ARG-Time": "argT"}

    def get_wikis(string):
        all_wikis = [w.split("wiki/")[-1] for w in re.findall(r"(wiki/\S+)", string)]
        all_wikis_with_split = set()
        for wik in all_wikis:
            all_wikis_with_split.add(wik)
            all_wikis_with_split.add(wik.split("#")[0])
        return all_wikis_with_split

    parsed_responses = []
    for m_id, g_res in gpt_responses.items():
        m_parsed = {}
        mention = mention_map[m_id]
        mention["mention_id"] = m_id
        gpt_response = g_res["response"]["choices"][0]["message"]["content"]
        lines = gpt_response.split("\n")
        for line in lines:
            if line.startswith("Roleset"):
                # print(gpt_response)
                # print(line)
                m_parsed["roleset_id"] = line[line.index(":") :].strip()
                m_parsed["roleset_id"] = mention["lemma"]
            if line.startswith("ARG"):
                arg_name = line[: line.index(":")]
                arg_val = line[line.index(":") :].strip()
                wikis = get_wikis(arg_val)
                if len(wikis) == 0:
                    if "(" in arg_val:
                        arg_val = arg_val[: arg_val.index("(")].strip('"').strip()
                else:
                    arg_val = "/".join(wikis)

                if arg_name.startswith("ARG-Time"):
                    date_vals = get_dates(arg_val)
                    if len(date_vals):
                        arg_val = "/".join([str(d.year) for d in date_vals])

                for val in arg_map.values():
                    m_parsed[val] = "<NA>"
                if arg_name in arg_map:
                    m_parsed[arg_map[arg_name]] = arg_val

        for k in ["mention_id", "doc_id", "sentence_id", "topic", "gold_cluster"]:
            m_parsed[k] = mention[k]
        parsed_responses.append(m_parsed)
    json.dump(parsed_responses, open(output_path, "w"))


@app.command()
def gpt_simple(
    mention_map_path: Path,
    mention_ids_path: Path,
    output_path: Path,
    model_name: str,
    cache_file: Optional[Path] = None,
    restart: Optional[bool] = False,
):
    mention_map, mention_ids, gpt_responses = _prepare(
        mention_map_path, mention_ids_path, output_path, cache_file, restart
    )

    with open(PROMPT_FILE_SIMPLE) as pff:
        prompt_ph = pff.read()

    pbar = tqdm(total=len(mention_ids), desc="running llm")

    while len(mention_ids):
        m_id = mention_ids[0]
        mention = mention_map[m_id]
        marked_doc = mention["bert_doc"]
        marked_sentence = mention["bert_sentence"]

        my_prompt = prompt_ph.replace(DOC_VAL, marked_doc)
        comp_prompt = COM_SENTENCE_PLACEHOLDER.replace(SENTENCE_VAL, marked_sentence)
        comp_prompt_evt = COM_SENTENCE_EVT_PLACEHOLDER.replace(
            SENTENCE_VAL, marked_sentence
        )

        my_prompt = prompt_ph + "\nSentence: " + marked_sentence
        # print(my_prompt)
        try:
            if m_id not in gpt_responses:
                response_evt = _run_gpt_with_prompt(my_prompt, model_name)
                gpt_responses[m_id] = {"prompt": my_prompt, "response": response_evt}

                pickle.dump(gpt_responses, open(cache_file, "wb"))
        except openai.error.RateLimitError:
            print("Rate limit reached. sleeping")
            time.sleep(15)  # adding delay to avoid time-out
        except openai.error.APIError:
            print("APIError. sleeping")
            time.sleep(15)  # adding delay to avoid time-out
        except openai.error.APIConnectionError:
            print("APIConnectionError. sleeping")
            time.sleep(15)
        except openai.error.ServiceUnavailableError:
            print("ServiceUnavailableError. sleeping")
            time.sleep(15)

        if m_id in gpt_responses:
            # print(gpt_responses[m_id])
            response = gpt_responses[m_id]["response"]["choices"][0]["message"][
                "content"
            ]
            print(response)
            pbar.update(1)
            mention_ids.pop(0)
    # gpt_response = response['content']
    json.dump(gpt_responses, open(output_path, "w"))


if __name__ == "__main__":
    app()
