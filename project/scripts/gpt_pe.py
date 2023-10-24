import json
import lexnlp.extract.en.dates
import openai
import os
import pickle
import re
import time
import typer

from collections import defaultdict
from pathlib import Path
from spacy.util import ensure_path
from tqdm import tqdm
from typing import Optional

from .utils import load_tasks


curr_file_dir = os.path.dirname(__file__)
PROMPT_FILE_SIMPLE = curr_file_dir + "/prompt.amr.txt"
PROMPT_FILE_JSON = curr_file_dir + "/prompt_xamr_v2.txt"
PROMPT_FILE_G2 = curr_file_dir + "/prompt_xamr_g2.txt"
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


def _prepare(mention_map_path, tasks_path, output_path, cache_file, restart):
    mention_map_path = ensure_path(mention_map_path)
    tasks_path = ensure_path(tasks_path)

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

    tasks = load_tasks(tasks_path)

    return mention_map, tasks, gpt_cached_responses


def _prepare_tasks(tasks_path, output_path, cache_file, restart):
    tasks_path = ensure_path(tasks_path)

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

    tasks = load_tasks(tasks_path)

    return tasks, gpt_cached_responses


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


def _run_gpt_with_prompt(prompt, model_name, temperature=0.7):
    messages = []
    message = {"role": "user", "content": prompt}
    messages.append(message)
    chat_comp = openai.ChatCompletion.create(
        model=model_name, messages=messages, temperature=temperature
    )
    response = chat_comp
    response = json.dumps(response, indent=2)
    response = json.loads(response)
    return response


def _run_gpt_with_prompt_instruct(prompt, model_name, temperature=0.7):
    instruct_completion = openai.Completion.create(
        model=model_name, prompt=prompt, temperature=temperature, max_tokens=1500
    )
    response = instruct_completion
    # print(response)
    response["choices"][0]["message"] = {"content": response["choices"][0]["text"]}

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

    with open(PROMPT_FILE_JSON) as pff:
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
            # print(response)
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
    tasks_path: Path,
    output_path: Path,
    model_name: str,
    cache_file: Optional[Path] = None,
    restart: Optional[bool] = False,
    temperature: Optional[float] = 0.7,
):
    mention_map, tasks, gpt_responses = _prepare(
        mention_map_path, tasks_path, output_path, cache_file, restart
    )

    with open(PROMPT_FILE_JSON) as pff:
        prompt_ph = pff.read()

    pbar = tqdm(total=len(tasks), desc="running llm")

    tasks_with_responses = []

    while len(tasks):
        task = tasks[0]
        m_id = task["mention_id"]
        mention = mention_map[m_id]
        marked_doc = mention["marked_doc"]
        marked_sentence = mention["marked_sentence"]

        my_prompt = prompt_ph.replace(DOC_VAL, marked_doc)
        # comp_prompt = COM_SENTENCE_PLACEHOLDER.replace(SENTENCE_VAL, marked_sentence)
        # comp_prompt_evt = COM_SENTENCE_EVT_PLACEHOLDER.replace(
        #     SENTENCE_VAL, marked_sentence
        # )

        my_prompt = my_prompt.replace(SENTENCE_VAL, marked_sentence)
        my_prompt = my_prompt.replace(' id="mark_id"', "")
        my_prompt = my_prompt.replace("mark>", "m>")
        # my_prompt = my_prompt + "\nSentence: " + marked_sentence
        # print(my_prompt)
        try:
            if m_id not in gpt_responses:
                if "instruct" in model_name or "davinci" in model_name:
                    response_evt = _run_gpt_with_prompt_instruct(
                        my_prompt, model_name, temperature=temperature
                    )
                else:
                    response_evt = _run_gpt_with_prompt(
                        my_prompt, model_name, temperature=temperature
                    )
                # print(response_evt)
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
        except openai.error.Timeout:
            print("TimeoutError. Sleeping")
            time.sleep(15)

        if m_id in gpt_responses:
            # print(gpt_responses[m_id])
            response = gpt_responses[m_id]["response"]["choices"][0]["message"][
                "content"
            ]
            print(response)
            pbar.update(1)
            task_with_resp = tasks.pop(0)
            task_with_resp["gpt"] = gpt_responses[m_id]
            tasks_with_responses.append(task_with_resp)
    # gpt_response = response['content']
    json.dump(gpt_responses, open(output_path, "w"), indent=1)
    # json.dump(task_with_resp, open(output_path_tasks, "w"))


@app.command()
def gpt_g2(
    g1_tasks_path: Path,
    output_path: Path,
    model_name: str,
    cache_file: Optional[Path] = None,
    restart: Optional[bool] = False,
    temperature: Optional[float] = 0.7,
):
    tasks, gpt_responses = _prepare_tasks(g1_tasks_path, output_path, cache_file, restart)
    topic2eds = defaultdict(list)
    for task in tasks:
        topic2eds[task["topic"]].append(task["Event_Description"])

    with open(PROMPT_FILE_G2) as pff:
        prompt_ph = pff.read()

    pbar = tqdm(total=len(tasks), desc="running llm")

    tasks_with_responses = []

    while len(tasks):
        task = tasks[0]
        m_id = task["mention_id"]

        event_desc = task["Event_Description"]

        topic_event_descs =  "\n".join([" ".join((str(i) + "." , val)) for i, val in
                                        enumerate(topic2eds[task["topic"]]) if val.startswith("On")])
        topic_event_descs = topic_event_descs.replace(event_desc, "")

        marked_sentence = task["marked_sentence"]

        my_prompt = prompt_ph.replace("EVENT_DESC", event_desc)
        my_prompt = my_prompt.replace("ED_LIST", topic_event_descs)
        my_prompt = my_prompt.replace(SENTENCE_VAL, marked_sentence)
        my_prompt = my_prompt.replace(' id="mark_id"', "")
        my_prompt = my_prompt.replace("mark>", "m>")
        # my_prompt = my_prompt + "\nSentence: " + marked_sentence
        print(my_prompt)
        try:
            if m_id not in gpt_responses:
                if "instruct" in model_name or "davinci" in model_name:
                    response_evt = _run_gpt_with_prompt_instruct(
                        my_prompt, model_name, temperature=temperature
                    )
                else:
                    response_evt = _run_gpt_with_prompt(
                        my_prompt, model_name, temperature=temperature
                    )
                # print(response_evt)
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
        except openai.error.Timeout:
            print("TimeoutError. Sleeping")
            time.sleep(15)

        if m_id in gpt_responses:
            # print(gpt_responses[m_id])
            response = gpt_responses[m_id]["response"]["choices"][0]["message"][
                "content"
            ]
            print(response)
            pbar.update(1)
            task_with_resp = tasks.pop(0)
            tasks_with_responses.append(task_with_resp)
    # gpt_response = response['content']
    json.dump(gpt_responses, open(output_path, "w"), indent=1)


escape_string = "\\"


def _clean_json_string(text):
    return f'''"{text.replace('"', f'{escape_string}"')}"'''


def _parse_json_format(content):
    res_text = content[content.index("{") + 1 : content.index("}")]

    lines = res_text.strip().split("\n")
    lines = ["{"] + lines
    lines.append('"<END>": "</END>"')
    lines.append("}")
    res_text_clean = []
    for line in lines:
        if line.strip() not in "{}" and "</END>" not in line:
            key, val = line.split('":')
            key = key.strip().strip('"')
            key = _clean_json_string(key)
            val = val.strip().strip('",')
            val = _clean_json_string(val)
            line = key.lower() + ":" + val + ","
        res_text_clean.append(line)
    res_text_clean = "\n".join(res_text_clean)
    try:
        res_json = json.loads(res_text_clean)
    except json.decoder.JSONDecodeError:
        print(res_text_clean)
        res_json = json.loads(res_text_clean)
    return res_json


def _clean_wiki(wiki_string):
    wiki_string = wiki_string.split("/")[-1].strip()
    wiki_string = wiki_string.replace("%27", "'")

    if "(" in wiki_string:
        wiki_string = wiki_string[: wiki_string.index("(") - 1]
    return wiki_string


def _get_gpt_jsons(gpt_responses):
    mid2gpt_task = {}
    for m_id, gpt_response in gpt_responses.items():
        res = gpt_response["response"]
        content = res["choices"][0]["message"]["content"]
        res_json = _parse_json_format(content)
        mid2gpt_task[m_id] = res_json
    return mid2gpt_task


def _do_entity_corefernce(gpt_tasks):
    arg_names = ["arg-0", "arg-1"]
    arg_val_wiki = []
    for task in gpt_tasks:
        for arg_n in arg_names:
            arg_val = task[arg_n]
            arg_wiki = ""
            if arg_n + "_coreference" in task:
                arg_wiki = _clean_wiki(task[arg_n + "_coreference"].strip())
                print(arg_wiki)
            if arg_wiki.lower() in {"", "null", "na", "<na>", "none"}:
                arg_wiki = ""

            arg_val_wiki.append((arg_val, arg_wiki))

    wiki2val = defaultdict(set)
    arg_val2wiki = {}
    for arg_val, arg_wiki in arg_val_wiki:
        wiki2val[arg_wiki].add(arg_val)
        if arg_wiki != "" and arg_val not in arg_val2wiki:
            arg_val2wiki[arg_val] = arg_wiki

    for task in gpt_tasks:
        for arg_n in arg_names:
            arg_val = task[arg_n]
            arg_wiki = ""

            if arg_n + "_coreference" in task:
                arg_wiki = _clean_wiki(task[arg_n + "_coreference"].strip())
                print(arg_wiki)
            if arg_wiki.lower() in {"", "null", "na", "<na>", "none"}:
                arg_wiki = ""

            if arg_wiki == "" and arg_val in arg_val2wiki:
                task[arg_n + "_coreference"] = arg_val2wiki[arg_val]


@app.command()
def gpt_json2task(json_file: Path, tasks_file_path: Path, output_tasks_file_path: Path):
    gpt_responses = load_tasks(json_file)
    tasks = load_tasks(tasks_file_path)
    output_tasks_file_path = ensure_path(output_tasks_file_path)
    mid2task = {task["mention_id"]: task for task in tasks}
    mid2gpt_tasks = _get_gpt_jsons(gpt_responses)
    _do_entity_corefernce(mid2gpt_tasks.values())
    print(len(mid2task))
    print(len(tasks))
    for mid, res_json in mid2gpt_tasks.items():
        task = mid2task[mid]
        task["arg0"] = res_json["arg-0"]
        task["roleset_id"] = res_json["event_roleset_id"]
        if (
            "arg-0_coreference" in res_json
            and "NA" not in res_json["arg-0_coreference"]
        ):
            task["arg0"] = _clean_wiki(res_json["arg-0_coreference"])
        task["arg1"] = res_json["arg-1"]
        if (
            "arg-1_coreference" in res_json
            and "NA" not in res_json["arg-1_coreference"]
        ):
            task["arg1"] = _clean_wiki(res_json["arg-1_coreference"])
        if "arg-1_roleset_id" in res_json and "NA" not in res_json["arg-1_roleset_id"]:
            task["arg1"] = res_json["arg-1_roleset_id"]
        if "arg-location" in res_json:
            task["argL"] = _clean_wiki(res_json["arg-location"])
        else:
            task["argL"] = "<NA>"
        dates = list(lexnlp.extract.en.dates.get_dates(res_json["arg-time"]))
        if len(dates):
            curr_date = dates[0]
            all_dates = ""
            if curr_date.year == 2023:
                all_dates = f"{curr_date.month}/{curr_date.month}-{curr_date.day}"
            else:
                all_dates = f"{str(curr_date)}/{curr_date.year}-{curr_date.month}/{curr_date.year}/{curr_date.month}-{curr_date.day}/{curr_date.month}"
                print(str(all_dates))
            task["argT"] = all_dates
        else:
            task["argT"] = res_json["arg-time"]

        if "complete event description" in res_json:
            task["Event_Description"] = res_json["complete event description"]

    json.dump(tasks, open(output_tasks_file_path, "w"), indent=1)


if __name__ == "__main__":
    app()
