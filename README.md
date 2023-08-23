# Propbank-Meaning Representation (PB-MR)

Accompanying code for the paper Zero-shot event coreference resolution with AMRs
## Getting Started
- Install the required packages:

  ```shell
  pip install -r requirements.txt
  ```
- Additionally, to install `prodigy`, acquire a license and follow
the instructions at [https://prodi.gy/docs/install](https://prodi.gy/docs/install)

- Change directory to the `project`:

    ```shell
    cd project
    ```
 
- Download the ECB+ Corpus, PropBank frames and PropBank Website:

    ```shell
    python -m spacy project assets
    ```

## Preprocesing
- Create mention_map from ECB+ corpus

    ```shell
    python -m spacy project run ecb-setup
    ```
    This will create the `mention_map.pkl` pickle file at `corpus/ecb/mention_map.pkl`
- Save propbank map (`pb.dict`) to access roleset definitions

    ```shell
    python -m spacy project run save-propbank-dict
    ```
    This will create the `pb.dict` pickle file at `outputs/common/pb.dict`

- Create propbank website locally to run on the port `8700` (this step can be skipped if not using 
  the annotation interface). You may have to start a new terminal session from the same directory 
  to continue after running this.

    ```shell
    python -m spacy project run propbank-website
    ```
    To check if this is working, enter `http://localhost:8700` in your browser.

## Annotations with Prodigy
- Create ECB+ Annotation Tasks

  ```shell
  python -m spacy project run create-ecb-tasks
  ```
  This will create the `train`, `dev`, and `test` json files at `corpus/ecb/tasks`

- Example Task in `dev.json`:
```json
{
  "mention_id": "12_10ecb.xml_5",
  "topic": "12",
  "doc_id": "12_10ecb.xml",
  "sentence_id": "0",
  "marked_sentence": "The Indian navy has <m> captured </m> 23 Somalian pirates .",
  "marked_doc": "The Indian navy has <m> captured </m> 23 Somalian ...",
  "lemma": "capture",
  "gold_cluster": "ACT17403639225065902",
  "text": "The Indian navy has captured 23 Somalian pirates .",
  "spans": [
   {
     "token_start": 4,
     "token_end": 4,
     "start": 20,
     "end": 28,
     "label": "EVT"
   }
  ],
  "meta": {
    "Doc": "12_10ecb.xml",
    "Sentence": "0"
  },
}
```
- Example Task with the annotations
```json
{
  "mention_id": "12_10ecb.xml_5",
  "topic": "12",
  "doc_id": "12_10ecb.xml",
  "sentence_id": "0",
  "marked_sentence": "The Indian navy has <m> captured </m> 23 Somalian pirates .",
  "marked_doc": "The Indian navy has <m> captured </m> 23 Somalian ...",
  "lemma": "capture",
  "gold_cluster": "ACT17403639225065902",
  "text": "The Indian navy has captured 23 Somalian pirates .",
  "spans": [
   {
     "token_start": 4,
     "token_end": 4,
     "start": 20,
     "end": 28,
     "label": "EVT"
   }
  ],
  "meta": {
    "Doc": "12_10ecb.xml",
    "Sentence": "0"
  },
  "roleset_id": "capture.01",
  "arg0": "Indian_Navy"
  "arg1": "23_Somalian_Pirates",
  "argL": "Guld_of_Aden",
  "argT": "2008"
}
```

- Run the prodigy UI for annotating the roleset ids for event triggers in the train set

  ```shell
  prodigy wsd-update ann1_train_rsid en_core_web_lg ./corpus/ecb/tasks/train.json ./outputs/common/pb.dict -UP -F ./recipes/wsd.py
  ```
  Once the annotation is done, you can save the annotated tasks in the `annotations` folder by this:
  ```shell
  prodigy db-out ann1_train_rsid > annotations/ann1_train_rsid.jsonl
  ```

- Finally, running the prodigy UI for annotating PB-MR on the annotated `rsids`:
  ```shell
  prodigy srl-update ann1_train_pbmr en_core_web_lg ./annotations/ann1_train_rsid.jsonl ./outputs/common/pb.dict -UP -F 
  ```
  And then to save these annotations to a file:
  ```shell
  prodigy db-out ann1_train_pbmr > annotations/ann1_train_pbmr.jsonl
  ```
## Annotations with GPT-4

## Coreference Results
To run the coreference algorithm on Ann1's dev set annotations:

```shell
python -m scripts.coreference single-ann-results ./annotations/ecb/ann1/dev_pbmr.json
```
and,

```shell
python -m scripts.coreference single-ann-results ./annotations/ecb/ann1/dev_pbmr.json --use-vn 
```
