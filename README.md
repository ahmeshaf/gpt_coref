# Propbank-Meaning Representation (PB-MR)

Accompanying code for the paper Zero-shot event coreference resolution with AMRs
## Getting Started
- Install the required packages:

    `pip install -r requirements.txt`

- Change directory to the `project`:

    `cd project`
 
- Download the ECB+ Corpus, PropBank frames and PropBank Website:

    `python -m spacy project assets`

## Preprocesing
- Create mention_map from ECB+ corpus

    `python -m spacy project run ecb-setup`

- Save propbank map (`pb.dict`) to access roleset definitions

    `python -m spacy project run save-propbank-dict`

- Create propbank website locally to run on the port `8700` (this step can be skipped if not using 
  the annotation interface). You may have to start a new terminal session from the same directory 
  to continue after running this.

    `python -m spacy project run propbank-website`

## Annotations with Prodigy
- Create ECB+ Annotation Tasks

  `python -m spacy project run create-ecb-tasks`

- Example Task
```json
{
  "mention_id": "12_10ecb.xml_5",
  "topic": "12",
  "doc_id": "12_10ecb.xml",
  "sentence_id": "0",
  "bert_sentence": "The Indian navy has <m> captured </m> 23 Somalian pirates .",
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
  "bert_sentence": "The Indian navy has <m> captured </m> 23 Somalian pirates .",
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
## Annotations with GPT-4

## Coreference Results