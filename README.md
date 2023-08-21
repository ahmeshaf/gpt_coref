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

- Create propbank website locally to run on the port `8700`

    `python -m spacy project run propbank-website`

You will have to start a new terminal session from the same directory to continue.
