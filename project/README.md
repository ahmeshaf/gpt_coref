<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: ECB+GPT AMR Experiments with spaCy and prodigy

## 📋 project.yml

The [`project.yml`](project.yml) defines the data assets required by the
project, as well as the available commands and workflows. For details, see the
[spaCy projects documentation](https://spacy.io/usage/projects).

### ⏯ Commands

The following commands are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run).
Commands are only re-run if their inputs have changed.

| Command | Description |
| --- | --- |
| `ecb-setup` | Preprocess and Create mention_map from ECB+ corpus |
| `save-propbank-dict` | generate an easy to use propbank dictionary for roleset definitions |
| `propbank-website` | run the propbank website |
| `create-ecb-tasks` | Create ECB+ Tasks for each Split |
| `run-g1` | Generate X-amr with G1 Method |
| `run-g2` | Generate X-amr with G2 Method |
| `gpt-simple-dev` | run a zero shot learning approach in the event graph generation |
| `single-ann2-results-dev` | run a zero shot learning approach in the event graph generation |
| `single-ann1-results-dev` | run a zero shot learning approach in the event graph generation |
| `ann1-or-ann2-results-dev` | run a zero shot learning approach in the event graph generation |
| `ann1-and-ann2-results-dev` | run a zero shot learning approach in the event graph generation |
| `stupidly-large` | run a zero shot learning approach in the event graph generation |
| `parsed-gpt-dev` |  |
| `gpt-results-dev` |  |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `gpt-amr` | `ecb-setup` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://spacy.io/api/cli#project-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/ecb` | Git |  |
| `assets/propbank_frames` | Git |  |
| `assets/propbank_scripts` | Git |  |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->