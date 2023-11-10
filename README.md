# PeaCoK Augmented PersonaChat based on BART

This is the repository for ConvAI2 PersonaChat dialogue modeling with BART and PeaCoK knowledge graph augmentation.

## Gathering the data
Our data can be downloaded from [this link](https://drive.google.com/drive/folders/1A51hZvSLvJoPAKDy2XR_eb-ooZqPRgbb?usp=sharing), please unzip the file and place the folder `data` under this root repository.

Our data include:

* Original PersonaChat (with either original or revised PersonaChat profiles):
  * Training set (original profiles): `data/persona_peacok/train_persona_original_chat_convai2.json`
  * Validation set (original profiles): `data/persona_peacok/valid_persona_original_chat_convai2.json`
  * Training set (revised profiles): `data/persona_peacok/train_persona_revised_chat_convai2.json`
  * Validation set (revised profiles): `data/persona_peacok/valid_persona_revised_chat_convai2.json`
* PersonaChat with profiles augmented with PeaCoK facts (up to 5 randomly chosen to augment each profile):
  * Training set (augmented original profiles): `data/persona_peacok/train_persona_original_chat_ext.json`
  * Validation set (augmented original profiles): `data/persona_peacok/valid_persona_original_chat_ext.json`
  * Training set (augmented revised profiles): `data/persona_peacok/train_persona_revised_chat_ext.json`
  * Validation set (augmented revised profiles): `data/persona_peacok/valid_persona_revised_chat_ext.json`
* Full set of PeaCoK facts linked to each PersonaChat profile:
  * For original profiles: `data/persona_peacok/persona_extend_full_original.json`
  * For revised profiles: `data/persona_peacok/persona_extend_full_revised.json`
* Full PeaCoK knowledge graph:
  * `data/peacok_kg.json`

## Environment setup

```
conda env create -f workstation_setup/bart_peacok.yml
conda activate bart_peacok
```

## Preparing datasets for train/eval

To save time prior to running training and evaluation, run the following the prepare the required datasets:

``python save_datasets.py --dataset {dataset}``

Supported options for the dataset are: 
``persona_chat``, 
``persona_chat_peacok``, 
``persona_chat_peacok_retrieved``, ``persona_chat_peacok_induced``, 
``persona_chat_peacok_random_induced``, ``persona_chat_peacok_retrieved_induced``

The first two options can be computed right away (given you obtained the datasets from the first section). For the remaining options you need to run the retrieved-induced components (instruction in the next section).

## Computing induced and retrieved

```
cd induce_retrieve_pipeline

# Compute all required embeddings in advance
python embed_full_personas.py
python embed_utterances.py
python embed_peacok.py

# Run the induce-retrieve pipeline
python induce_and_retrieve.py

# Update the data based on the pipeline output
extend_persona_chat_with_induced_and_retrieved.py
```

## Run training or eval

To run the training and evaluation scripts, please refer to the Makefile. Prior to running it, set the desired arguments in the Makefile. The following options are supported:

```
make train-pc
make train-peacok

make eval-ppl
make eval-f1
make eval-hits1
```
