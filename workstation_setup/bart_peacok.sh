#!/bin/bash
set -x
MNT_DIR=/home/cutura/mnt/home/cutura/bart_peacok

cp -r ${MNT_DIR}/data .
cp -r ${MNT_DIR}/saved_datasets .
cp -r ${MNT_DIR}/ParlAI .

conda env create -f workstation_setup/bart_peacok.yml
