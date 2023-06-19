#!/bin/bash
: '
Calls train.py
'

# SETTINGS
HIDDEN_SIZE=$1
DATASET=$2
DATASET_VAL=$3
N_EPOCHS=$4
DESCRIPTIVE_NAME=$5

# Run
python -u ../code/train.py\
    --HIDDEN_SIZE $HIDDEN_SIZE --N_EPOCHS $N_EPOCHS\
    --DATASET $DATASET --DATASET_VAL $DATASET_VAL\
    --DEMODULATE\
    --DESCRIPTIVE_NAME $DESCRIPTIVE_NAME --DRY_RUN