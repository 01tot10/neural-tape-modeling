#!/bin/bash
: '
Calls test-dataset.py
'

# ARGUMENTS
TAPE=$1
IPS=$2

# SETTINGS
DATASET_NAME="ReelToReel_Dataset_MiniPulse100_AKAI_IPS[$IPS]_$TAPE"       # NORMAL
# DATASET_NAME="ReelToReel_Dataset_Mini192kHzPulse100_AKAI_IPS[$IPS]_$TAPE" # HIRES
N_FILES=2

# ANALYZE
declare -a ARR_SUBSET=("Train" "Val" "Test")
counter=0
N_PARALLEL=2
for subset in "${ARR_SUBSET[@]}"; do
    if [ $subset == "Train" ]; then
        for idx in $(seq 0 $(($N_FILES - 1))); do
            python -u ../code/test-dataset.py --DATASET $DATASET_NAME --SUBSET $subset --PRELOAD\
                --NO_SHUFFLE --IDX $idx\
                --SAVE_TRAJECTORY --DESCRIPTIVE_NAME "$subset""_""$idx"

            ((counter+=1))
            if [ $counter -eq $(($N_PARALLEL)) ]; then
                echo "Waitin'.."
                wait
                ((counter=0))
            fi
        done
    else
        idx=0
        python -u ../code/test-dataset.py --DATASET $DATASET_NAME --SUBSET $subset --PRELOAD\
            --NO_SHUFFLE --IDX $idx\
            --SAVE_TRAJECTORY --DESCRIPTIVE_NAME "$subset""_""$idx"

        ((counter+=1))
        if [ $counter -eq $(($N_PARALLEL)) ]; then
            echo "Waitin'.."
            wait
            ((counter=0))
        fi
    fi
done
wait