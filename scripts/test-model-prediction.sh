#!/bin/bash
: '
Calls test-dataset.py
'
# ARGUMENTS
EXPERIMENT=$1
TAPE=$2
IPS=$3
MODEL=$4
LOSS=$5

# SETTINGS
DESCRIPTIVE_NAME="PREDICTION"
SEGMENT_LENGTH=$((5*44100)) # in [n]
NOISE_TYPE="Generated" # in ["Real","Generated"]
DELAY_TYPE="True" # in ["True", "Real", "Generated"]
NUM_EXAMPLES=2

if [ $EXPERIMENT == "TOY" ]; then # CHOWTAPE
    # DATASET_NAME="ReelToReel_Dataset_MiniPulse100_CHOWTAPE" # EXP-1
    # declare -a ARR_WEIGHTS=(
    # "GRU-HS[64]-DS[ReelToReel_Dataset_Mini_CHOWTAPE]"
    # )
    declare -a ARR_INDICES=(16 40 70 77 109)  # cherry-picked indices
    DATASET_NAME="ReelToReel_Dataset_MiniPulse100_CHOWTAPE_WOWFLUTTER" # EXP-2
    declare -a ARR_WEIGHTS=(
        "$MODEL-HS[64]-L[$LOSS]-DS[ReelToReel_Dataset_MiniPulse100_CHOWTAPE_WOWFLUTTER]_BEST"
    )
    SYNC=0.0
else # REAL
    declare -a ARR_INDICES=(52 70 84 98 109)  # cherry-picked indices
    DATASET_NAME="ReelToReel_Dataset_MiniPulse100_AKAI_IPS[$IPS]_$TAPE" # REAL
    declare -a ARR_WEIGHTS=(
        "$MODEL-HS[64]-L[$LOSS]-DS[ReelToReel_Dataset_MiniPulse100_AKAI_IPS[$IPS]_$TAPE]_BEST"
    )
    DATASET_NOISE="Silence_AKAI_IPS[$IPS]_$TAPE"
    SYNC=0.0
fi

# ANALYZE
for weight in "${ARR_WEIGHTS[@]}"; do
    for idx in "${ARR_INDICES[@]}"; do
        for repeat in $(seq $NUM_EXAMPLES); do
            if [ $EXPERIMENT == "TOY" ]; then # CHOWTAPE
                python -u ../code/test-model.py \
                    --MODEL $MODEL --WEIGHTS $weight \
                    --DATASET $DATASET_NAME --SEGMENT_LENGTH $SEGMENT_LENGTH\
                    --SYNC $SYNC --NO_SHUFFLE --IDX $idx\
                    --ADD_DELAY --DELAY_TYPE $DELAY_TYPE\
                    --SAVE_AUDIO\
                    --DESCRIPTIVE_NAME $DESCRIPTIVE_NAME"_$idx""_$repeat""_DELAY[$DELAY_TYPE]"
            else
                python -u ../code/test-model.py \
                    --MODEL $MODEL --WEIGHTS $weight \
                    --DATASET $DATASET_NAME --SEGMENT_LENGTH $SEGMENT_LENGTH\
                    --SYNC $SYNC --NO_SHUFFLE --IDX $idx\
                    --DATASET_NOISE $DATASET_NOISE\
                    --ADD_DELAY --DELAY_TYPE $DELAY_TYPE\
                    --ADD_NOISE --NOISE_TYPE $NOISE_TYPE\
                    --SAVE_AUDIO\
                    --DESCRIPTIVE_NAME $DESCRIPTIVE_NAME"_$idx""_$repeat""_DELAY[$DELAY_TYPE]""_NOISE[$NOISE_TYPE]"
            fi
        done
    done
done