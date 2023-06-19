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
NUM_EXAMPLES=5
declare -a ARR_INDICES=(19 98 118) #  cherry-picked

if [ $EXPERIMENT == "TOY" ]; then # CHOWTAPE
    echo "Toy data doesn't contain noise component!"
    exit
else # REAL
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
        # Real noise
        NOISE_TYPE="Real"
        python -u ../code/test-model.py \
                --MODEL $MODEL --WEIGHTS $weight \
                --DATASET $DATASET_NAME --SEGMENT_LENGTH $SEGMENT_LENGTH\
                --SYNC $SYNC --NO_SHUFFLE --IDX $idx\
                --DATASET_NOISE $DATASET_NOISE\
                --ADD_DELAY --ADD_NOISE --NOISE_TYPE $NOISE_TYPE\
                --SAVE_AUDIO\
                --DESCRIPTIVE_NAME $DESCRIPTIVE_NAME"_$idx""_$repeat""_REAL"
                
        # Generated noise
        NOISE_TYPE="Generated"
        python -u ../code/test-model.py \
                --MODEL $MODEL --WEIGHTS $weight \
                --DATASET $DATASET_NAME --SEGMENT_LENGTH $SEGMENT_LENGTH\
                --SYNC $SYNC --NO_SHUFFLE --IDX $idx\
                --DATASET_NOISE $DATASET_NOISE\
                --ADD_DELAY --ADD_NOISE --NOISE_TYPE $NOISE_TYPE\
                --SAVE_AUDIO\
                --DESCRIPTIVE_NAME $DESCRIPTIVE_NAME"_$idx""_$repeat""_GENERATED"
        done
    done
done
