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
DESCRIPTIVE_NAME="HYSTERESIS"
# declare -a ARR_INDICES=(13 63 98) # cherry-picked indices OLD
# SEGMENT_LENGTH=48510 # OLD
declare -a ARR_INDICES=(5) # cherry-picked indices
SEGMENT_LENGTH=44100
COMBINED=True

if [ $EXPERIMENT == "TOY" ]; then # CHOWTAPE
    # OLD
    # DATASET_NAME="SinesFadedShortContinuousPulse100_CHOWTAPE" # EXP-1
    # DATASET_NAME="SinesFadedShortContinuousPulse100_CHOWTAPE_WOWFLUTTER" # EXP-2

    if [ $TAPE == "EXP1" ]; then # EXP-1
        DATASET_NAME="SinesFadedShortContinuousPulse100CherryPicked_CHOWTAPE" # EXP-1
        declare -a ARR_WEIGHTS=(
            # "$MODEL-HS[64]-L[$LOSS]-DS[ReelToReel_Dataset_Mini_CHOWTAPE]_BEST"
            "$MODEL-HS[64]-L[$LOSS]-DS[ReelToReel_Dataset_Mini_CHOWTAPE]_1"
            "$MODEL-HS[64]-L[$LOSS]-DS[ReelToReel_Dataset_Mini_CHOWTAPE]_2"
            "$MODEL-HS[64]-L[$LOSS]-DS[ReelToReel_Dataset_Mini_CHOWTAPE]_3"
        )
    else
        DATASET_NAME="SinesFadedShortContinuousPulse100CherryPicked_CHOWTAPE_WOWFLUTTER" # EXP-2
        declare -a ARR_WEIGHTS=(
            "GRU-HS[64]-L[DCPreESR]-DS[ReelToReel_Dataset_MiniPulse100_CHOWTAPE_WOWFLUTTER]_BEST"
            "DiffDelGRU-HS[64]-L[DCPreESR]-DS[ReelToReel_Dataset_MiniPulse100_CHOWTAPE_WOWFLUTTER]_BEST"
            "DiffDelGRU-HS[64]-L[LogSpec]-DS[ReelToReel_Dataset_MiniPulse100_CHOWTAPE_WOWFLUTTER]_BEST"
            # "$MODEL-HS[64]-L[$LOSS]-DS[ReelToReel_Dataset_MiniPulse100_CHOWTAPE_WOWFLUTTER]_BEST"
            # "$MODEL-HS[64]-L[$LOSS]-DS[ReelToReel_Dataset_MiniPulse100_CHOWTAPE_WOWFLUTTER]_1"
            # "$MODEL-HS[64]-L[$LOSS]-DS[ReelToReel_Dataset_MiniPulse100_CHOWTAPE_WOWFLUTTER]_2"
            # "$MODEL-HS[64]-L[$LOSS]-DS[ReelToReel_Dataset_MiniPulse100_CHOWTAPE_WOWFLUTTER]_3"
        )
    fi
    SYNC=0.0 # CHOWTAPE
else # REAL
    DATASET_NAME="SinesFadedShortContinuousPulse100_AKAI_IPS[$IPS]_$TAPE"
    declare -a ARR_WEIGHTS=(
        "$MODEL-HS[64]-L[$LOSS]-DS[ReelToReel_Dataset_MiniPulse100_AKAI_IPS[$IPS]_$TAPE]"
    )
    SYNC=1.0 # REAL
fi

# ANALYZE
if [ $COMBINED == True ]; then
    for idx in "${ARR_INDICES[@]}"; do

        if [ $TAPE == "EXP1" ]; then
            python -u ../code/test-model.py \
                --MODEL $MODEL --WEIGHTS $(echo "${ARR_WEIGHTS[*]}") \
                --DATASET $DATASET_NAME \
                --SYNC $SYNC --SEGMENT_LENGTH $SEGMENT_LENGTH --ZOOM 0.1 --NO_SHUFFLE --IDX $idx \
                --PLOT_TRANSFER --SAVE_FIG \
                --DESCRIPTIVE_NAME $DESCRIPTIVE_NAME"_$idx"
        else
            python -u ../code/test-model.py \
                --MODEL $MODEL --WEIGHTS $(echo "${ARR_WEIGHTS[*]}") \
                --DATASET $DATASET_NAME \
                --SYNC $SYNC --SEGMENT_LENGTH $SEGMENT_LENGTH --ZOOM 0.1 --NO_SHUFFLE --IDX $idx \
                --DEMODULATE \
                --PLOT_TRANSFER --SAVE_FIG \
                --DESCRIPTIVE_NAME $DESCRIPTIVE_NAME"_$idx"
        fi
    done
else
    for weight in "${ARR_WEIGHTS[@]}"; do
        for idx in "${ARR_INDICES[@]}"; do
            if [ $TAPE == "EXP1" ]; then
                python -u ../code/test-model.py \
                    --MODEL $MODEL --WEIGHTS $weight \
                    --DATASET $DATASET_NAME \
                    --SYNC $SYNC --SEGMENT_LENGTH $SEGMENT_LENGTH --ZOOM 0.1 --NO_SHUFFLE --IDX $idx \
                    --PLOT_TRANSFER --SAVE_FIG \
                    --DESCRIPTIVE_NAME $DESCRIPTIVE_NAME"_$idx"
            else
                python -u ../code/test-model.py \
                    --MODEL $MODEL --WEIGHTS $weight \
                    --DATASET $DATASET_NAME \
                    --SYNC $SYNC --SEGMENT_LENGTH $SEGMENT_LENGTH --ZOOM 0.1 --NO_SHUFFLE --IDX $idx \
                    --DEMODULATE \
                    --PLOT_TRANSFER --SAVE_FIG \
                    --DESCRIPTIVE_NAME $DESCRIPTIVE_NAME"_$idx"
            fi
        done
    done
fi
