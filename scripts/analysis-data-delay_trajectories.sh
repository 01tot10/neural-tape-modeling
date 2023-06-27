#!/bin/bash
: '
Calls test-dataset.py
'
# ARGUMENTS
EXPERIMENT=$1
TAPE=$2
IPS=$3

# SETTINGS
# SEGMENT_LENGTH=$((44100/2)) # SHORT
SEGMENT_LENGTH=$((44100*10))  # LONG
# SEGMENT_LENGTH=$((192000*10)) # LONG, HIRES
N_PARALLEL=1
N_RUNS=5

# ANALYZE
DESCRIPTIVE_NAME="DELAY_TRAJECTORIES"

if [ $EXPERIMENT == "TOY" ]; then # CHOWTAPE
    # DATASET_NAME="ReelToReel_Dataset_MiniPulse100_CHOWTAPE" # EXP1a
    DATASET_NAME="ReelToReel_Dataset_MiniPulse100_CHOWTAPE_WOWFLUTTER" # EXP1b
else # REAL
    DATASET_NAME="ReelToReel_Dataset_MiniPulse100_AKAI_IPS[$IPS]_$TAPE"       # NORMAL
    # DATASET_NAME="ReelToReel_Dataset_Mini192kHzPulse100_AKAI_IPS[$IPS]_$TAPE" # HIRES
fi

counter=0
for idx in $(seq $N_RUNS); do
    python -u ../code/test-dataset.py\
        --DATASET $DATASET_NAME --PRELOAD\
        --SEGMENT_LENGTH $SEGMENT_LENGTH\
        --PLOT_DELAY\
        --SAVE_FIG --DESCRIPTIVE_NAME $DESCRIPTIVE_NAME"_""$idx" &
    # --NO_SHUFFLE --IDX 0\ # SHORT

    ((counter+=1))
    if [ $counter -eq $(($N_PARALLEL)) ]; then
        echo "Waitin'.."
        wait
        ((counter=0))
    fi
done
wait