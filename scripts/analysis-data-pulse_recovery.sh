#!/bin/bash
: '
Calls test-dataset.py
'
# ARGUMENTS
EXPERIMENT=$1
TAPE=$2
IPS=$3

# SETTINGS
SEGMENT_LENGTH=$((44100 / 2))  # SHORT
# SEGMENT_LENGTH=441000 # LONG
N_PARALLEL=1

# ANALYZE
DESCRIPTIVE_NAME="PULSEREC"
if [ $EXPERIMENT == "TOY" ]; then # CHOWTAPE
  # DATASET_NAME="ReelToReel_Dataset_MiniPulse100_CHOWTAPE" # EXP1a
  DATASET_NAME="ReelToReel_Dataset_MiniPulse100_CHOWTAPE_WOWFLUTTER" # EXP1b
else # REAL
  DATASET_NAME="ReelToReel_Dataset_MiniPulse100_AKAI_IPS[$IPS]_$TAPE"
fi
declare -a ARR_SUBSET=("Train" "Val" "Test")

counter=0
for subset in "${ARR_SUBSET[@]}"; do
  python -u ../code/test-dataset.py\
  --DATASET $DATASET_NAME --SUBSET $subset --PRELOAD\
  --SEGMENT_LENGTH $SEGMENT_LENGTH --NO_SHUFFLE --IDX 0 --ZOOM 0.5\
  --PLOT_DELAY\
  --SAVE_FIG --DESCRIPTIVE_NAME $DESCRIPTIVE_NAME"_"$subset &
  
  ((counter+=1))
  if [ $counter -eq $(($N_PARALLEL)) ]; then
    echo "Waitin'.."
    wait
    ((counter=0))
  fi
done

# # LogSweepsContinuousPulse100
# DATASET_NAME="LogSweepsContinuousPulse100_AKAI_IPS[$IPS]_$TAPE"
# python -u ../code/test-dataset.py --DATASET $DATASET_NAME --SYNC 1.0 --SEGMENT_LENGTH 264600 --PLOT_DELAY --NO_SHUFFLE --IDX 0 --ZOOM 0.5 --SAVE_FIG --DESCRIPTIVE_NAME $DESCRIPTIVE_NAME &
# wait
# 
# # SinesFadedShortContinuousPulse100
# DATASET_NAME="SinesFadedShortContinuousPulse100_AKAI_IPS[$IPS]_$TAPE"
# python -u ../code/test-dataset.py --DATASET $DATASET_NAME --SYNC 1.0 --SEGMENT_LENGTH 48510 --PLOT_DELAY --NO_SHUFFLE --IDX 0 --ZOOM 0.5 --SAVE_FIG --DESCRIPTIVE_NAME $DESCRIPTIVE_NAME
