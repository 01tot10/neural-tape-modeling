#!/bin/bash
: '
Calls test-dataset.py
'
# ARGUMENTS
TAPE=$1
IPS=$2

# SETTINGS
DESCRIPTIVE_NAME="TRANSFER"
declare -a ARR_INDICES=(13 63 98) # cherry-picked indices
DATASET_NAME="SinesFadedShortContinuousPulse100_AKAI_IPS[$IPS]_$TAPE"
N_PARALLEL=2

## ANALYZE
counter=0
for idx in "${ARR_INDICES[@]}"; do
  python -u ../code/test-dataset.py \
   --DATASET $DATASET_NAME --PRELOAD \
   --SYNC 1.0 --SEGMENT_LENGTH 48510 --ZOOM 0.1 --NO_SHUFFLE --IDX $idx \
   --DEMODULATE --PLOT_TRANSFER \
   --SAVE_FIG --DESCRIPTIVE_NAME $DESCRIPTIVE_NAME"_""$idx" &
  
  ((counter+=1))
  if [ $counter -eq $(($N_PARALLEL)) ]; then
    echo "Waitin'.."
    wait
    ((counter=0))
  fi
done
wait