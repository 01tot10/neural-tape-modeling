#!/bin/bash
: '
Calls test-dataset.py
'
# ARGUMENTS
TAPE=$1
IPS=$2

# SETTINGS
SEGMENT_LENGTH=441000 # [n]
ZOOM=5.0              # [s]
N_PARALLEL=2

# Analyze
DATASET_NAME="ReelToReel_Dataset_MiniPulse100_AKAI_IPS[$IPS]_$TAPE"
DESCRIPTIVE_NAME="IO"

counter=0
for idx in $(seq 5); do
  python -u ../code/test-dataset.py --DATASET $DATASET_NAME --SEGMENT_LENGTH $SEGMENT_LENGTH --PLOT_DELAY --ZOOM $ZOOM --SAVE_AUDIO --DESCRIPTIVE_NAME $DESCRIPTIVE_NAME"_""$idx" &

  ((counter+=1))
  if [ $counter -eq $(($N_PARALLEL)) ]; then
    echo "Waitin'.."
    wait
    ((counter=0))
  fi
done
wait