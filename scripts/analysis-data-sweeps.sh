#!/bin/bash
: '
Calls test-dataset.py
'
# ARGUMENTS
TAPE=$1
IPS=$2

# SETTINGS
DESCRIPTIVE_NAME="SWEEP"
N_PARALLEL=4
SEGMENT_LENGTH=1367100
ZOOM=30.0
START_IDX=0
END_IDX=9

DATASET_NAME="LogSweepsContinuousPulse100_AKAI_IPS[$IPS]_$TAPE"  # REAL
SYNC=1.0

# DATASET_NAME="LogSweepsContinuousPulse100_CHOWTAPE" # CHOWTAPE
# SYNC=0.0

## ANALYZE
counter=0
for idx in $(seq $START_IDX $END_IDX); do
  python -u ../code/test-dataset.py \
  --DATASET $DATASET_NAME --SEGMENT_LENGTH $SEGMENT_LENGTH --PRELOAD \
  --SYNC $SYNC --ZOOM $ZOOM --NO_SHUFFLE --IDX $idx \
  --DEMODULATE \
  --PLOT_SWEEP --DESCRIPTIVE_NAME $DESCRIPTIVE_NAME"_""$((idx-START_IDX))" --SAVE_FIG &
  
  ((counter+=1))
  if [ $counter -eq $(($N_PARALLEL)) ]; then
    echo "Waitin'.."
    wait
    ((counter=0))
  fi
done
wait
