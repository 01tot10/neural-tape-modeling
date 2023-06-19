#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --output=sbatch-sweep_%j.out

# ARGUMENTS
EXPERIMENT=$1
TAPE=$2
IPS=$3
MODEL=$4
LOSS=$5

# LOAD ENVIRONMENT IF IN TRITON
if command -v module &> /dev/null; then
    module load miniconda
    source activate neural-tape
fi

# SETTINGS
DESCRIPTIVE_NAME="SWEEP"
N_PARALLEL=1

if [ $EXPERIMENT == "TOY" ]; then # CHOWTAPE
    SYNC=0.0
    # DATASET_NAME="LogSweepsContinuousPulse100_CHOWTAPE" # EXP-1
    DATASET_NAME="LogSweepsContinuousPulse100_CHOWTAPE_WOWFLUTTER" # EXP-2
    declare -a ARR_WEIGHTS=(
        "$MODEL-HS[64]-L[$LOSS]-DS[ReelToReel_Dataset_MiniPulse100_CHOWTAPE_WOWFLUTTER]"
    )
else # REAL
    SYNC=1.0
    DATASET_NAME="LogSweepsContinuousPulse100_AKAI_IPS[$IPS]_$TAPE" # REAL
    declare -a ARR_WEIGHTS=(
        # "$MODEL-HS[64]-L[$LOSS]-DS[ReelToReel_Dataset_MiniPulse100_AKAI_IPS[$IPS]_$TAPE]"
        "$MODEL-HS[64]-L[$LOSS]-DS[ReelToReel_Dataset_MiniPulse100_AKAI_IPS[$IPS]_$TAPE]_BEST"
        # "$MODEL-HS[64]-L[$LOSS]-DS[ReelToReel_Dataset_MiniPulse100_AKAI_IPS[$IPS]_$TAPE]_1"
        # "$MODEL-HS[64]-L[$LOSS]-DS[ReelToReel_Dataset_MiniPulse100_AKAI_IPS[$IPS]_$TAPE]_2"
        # "$MODEL-HS[64]-L[$LOSS]-DS[ReelToReel_Dataset_MiniPulse100_AKAI_IPS[$IPS]_$TAPE]_3"
    )
fi

SEGMENT_LENGTH=1367100 # NEW
ZOOM=30.0
START_IDX=8
END_IDX=8
# SEGMENT_LENGTH=264600 # OLD
# ZOOM=5.0
# START_IDX=5
# END_IDX=14

# ANALYZE
counter=0
for weight in "${ARR_WEIGHTS[@]}"; do
    for idx in $(seq $START_IDX $END_IDX); do
        python -u ../code/test-model.py \
            --MODEL $MODEL --WEIGHTS $weight \
            --DATASET $DATASET_NAME \
            --SYNC $SYNC --SEGMENT_LENGTH $SEGMENT_LENGTH --ZOOM $ZOOM --NO_SHUFFLE --IDX $idx \
            --DEMODULATE \
            --PLOT_SWEEP --SAVE_FIG \
            --DESCRIPTIVE_NAME $DESCRIPTIVE_NAME"_""$((idx))" &

        ((counter+=1))
        if [ $counter -eq $(($N_PARALLEL)) ]; then
            echo "Waitin'.."
            wait
            ((counter=0))
        fi
    done
done
