#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --output=sbatch-loss_%j.out

# SETTINGS
EXPERIMENT=$1
TAPE=$2
IPS=$3
MODEL=$4
LOSS=$5

# LOAD ENVIRONMENT IF IN TRITON
# if command -v module &> /dev/null; then
#   module load miniconda
#   source activate neural-tape
# fi

DESCRIPTIVE_NAME="LOSS"
SEGMENT_LENGTH=$((44100*10))
declare -a ARR_SUBSETS=(
    # "Val"
    "Test"
)

if [ $EXPERIMENT == "TOY" ]; then # CHOWTAPE
    # declare -a ARR_WEIGHTS=(
    # "$MODEL-HS[64]-L[$LOSS]-DS[ReelToReel_Dataset_MiniPulse100_CHOWTAPE]_1"
    # "$MODEL-HS[64]-L[$LOSS]-DS[ReelToReel_Dataset_MiniPulse100_CHOWTAPE]_2"
    # "$MODEL-HS[64]-L[$LOSS]-DS[ReelToReel_Dataset_MiniPulse100_CHOWTAPE]_3"
    # ) # EXP1
    declare -a ARR_WEIGHTS=(
        "$MODEL-HS[64]-L[$LOSS]-DS[ReelToReel_Dataset_MiniPulse100_CHOWTAPE_WOWFLUTTER]_BEST"
        # "$MODEL-HS[64]-L[$LOSS]-DS[ReelToReel_Dataset_MiniPulse100_CHOWTAPE_WOWFLUTTER]_2"
        # "$MODEL-HS[64]-L[$LOSS]-DS[ReelToReel_Dataset_MiniPulse100_CHOWTAPE_WOWFLUTTER]_3"
    ) # EXP2
else # REAL
    DATASET_NAME="ReelToReel_Dataset_MiniPulse100_AKAI_IPS[$IPS]_$TAPE" # REAL
    declare -a ARR_WEIGHTS=(
        "$MODEL-HS[64]-L[$LOSS]-DS[ReelToReel_Dataset_MiniPulse100_AKAI_IPS[$IPS]_$TAPE]_BEST"
        # "$MODEL-HS[64]-L[$LOSS]-DS[ReelToReel_Dataset_MiniPulse100_AKAI_IPS[$IPS]_$TAPE]_1"
        # "$MODEL-HS[64]-L[$LOSS]-DS[ReelToReel_Dataset_MiniPulse100_AKAI_IPS[$IPS]_$TAPE]_2"
        # "$MODEL-HS[64]-L[$LOSS]-DS[ReelToReel_Dataset_MiniPulse100_AKAI_IPS[$IPS]_$TAPE]_3"
    )
    DATASET_NOISE="Silence_AKAI_IPS[$IPS]_$TAPE"
fi

# ANALYZE
for weight in "${ARR_WEIGHTS[@]}"; do
    for subset in "${ARR_SUBSETS[@]}"; do

        if [ $EXPERIMENT == "TOY" ]; then # CHOWTAPE
            DATASET_NAME="ReelToReel_Dataset_MiniPulse100_CHOWTAPE_WOWFLUTTER" # EXP 2
            # Delayed
            python -u ../code/test-model.py \
                --MODEL $MODEL --WEIGHTS $weight \
                --DATASET $DATASET_NAME --SUBSET $subset --NO_SHUFFLE --SEGMENT_LENGTH $SEGMENT_LENGTH\
                --ADD_DELAY \
                --COMPUTE_LOSS \
                --SAVE_AUDIO \
                --DESCRIPTIVE_NAME $DESCRIPTIVE_NAME

            if [ $MODEL == "GRU" ]; then
                # Demodulated
                python -u ../code/test-model.py \
                    --MODEL $MODEL --WEIGHTS $weight \
                    --DATASET $DATASET_NAME --SUBSET $subset --NO_SHUFFLE --SEGMENT_LENGTH $SEGMENT_LENGTH\
                    --DEMODULATE \
                    --COMPUTE_LOSS \
                    --SAVE_AUDIO \
                    --DESCRIPTIVE_NAME $DESCRIPTIVE_NAME

                # DATASET_NAME="ReelToReel_Dataset_MiniPulse100_CHOWTAPE" # EXP 1
                # python -u ../code/test-model.py \
                    # --MODEL $MODEL --WEIGHTS $weight \
                    # --DATASET $DATASET_NAME --SUBSET $subset --NO_SHUFFLE --SEGMENT_LENGTH $SEGMENT_LENGTH\
                    # --COMPUTE_LOSS \
                    # --SAVE_AUDIO \
                    # --DESCRIPTIVE_NAME $DESCRIPTIVE_NAME
            fi

        else # REAL

            # Delayed + No Noise
            python -u ../code/test-model.py \
                --MODEL $MODEL --WEIGHTS $weight \
                --DATASET $DATASET_NAME --SUBSET $subset --NO_SHUFFLE --SEGMENT_LENGTH $SEGMENT_LENGTH\
                --ADD_DELAY \
                --COMPUTE_LOSS \
                --SAVE_AUDIO \
                --DESCRIPTIVE_NAME $DESCRIPTIVE_NAME

            # Delayed + Noised
            # python -u ../code/test-model.py \
                # --MODEL $MODEL --WEIGHTS $weight \
                # --DATASET $DATASET_NAME --SUBSET $subset --NO_SHUFFLE --SEGMENT_LENGTH $SEGMENT_LENGTH\
                # --DATASET_NOISE $DATASET_NOISE\
                # --ADD_DELAY --ADD_NOISE \
                # --COMPUTE_LOSS \
                # --SAVE_AUDIO \
                # --DESCRIPTIVE_NAME $DESCRIPTIVE_NAME

            if [ $MODEL == "GRU" ]; then
                # Demodulated + No Noise
                python -u ../code/test-model.py \
                    --MODEL $MODEL --WEIGHTS $weight \
                    --DATASET $DATASET_NAME --SUBSET $subset --NO_SHUFFLE --SEGMENT_LENGTH $SEGMENT_LENGTH\
                    --DEMODULATE \
                    --COMPUTE_LOSS \
                    --SAVE_AUDIO \
                    --DESCRIPTIVE_NAME $DESCRIPTIVE_NAME

                # Demodulated + Noised
                # python -u ../code/test-model.py \
                    # --MODEL $MODEL --WEIGHTS $weight \
                    # --DATASET $DATASET_NAME --SUBSET $subset --NO_SHUFFLE --SEGMENT_LENGTH $SEGMENT_LENGTH\
                    # --DATASET_NOISE $DATASET_NOISE\
                    # --DEMODULATE --ADD_NOISE \
                    # --COMPUTE_LOSS \
                    # --SAVE_AUDIO \
                    # --DESCRIPTIVE_NAME $DESCRIPTIVE_NAME
            fi
        fi
    done
done