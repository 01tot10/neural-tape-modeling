#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --output=sbatch-train_%j.out

# LOAD ENVIRONMENT IF IN TRITON
if command -v module &> /dev/null; then
    module load miniconda
    source activate neural-tape
fi

# SETTINGS
DATASET="ReelToReel_Dataset_MiniPulse100_CHOWTAPE_WOWFLUTTER"
LOSS="ESR"
MODEL="DiffDelGRU"
HIDDEN_SIZE=64
N_EPOCHS=200
DESCRIPTIVE_NAME="TEMP"

# Run
python -u ../code/train.py\
    --MODEL $MODEL --HIDDEN_SIZE $HIDDEN_SIZE\
    --LOSS $LOSS --N_EPOCHS $N_EPOCHS\
    --DATASET $DATASET --SEGMENT_LENGTH 441000 --PRELOAD\
    --DESCRIPTIVE_NAME $DESCRIPTIVE_NAME\
    --DRY_RUN