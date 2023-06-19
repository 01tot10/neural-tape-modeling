#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --output=sbatch-train_%j.out

## Train models in Triton.
## Calls train.sh

module load miniconda
source activate neural-tape

# SETTINGS
HIDDEN_SIZE=32
DATASET="ReelToReel_Dataset_MiniPulse100_CHOWTAPE_WOWFLUTTER"
DATASET_VAL="ReelToReel_Dataset_Mini_CHOWTAPE"
N_EPOCHS=200
DESCRIPTIVE_NAME="EXP5"

./train.sh $HIDDEN_SIZE $DATASET $DATASET_VAL $N_EPOCHS $DESCRIPTIVE_NAME
