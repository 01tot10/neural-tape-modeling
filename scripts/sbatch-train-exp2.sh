#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --output=sbatch-train_%A_%a.out
#SBATCH --array=0-2

## Train models in Triton.

case $SLURM_ARRAY_TASK_ID in
   0) N_MODEL=1  ;;
   1) N_MODEL=2  ;;
   2) N_MODEL=3  ;;
esac

module load miniconda
source activate neural-tape

# SETTINGS
DATASET="ReelToReel_Dataset_MiniPulse100_AKAI_IPS[3.75]_SCOTCH"
LOSS="DCPreESR"
MODEL="DiffDelGRU"
HIDDEN_SIZE=64
N_EPOCHS=200
FRACTION_VAL=0.25
DESCRIPTIVE_NAME="EXP3""_""$MODEL""_""$LOSS""_""$N_MODEL"

python -u ../code/train.py\
 --MODEL $MODEL --HIDDEN_SIZE $HIDDEN_SIZE\
 --LOSS $LOSS --N_EPOCHS $N_EPOCHS\
 --DATASET $DATASET --SEGMENT_LENGTH 441000 --PRELOAD\
 --FRACTION_VAL $FRACTION_VAL\
 --DESCRIPTIVE_NAME $DESCRIPTIVE_NAME