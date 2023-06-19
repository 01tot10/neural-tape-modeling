#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --output=sbatch-test-dataset_%j.out

module load miniconda
source activate neural-tape

# SETTINGS
DATASET="ReelToReel_Dataset_MiniPulse100_CHOWTAPE"
SUBSET="Train"

# PROCESS
python -u ../code/test-dataset.py --DATASET $DATASET --SUBSET $SUBSET --PRELOAD