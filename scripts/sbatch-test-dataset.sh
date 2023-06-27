#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --output=sbatch-test-dataset_%j.out

module load miniconda
source activate neural-tape

# SETTINGS
DATASET="ReelToReel_Dataset_Mini192kHzPulse100_AKAI_IPS[3.75]_SCOTCH"
declare -a ARR_SUBSET=("Train" "Val" "Test")

# PROCESS
for SUBSET in "${ARR_SUBSET[@]}"; do
    python -u ../code/test-dataset.py --DATASET $DATASET --SUBSET $SUBSET --PRELOAD
done