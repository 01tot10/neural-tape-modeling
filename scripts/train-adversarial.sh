#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --output=sbatch-train_adversarial_%j.out

# LOAD ENVIRONMENT IF IN TRITON
if command -v module &> /dev/null; then
    module load miniconda
    source activate neural-tape
fi

# SETTINGS
# NADA

# Run
python -u ../code/trainAdversarial.py