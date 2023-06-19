#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --output=sbatch-analysis-tape_%j.out

## Analyze VA reel-to-reel tape performace.
## Calls analysis-tape.sh

module load miniconda
source activate neural-tape

# Process
./analysis-tape.sh