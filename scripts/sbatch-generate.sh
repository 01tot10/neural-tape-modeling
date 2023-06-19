#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --output=sbatch-generate_%j.out

## Generate dataset of input-output pairs using VA reel-to-reel tape.
## Calls generate.sh

module load miniconda
source activate neural-tape

# SETTINGS
DATASET="SinesFaded"
FRACTION=0.3
SIGNAL_AMPLITUDE=1e-3
declare -i SEGMENT_LENGTH=$((1*5*44100))

# PROCESS
signal_amplitude_converted=$(printf '%.6f' $SIGNAL_AMPLITUDE)
printf "\nSignal amplitude: $SIGNAL_AMPLITUDE = $signal_amplitude_converted\n"

BIAS_AMPLITUDE=$(echo 5*$signal_amplitude_converted|bc)
printf "Bias amplitude: $BIAS_AMPLITUDE\n"

./generate.sh $DATASET $FRACTION $SEGMENT_LENGTH $SIGNAL_AMPLITUDE $BIAS_AMPLITUDE
