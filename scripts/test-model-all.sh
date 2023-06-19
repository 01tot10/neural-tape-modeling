#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --output=sbatch-loss_%j.out

# LOAD ENVIRONMENT IF IN TRITON
if command -v module &> /dev/null; then
    module load miniconda
    source activate neural-tape
fi

# SETTINGS
declare -a ARR_MODELS=(
    # "GRU"
    "DiffDelGRU"
)
declare -a ARR_LOSSES=(
    # "ESR"
    # "DCPreESR"
    "LogSpec"
)

for CONFIG_ID in $(seq 1 1); do
    case $CONFIG_ID in
        0) EXPERIMENT="TOY"
            TAPE="EMPTY"
            IPS="EMPTY"
            ;;
        1) EXPERIMENT="REAL"
            TAPE="MAXELL"
            IPS="7.5"
            ;;
        2) EXPERIMENT="REAL"
            TAPE="SCOTCH"
            IPS="3.75"
            ;;
    esac

    for MODEL in "${ARR_MODELS[@]}"; do
        for LOSS in "${ARR_LOSSES[@]}"; do

            echo
            echo "========================"
            echo "==      SETTINGS      =="
            echo "EXPERIMENT: $EXPERIMENT"
            echo "TAPE:       $TAPE"
            echo "IPS:        $IPS"
            echo "MODEL:      $MODEL"
            echo "LOSS:       $LOSS"
            echo "========================"

            # Loss
            ./test-model-loss.sh $EXPERIMENT $TAPE $IPS $MODEL $LOSS

            # Predictions
            # ./test-model-prediction.sh $EXPERIMENT $TAPE $IPS $MODEL $LOSS
            
            # Noise
            # ./test-model-noise.sh $EXPERIMENT $TAPE $IPS $MODEL $LOSS

            # Hysteresis
            # ./test-model-hysteresis.sh $EXPERIMENT $TAPE $IPS $MODEL $LOSS

            # Sweep
            # ./test-model-sweep.sh $EXPERIMENT $TAPE $IPS $MODEL $LOSS
        done
    done
done