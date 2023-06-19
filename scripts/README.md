# Scripts

This folders contains scripts for running various processing pipelines.
For some of the scripts, an option exists for running them in a computing cluster with [slurm](https://slurm.schedmd.com/documentation.html).

- [SCRIPTS](#scripts)
  - [TAPE SIMULATIONS](#tape-simulations)
  - [DATA GENERATION](#data-generation)
  - [DATA VISUALIZATION / ANALYSIS](#data-visualization--analysis)
  - [MODEL TRAINING](#model-training)
  - [MODEL EVALUATION](#model-evaluation)

## TAPE SIMULATIONS

Simulate tape:
```
> simulate-tape.sh               # local
> sbatch sbatch-simulate-tape.sh # slurm
```

Simulate tape playback:
```
> simulate-playback.sh
```

## DATA GENERATION

Generate target outputs using the VST effect processor:
```
> ./generate-targets.sh [+ OPTIONS] # local
> sbatch sbatch-generate.sh         # slurm
```

Pre-compute delay trajectories:
```
> sbatch sbatch-test-dataset.sh
```

Save delay trajectories as `.wav`-files for training the delay generator:
```
> ./generate-trajectories.sh
```

## DATA VISUALIZATION / ANALYSIS

Generate visualizations and auralizations of data in dataset.

All:
```
> ./analysis-data-all.sh [TAPE] [IPS]
```

Pulse recovery:
```
> ./analysis-data-pulse_recovery.sh [TAPE] [IPS]
```

Delay trajectories:
```
> ./analysis-data-delay_trajectories.sh [TAPE] [IPS]
```

Input / Output examples:
```
> ./analysis-data-io.sh [TAPE] [IPS]
```

Magnitude response analysis:
```
> ./analysis-data-sweep.sh [TAPE] [IPS]
```

Hysteresis:
```
./analysis-data-hysteresis.sh [TAPE] [IPS]
```

## MODEL TRAINING

Train the nonlinear model for different experiments.

Generic:
```
> sbatch sbatch-train.sh # slurm
> ./train.sh             # local
```

All:
```
> sbatch sbatch-train-all.sh
```

Experiment 1a:
```
> sbatch sbatch-train-exp1a.sh
```

Experiment 1b:
```
> sbatch sbatch-train-exp1b.sh
```

Experiment 2:
```
> sbatch sbatch-train-exp2.sh
```

## MODEL EVALUATION

Test model performance under different test conditions.

All:
```
> ./test-model-all.sh
```

Hysteresis:
```
> ./test-model-hysteresis.sh [+ OPTIONS]
```

Magnitude response:
```
> ./test-model-sweep.sh [+ OPTIONS]
```

Predictions:
```
> ./test-model-predictions.sh [+ OPTIONS]
```

Losses:
```
> ./test-model-loss.sh [+ OPTIONS]
```

Noise:
```
> ./test-model-noise.sh [+ OPTIONS]
```
