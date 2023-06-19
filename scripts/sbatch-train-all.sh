#!/bin/bash
# SETTINGS

# EXP 1a - CHOWTAPE, No Delay
sbatch sbatch-train-exp1a.sh

# EXP 1a - CHOWTAPE, Delay
sbatch sbatch-train-exp1b.sh

# EXP 1a - REAL DATA
sbatch sbatch-train-exp2.sh
