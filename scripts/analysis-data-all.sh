#!/bin/bash
: '
Calls:
- analysis-data-pulse_recovery.sh
- analysis-data-delay_trajectories.sh
- analysis-data-io.sh
- analysis-data-sweeps.sh
- analysis-data-transfer.sh
'
# SETTINGS
TAPE=$1
IPS=$2

# Pulse recovery
./analysis-data-pulse_recovery.sh $TAPE $IPS

# Delay trajectories
./analysis-data-delay_trajectories.sh $TAPE $IPS

# Input / Output examples
./analysis-data-io.sh $TAPE $IPS

# Sweeps
./analysis-data-sweeps.sh $TAPE $IPS

# Hysteresis
./analysis-data-hysteresis.sh $TAPE $IPS
