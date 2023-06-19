#!/bin/bash
: '
Calls analysis-tape.py
'

# SETTINGS
declare -a arr_signal_amplitudes=(1e-4 1e-3 1e-2)
T_MAX=0.1 #s

bias_amplitude=0
for signal_amplitude in "${arr_signal_amplitudes[@]}"; do
  signal_amplitude_converted=$(printf '%.6f' $signal_amplitude)
  printf "\nSignal amplitude: $signal_amplitude = $signal_amplitude_converted\n"
  bias_amplitude=$(echo 5*$signal_amplitude_converted|bc)
  printf "Bias amplitude: $bias_amplitude\n"
  
  python -u ../code/analysis-tape.py --T_MAX $T_MAX --SIGNAL_AMPLITUDE $signal_amplitude --BIAS_AMPLITUDE $bias_amplitude --BIAS_ENABLE --RETURN_INTERNAL --PLOT_FFT --SAVE_FIGS --SAVE_AUDIO &
done
