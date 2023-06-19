#!/bin/bash
: '
Calls analysis-playback-head.py
'

# SETTINGS
declare -a arr_tape_v=(1.875 3.75 7.5 15 30)
declare -a arr_tape_delta=(8.75 17.5 35 70 140)
declare -a arr_play_g=(5 10 20 40 80)
declare -a arr_play_d=(1.5 3 6 12 24)

# Choose array to iterate over
chosen=("${arr_tape_v[@]}")

for item in "${chosen[@]}"; do
  processed=$(echo $item*0.0254|bc)
  # processed=$(echo $item*$(printf '%.16f' 1e-6)|bc)
  echo $processed
  python -u ../code/analysis-playback-head.py --TAPE_V $processed --SAVE_FIGS &
done
