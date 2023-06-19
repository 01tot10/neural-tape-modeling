#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 11:20:41 2023

@author: 01tot10
"""

#%% Imports

import argparse
import os
import sys

import numpy as np
import scipy
import soundfile as sf

#%% Argument parser

# Add argument parser
parser = argparse.ArgumentParser(description='Generate input files.')

# GLOBAL
parser.add_argument('--DRY_RUN', action='store_true', default=False)
parser.add_argument('--DESCRIPTIVE_NAME', type=str, default=None)

# SIGNALS
parser.add_argument('--FRAME_RATE', type=int, default=int(44100))
parser.add_argument('--N_SIGNALS', type=int, default=int(100))
parser.add_argument('--SEGMENT_LENGTH', type=float, default=float(1 * 5))
parser.add_argument('--TYPE', type=str, default="sine")
parser.add_argument('--FREQUENCY', type=float, default=None)
parser.add_argument('--AMPLITUDE', type=float, default=None)
parser.add_argument('--ENABLE_ENVELOPE', action='store_true', default=False)

args = parser.parse_args()

print("\nArguments:")
print(args)

assert args.TYPE.lower() in ["sine", "pulse", "sweep"
                            ], "Choose either sine, pulse or sweep as TYPE"

#%% Config

# Global
RESULTS_PATH = '../results/'
DRY_RUN = args.DRY_RUN

# Dataset
fs = args.FRAME_RATE
N_SIGNALS = args.N_SIGNALS
SEGMENT_LENGTH = args.SEGMENT_LENGTH
DATASET_NAME = 'Signals' # results subdirectory
DESCRIPTIVE_NAME = args.DESCRIPTIVE_NAME

# SIGNALS
TYPE = args.TYPE
AMPLITUDE = args.AMPLITUDE
FREQUENCY = args.FREQUENCY
F_MIN = 1
F_MAX = 20000
ENABLE_ENVELOPE = args.ENABLE_ENVELOPE

#%% Setup

rng = np.random.default_rng()

#%% Process

# Initializations
t = np.arange(0, int(SEGMENT_LENGTH * fs)) / fs

amplitude_env = (
    0.5 +
    0.5 * scipy.signal.sawtooth(2 * np.pi * 1 / SEGMENT_LENGTH * t, width=0.5))
if isinstance(AMPLITUDE, str):
    START_AMPLITUDE_DB = -54
    END_AMPLITUDE_DB = 0
    amplitude_increment_dB = (END_AMPLITUDE_DB -
                              START_AMPLITUDE_DB) / (N_SIGNALS - 1)

# Results Path
if DESCRIPTIVE_NAME:
    DATASET_NAME = f"{DATASET_NAME}_{DESCRIPTIVE_NAME}"
results_path = os.path.join(RESULTS_PATH, DATASET_NAME, "Train")
if os.path.exists(results_path):
    print(f"Path {results_path} exists ...")
else:
    os.makedirs(results_path)

for signal_idx in range(N_SIGNALS):
    sys.stdout.write(f"Generating {signal_idx+1}/{N_SIGNALS}...")
    sys.stdout.flush()

    # Get signal properties
    if isinstance(AMPLITUDE, float):
        amplitude = AMPLITUDE
    elif AMPLITUDE is None:
        amplitude = rng.random()
    else:
        amplitude_dB = START_AMPLITUDE_DB + signal_idx * amplitude_increment_dB
        amplitude = 10**(amplitude_dB / 20)
        print(
            f"signal_idx: {signal_idx}, amplitude(lin): {amplitude}, amplitude(dB): {20*np.log10(amplitude)}"
        )

    if FREQUENCY:
        frequency = FREQUENCY
    else:
        frequency = F_MIN + (scipy.stats.loguniform.rvs(
            1e-3, 1.0, size=1)[0]) * (F_MAX - F_MIN)

    # Generate signal
    if TYPE.lower() == "sine":
        signal = amplitude * np.sin(2 * np.pi * frequency * t)
    elif TYPE.lower() == "pulse":
        period_n = int((1.0 / frequency) * fs)
        signal = np.zeros_like(t)
        for n in range(len(signal)):
            if (n % period_n == 0) and n > 0:
                signal[n] = 1.0
    else:
        signal = amplitude * scipy.signal.chirp(t,
                                                f0=F_MIN,
                                                f1=F_MAX,
                                                t1=SEGMENT_LENGTH,
                                                method='logarithmic',
                                                phi=-90)

    if ENABLE_ENVELOPE:
        signal *= amplitude_env
        signal = signal / np.max(signal) * amplitude # retain amplitude

    # Save
    if not DRY_RUN:
        sys.stdout.write(" Saving...")

        filename = f"input_{signal_idx}"
        filename = f"{filename}_{TYPE.lower()}"
        filename = f"{filename}_F[{'{:d}'.format(int(frequency))}]"
        # filename = f"{filename}_A[{('{:.3f}'.format(amplitude)).replace('.','_')}]" # linear
        filename = f"{filename}_A[{int(20*np.log10(amplitude))}dB]" # log

        filepath = os.path.join(results_path, f"{filename}.wav")

        sf.write(filepath, signal, fs, subtype='FLOAT')

    sys.stdout.write(" Done! \n")
    sys.stdout.flush()
