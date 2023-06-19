#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 14:31:58 2023

@author: 01tot10
"""

#%% Imports

import argparse
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from dataset import VADataset
from tape import Tape, VSTWrapper

print("=" * 10, " SCRIPT START ", "=" * 10)

#%% Argument parser


def none_or_int(value):
    """ Parse NoneType or int input arguments from CLI """
    if value == 'None':
        return None
    return int(value)


# Add argument parser
parser = argparse.ArgumentParser(
    description='Generate target files using VA reel-to-reel tape recorder.')

# GLOBAL
parser.add_argument('--DRY_RUN', action='store_true', default=False)

# DATASET
parser.add_argument('--DATASET', type=str, default="ReelToReel_Dataset_Mini")
parser.add_argument('--SUBSET', type=str, default="Train")
parser.add_argument('--DESCRIPTIVE_NAME', type=str, default=None)
parser.add_argument('--FRACTION', type=float, default=1.0)
parser.add_argument('--SEGMENT_LENGTH', type=none_or_int,
                    default=None)                         # in samples
parser.add_argument('--PRELOAD', action='store_true', default=True)
parser.add_argument('--NO_SHUFFLE', action='store_true', default=False)

## TAPE
parser.add_argument('--BACKEND', type=str, default="VST")

# VST PARAMS
parser.add_argument('--TAPE_DISABLE', action='store_true', default=False)
parser.add_argument('--TAPE_SATURATION', type=float, default=0.5)
parser.add_argument('--TAPE_DRIVE', type=float, default=0.5)
parser.add_argument('--TAPE_BIAS', type=float, default=0.5)

# DSP PARAMS
parser.add_argument('--BIAS_ENABLE', action='store_true', default=True)
parser.add_argument('--DELAY_ENABLE', action='store_true', default=False)
parser.add_argument('--WOW_FLUTTER_ENABLE', action='store_true', default=False)
parser.add_argument('--PLAYBACK_LOSS_ENABLE',
                    action='store_true',
                    default=False)
parser.add_argument('--STARTUP_ENABLE', action='store_true', default=False)
parser.add_argument('--SIGNAL_AMPLITUDE', type=float, default=10e-3)
parser.add_argument('--BIAS_AMPLITUDE', type=float, default=50e-3)

args = parser.parse_args()

print("\nArguments:")
print(args)

assert args.BACKEND in ["VST",
                        "Python"], "Choose either VST or Python for Backend"

#%% Config

script_path = os.path.dirname(__file__)

# global
fs = int(44.1e3)
BATCH_SIZE = 1
DRY_RUN = args.DRY_RUN

# results
RESULTS_PATH = '../results/'
DESCRIPTIVE_NAME = args.DESCRIPTIVE_NAME

## tape
BACKEND = args.BACKEND
OVERSAMPLING = 16
DELAY_ENABLE = args.DELAY_ENABLE

# VST
VST_PATH = os.path.join(
    script_path,
    'AnalogTapeModel/Plugin/build/CHOWTapeModel_artefacts/Release/VST3/CHOWTapeModel.vst3'
)

TAPE_DISABLE = args.TAPE_DISABLE
TAPE_SATURATION = args.TAPE_SATURATION
TAPE_BIAS = args.TAPE_BIAS
TAPE_DRIVE = args.TAPE_DRIVE

WOW_FLUTTER_ENABLE = args.WOW_FLUTTER_ENABLE

# DSP
BIAS_ENABLE = args.BIAS_ENABLE

PLAYBACK_LOSS_ENABLE = args.PLAYBACK_LOSS_ENABLE
STARTUP_ENABLE = args.STARTUP_ENABLE
SIGNAL_AMPLITUDE = args.SIGNAL_AMPLITUDE
BIAS_AMPLITUDE = args.BIAS_AMPLITUDE

# dataset
AUDIO_PATH = "../audio/"
PRELOAD = args.PRELOAD
SHUFFLE = not args.NO_SHUFFLE
DATASET = args.DATASET
SUBSET = args.SUBSET
INPUT_ONLY = True
FRACTION = args.FRACTION
SEGMENT_LENGTH = args.SEGMENT_LENGTH

#%% Setup

dataset_path = os.path.join(AUDIO_PATH, DATASET)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\nUsing device: {device}")
print(f"Batch size: {BATCH_SIZE}")

print(f"\ninitializing DSP... Using backend {BACKEND}")
if BACKEND == "VST":
    tape = VSTWrapper(VST_PATH,
                      fs,
                      oversampling=OVERSAMPLING,
                      wow_flutter_enable=WOW_FLUTTER_ENABLE,
                      tape_enable=not TAPE_DISABLE,
                      tape_drive=TAPE_DRIVE,
                      tape_saturation=TAPE_SATURATION,
                      tape_bias=TAPE_BIAS)
else:
    tape = Tape(batch_size=BATCH_SIZE,
                fs=fs,
                oversampling=OVERSAMPLING,
                signal_amplitude=SIGNAL_AMPLITUDE,
                bias_amplitude=BIAS_AMPLITUDE,
                bias_enable=BIAS_ENABLE,
                delay_enable=DELAY_ENABLE,
                playback_loss_enable=PLAYBACK_LOSS_ENABLE,
                startup_enable=STARTUP_ENABLE)

# Data
print("initializing dataset...\n")
dataset = VADataset(dataset_path,
                    subset=SUBSET,
                    double=True,
                    length=SEGMENT_LENGTH,
                    fraction=FRACTION,
                    input_only=INPUT_ONLY,
                    shuffle=SHUFFLE,
                    preload=PRELOAD)
assert dataset.fs == tape.fs, "Model and dataset sampling rates don't match!"

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

#%% Process

# Results path
dataset_name = os.path.basename(dataset.data_dir)
if DESCRIPTIVE_NAME:
    dataset_name = f"{dataset_name}_{DESCRIPTIVE_NAME}"
results_path = os.path.join(RESULTS_PATH, dataset_name,
                            dataset.subset.capitalize())
if os.path.exists(results_path):
    print(f"Path {results_path} exists ...")
else:
    os.makedirs(results_path)

# Initializations
target_size = len(dataset) * dataset.length / fs
generated_size = 0.0
time_per_batch = 0.0
start = time.time()

print("\nProcessing start!")
for batch_idx, batch in enumerate(dataloader):
    sys.stdout.write(f"Processing batch {batch_idx+1} / {len(dataloader)}... ")

    # Retrieve batch
    input, meta = batch
    input = input.to(device)
    filename = meta['input_name']

    # Process
    target = tape(input)

    end = time.time()
    sys.stdout.write("Done!")

    # Stats
    generated_size += input.shape[0] * input.shape[-1] / fs
    time_per_batch = (end - start) / (batch_idx + 1)
    real_time_factor = (np.prod(input.shape) / fs) / time_per_batch
    batches_left = len(dataloader) - (batch_idx + 1)
    time_left = time_per_batch * batches_left

    sys.stdout.write(
        f" Average time/batch {'{:.3f}'.format(time_per_batch)} s, ")
    sys.stdout.write(
        f"Generated {'{:.3f}'.format(generated_size / 60)} min of {'{:.3f}'.format(target_size / 60)} min, "
    )
    sys.stdout.write(
        f"Time left ~ {'{:.3f}'.format(time_left / 60)} min (RT factor: {'{:.3f}'.format(real_time_factor)})"
    )
    sys.stdout.flush()

    if not DRY_RUN:
        # Save
        sys.stdout.write(" Saving results...")

        inputs, filenames, targets = input.numpy(), filename, target.numpy()
        for idx, (input, filename,
                  target) in enumerate(zip(inputs, filenames, targets)):
            inputname = f"input_{filename.split('_', 1)[1]}"
            filepath = os.path.join(results_path, inputname)
            sf.write(filepath, input.T, fs, subtype='FLOAT')

            targetname = f"target_{filename.split('_', 1)[1]}"
            filepath = os.path.join(results_path, targetname)
            sf.write(filepath, target.T, fs, subtype='FLOAT')
        sys.stdout.write(" Done!\n")
    else:
        sys.stdout.write("\n")
    sys.stdout.flush()

print("Processing finished!")

print("=" * 10, " SCRIPT END ", "=" * 10)
