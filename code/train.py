#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 11:40:30 2023

@author: 01tot10
"""

#%% Imports

import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import wandb
from Automated_GuitarAmpModelling.CoreAudioML.training import ESRLoss
from dataset import VADataset
from GreyBoxDRC.loss_funcs import ESRLoss as DCPreESR
from model import RNN, DiffDelRNN

print("=" * 10, " SCRIPT START ", "=" * 10)

#%% Argument parser

# Add argument parser
parser = argparse.ArgumentParser()

# GLOBAL
parser.add_argument('--DRY_RUN', action='store_true', default=False)
parser.add_argument('--DESCRIPTIVE_NAME', type=str, default=None)

# DATASET
parser.add_argument(
    '--DATASET',
    type=str,
    default="ReelToReel_Dataset_Mini__F[0.1]_SL[10]_TD[0.75]_TS[0.75]_TB[0.0]")
parser.add_argument('--DATASET_VAL', type=str, default=None)
parser.add_argument('--FRACTION', type=float, default=1.0)
parser.add_argument('--FRACTION_VAL', type=float, default=1.0)
parser.add_argument('--SEGMENT_LENGTH', type=int, default=None)
parser.add_argument('--PRELOAD', action='store_true', default=False)

# MODEL
parser.add_argument('--MODEL', type=str, default="GRU")
parser.add_argument('--HIDDEN_SIZE', type=int, default=16)

# TRAINING
parser.add_argument('--N_EPOCHS', type=int, default=1000)
parser.add_argument('--LOSS', type=str, default="DCPreESR")
parser.add_argument('--DEMODULATE', action='store_true', default=False)
parser.add_argument('--DEMODULATE_VAL', action='store_true', default=None)

args = parser.parse_args()

print("\nArguments:")
print(args)

assert args.LOSS in ["DCPreESR", "ESR"], "Chosen loss not supported!"
assert args.MODEL in ["GRU", "DiffDelGRU"], "Chosen loss not supported!"

#%% Config

# Global
DRY_RUN = args.DRY_RUN
DESCRIPTIVE_NAME = args.DESCRIPTIVE_NAME

# Data
AUDIO_PATH = "../audio/"
DATASET = args.DATASET
DATASET_VAL = args.DATASET_VAL
DEMODULATE = args.DEMODULATE
DEMODULATE_VAL = DEMODULATE if args.DEMODULATE_VAL is None else args.DEMODULATE_VAL
FRACTION = args.FRACTION
FRACTION_VAL = args.FRACTION_VAL
SEGMENT_LENGTH = args.SEGMENT_LENGTH
PRELOAD = args.PRELOAD

# Model
MODEL_PATH = "../weights/"
MODEL = args.MODEL
INPUT_SIZE = 1
OUTPUT_SIZE = 1
HIDDEN_SIZE = args.HIDDEN_SIZE
SKIP = False

# Training
LOSS = args.LOSS
N_EPOCHS = args.N_EPOCHS
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

#%% Setup

# Names and paths
dataset_path = os.path.join(AUDIO_PATH, DATASET)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = f"{MODEL}-HS[{HIDDEN_SIZE}]-L[{LOSS}]-DS[{DATASET}]"
if DESCRIPTIVE_NAME:
    model_name = f"{model_name}_{DESCRIPTIVE_NAME}"
model_path = os.path.join(MODEL_PATH, model_name)
model_running_path = os.path.join(model_path, "running.pth")
model_best_path = os.path.join(model_path, "best.pth")

# Logger
run_config = {
    "loss": LOSS,
    "model": MODEL,
    "model_name": model_name,
    "dataset": DATASET,
    "learning_rate": LEARNING_RATE,
    "epochs": N_EPOCHS,
    "batch_size": BATCH_SIZE,
    "hidden_size": HIDDEN_SIZE,
    "segment_length[n]": SEGMENT_LENGTH
}
wandb.init(project="neural-tape",
           name=DESCRIPTIVE_NAME if DESCRIPTIVE_NAME else None,
           mode="online" if not DRY_RUN else "disabled",
           group=DESCRIPTIVE_NAME,
           config=run_config)

# Data
print("\nDataset for training:")
dataset_train = VADataset(dataset_path,
                          fraction=FRACTION,
                          length=SEGMENT_LENGTH,
                          demodulate=DEMODULATE,
                          preload=PRELOAD)
dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE)
SEGMENT_LENGTH = dataset_train.length

print("\nDataset for validation:")
dataset_path = dataset_path if DATASET_VAL is None else os.path.join(
    AUDIO_PATH, DATASET_VAL)
dataset_val = VADataset(dataset_path,
                        subset="val",
                        length=SEGMENT_LENGTH,
                        preload=PRELOAD,
                        fraction=FRACTION_VAL,
                        demodulate=DEMODULATE_VAL,
                        shuffle=False)
dataloader_val = DataLoader(dataset_val, batch_size=6, shuffle=False)

assert dataset_train.fs == dataset_val.fs, "Sampling rates don't match"
fs = dataset_train.fs

# Model
if MODEL == "GRU":
    model = RNN(input_size=INPUT_SIZE,
                hidden_size=HIDDEN_SIZE,
                output_size=OUTPUT_SIZE,
                skip=SKIP).to(device)
elif MODEL == "DiffDelGRU":
    max_delay_n = int(1.25 * dataset_train.delay_analyzer.max_delay * fs)
    model = DiffDelRNN(input_size=INPUT_SIZE,
                       hidden_size=HIDDEN_SIZE,
                       output_size=OUTPUT_SIZE,
                       skip=SKIP,
                       max_delay=max_delay_n).to(device)
else:
    sys.exit()

print("\nModel:")
print(model)

# Loss
if LOSS == "DCPreESR":
    loss_fcn = DCPreESR(dc_pre=True)
elif LOSS == "ESR":
    loss_fcn = ESRLoss()
else:
    sys.exit()

# Optimizer & Scheduler
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       verbose=True,
                                                       factor=0.75)

#%% Process

# Compute epoch stats
SEQUENCE_LENGTH, TBPTT_INIT, TBPTT_LEN = dataset_train.length, 2**10, 2**10
n_steps_batch = np.ceil(
    (SEQUENCE_LENGTH - TBPTT_INIT) / (TBPTT_LEN)).astype(int)
n_batches = np.ceil(len(dataset_train) * (SEQUENCE_LENGTH / fs) /
                    BATCH_SIZE).astype(int)
optimizer_steps = n_batches * n_steps_batch

# Compute total stats
n_batches_total = n_batches * N_EPOCHS
optimizer_steps_total = optimizer_steps * N_EPOCHS

print("\nTraining start!")
print(f"Using device: {device}")
print(
    f"Epoch statistics: batch size: {BATCH_SIZE}, #batches: {n_batches}, #optimizer_steps: {optimizer_steps}"
)
print(
    f"Total statistics: #epochs: {N_EPOCHS}, #batches: {n_batches_total}, #optimizer_steps: {optimizer_steps_total}"
)

if not DRY_RUN:
    # Checkpoint
    if os.path.exists(model_running_path):
        print(
            "\nExisting model found! Continuing training from previous checkpoint"
        )

        # Load checkpoint
        checkpoint_dict = torch.load(model_running_path)

        model.load_state_dict(checkpoint_dict['model_state_dict'])
        optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
    else:
        print("\nNo existing model found! Starting training from scratch")

        # Create checkpoint
        os.makedirs(model_path)
        checkpoint_dict = {
            'epoch': 0,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_history': np.array([])
        }
else:
    checkpoint_dict = {'epoch': 0, 'val_history': np.array([])}

print("=" * 20)
start = time.time()
n_processed = 1
for epoch in range(checkpoint_dict['epoch'], N_EPOCHS):
    sys.stdout.write(f"Epoch {epoch+1}/{N_EPOCHS}...")

    train_loss = model.train_epoch(dataloader_train, loss_fcn, optimizer)
    val_loss, examples = model.validate(dataloader_val, loss_fcn)

    scheduler.step(val_loss)

    avg_time_epoch = (time.time() - start) / (n_processed)
    time_left = avg_time_epoch * (N_EPOCHS - epoch)

    sys.stdout.write(
        f" Done! Train loss {'{:.3f}'.format(train_loss)}, Validation loss {'{:.3f}'.format(val_loss)}"
    )
    sys.stdout.write(
        f" Average time/epoch {'{:.3f}'.format(avg_time_epoch/60)} min")
    sys.stdout.write(f" (~ {'{:.3f}'.format(time_left / 60)} min left)")

    # Collect results
    checkpoint_dict['val_history'] = np.append(checkpoint_dict['val_history'],
                                               val_loss)
    checkpoint_dict['epoch'] = epoch + 1
    n_processed += 1

    if not DRY_RUN:
        # Save checkpoints
        torch.save(checkpoint_dict, model_running_path)
        if val_loss < np.min(checkpoint_dict['val_history'][:-1], initial=1e3):
            sys.stdout.write(" Validation loss decreased! Updating best model.")
            torch.save(model.state_dict(), model_best_path)

    # Logging
    wandb.log({"loss (train)": train_loss}, step=epoch + 1)
    wandb.log({"loss (val)": val_loss}, step=epoch + 1)
    for example_idx, example in enumerate(examples):

        for key in example:
            waveform = example[key].cpu().squeeze().numpy()

            # Audio
            if key in ['input', 'target'] and (epoch + 1 == 1):
                # Log input and target only at first pass
                wandb.log(
                    {
                        f"{key}_{example_idx}":
                            wandb.Audio(
                                waveform, caption=f"{key}", sample_rate=fs)
                    },
                    step=epoch + 1)
            elif key in ['input', 'target'] and (epoch + 1 != 1):
                # Don't Log input and target at other passes
                pass
            else:
                # Log everything else always
                wandb.log(
                    {
                        f"{key}_{example_idx}":
                            wandb.Audio(
                                waveform, caption=f"{key}", sample_rate=fs)
                    },
                    step=epoch + 1)

    sys.stdout.write("\n")
    sys.stdout.flush()

print("Training Done!\n")

print("=" * 10, " SCRIPT END ", "=" * 10)
