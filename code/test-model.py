#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 16:47:20 2023

@author: 01tot10
"""

#%% Imports

import argparse
import os
import sys

import labellines
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy
import soundfile as sf
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from auraloss.auraloss.freq import MultiResolutionSTFTLoss
from Automated_GuitarAmpModelling.CoreAudioML.training import ESRLoss
from dataset import VADataset
from GreyBoxDRC.loss_funcs import ESRLoss as DCPreESR
from model import RNN, DiffDelRNN, DiffusionGenerator, TimeVaryingDelayLine
from utilities.utilities import (Prettyfier, analyze_sweep, nextpow2,
                                 parse_hidden_size, parse_loss, parse_model,
                                 smooth_spectrum)

#%% Argument parser


def none_or_int(argument):
    """ Parse NoneType or int input arguments from CLI """
    if argument == 'None':
        return None
    return int(argument)


# Add argument parser
parser = argparse.ArgumentParser()

# GLOBAL
parser.add_argument('--DESCRIPTIVE_NAME', type=str, default=None)
parser.add_argument('--SAVE_FIG', action='store_true', default=False)
parser.add_argument('--SAVE_AUDIO', action='store_true', default=False)

# MODEL
parser.add_argument('--MODEL', type=str, default="GRU")
parser.add_argument('--WEIGHTS',
                    nargs='+',
                    default="GRU-HS[64]-DS[ReelToReel_Dataset_Mini_CHOWTAPE]")
parser.add_argument('--ADD_DELAY', action='store_true', default=False)
parser.add_argument('--DELAY_TYPE', type=str, default="Real")
parser.add_argument('--ADD_NOISE', action='store_true', default=False)
parser.add_argument('--NOISE_TYPE', type=str, default="Real")

# DATASET
parser.add_argument('--DATASET',
                    type=str,
                    default="ReelToReel_Dataset_Mini_CHOWTAPE")
parser.add_argument('--SUBSET', type=str, default="Val")
parser.add_argument('--FRACTION', type=float, default=1.0)
parser.add_argument('--SEGMENT_LENGTH', type=int, default=None)
parser.add_argument('--NO_SHUFFLE', action='store_true', default=False)
parser.add_argument('--DEMODULATE', action='store_true', default=False)
parser.add_argument('--IDX', type=none_or_int, default=None)
parser.add_argument('--SYNC', type=float, default=0.0)
parser.add_argument('--COMPUTE_LOSS', action='store_true', default=False)
parser.add_argument('--DATASET_NOISE',
                    type=str,
                    default="Silence_AKAI_IPS[7.5]_MAXELL")

# VISUALIZATIONS
parser.add_argument('--PLOT_SWEEP', action='store_true', default=False)
parser.add_argument('--PLOT_TRANSFER', action='store_true', default=False)
parser.add_argument('--PLOT_PHASE', action='store_true', default=False)
parser.add_argument('--PLOT_DELAY', action='store_true', default=False)
parser.add_argument('--ZOOM', type=float, default=None)

args = parser.parse_args()

print("\nArguments:")
print(args)

assert not (args.PLOT_SWEEP and args.PLOT_TRANSFER and args.PLOT_DELAY
           ), "Choose either PLOT_SWEEP, PLOT_TRANSFER or PLOT_DELAY"
assert args.DELAY_TYPE.lower() in [
    "real", "generated", "true"
], "Choose 'Real', 'Generated' or 'True' as DELAY_TYPE"
assert args.NOISE_TYPE.lower() in [
    "real", "generated"
], "Choose 'Real' or 'Generated' as NOISE_TYPE"

#%% Config

# Global
RESULTS_PATH = "../results/"
DESCRIPTIVE_NAME = args.DESCRIPTIVE_NAME
SAVE_AUDIO = args.SAVE_AUDIO
SAVE_FIG = args.SAVE_FIG

# Data
AUDIO_PATH = "../audio/"
DATASET = args.DATASET
SUBSET = args.SUBSET
FRACTION = args.FRACTION
SYNC = args.SYNC
SEGMENT_LENGTH = args.SEGMENT_LENGTH
DEMODULATE = args.DEMODULATE
BATCH_SIZE = 1
SHUFFLE = not args.NO_SHUFFLE
COMPUTE_LOSS = args.COMPUTE_LOSS
DATASET_NOISE = args.DATASET_NOISE

# Model
MODEL_PATH = "../weights/"
MODEL = args.MODEL
WEIGHTS = args.WEIGHTS
INPUT_SIZE = 1
OUTPUT_SIZE = 1
SKIP = False
ADD_DELAY = args.ADD_DELAY
DELAY_TYPE = args.DELAY_TYPE
ADD_NOISE = args.ADD_NOISE
NOISE_TYPE = args.NOISE_TYPE

# Visualizations
ZOOM = args.ZOOM
IDX = args.IDX
PLOT_TRANSFER = args.PLOT_TRANSFER
PLOT_SWEEP = args.PLOT_SWEEP
PLOT_PHASE = args.PLOT_PHASE
PLOT_DELAY = args.PLOT_DELAY

#%% Setup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Datasets

# Audio
print("Dataset (audio)")
dataset_path = os.path.join(AUDIO_PATH, DATASET)
dataset = VADataset(dataset_path,
                    subset=SUBSET,
                    fraction=FRACTION,
                    length=SEGMENT_LENGTH,
                    shuffle=SHUFFLE,
                    preload=True,
                    sync=SYNC)
fs = dataset.fs

# Noise
if ADD_NOISE:
    if NOISE_TYPE.lower() == "real":
        print("\nDataset (noise)")
        dataset_path = os.path.join(AUDIO_PATH, DATASET_NOISE)
        dataset_noise = VADataset(dataset_path,
                                  subset="Train",
                                  length=dataset.length,
                                  shuffle=True,
                                  preload=True,
                                  fraction=0.1)
        assert dataset_noise.fs == fs, "Sampling rates don't match!"
    else:
        # Noise generator
        config_path = os.path.join("../configs/", 'conf_noise.yaml')
        args = OmegaConf.load(config_path)

        noise_generator = DiffusionGenerator(args, device)

# Dataloader
dataset.demodulate = DEMODULATE
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# Delay
if DELAY_TYPE == "Generated":
    # Delay generator
    config_path = os.path.join(
        "../configs/", 'conf_trajectories.yaml' if
        ("AKAI" in DATASET) else 'conf_toytrajectories.yaml')
    args = OmegaConf.load(config_path)

    delay_generator = DiffusionGenerator(args, device)

# Nonlinearity
models = []
for weight in WEIGHTS:
    model_dict = {}
    model_dict['weight'] = weight

    hidden_size = parse_hidden_size(weight)
    model_path = os.path.join(MODEL_PATH, weight)
    model_best_path = os.path.join(model_path, "best.pth")

    model_type = parse_model(weight)
    model_dict['model_type'] = model_type

    training_loss = parse_loss(weight)

    # Model id
    options = {
        'GRU-DCPreESR': 'Supervised 1',
        'GRU-ESR': 'Supervised 1',
        'DiffDelGRU-DCPreESR': 'Supervised 2',
        'DiffDelGRU-ESR': 'Supervised 2',
        'DiffDelGRU-LogSpec': 'Adversarial'
    }
    model_id = options[f'{model_type}-{training_loss}']
    model_dict['model_id'] = model_id

    if model_type == "GRU":
        model = RNN(input_size=INPUT_SIZE,
                    hidden_size=hidden_size,
                    output_size=OUTPUT_SIZE,
                    skip=SKIP).to(device)
    elif model_type == "DiffDelGRU":
        max_delay_n = int(1.25 * dataset.delay_analyzer.max_delay * fs)
        if max_delay_n == 0:                        # if there's no delay ...
            max_delay_n = 2**8
        model = DiffDelRNN(input_size=INPUT_SIZE,
                           hidden_size=hidden_size,
                           output_size=OUTPUT_SIZE,
                           skip=SKIP,
                           max_delay=max_delay_n).to(device)
    else:
        sys.exit('Something is not right!')
    model.load_state_dict(torch.load(model_best_path, map_location=device))
    model_dict['model'] = model

    if ADD_DELAY:
        max_delay_n = int(1.25 * dataset.delay_analyzer.max_delay * fs)
        delay = TimeVaryingDelayLine(max_delay=max_delay_n)

        model_dict['delay'] = delay

    print("=" * 25)
    for key, value in model_dict.items():
        print(f"{key.ljust(9)}: {value}")
    print("=" * 25, "\n")

    models.append(model_dict)

# Loss
loss_fcn_dict = {
    'ESR': ESRLoss(),
    'DCPreESR': DCPreESR(dc_pre=True),
    'MultiSTFT': MultiResolutionSTFTLoss()
}

#%% Process


def apply_delay(delay_trajectory, output):
    """Apply delay."""
    ## Process in minibatches
    SEGMENT_LENGTH = 2**12
    num_minibatches = int(np.ceil(output.shape[-1] / SEGMENT_LENGTH))
    output_delayed = torch.empty(output.shape).to(device)
    # output_delayed = torch.zeros(output.shape).to(device)

    # Initialize delay
    delay.init_buffer(output.shape[0])

    for idx_minibatch in range(num_minibatches):

        # Take segment
        delay_trajectory_mini = delay_trajectory[:, :, idx_minibatch *
                                                 SEGMENT_LENGTH:(idx_minibatch +
                                                                 1) *
                                                 SEGMENT_LENGTH]
        output_mini = output[:, :, idx_minibatch *
                             SEGMENT_LENGTH:(idx_minibatch + 1) *
                             SEGMENT_LENGTH]

        # Process
        output_delayed_mini = delay(output_mini, delay_trajectory_mini)

        output_delayed[:, :,
                       idx_minibatch * SEGMENT_LENGTH:(idx_minibatch + 1) *
                       SEGMENT_LENGTH] = output_delayed_mini

    output = output_delayed

    return output


sys.stdout.write("\nProcessing start ...\n")
sys.stdout.flush()

with torch.inference_mode():

    if COMPUTE_LOSS:
        ## LOSS OVER DATASET

        sys.stdout.write("\nComputing loss over dataset ...")
        sys.stdout.flush()

        save_path = os.path.join('.temp/', 'loss', f"{DATASET}",
                                 f"{dataset.subset}")
        save_name = f"{WEIGHTS}_DELAY[{ADD_DELAY}]_DEMODULATE[{DEMODULATE}]_NOISE[{ADD_NOISE}].npy"

        # Compute loss over dataset
        if os.path.exists(os.path.join(save_path,
                                       save_name)): # Load pre-computed
            sys.stdout.write(" Loading pre-computed!\n")
            sys.stdout.flush()

            # Load from disk
            results_dict = np.load(os.path.join(save_path, save_name),
                                   allow_pickle=True).item()

        else: # Analyze and save
            sys.stdout.write(" Starting analysis...\n")
            sys.stdout.flush()

            # Initializations
            INIT_LEN = nextpow2(
                int(dataloader.dataset.delay_analyzer.max_delay * fs)) # 2**10

            # Run epoch
            num_batches = len(dataloader)
            val_loss = 0
            results_dict = dict.fromkeys(loss_fcn_dict.keys(), 0)

            print()
            for batch_idx, batch in enumerate(dataloader):

                sys.stdout.write(
                    f"* Computing loss {batch_idx + 1}/{len(dataloader)} ...\r")
                sys.stdout.flush()

                # Take current batch
                input, target, meta = batch
                if input.shape[1] > 1: # Only consider audio for computing loss
                    input, target = input[:, :1, :], target[:, :1, :]
                input, target = input.to(device), target.to(device)

                # Apply model
                if MODEL == "GRU":
                    output = model.predict(input)
                elif MODEL == "DiffDelGRU":
                    # Take delay trajectory
                    d_traj = meta['delay_trajectory'].float()
                    d_traj = d_traj.view(1, 1, -1) * fs
                    d_traj = d_traj.to(device)

                    output, _ = model.predict(input, d_traj)

                if ADD_DELAY and MODEL == "GRU":
                    # Apply delay
                    delay_trajectory = torch.unsqueeze(meta["delay_trajectory"],
                                                       0).to(device)
                    delay_trajectory = delay_trajectory * fs

                    output = apply_delay(delay_trajectory, output)

                    # Cut delay trajectory
                    delay_trajectory = delay_trajectory[:, :, INIT_LEN:]

                # Cut segments
                input = input[:, :, INIT_LEN:]
                output = output[:, :, INIT_LEN:]
                target = target[:, :, INIT_LEN:]

                if ADD_NOISE:
                    if NOISE_TYPE.lower() == "real":
                        noise_idx = np.random.randint(0,
                                                      high=len(dataset_noise))
                        _, noise, __ = dataset_noise[noise_idx]
                    else:
                        NotImplementedError("Step through just in case!")
                        noise = noise_generator.sample_long(SEGMENT_LENGTH * fs)
                    noise = torch.unsqueeze(noise, 0).to(device)

                    # Add (Assume noise is additive ^__^)
                    output += noise[:, :, :output.
                                    shape[-1]]     # Demodulation might alter shape..

                # Compute Loss and aggregate
                for key, loss_fcn in loss_fcn_dict.items():
                    loss = loss_fcn(output, target)
                    results_dict[key] += loss.item()

            sys.stdout.write(
                f"* Computing loss {batch_idx}/{len(dataloader)} ... Done! \r")
            sys.stdout.flush()

            # Collect final loss
            results_dict = {
                key: value / num_batches
                for (key, value) in results_dict.items()
            }

            # Save to disk
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            np.save(os.path.join(save_path, save_name), results_dict)

        print()
        print("=" * 5, "Stats:", "=" * 5)
        print(f"Model:      {WEIGHTS}")
        print(f"Dataset:    {DATASET}")
        print(f"Subset:     {SUBSET}")
        print(f"ADD_DELAY:  {ADD_DELAY}")
        print(f"DEMODULATE: {DEMODULATE}")
        print(f"ADD_NOISE:  {ADD_NOISE}\n")

        for key, value in results_dict.items():
            print(f"{key.ljust(9)}: {'{:.6f}'.format(value)}")

        print()
        print("=" * 18)

    ## EXAMPLE PREDICTION

    sys.stdout.write("\nMaking example prediction ...\n")
    sys.stdout.flush()

    # Load example
    sample_idx = np.random.randint(0, high=len(dataset)) if IDX is None else IDX
    input, target, meta = dataset[sample_idx]

    input, target = torch.unsqueeze(input, 0), torch.unsqueeze(target, 0)
    input, target = input.to(device), target.to(device)

    if input.shape[1] > 0:
        input, target = input[:, :1, :], target[:, :1, :]

    print(f"idx = {sample_idx}")
    print(f"filename: {meta['target_name']}")

    if ADD_NOISE:
        # Sample noise
        if NOISE_TYPE.lower() == "real":
            noise_idx = np.random.randint(0, high=len(dataset_noise))
            _, noise, __ = dataset_noise[noise_idx]
        else:
            noise = noise_generator.sample_long(SEGMENT_LENGTH / fs)
        noise = torch.unsqueeze(noise, 0).to(device)

    # Sample delay
    if DELAY_TYPE == "True":
        # Extract trajectory
        d_traj = torch.tensor(meta['delay_trajectory']).float()
        d_traj = d_traj.view(1, 1, -1) * fs
        d_traj = d_traj.to(device)
    elif DELAY_TYPE == "Real":
        # Take random example
        delay_idx = np.random.randint(0, high=len(dataset))
        _, __, meta = dataset[delay_idx]

        # Extract trajectory
        d_traj = torch.tensor(meta['delay_trajectory']).float()
        d_traj = d_traj.view(1, 1, -1) * fs
        d_traj = d_traj.to(device)
    else:
        # Generate trajectory
        trajectory_length = SEGMENT_LENGTH / fs
        if trajectory_length >= 10: # generator minimum sample length 10s
            d_traj = delay_generator.sample_long(SEGMENT_LENGTH / fs)
        else:
            d_traj = delay_generator.sample_long(10)[:, :SEGMENT_LENGTH]
        d_traj += dataset.delay_analyzer.mean_delay

        d_traj = torch.unsqueeze(d_traj * fs, 0).to(device)

    # Apply model
    sys.stdout.write("\nApplying models ..."), sys.stdout.flush()
    for model_dict in models:

        model = model_dict['model']

        # Prediction
        if model_dict['model_type'] == "GRU":
            output = model.predict(input)
            # output = model(input) # Second go
        elif model_dict['model_type'] == "DiffDelGRU":
            output, output_pre_d = model.predict(input, d_traj)

            if not (PLOT_TRANSFER or PLOT_SWEEP):
                INIT_LEN = nextpow2(int(dataset.delay_analyzer.max_delay * fs))

                # Remove silence from start
                input, target, output, output_pre_d = input[:, :,
                                                            INIT_LEN:], target[:, :,
                                                                               INIT_LEN:], output[:, :,
                                                                                                  INIT_LEN:], output_pre_d[:, :,
                                                                                                                           INIT_LEN:]
                d_traj = d_traj[:, :, INIT_LEN:]
            else:
                output = output_pre_d
        sys.stdout.write(" Done!"), sys.stdout.flush()

        if ADD_NOISE:
            sys.stdout.write("\nAdding noise ..."), sys.stdout.flush()

            # Add (Assume noise is additive ^__^)
            output += noise[:, :, :
                            output.shape[-1]] # Demodulation might alter shape..

            sys.stdout.write(" Done!"), sys.stdout.flush()

        if ADD_DELAY and MODEL == "GRU":
            # Add delay
            sys.stdout.write("\nAdding delay ..."), sys.stdout.flush()

            output = apply_delay(d_traj, output)

            # Remove silence from start
            start_delay = int(d_traj[0, 0, 0])
            input, target, output = input[:, :,
                                          start_delay:], target[:, :,
                                                                start_delay:], output[:, :,
                                                                                      start_delay:]
            d_traj = d_traj[:, :, start_delay:]

            sys.stdout.write(" Done!"), sys.stdout.flush()

        output = torch.squeeze(output, dim=0).cpu()
        model_dict['output'] = output

        if COMPUTE_LOSS:
            ## LOSS
            SEGMENT_LENGTH = 44100
            num_segments = int(np.ceil(output.shape[-1] / SEGMENT_LENGTH))

            loss_segment = dict.fromkeys(loss_fcn_dict.keys(), 0)
            for idx_segment in range(num_segments):

                # Take segment
                output_mini = output[:, :, idx_segment *
                                     SEGMENT_LENGTH:(idx_segment + 1) *
                                     SEGMENT_LENGTH]
                target_mini = target[:, :, idx_segment *
                                     SEGMENT_LENGTH:(idx_segment + 1) *
                                     SEGMENT_LENGTH]

                # Aggregate loss
                for key, loss_fcn in loss_fcn_dict.items():
                    loss = loss_fcn(output_mini, target_mini)
                    loss_segment[key] += loss.item()

            # Final loss
            loss_segment = {
                key: value / num_segments
                for (key, value) in loss_segment.items()
            }

input, target = torch.squeeze(input, dim=0).cpu(), torch.squeeze(target,
                                                                 dim=0).cpu()

sys.stdout.write("\nProcessing Done! ...")
sys.stdout.flush()

#%% Analyze

## Config
FIG_MULTIPLIER = 1.0 if PLOT_SWEEP else 0.8
COLS = 2
ROWS = 2 if (PLOT_TRANSFER or PLOT_SWEEP or PLOT_DELAY) else 1
RATIO = (1.25 if PLOT_SWEEP else 1, 1)
NFFT = int(2**15)
labels_input = ["input, L", "input, R"]
labels_target = ["output, L", "output, R"]

## Setup
prettyfier = Prettyfier(COLS, ROWS, ratio=RATIO, size_multiplier=FIG_MULTIPLIER)
ALPHA = prettyfier.alpha
DB_LIMS = prettyfier.db_lims

gs = plt.GridSpec(ROWS, COLS * 2) # Grid

prop_cycle = plt.rcParams['axes.prop_cycle'] # Colors
colors = prop_cycle.by_key()['color']

if PLOT_SWEEP:
    prettyfier.font_size *= 1.25
mpl.rcParams['font.family'] = prettyfier.font_type
mpl.rcParams['font.size'] = prettyfier.font_size
mpl.rcParams['axes.labelsize'] = prettyfier.font_size * 6.0 / 6.0
mpl.rcParams['xtick.labelsize'] = prettyfier.font_size * 5 / 6.0
mpl.rcParams['ytick.labelsize'] = prettyfier.font_size * 5 / 6.0
mpl.rcParams['legend.fontsize'] = prettyfier.font_size * 5 / 6.0
mpl.rcParams['lines.linewidth'] = prettyfier.line_width

box = {
    "boxstyle": 'round',
    "edgecolor": 'black',
    "facecolor": 'None',
    "alpha": 0.25
}

if ZOOM:
    ZOOM_LEN = int(ZOOM * fs)
    ZOOM_START = 0 # np.random.randint(0, input.shape[-1] - ZOOM_LEN) # int(1 * fs + 0* fs)

X_MAX = np.minimum(1.0, np.max(np.abs(input[0, :].flatten().numpy())))
concatenated = np.vstack(
    (output[0, :].flatten().numpy(), target[0, :].flatten().numpy()))
Y_MAX = np.minimum(1.0, np.max(np.abs(concatenated)))

# Start plotting
fig = plt.figure(10, prettyfier.fig_size)
fig.clf()

# Time Domain
t = np.arange(0, input.shape[-1] / fs, 1 / fs)

ax = fig.add_subplot(
    gs[1 if PLOT_TRANSFER or PLOT_SWEEP or PLOT_DELAY else 0, :2])

ax.plot(t, input.flatten().numpy())
# ax.set_title('Input')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Amplitude [1]')
ax.set_ylim(-1.25 * X_MAX, 1.25 * X_MAX)
if ZOOM:
    ax.set_xlim([ZOOM_START / fs, (ZOOM_START + ZOOM_LEN) / fs])

ax_prev = ax

ax = fig.add_subplot(gs[1 if PLOT_TRANSFER or PLOT_SWEEP or PLOT_DELAY else 0,
                        2:],
                     sharex=ax_prev)
ax.plot(t, target.flatten().numpy(), label='Target')
for model_dict in models:
    ax.plot(t,
            model_dict['output'].flatten().numpy(),
            label=model_dict['model_id'])
ax.set_xlabel('Time [s]')
# ax.set_title('Output')
ax.legend(loc="lower right")
ax.set_ylim(-1.25 * Y_MAX, 1.25 * Y_MAX)

if ZOOM:
    ax.set_xlim([ZOOM_START / fs, (ZOOM_START + ZOOM_LEN) / fs])

if PLOT_TRANSFER:

    ## Compute statistics

    # Input frequency
    w = scipy.signal.windows.hann(
        len(input[0, ZOOM_START:ZOOM_START + ZOOM_LEN]))
    X = scipy.fft.fft(
        w * input[0, ZOOM_START:ZOOM_START + ZOOM_LEN].flatten().numpy(), NFFT)
    f = scipy.fft.fftfreq(NFFT, 1 / fs)

    f_max_idx = np.argmax(np.abs(X[:NFFT // 2]))
    f_max = f[f_max_idx]

    ## Cross correlation
    # xcorr = scipy.signal.correlate(
    #     input[0, ZOOM_START:ZOOM_START + ZOOM_LEN].flatten().numpy(),
    #     target[0, ZOOM_START:ZOOM_START + ZOOM_LEN].flatten().numpy())
    # max_crosscorr_idx = np.argmax(xcorr)

    # # find the lag
    # Ts = 1 / fs
    # lag_values = np.arange(-ZOOM + Ts, ZOOM, Ts)
    # lag = lag_values[max_crosscorr_idx]
    # lag_timesteps = int(round(lag_values[max_crosscorr_idx] / Ts))

    # target_corr = torch.roll(target, lag_timesteps, dims=-1)
    # output_corr = torch.roll(output, lag_timesteps, dims=-1)

    # Transfer
    ax = fig.add_subplot(gs[0, :])
    ax.plot(input[0, ZOOM_START:ZOOM_START + ZOOM_LEN].flatten().numpy(),
            target[0, ZOOM_START:ZOOM_START + ZOOM_LEN].flatten().numpy(),
            label='Target')
    for model_dict in models:
        ax.plot(input[0, ZOOM_START:ZOOM_START + ZOOM_LEN].flatten().numpy(),
                model_dict['output'][0, ZOOM_START:ZOOM_START +
                                     ZOOM_LEN].flatten().numpy(),
                '--',
                label=model_dict['model_id'])

    # ax_prev.plot(t, output_corr.flatten().numpy(), label='prediction, corr')
    # ax_prev.plot(t, target_corr.flatten().numpy(), label='target, corr')

    # ax.set_title('Transfer Function')
    ax.set_xlabel('Input [1]')
    ax.set_ylabel('Output [1]')
    ax.grid()
    xy_ticks = np.arange(-1.0, 1.0 + 0.25, 0.25)
    ax.set_xticks(xy_ticks)
    ax.set_yticks(xy_ticks)
    ax.set_xlim(-1.25 * X_MAX, 1.25 * X_MAX)
    ax.set_ylim(-1.05 * Y_MAX, 1.05 * Y_MAX)

    ax.legend(loc="lower right")

    # textbox = f"f = {'{:.0f}'.format(f_max)} Hz, a = {'{:.3f}'.format(X_MAX)}"
    # ax.text(0.985,
    #         1.075,
    #         textbox,
    #         bbox=box,
    #         transform=ax.transAxes,
    #         fontsize=10,
    #         verticalalignment='top',
    #         horizontalalignment='right')

elif PLOT_SWEEP:
    ax3 = fig.add_subplot(gs[0, :])

    # Settings
    N_harmonics = 6 # number of harmonics to compute
    N_octaves = 3   # fractional order filtering

    # colors_in = mpl.colormaps['Greys'](np.linspace(0.25, 0.75,
    #                                                N_harmonics))[::-1]
    colors_out = mpl.colormaps['Greys'](np.linspace(0.25, 0.75,
                                                    N_harmonics))[::-1]
    colors_pred = mpl.colormaps['Reds'](np.linspace(0.25, 0.75,
                                                    N_harmonics))[::-1]

    # Compute frequency responses
    sweep_extracted_in = input[0, ZOOM_START:ZOOM_START +
                               ZOOM_LEN].flatten().numpy()
    sweep_extracted_out = target[0, ZOOM_START:ZOOM_START +
                                 ZOOM_LEN].flatten().numpy()
    responses = analyze_sweep(sweep_extracted_in,
                              sweep_extracted_out,
                              fs,
                              NFFT,
                              normalize=True,
                              N_harmonics=N_harmonics)
    for model_dict in models:
        sweep_extracted_pred = model_dict['output'][0, ZOOM_START:ZOOM_START +
                                                    ZOOM_LEN].flatten().numpy()
        responses_pred = analyze_sweep(sweep_extracted_in,
                                       sweep_extracted_pred,
                                       fs,
                                       NFFT,
                                       normalize=True,
                                       N_harmonics=N_harmonics)
        model_dict['responses_pred'] = responses_pred

    ## Magnitude
    for N in range(1, N_harmonics + 1):

        response = responses[N - 1]
        f = response['frequencies'][:NFFT // 2]

        ## Input
        # ax3.semilogx(
        #     f,
        #     20 *
        #     np.log10(np.abs(response['frequency_response_in'][:NFFT // 2])),
        #     '--',
        #     label=f"N = {N}",
        #     alpha=ALPHA,
        #     color=colors_in[N - 1])

        ## Target

        # Smoothing
        X = 20 * np.log10(np.abs(
            response['frequency_response_out'][:NFFT // 2]))
        X_smoothed = smooth_spectrum(X, f, N_octaves)

        ax3.semilogx(
            f,
            X_smoothed,
            '-' if int(N - 1 % 2) == 0 else '--',
            label=None if
            (N - 1) == 0 else N,                  # "target" if (N - 1) == 0 else None,
            alpha=ALPHA,
            color=colors_out[N - 1])
        legend = ['Target']

        # ax3.semilogx(f,
        #              X,
        #              '-' if int(N - 1 % 2) == 0 else '--',
        #              label="target" if (N - 1) == 0 else None,
        #              alpha=ALPHA,
        #              color=colors_out[N - 1])

        ## Output
        for model_dict in models:
            response_pred = model_dict['responses_pred'][N - 1]

            # Smoothing
            X_pred = 20 * np.log10(
                np.abs(response_pred['frequency_response_out'][:NFFT // 2]))
            X_pred_smoothed = smooth_spectrum(X_pred, f, N_octaves)

            ax3.semilogx(
                f,
                X_pred_smoothed,
                '-' if int(N - 1 % 2) == 0 else '--',
                label=None if (N - 1) == 0 else
                N,                                    # model_dict['model_id'] if (N - 1) == 0 else None,
                alpha=ALPHA,
                color=colors_pred[N - 1])

            # ax3.semilogx(f,
            #              X_pred,
            #              '-' if int(N - 1 % 2) == 0 else '--',
            #              label="prediction" if (N - 1) == 0 else None,
            #              alpha=ALPHA,
            #              color=colors_pred[N - 1])

            legend.append(model_dict['model_id'])

    # ax3.set_title('Frequency response')
    ax3.set_xlabel('Frequency [Hz]')
    ax3.set_ylabel('Magnitude [dB]')
    ax3.grid()
    ax3.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())

    ax3.set_xlim(20, 20000)

    ACCURACY = 10
    ax3.set_ylim(DB_LIMS)
    ax3.yaxis.set_major_locator(mpl.ticker.MultipleLocator(12))
    lines = ax3.get_lines()
    labellines.labelLines(lines, xvals=(1e2 / 2, 1e3 / 2), align=False)
    ax3.legend(legend, loc="upper right", ncols=len(models) + 1)

    if PLOT_PHASE:
        ## Phase
        ax4 = plt.twinx(ax3)

        ax4.semilogx(response['frequencies'][:NFFT // 2],
                     np.rad2deg(
                         np.unwrap(np.angle(
                             response['frequency_response_out'][:NFFT // 2]),
                                   period=2 * np.pi)),
                     color=colors[2],
                     label=labels_target[0][:-3] + ", phase",
                     alpha=ALPHA / 2)

        ax4.set_ylim(-360, 360)
        ax4.set_ylabel('Phase [deg]')
        ax4.legend(loc="upper right")
elif PLOT_DELAY:
    ax3 = fig.add_subplot(gs[0, :])

    ax3.plot(t, d_traj / fs * 1000, color=colors[2], label='delay trajectory')

    # ax3.set_title('Time Delay')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Delay [ms]')

    # Get mean delay and range and set y-axis
    ACCURACY = 1
    mean_delay = dataset.delay_analyzer.mean_delay * 1000
    mean_delay = round(mean_delay / ACCURACY) * ACCURACY

    ACCURACY = 1.0
    RANGE = (np.max(d_traj / fs) - np.min(d_traj / fs)) * 1000
    RANGE = np.ceil(RANGE / ACCURACY) * ACCURACY

    ax3.set_ylim(mean_delay - RANGE, mean_delay + RANGE)

    # ax3.legend(loc="upper right")
    textbox = f"delay(mean) = {mean_delay} ms"
    ax3.text(0.985,
             0.925,
             textbox,
             bbox=box,
             transform=ax3.transAxes,
             fontsize=prettyfier.font_size * 3 / 4,
             verticalalignment='top',
             horizontalalignment='right')
    ax3.grid()

plt.tight_layout()

BASENAME = f"{DATASET}"
if DESCRIPTIVE_NAME:
    BASENAME = f"{BASENAME}_{DESCRIPTIVE_NAME}"

if SAVE_AUDIO:
    print("\nSaving audio...")
    file_path = os.path.join(RESULTS_PATH, f"{BASENAME}_input.wav")
    sf.write(file_path, input.flatten().numpy(), fs)

    file_path = os.path.join(RESULTS_PATH, f"{BASENAME}_target.wav")
    sf.write(file_path, target.flatten().numpy(), fs)

    for model_dict in models:
        file_path = os.path.join(
            RESULTS_PATH, f"{BASENAME}_prediction_{model_dict['model_id']}.wav")
        sf.write(file_path, model_dict['output'].flatten().numpy(), fs)

if SAVE_FIG:
    print("\nSaving figure...")
    fig_name = f"{BASENAME}_{WEIGHTS}.pdf" if len(
        WEIGHTS) == 1 else f"{BASENAME}.pdf"
    fig_path = os.path.join(RESULTS_PATH, fig_name)
    fig.savefig(fig_path, format='pdf')

if not (SAVE_AUDIO or SAVE_FIG or COMPUTE_LOSS):
    plt.show()
