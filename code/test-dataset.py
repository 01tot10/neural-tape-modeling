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

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy
import soundfile as sf
import torch
from matplotlib import ticker

from dataset import VADataset
from utilities.utilities import Prettyfier, analyze_sweep, smooth_spectrum

#%% Argument parser


def none_or_int(value):
    """ Parse NoneType or int input arguments from CLI """
    if value == 'None':
        return None
    return int(value)


# Add argument parser
parser = argparse.ArgumentParser()

# DATASET
parser.add_argument('--DATASET',
                    type=str,
                    default="ReelToReel_Dataset_MiniPulse100_CHOWTAPE")
parser.add_argument('--SUBSET', type=str, default="Train")
parser.add_argument('--INPUT_ONLY', action='store_true', default=False)
parser.add_argument('--FRACTION', type=float, default=1.0)
parser.add_argument('--SEGMENT_LENGTH', type=int, default=None)
parser.add_argument('--NO_SHUFFLE', action='store_true', default=False)
parser.add_argument('--SYNC', type=float, default=0.0)

# VISUALIZATIONS
parser.add_argument('--PLOT_FFT', action='store_true', default=False)
parser.add_argument('--PLOT_SWEEP', action='store_true', default=False)
parser.add_argument('--PLOT_PHASE', action='store_true', default=False)
parser.add_argument('--PLOT_TRANSFER', action='store_true', default=False)
parser.add_argument('--X_CORR', action='store_true', default=False)
parser.add_argument('--PLOT_DELAY', action='store_true', default=False)
parser.add_argument('--DEMODULATE', action='store_true', default=False)
parser.add_argument('--ZOOM', type=float, default=None)
parser.add_argument('--IDX', type=none_or_int, default=None)

# GLOBAL
parser.add_argument('--DESCRIPTIVE_NAME', type=str, default=None)
parser.add_argument('--SAVE_AUDIO', action='store_true', default=False)
parser.add_argument('--SAVE_FIG', action='store_true', default=False)
parser.add_argument('--SAVE_TRAJECTORY', action='store_true', default=False)
parser.add_argument('--PRELOAD', action='store_true', default=False)

args = parser.parse_args()

print("\nArguments:")
print(args)

assert not (args.PLOT_FFT and args.PLOT_TRANSFER
           ), "Choose either PLOT_TRANSFER, PLOT_FFT, PLOT_DELAY, PLOT_SWEEP"
assert not (args.INPUT_ONLY and
            args.PLOT_TRANSFER), "Can't PLOT_TRANSFER with INPUT_ONLY"
assert not (args.INPUT_ONLY and
            args.PLOT_DELAY), "Can't PLOT_DELAY with INPUT_ONLY"
assert not (args.INPUT_ONLY and
            args.PLOT_SWEEP), "Can't PLOT_SWEEP with INPUT_ONLY"

#%% Config

# Global
RESULTS_PATH = "../results/"
DESCRIPTIVE_NAME = args.DESCRIPTIVE_NAME
SAVE_AUDIO = args.SAVE_AUDIO
SAVE_FIG = args.SAVE_FIG
SAVE_TRAJECTORY = args.SAVE_TRAJECTORY

# Dataset
AUDIO_PATH = "../audio/"
DATASET = args.DATASET
SUBSET = args.SUBSET
INPUT_ONLY = args.INPUT_ONLY
FRACTION = args.FRACTION
SEQUENCE_LENGTH = args.SEGMENT_LENGTH
SHUFFLE = not args.NO_SHUFFLE
SYNC = args.SYNC
RETURN_FULL = True

# Visualizations
IDX = args.IDX
PLOT_FFT = args.PLOT_FFT
PLOT_TRANSFER = args.PLOT_TRANSFER
X_CORR = args.X_CORR
PLOT_DELAY = args.PLOT_DELAY
PLOT_SWEEP = args.PLOT_SWEEP
PLOT_PHASE = args.PLOT_PHASE
DEMODULATE = args.DEMODULATE
ZOOM = args.ZOOM
PRELOAD = args.PRELOAD

#%% Setup

dataset_path = os.path.join(AUDIO_PATH, DATASET)
dataset = VADataset(dataset_path,
                    fraction=FRACTION,
                    subset=SUBSET,
                    input_only=INPUT_ONLY,
                    demodulate=DEMODULATE,
                    length=SEQUENCE_LENGTH,
                    shuffle=SHUFFLE,
                    preload=PRELOAD,
                    sync=SYNC,
                    return_full=RETURN_FULL)
fs = dataset.fs

#%% Process

sample_idx = np.random.randint(0, high=len(dataset)) if IDX is None else IDX
if INPUT_ONLY:
    input, meta = dataset[sample_idx]
    print(f"\nLoaded: {meta['input_name']}")
else:
    input, target, meta = dataset[sample_idx]
    print(
        f"\nLoaded: {meta['input_name']} of shape {input.shape} and dtype {input.dtype}"
    )
    print(
        f"       {meta['target_name']} of shape {target.shape} and dtype {target.dtype}"
    )

#%% Analyze

## Config
STEREO = True if input.shape[0] > 1 else False
COLS = 1
ROWS = 2 if (PLOT_FFT or PLOT_TRANSFER or PLOT_DELAY or PLOT_SWEEP) else 1
ROWS = ROWS + 1 if STEREO else ROWS
RATIO = (2.0 if not PLOT_DELAY else 3.0, 1)
FIG_MULTIPLIER = 1.5
NFFT = int(2**15)
labels_input = ["input"] if not STEREO else ["input, L", "input, R"]
labels_target = ["output"] if not STEREO else ["output, L", "output, R"]

## Setup
prettyfier = Prettyfier(COLS, ROWS, ratio=RATIO, size_multiplier=FIG_MULTIPLIER)
ALPHA = prettyfier.alpha
DB_LIMS = prettyfier.db_lims

HEIGHT_RATIOS = [1.0 / ROWS] * ROWS # Grid
gs = plt.GridSpec(ROWS, COLS * 2, height_ratios=HEIGHT_RATIOS)

prop_cycle = plt.rcParams['axes.prop_cycle'] # Colors
colors = prop_cycle.by_key()['color']

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

if not INPUT_ONLY:
    concatenated = np.vstack(
        (input[0, :].flatten().numpy(), target[0, :].flatten().numpy()))
else:
    concatenated = input[0, :].flatten().numpy()
Y_MAX_L = np.minimum(1.0, np.max(np.abs(concatenated)))
Y_MAX = np.minimum(1.0, np.max(np.abs(target[0, :].flatten().numpy())))

if STEREO and not INPUT_ONLY:
    concatenated = np.vstack(
        (input[1, :].flatten().numpy(), target[1, :].flatten().numpy()))
else:
    concatenated = input[0, :].flatten().numpy()
Y_MAX_R = np.minimum(1.0, np.max(np.abs(concatenated)))

# Start plotting
fig = plt.figure(1, prettyfier.fig_size)
fig.clf()
fig_count = 0

## Time Domain

# Audio (Mono or L)
t = np.arange(0, input.shape[-1] / fs, 1 / fs)
ax1 = fig.add_subplot(gs[-1, :])

ax1.plot(t, input[0].numpy().T, label=labels_input[0], alpha=ALPHA)
if not INPUT_ONLY:
    ax1.plot(t, target[0].numpy().T, label=labels_target[0], alpha=ALPHA)

# ax1.set_title('Audio' if not STEREO else 'Audio (L)')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Amplitude [1]')
ax1.set_ylim(-1.25 * Y_MAX_L, 1.25 * Y_MAX_L)
ax1.legend(loc="lower right")
if ZOOM:
    ax1.set_xlim([ZOOM_START / fs, (ZOOM_START + ZOOM_LEN) / fs])

ax_prev = ax1

# Audio (R)
if STEREO:
    ax2 = fig.add_subplot(gs[-2, :], sharex=ax_prev)

    ax2.plot(t, input[1, :].numpy().T, label=labels_input[1], alpha=ALPHA)

    if not INPUT_ONLY:
        ax2.plot(meta["input_peaks"] / fs,
                 input[1, :].numpy()[meta["input_peaks"]],
                 'x',
                 color=colors[0])

        ax2.plot(t, target[1, :].numpy().T, label=labels_target[1], alpha=ALPHA)
        ax2.plot(meta["output_peaks"] / fs,
                 target[1, :].numpy()[meta["output_peaks"]],
                 'x',
                 color=colors[1])

    # ax2.set_title('Audio (R)')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Amplitude [1]')
    ax2.set_ylim(-Y_MAX_R / 4.0, 1.25 * Y_MAX_R)
    ax2.legend(loc="upper right")
    if ZOOM:
        ax2.set_xlim([ZOOM_START / fs, (ZOOM_START + ZOOM_LEN) / fs])

    ax_prev = ax2

if PLOT_FFT:
    ax3 = fig.add_subplot(gs[0, :])

    # Compute windowed FFT
    w = scipy.signal.windows.hann(
        len(input[0, ZOOM_START:ZOOM_START + ZOOM_LEN]))
    X = scipy.fft.fft(
        w * input[0, ZOOM_START:ZOOM_START + ZOOM_LEN].flatten().numpy(), NFFT)
    f = scipy.fft.fftfreq(NFFT, 1 / fs)

    # Take frequency with max peak
    f_max_idx = np.argmax(np.abs(X[:NFFT // 2]))
    f_max = f[f_max_idx]

    ax3.semilogx(f[:NFFT // 2],
                 20 * np.log10(2 / NFFT * np.abs(X[:NFFT // 2])),
                 color=colors[0],
                 label=labels_input[0])
    ax3.semilogx(f_max,
                 20 * np.log10(2 / NFFT * np.abs(X[f_max_idx])),
                 'x',
                 color=colors[0])

    if not INPUT_ONLY:
        X = scipy.fft.fft(
            w * target[0, ZOOM_START:ZOOM_START + ZOOM_LEN].flatten().numpy(),
            NFFT)

        ax3.semilogx(f[:NFFT // 2],
                     20 * np.log10(2 / NFFT * np.abs(X[:NFFT // 2])),
                     color=colors[1],
                     label=labels_target[0])

    # ax3.set_title('Frequency Domain')
    ax3.set_xlabel('f [Hz]')
    ax3.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax3.grid()
    ax3.set_xlim(20, fs / 2)
    ax3.set_ylim(DB_LIMS)
    ax3.legend(loc="upper right")
elif PLOT_TRANSFER:
    ax3 = fig.add_subplot(gs[0, :])

    ## Compute statistics

    # Input frequency
    w = scipy.signal.windows.hann(
        len(input[0, ZOOM_START:ZOOM_START + ZOOM_LEN]))
    X = scipy.fft.fft(
        w * input[0, ZOOM_START:ZOOM_START + ZOOM_LEN].flatten().numpy(), NFFT)
    f = scipy.fft.fftfreq(NFFT, 1 / fs)

    f_max_idx = np.argmax(np.abs(X[:NFFT // 2]))
    f_max = f[f_max_idx]

    if X_CORR:
        ## Cross correlation

        # Take max correlation
        xcorr = scipy.signal.correlate(
            input[0, ZOOM_START:ZOOM_START + ZOOM_LEN].flatten().numpy(),
            target[0, ZOOM_START:ZOOM_START + ZOOM_LEN].flatten().numpy())
        max_crosscorr_idx = np.argmax(xcorr)

        # Convert to lag
        Ts = 1 / fs
        lag_values = np.arange(-ZOOM + Ts, ZOOM, Ts)
        lag = lag_values[max_crosscorr_idx]
        lag_timesteps = int(round(lag_values[max_crosscorr_idx] / Ts))

        # apply lag
        target = torch.roll(target, lag_timesteps, dims=-1)

        ax1.plot(t, target[0].numpy().T, label='cross-correlated', alpha=ALPHA)
        legend = ax1.legend()
        legend.remove()

    ax3.plot(input[0, ZOOM_START:ZOOM_START + ZOOM_LEN].flatten().numpy(),
             target[0, ZOOM_START:ZOOM_START + ZOOM_LEN].flatten().numpy(),
             color=colors[2],
             alpha=ALPHA)

    # ax3.set_title('Transfer Function')
    ax3.set_xlabel('Input [1]')
    ax3.set_ylabel('Output [1]')
    ax3.grid()
    xy_ticks = np.arange(-1.0, 1.0 + 0.25, 0.25)
    ax3.set_xticks(xy_ticks)
    ax3.set_yticks(xy_ticks)
    ax3.set_xlim(-1.25 * X_MAX, 1.25 * X_MAX)
    ax3.set_ylim(-1.25 * Y_MAX, 1.25 * Y_MAX)

    # textbox = f"f = {'{:.0f}'.format(f_max)} Hz, a = {'{:.3f}'.format(X_MAX)}"
    # ax3.text(0.985,
    #          1.15,
    #          textbox,
    #          bbox=box,
    #          transform=ax3.transAxes,
    #          fontsize=10,
    #          verticalalignment='top',
    #          horizontalalignment='right')

elif PLOT_SWEEP:
    ax3 = fig.add_subplot(gs[0, :])

    # Settings
    N_harmonics = 6 # number of harmonics to compute
    N_octaves = 3   # fractional order filtering

    ## Compute frequency responses
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

    ## Plotting

    # Custom colormaps
    colors_in = mpl.colormaps['Greys'](np.linspace(0.25, 0.75,
                                                   N_harmonics))[::-1]
    colors_out = mpl.colormaps['Reds'](np.linspace(0.25, 0.75,
                                                   N_harmonics))[::-1]
    # colors_out_smoothed = mpl.colormaps['Greens'](np.linspace(0.25, 0.75,
    #                                                N_harmonics))[::-1]

    ## Magnitude
    for N in range(1, N_harmonics + 1):

        response = responses[N - 1]

        ## 1/N-Octave smoothing

        # Take log-magnitude and freqs
        X = 20 * np.log10(np.abs(
            response['frequency_response_out'][:NFFT // 2]))
        f = response['frequencies'][:NFFT // 2]

        # Smoothing
        X_smoothed = smooth_spectrum(X, f, N_octaves)

        # input
        ax3.semilogx(
            f,
            20 *
            np.log10(np.abs(response['frequency_response_in'][:NFFT // 2])),
            '--',
            label=f"N = {N}",
            alpha=ALPHA,
            color=colors_in[N - 1])

        # output
        ax3.semilogx(f,
                     X_smoothed,
                     '-' if int(N - 1 % 2) == 0 else '--',
                     label=f"N = {N}",
                     alpha=ALPHA,
                     color=colors_out[N - 1])
        # ax3.semilogx(
        #     f,
        #     X,
        #     '-' if int(N - 1 % 2) == 0 else '--',
        #     label=f"N = {N}",
        #     alpha=ALPHA,
        #     color=colors_out[N - 1])

    # ax3.set_title('Frequency response')
    ax3.set_xlabel('f [Hz]')
    ax3.set_ylabel('A [dB]')
    ax3.grid()
    ax3.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax3.set_xlim(20, 20000)
    ax3.set_ylim(DB_LIMS)
    # ax3.legend(loc="lower left")

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
    ax3 = fig.add_subplot(gs[0, :], sharex=ax_prev)

    ax3.plot(t,
             meta["delay_trajectory"] * 1000,
             color=colors[2],
             label='delay trajectory')

    # ax3.set_title('Time Delay')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Delay [ms]')

    # Get mean delay and range and set y-axis
    ACCURACY = 0.5
    mean_delay = dataset.delay_analyzer.mean_delay * 1000
    mean_delay = round(mean_delay / ACCURACY) * ACCURACY

    max_delay = dataset.delay_analyzer.max_delay * 1000
    RANGE = np.ceil((max_delay - mean_delay) / ACCURACY) * ACCURACY
    y_ticks = np.arange(mean_delay - RANGE, mean_delay + RANGE + ACCURACY / 2.0,
                        ACCURACY)

    ax3.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
    ax3.set_ylim(mean_delay - RANGE, mean_delay + RANGE)

    # ax3.legend(loc="upper right")
    # textbox = f"delay(mean) = {mean_delay} ms"
    # ax3.text(0.985,
    #          0.925,
    #          textbox,
    #          bbox=box,
    #          transform=ax3.transAxes,
    #          fontsize=10,
    #          verticalalignment='top',
    #          horizontalalignment='right')
    ax3.grid()

fig.tight_layout()

BASENAME = f"{DATASET}"
if DESCRIPTIVE_NAME:
    BASENAME = f"{BASENAME}_{DESCRIPTIVE_NAME}"

if SAVE_AUDIO:
    sys.stdout.write("\nSaving audio... ")

    SAVE_INDICES = [0, input.shape[-1]
                   ] if not ZOOM else [ZOOM_START, ZOOM_START + ZOOM_LEN]

    if input.shape[0] > 1: # Multi-channel
        sys.stdout.write("Multi-channel! \n")

        input_path = os.path.join(RESULTS_PATH, f"{BASENAME}_input_L.wav")
        sf.write(input_path,
                 input[:1, SAVE_INDICES[0]:SAVE_INDICES[1]].numpy().T,
                 fs,
                 subtype='FLOAT')

        input_path = os.path.join(RESULTS_PATH, f"{BASENAME}_input_R.wav")
        sf.write(input_path,
                 input[1:, SAVE_INDICES[0]:SAVE_INDICES[1]].numpy().T,
                 fs,
                 subtype='FLOAT')

        if not INPUT_ONLY:
            target_path = os.path.join(RESULTS_PATH, f"{BASENAME}_target_L.wav")
            sf.write(target_path,
                     target[:1, SAVE_INDICES[0]:SAVE_INDICES[1]].numpy().T,
                     fs,
                     subtype='FLOAT')

            target_path = os.path.join(RESULTS_PATH, f"{BASENAME}_target_R.wav")
            sf.write(target_path,
                     target[1:, SAVE_INDICES[0]:SAVE_INDICES[1]].numpy().T,
                     fs,
                     subtype='FLOAT')
    else: # Single-channel
        sys.stdout.write("\n")

        input_path = os.path.join(RESULTS_PATH, f"{BASENAME}_input.wav")
        sf.write(input_path,
                 input[:, SAVE_INDICES[0]:SAVE_INDICES[1]].numpy().T,
                 fs,
                 subtype='FLOAT')

        if not INPUT_ONLY:
            target_path = os.path.join(RESULTS_PATH, f"{BASENAME}_target.wav")
            sf.write(target_path,
                     target[:, SAVE_INDICES[0]:SAVE_INDICES[1]].numpy().T,
                     fs,
                     subtype='FLOAT')

if SAVE_TRAJECTORY:
    sys.stdout.write("\nSaving trajectory... \n")
    trajectory_path = os.path.join(RESULTS_PATH, f"{BASENAME}_TRAJECTORY.wav")
    sf.write(trajectory_path, meta["delay_trajectory"], fs, subtype='FLOAT')

sys.stdout.flush()

if SAVE_FIG:
    print("\nSaving figure...")
    fig_path = os.path.join(RESULTS_PATH, f"{BASENAME}.pdf")
    fig.savefig(fig_path, format='pdf')

if not (SAVE_FIG or SAVE_AUDIO or SAVE_TRAJECTORY):
    plt.show()
