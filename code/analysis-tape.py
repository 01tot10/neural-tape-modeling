#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 15:10:24 2023

@author: 01tot10
"""

#%% Imports

import argparse
import os
import sys
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.signal
import soundfile as sf
import torch
from matplotlib import ticker

from tape import Tape
from utilities.utilities import Prettyfier

#%% Argument parser

# Add argument parser
parser = argparse.ArgumentParser(
    description='Simulate a VA reel-to-reel tape recorder.')

# INPUT
parser.add_argument('--F_IN', type=float, default=500)
parser.add_argument('--T_MAX', type=float, default=10 * (1 / 1000))

# TAPE
parser.add_argument('--BIAS_ENABLE', action='store_true', default=False)
parser.add_argument('--DELAY_ENABLE', action='store_true', default=False)
parser.add_argument('--PLAYBACK_LOSS_ENABLE',
                    action='store_true',
                    default=False)
parser.add_argument('--STARTUP_ENABLE', action='store_true', default=True)
parser.add_argument('--SIGNAL_AMPLITUDE', type=float, default=0.0025)
parser.add_argument('--BIAS_AMPLITUDE', type=float, default=0.0125)
parser.add_argument('--RETURN_INTERNAL', action='store_true', default=True)

# FIGS
parser.add_argument('--SAVE_FIGS', action='store_true', default=False)
parser.add_argument('--PLOT_TRANSFER', action='store_true', default=True)
parser.add_argument('--PLOT_FFT', action='store_true', default=False)

# AUDIO
parser.add_argument('--SAVE_AUDIO', action='store_true', default=False)

args = parser.parse_args()
print(args)

assert not (args.PLOT_FFT and
            args.PLOT_TRANSFER), "Choose either PLOT_TRANSFER or PLOT_FFT"

#%% Config

# global
fs = int(48e3)
SAVE_PATH = '../results/'
SAVE_FIGS = args.SAVE_FIGS
SAVE_AUDIO = args.SAVE_AUDIO

# tape
BATCH_SIZE = 1
OVERSAMPLING = 32
RETURN_FULL = False
STARTUP_ENABLE = args.STARTUP_ENABLE
SIGNAL_AMPLITUDE = args.SIGNAL_AMPLITUDE
BIAS_AMPLITUDE = args.BIAS_AMPLITUDE
BIAS_ENABLE = args.BIAS_ENABLE
DELAY_ENABLE = args.DELAY_ENABLE
PLAYBACK_LOSS_ENABLE = args.PLAYBACK_LOSS_ENABLE
RETURN_INTERNAL = args.RETURN_INTERNAL

# plotting
PLOT_TRANSFER = args.PLOT_TRANSFER
PLOT_FFT = args.PLOT_FFT

# input
F_IN = args.F_IN
T_MAX = args.T_MAX
A = 1.0

#%% Setup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.inference_mode(True)

tape = Tape(batch_size=BATCH_SIZE,
            fs=fs,
            oversampling=OVERSAMPLING,
            signal_amplitude=SIGNAL_AMPLITUDE,
            bias_amplitude=BIAS_AMPLITUDE,
            bias_enable=BIAS_ENABLE,
            delay_enable=DELAY_ENABLE,
            playback_loss_enable=PLAYBACK_LOSS_ENABLE,
            startup_enable=STARTUP_ENABLE,
            return_full=RETURN_FULL,
            return_internal=RETURN_INTERNAL)

#%% Process

## INPUTS

# time
Ts = 1 / fs
t = np.arange(0, T_MAX, Ts)

# oversampled time
fs_OS = fs * OVERSAMPLING
Ts_OS = 1 / fs_OS
t_OS = np.arange(0, T_MAX, Ts_OS)

# input
x = np.sin(2 * np.pi * F_IN * t)
amp = A * (0.5 +
           0.5 * scipy.signal.sawtooth(2 * np.pi * 1 / T_MAX * t, width=0.5))
x = x * amp

## Process
print(
    f"Generating {'{:.3f}'.format(T_MAX)} s of audio w/ batch size of {BATCH_SIZE}"
)

x = torch.tensor(x, requires_grad=False, device=device,
                 dtype=tape.dtype).expand(BATCH_SIZE, -1)

start = time.time()
y = tape(x)
end = time.time()

print(f"Processing took {'{:.3f}'.format((end - start) * 1)} s")
print(f"RT Factor  {'{:.3f}'.format((end - start) / (T_MAX))} x")

if RETURN_INTERNAL:
    # Unpack results.
    I_in, I_in_OS, I_rec_OS, H_rec_OS, M_OS, M, M_del, V_play, V_out = y

    ## Collect properties for plotting
    titles = [
        'Input', 'Pre Amp', 'Oversample', 'Bias', 'Rec Head',
        'Tape Magnetization', 'Downsample', 'Tape Delay', 'Play Head', 'Output'
    ]

    # Adjust units.
    I_in *= 1e3          # Pre-amp
    I_in_OS *= 1e3       # Oversample
    I_rec_OS *= 1e3      # Bias
    H_rec_OS /= 1e3      # Magnetic Field
    M_OS /= tape.TAPE_Ms # Tape Magnetization (normalized)
    M /= tape.TAPE_Ms    # Tape Magnetization (normalized)
                         # M_OS /= 1e3     # Tape Magnetization
                         # M /= 1e3        # Tape Magnetization
    M_del /= 1e3         # Delay
    V_play *= 1e6        # Playback head

    units = [
        'Voltage [V]',           # Input
        'Current [mA]',          # Pre-amp
        'Current [mA]',          # Oversample
        'Current [mA]',          # Bias
        'Magnetic field [kA/m]', # Rec Head
        'Magnetization [M/Ms]',  # Tape Magnetization (oversampled)
        'Magnetization [M/Ms]',  # Tape Magnetization (downsampled)
        'M [kA/m]',              # Tape Magnetization (oversampled)
        'M [kA/m]',              # Tape Magnetization (downsampled)
        'M [kA/m]',              # Tape Delay
        'Voltage [mV]',          # Play Head
        'Voltage [V]'            # Post Amp
    ]

    # Frame-rates
    framerates = [fs, fs, fs_OS, fs_OS, fs_OS, fs_OS, fs, fs, fs, fs]

    # Expand input
    x = torch.cat((torch.zeros((BATCH_SIZE, I_in.shape[1] - x.shape[1])), x),
                  dim=1)

    # Limits
    y_limits = [
        [-1.25, 1.25],   # Input
        [-0.125, 0.125], # Pre-Amp
        [-0.125, 0.125], # Oversample
        [-1.25, 1.25],   # Bias
        [-1.0, 1.0],     # Rec Head
        [-0.05, 0.05],   # Tape Magnetization (oversampled)
        [-0.05, 0.05],   # Tape Magnetization (downsampled)
        [-0.05, 0.05],   # Tape Delay
        [-1.25, 1.25],   # Play Head
        [-1.25, 1.25]    # Post Amp
    ]

    # Collect signals and metadata
    y = list(y)
    y.insert(0, x)
    signals = []
    for idx, waveform in enumerate(y):
        signal_dict = {}

        waveform = waveform[0, :].numpy()

        signal_dict['waveform'] = waveform
        signal_dict['framerate'] = framerates[idx]
        signal_dict['unit'] = units[idx]
        signal_dict['title'] = titles[idx]
        signal_dict['limits'] = y_limits[idx]
        signals.append(signal_dict)
else:
    sys.exit('Not implemented')

#%% Analyze

# Settings
FIG_MULTIPLIER = 1.1
COLS = 2 if (PLOT_TRANSFER or PLOT_FFT) else 1
ROWS = 1
RATIO = (1.5, 1)

# Setup
prettyfier = Prettyfier(COLS, ROWS, ratio=RATIO, size_multiplier=FIG_MULTIPLIER)
prettyfier.font_size *= 1.2

gs = plt.GridSpec(ROWS, COLS)

mpl.rcParams['font.family'] = prettyfier.font_type
mpl.rcParams['font.size'] = prettyfier.font_size
mpl.rcParams['axes.labelsize'] = prettyfier.font_size * 6.0 / 6.0
mpl.rcParams['xtick.labelsize'] = prettyfier.font_size * 5.0 / 6.0
mpl.rcParams['ytick.labelsize'] = prettyfier.font_size * 5.0 / 6.0
mpl.rcParams['legend.fontsize'] = prettyfier.font_size * FIG_MULTIPLIER
mpl.rcParams['lines.linewidth'] = prettyfier.line_width

prop_cycle = plt.rcParams['axes.prop_cycle'] # Colors
colors = prop_cycle.by_key()['color']

signal_params = [
    r"$\bf{Signal Parameters}$",
    f"I_amp = {'{:.2e}'.format(SIGNAL_AMPLITUDE * 1000)} mA",
    f"B_amp = {'{:.2e}'.format(BIAS_AMPLITUDE * 1000)} mA"
]
textbox1 = "\n".join(signal_params)

tape_params = [
    r"$\bf{Tape Parameters}$", f"Ms = {'{:.2e}'.format(tape.TAPE_Ms)}",
    f"a = {'{:.2e}'.format(tape.TAPE_A)}",
    f"alpha = {'{:.2e}'.format(tape.TAPE_ALPHA)}",
    f"k = {'{:.2e}'.format(tape.TAPE_K)}", f"c = {'{:.2e}'.format(tape.TAPE_C)}"
]
textbox2 = "\n".join(tape_params)
box = {
    "boxstyle": 'round',
    "edgecolor": 'black',
    "facecolor": 'None',
    "alpha": 0.25
}

# Start plotting
if RETURN_INTERNAL:

    ALLOW_SETTLE = False

    # Initialize
    figs = []
    fig_counter = 0
    prev_signal = signals[0]
    idx = 1

    for signal_dict in signals:

        print(f"Processing {signal_dict['title']} ...")

        fig = plt.figure(fig_counter, prettyfier.fig_size)
        fig.clf()

        t = np.arange(
            0, signal_dict['waveform'].shape[0] / signal_dict['framerate'],
            1 / signal_dict['framerate'])
        first_idx = len(t) // 2 if ALLOW_SETTLE else 0

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(t[first_idx:],
                 signal_dict['waveform'][first_idx:],
                 color=colors[0])

        ax1.set_xlabel('Time [s]')
        ax1.set_title('Time Domain')

        if PLOT_TRANSFER:
            ax2 = fig.add_subplot(gs[0, 1])
            first_idx = len(t_OS) // 2 if ALLOW_SETTLE else 0

            prev, current = prev_signal['waveform'], signal_dict['waveform']
            if current.shape > prev.shape:
                prev = scipy.signal.resample_poly(prev, OVERSAMPLING, 1)
            elif prev.shape > current.shape:
                current = scipy.signal.resample_poly(current, OVERSAMPLING, 1)

            ax2.plot(prev[first_idx:], current[first_idx:], color=colors[1])

            ax2.set_title('Transfer Function')
            ax2.set_xlabel(prev_signal['unit'])

            axes = [ax1, ax2]
        elif PLOT_FFT:
            ax2 = fig.add_subplot(gs[0, 1])

            NFFT = int(2**12 * (signal_dict['framerate'] / fs))

            w = scipy.signal.windows.hann(
                len(signal_dict['waveform'][first_idx:]))
            X = scipy.fft.fft(w * signal_dict['waveform'][first_idx:], NFFT)
            f = scipy.fft.fftfreq(NFFT, 1 / signal_dict['framerate'])

            ax2.semilogx(f[:NFFT // 2],
                         20 * np.log10(2 / NFFT * np.abs(X[:NFFT // 2])),
                         color=colors[1])

            ax2.set_title('Frequency Domain')
            ax2.set_xlabel('Frequency [Hz]')
            ax2.xaxis.set_major_formatter(ticker.ScalarFormatter())
            ax2.set_xlim(f[0], int(signal_dict['framerate'] / 2))
            ax2.set_ylim(-96, 48)

            axes = [ax1]
        else:
            axes = [ax1]

        for ax in axes:
            ax.grid()
            if signal_dict['title'] in ['Input', 'Play Head', 'Output']:
                ax.set_ylim(signal_dict['limits'])
            ax.set_ylabel(signal_dict['unit'])
            ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(5))

        # fig.text(0.8375,
        #          0.725,
        #          textbox1,
        #          label='test',
        #          bbox=box,
        #          horizontalalignment='left',
        #          fontsize=3 * prettyfier.font_size / 5)
        # fig.text(0.8375,
        #          0.45,
        #          textbox2,
        #          bbox=box,
        #          horizontalalignment='left',
        #          fontsize=3 * prettyfier.font_size / 5)
        # fig.suptitle(signal_dict['title'],
        #               fontsize=prettyfier.font_size,
        #               fontweight='bold')
        fig.tight_layout()

        fig_counter += 1
        idx += 1
        prev_signal = signal_dict # Update previous signal
        figs.append(fig)
else:
    sys.exit('Not implemented')

if SAVE_FIGS:
    for idx, fig in enumerate(figs):
        fig_name = f"fig{idx}_{titles[idx].lower().replace(' ','_')}"
        if PLOT_TRANSFER:
            fig_name = f"transfer_{fig_name}"
        elif PLOT_FFT:
            fig_name = f"freq_{fig_name}"
        else:
            fig_name = f"time_{fig_name}"

        fig_name = f"{fig_name}_S{('{:0.6f}'.format(SIGNAL_AMPLITUDE)).replace('.','_')}"

        if titles[idx] not in ["Input", "Pre Amp", "Oversample"]:
            if BIAS_ENABLE is False:
                fig_name = f"{fig_name}_NOBIAS"
        if titles[idx] in ["Play Head", "Output"]:
            if not PLAYBACK_LOSS_ENABLE:
                fig_name = f"{fig_name}_NOPLAYLOSS"
        full_name = os.path.join(SAVE_PATH, f"{fig_name}.pdf")
        print(f'Saving {fig_name} ...')
        fig.savefig(full_name, format='pdf')

    plt.close('all')

if SAVE_AUDIO:
    print("\nSaving results...")

    input_name = f"tape_S{('{:0.6f}'.format(SIGNAL_AMPLITUDE)).replace('.','_')}_input"
    input_path = os.path.join(SAVE_PATH, f"{input_name}.wav")
    sf.write(input_path, x.numpy().T, fs)

    output_name = f"tape_S{('{:0.6f}'.format(SIGNAL_AMPLITUDE)).replace('.','_')}_output"
    output_path = os.path.join(SAVE_PATH, f"{output_name}.wav")
    sf.write(output_path, V_out.numpy().T, fs)
