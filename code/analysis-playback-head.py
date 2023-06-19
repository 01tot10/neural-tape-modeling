#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 14:44:53 2023

@author: 01tot10
"""

#%% Imports

import argparse
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
from matplotlib import ticker

from tape import Tape
from utilities.utilities import Prettyfier

#%% Argument parser

# Initialize tape to get default values
tape = Tape()

# Add argument parser
parser = argparse.ArgumentParser(
    description='Simulate a reel-to-reel tape recorder playback head.')

parser.add_argument('--TAPE_V', type=float, default=tape.TAPE_V)
parser.add_argument('--TAPE_DELTA', type=float, default=tape.TAPE_DELTA)
parser.add_argument('--PLAY_D', type=float, default=tape.PLAY_D)
parser.add_argument('--PLAY_G', type=float, default=tape.PLAY_G)
parser.add_argument('--SAVE_FIGS', action='store_true', default=False)
parser.add_argument('--ENABLE_PLAYBACK_LOSS',
                    action='store_false',
                    default=True)

args = parser.parse_args()
print(args)

del tape

#%% Config

# global
fs = int(48e3)
OVERSAMPLING = 16
SAVE_PATH = '../results/'
SAVE_FIGS = args.SAVE_FIGS

# tape
TAPE_V = args.TAPE_V         # tape speed
TAPE_DELTA = args.TAPE_DELTA # tape thickness

# head
PLAY_D = args.PLAY_D # playback head spacing
PLAY_G = args.PLAY_G # playback head gap width

# filter
ENABLE_PLAYBACK_LOSS = args.ENABLE_PLAYBACK_LOSS
FIR_order = 2**12 # approximation filter order

# input
F_IN = 100
T_MAX = 10 * (1 / F_IN)
A = 1.0
ALLOW_SETTLE = True # Add warmup/settling time before exciting circuits

#%% Playback Loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.inference_mode(True)

tape = Tape(FIR_order=FIR_order, playback_loss_enable=ENABLE_PLAYBACK_LOSS)

# Modify internal parameters
tape.TAPE_V = TAPE_V
tape.TAPE_DELTA = TAPE_DELTA
tape.PLAY_D = PLAY_D
tape.PLAY_G = PLAY_G

# Get playback loss effects
f, k, spacing_loss, thickness_loss, gap_loss, loss, H, h = tape.return_filter()

# Gain term
gain = tape.PLAY_N * tape.PLAY_W * tape.PLAY_E * tape.TAPE_V * tape.PLAY_MU0 * tape.PLAY_G

#%% Test Audio

# time
Ts = 1 / fs
t = np.arange(0, T_MAX, Ts)
if ALLOW_SETTLE is True:
    t = np.hstack((-t[::-1] - Ts, t))

# Input
x = np.sin(2 * np.pi * F_IN * t)
amp = A * (0.5 +
           0.5 * scipy.signal.sawtooth(2 * np.pi * 1 / T_MAX * t, width=0.5))
x = x * amp
if ALLOW_SETTLE is True:
    x[:len(x) // 2] = np.zeros(x[:len(x) // 2].shape)
x = torch.tensor(x, requires_grad=False, device=device,
                 dtype=tape.dtype).expand(1, -1)

# Process
y = tape.H_play(x)
y /= gain # normalize

x, y = x.squeeze().numpy(), y.squeeze().numpy()

#%% Analyze

# Settings
FIG_MULTIPLIER = 1.0
COLS = 2
ROWS = 2
RATIO = (1.5, 1)

# Setup
prettyfier = Prettyfier(COLS, ROWS, ratio=RATIO, size_multiplier=FIG_MULTIPLIER)
prettyfier.font_size *= 1.1

mpl.rcParams['font.family'] = prettyfier.font_type
mpl.rcParams['font.size'] = prettyfier.font_size
mpl.rcParams['axes.labelsize'] = prettyfier.font_size * 6.0 / 6.0
mpl.rcParams['xtick.labelsize'] = prettyfier.font_size * 5 / 6.0
mpl.rcParams['ytick.labelsize'] = prettyfier.font_size * 5 / 6.0
mpl.rcParams['legend.fontsize'] = prettyfier.font_size * FIG_MULTIPLIER
mpl.rcParams['lines.linewidth'] = prettyfier.line_width

# Collect signals
titles = ['Spacing Loss', 'Thickness Loss', 'Gap Loss', 'Total Loss']
signals = [spacing_loss, thickness_loss, gap_loss, loss]

# Gridspec
PLOT_WIDTH = 0.2
width_ratios = [PLOT_WIDTH] * (len(signals) - 1)
gs = plt.GridSpec(2, len(signals) - 1, width_ratios=width_ratios)

# Colors
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

# Axis limits
X_LIMS = [20, fs / 2]
Y_LIMS = [-48, 6]

# Textbox and figure name
params = [
    f"tape_v = {'{:.3f}'.format(TAPE_V / 2.54e-2)}",
    f"tape_delta = {'{:.2f}'.format(TAPE_DELTA / 1e-6)}",
    f"play_d = {'{:.2f}'.format(PLAY_D / 1e-6)}",
    f"play_g = {'{:.2f}'.format(PLAY_G / 1e-6)}"
]
units = [
    'ips',                  # TAPE_V
    'um',                   # TAPE_D
    'um',                   # PLAY_V
    'um',                   # PLAY_G
]
textbox = [f"{a} {b}" for a, b in zip(params, units)]
textbox = "\n".join(textbox)
box = {
    "boxstyle": 'round',
    "edgecolor": 'black',
    "facecolor": 'None',
    "alpha": 0.25
}

#% PLOTTING

# Loss Effects
fig = plt.figure(0, prettyfier.fig_size)
fig.clf()

for idx, signal in enumerate(signals):

    if idx <= 2: # individual losses
        ax = fig.add_subplot(gs[0, idx])
    else:        # combined loss
        ax = fig.add_subplot(gs[1, :-1])

    ax.semilogx(f[1:int(FIR_order / 2)],
                20 * np.log10(signal),
                color=colors[idx])

    ax.grid()
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel('Frequency [Hz]')
    if idx in [0, 3]:
        ax.set_ylabel('Magnitude [dB]')
    ax.set_title(titles[idx])
    ax.set_xlim(X_LIMS)
    ax.set_ylim(Y_LIMS)

fig.text(.7, 0.2, textbox, bbox=box, horizontalalignment='left')
# fig.suptitle('Playback Losses',
#              fontsize=prettyfier.font_size,
#              fontweight='bold')
fig.tight_layout()

# Time Domain
fig2 = plt.figure(1, prettyfier.fig_size)
fig2.clf()

ax = fig2.add_subplot(1, 1, 1)
ax.plot(t, x, label='input')
ax.plot(t, y, label='output')

ax.grid()
ax.set_xlabel('Time [s]')
ax.set_ylabel('Amplitude [1]')
ax.set_ylim([-1.25, 1.25])
ax.legend()

fig2.suptitle('Time Domain')

if SAVE_FIGS:

    fig_name = "fig_playback_losses"
    fig_name = f"{fig_name}_[{'_'.join(params).replace(' ', '').upper()}]"
    full_name = os.path.join(SAVE_PATH, f"{fig_name}.pdf")

    print(f'Saving {full_name} ...')
    fig.savefig(full_name, format='pdf')

    fig_name = "fig_playback_losses_timedomain"
    fig_name = f"{fig_name}_[{'_'.join(params).replace(' ', '').upper()}]"
    full_name = os.path.join(SAVE_PATH, f"{fig_name}.pdf")

    print(f'Saving {full_name} ...')
    fig2.savefig(full_name, format='pdf')

    plt.close('all')
