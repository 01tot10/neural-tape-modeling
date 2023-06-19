#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 14:42:23 2022

@author: 01tot10
"""

#%% IMPORTS

import glob
import os
import re

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from natsort import index_natsorted, order_by_index

#%% CONFIG

# global
VERBOSE = False
SAVE_MODE = False
RETAIN_BALANCE = False # normalize loudness relative to output
INCLUDE_INPUT = True   # include input audio in normalization

# audio
AUDIO_PATH = "../results/"
AUDIO = "EXP1B_ToyData/predictions"

# analysis
TARGET_LUFS = -23 # target LUFS

#%% SETUP

input_path = os.path.join(AUDIO_PATH, AUDIO)

# get all filenames
search_dir = re.sub('([\[\]])', '[\\1]', input_path) # escape [ and ]
filenames = glob.glob(os.path.join(search_dir, "*.wav"))

#%% PROCESS


def get_number(string):
    """
    Get a number from `string` representing the groups of inputs/outputs/predictions.

    Parameters
    ----------
    string : string
        Filename.

    Returns
    -------
    match : int
        Extracted number.

    """
    match = os.path.basename(string)
    match = int(match[:match.find('_')])
    return match


def get_target(string):
    """
    Get the portion from `string` defining the target

    Parameters
    ----------
    string : str
        Model full string.

    Returns
    -------
    trainset : str
        Portion of string defining the target.

    """
    match = re.search('T\[(.*)\]_D', string).group(1)
    return match


# sorting
natindex = index_natsorted(filenames)
filenames = order_by_index(filenames, natindex)

# take outputs
output_names = [item for item in filenames if 'target' in item]

# create groups
groups = []
for output_name in output_names:

    # define current group
    result_number = get_number(output_name) # by file number

    # get current group
    group = [item for item in filenames if result_number is get_number(item)]

    # exclude input
    if INCLUDE_INPUT is False:
        group = [item for item in group if 'input' not in item]

    # appends to groups
    groups.append(group)

#%% LOUDNESS

## Normalize loudness
groups_audio = []

# iterate over output_names
print("Processing ...")
for idx, output_name in enumerate(output_names):
    print()
    filename = output_name[1:] if output_name[
        0] == '/' else output_name             # fix filename
    print(filename)

    # read and analyze
    data, rate = sf.read(filename)                  # load audio (with shape (samples, channels))
    meter = pyln.Meter(rate)                        # create BS.1770 meter
    loudness_orig = meter.integrated_loudness(data) # measure loudness

    # compute difference in loudness and linear correction gain
    loudness_difference_db = TARGET_LUFS - loudness_orig
    gain = 10**(loudness_difference_db / 20)

    if VERBOSE:
        print(f"Loudness (original): {np.round(loudness_orig, 3)}")
        print(
            f"Loudness difference (dB): {np.round(loudness_difference_db, 3)}")
        print(f"Gain: {np.round(gain, 3)}")

    # normalize audio beloging to same group
    group_audio = []

    for group_item in groups[idx]:

        filename = group_item[1:] if group_item[
            0] == '/' else group_item            # fix filename
        if VERBOSE:
            print(filename)

        # read and normalize
        data, rate = sf.read(
            filename)         # load audio (with shape (samples, channels))

        # normalize
        if RETAIN_BALANCE:                                           # relate to output
            data_processed = gain * data
        else:                                                        # individually
            loudness = meter.integrated_loudness(data)
            data_processed = pyln.normalize.loudness(data, loudness,
                                                     TARGET_LUFS)

        # measure loudness
        loudness_processed = meter.integrated_loudness(data_processed)

        if VERBOSE:
            print(f"Loudness (processed): {np.round(loudness_processed, 3)}")

        # add to group
        group_audio.append(data_processed)

    groups_audio.append(group_audio)

if SAVE_MODE:
    print("Saving results...")
    assert len(groups) == len(groups_audio)

    results_path = os.path.join(AUDIO_PATH, f"{AUDIO[:-1]}-normalized")
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    for idx, group in enumerate(groups):
        current_names = group
        group_audio = groups_audio[idx]
        assert len(current_names) == len(group_audio)

        for idxx, item in enumerate(current_names):
            item = item[1:] if item[0] == '/' else item
            data = group_audio[idxx]

            filename = os.path.basename(item)
            full_name = os.path.join(results_path, filename)

            print(full_name)
            sf.write(full_name, data, rate)
        print()
