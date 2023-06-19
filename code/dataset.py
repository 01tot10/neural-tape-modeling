#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 16:34:03 2023

@author: 01tot10
"""

#%% Imports

import glob
import os
import re
import sys

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset

from utilities.utilities import DelayAnalyzer

torchaudio.set_audio_backend("sox_io")

#%% Classes


class VADataset(Dataset):
    """
    Based on SignalTrain LA2A dataset implementation [*].
    [*] https://github.com/csteinmetz1/micro-tcn/blob/main/microtcn/data.py
    """

    # TODO - Move printouts to separate method..

    def __init__(self,
                 data_dir,
                 subset="train",
                 length=44100,
                 input_only=False,
                 demodulate=False,
                 shuffle=True,
                 preload=True,
                 half=False,
                 double=False,
                 fraction=1.0,
                 sync=0.0,
                 use_soundfile=False,
                 return_full=False):
        """
        Args:
            data_dir (str): Path to the root directory of the dataset.
            subset (str, optional): Pull data either from "train", "val", "test", or "full" subsets. (Default: "train")
            length (int, optional): Number of samples in the returned examples. (Default: 44100)
            input_only (bool, optional): Read input files only. (Default: False)
            demodulate (bool, optional): Demodulate targets. (Default: False)
            shuffle (bool, optional): Permute examples when creating the dataset. (Default: True)
            preload (bool, optional): Read in all data into RAM during init. (Default: True)
            half (bool, optional): Store audio as float16. (Default: False)
            double (bool, optional): Store audio as float64. (Default: False)
            fraction (float, optional): Fraction of the data to load from the subset. (Default: 1.0)
            sync (float, optional): "Cut `sync` seconds before creating patches". (Default: 0.0)
            use_soundfile (bool, optional): Use the soundfile library to load instead of torchaudio. (Default: False)
            return_full (bool, optional): Return extra signals in metadata. (Default: False)
        """
        assert not (half and double), "Set bit-depth to either half or double"
        assert os.path.exists(data_dir), "Can't find chosen data_dir"
        assert not (input_only and demodulate), "Can't demodulate without inputs"
        assert preload, "Class doesn't handle all cases without preload=True!"

        if not input_only:
            self.delay_analyzer = DelayAnalyzer(data_dir, subset)

        # attributes
        self.data_dir = data_dir
        self.subset = subset
        self.length = length
        self.input_only = input_only
        self.shuffle = shuffle
        self.demodulate = demodulate
        self.preload = preload
        self.half = half
        self.double = double
        self.fraction = fraction
        self.sync = sync
        self.use_soundfile = use_soundfile
        self.return_full = return_full

        print("=" * 5, " Dataset ", "=" * 5)
        print(f"Using:          {os.path.basename(self.data_dir)}")
        print(f"Chosen subset:  {self.subset}")
        print(
            f"Demodulate:     {self.demodulate},     Shuffle: {self.shuffle},     Synced: {self.sync}"
        )

        # initialize files
        self.multi_channel = False
        self.fs = None

        self._init_files()
        if not self.input_only:
            print(
                f"Found:          {len(self.input_files), len(self.target_files)} (input, output) files"
            )
        else:
            print(f"Found:          {len(self.input_files)} input files")

        if self.multi_channel and not self.input_only:
            print(f"Found:          {len(self.trajectory_files)} trajectories")
            print(
                f"        input - (Reconstructed, Wiggled) ({'{:.3f}'.format(self.delay_analyzer.input_meta['reconstruction_percentage'])}, {'{:.3f}'.format(self.delay_analyzer.input_meta['wiggle_percentage'])})% of pulses",
            )
            print(
                f"       output - (Reconstructed, Wiggled) ({'{:.3f}'.format(self.delay_analyzer.output_meta['reconstruction_percentage'])}, {'{:.3f}'.format(self.delay_analyzer.output_meta['wiggle_percentage'])})% of pulses"
            )
        assert not ((not self.multi_channel) and
                    self.demodulate), "Can't demodulate without stereo audio!"

        # load data
        self.create_patches()
        self.create_fractional_patches()
        self.minutes = ((self.length * len(self.examples)) / self.fs) / 60
        print("=" * 21)

    #%% INITIALIZATIONS

    def _init_files(self):
        """ Initialize input and output files for analysis. """

        # get all input files
        search_dir = re.sub('([\[\]])', '[\\1]', self.data_dir)  # escape [ and ]
        search_string = "**" if self.subset == "full" else self.subset.capitalize()
        self.input_files = glob.glob(os.path.join(search_dir, search_string, "input_*.wav"))
        self.input_files.sort()

        # get all output files
        if not self.input_only:

            # outputs
            self.target_files = glob.glob(os.path.join(search_dir, search_string, "target_*.wav"))
            self.target_files.sort()
            assert len(self.target_files) > 0, "No target files found!"

            # Currently no support for input conditioning
            self.params = [''] * len(self.target_files)

        else:  # initialize with empty strings (used for later logic)
            self.target_files = [''] * len(self.input_files)
            self.params = [''] * len(self.input_files)

        # Set frame rate and length
        metadata = torchaudio.info(self.input_files[0])
        self.fs = metadata.sample_rate
        if self.length is None:
            self.length = metadata.num_frames
        print(f"Segment length  {self.length/self.fs} s,         fs: {self.fs}")

        # Get all trajectories
        input, _ = self.load(self.input_files[0])  # test dimensionality
        if input.shape[0] > 1 and not self.input_only:
            self.multi_channel = True

            self.trajectory_files = glob.glob(
                os.path.join(search_dir, search_string, "trajectory_*.npy"))
            self.trajectory_files.sort()
        else:  # initalize with something (used for later logic)
            self.trajectory_files = [
                f'trajectory_{int(os.path.basename(item).split("_")[1])}_'
                for item in self.input_files
            ]

    def create_patches(self):
        """
        Load data from source 'self.input_files' and 'self.target_files'.
        Sets global frame-rate.
        Pre-loads to RAM if self.preload.
        Slices files into self.length chunks.
        Stores results as dictionaries in self.examples.
        """
        self.examples = []

        for idx, item in enumerate(
                zip(self.input_files, self.target_files, self.trajectory_files,
                    self.params)):  # iterate over files

            # Get file id(s)
            (ifile, tfile, dfile, params) = item
            ifile_id = int(os.path.basename(ifile).split("_")[1])
            if not self.input_only:
                tfile_id = int(os.path.basename(tfile).split("_")[1])
                dfile_id = int(os.path.basename(dfile).split("_")[1])
                if ifile_id != tfile_id != dfile_id:
                    raise RuntimeError(
                        f"Found non-matching file ids: {ifile_id} != {tfile_id}! Check dataset.")

            # Get number of frames and test sample-rate
            metadata = torchaudio.info(ifile)
            num_frames = metadata.num_frames
            if num_frames / self.length < 1:
                raise ValueError(
                    f"Sequence length `{self.length}` is longer than file length `{num_frames}`.")
            if metadata.sample_rate != self.fs:
                raise RuntimeError("Framerate not constant across dataset.")

            # Pre-load to RAM (TODO: Consider moving this to after fractional patch is made..)
            if self.preload:
                sys.stdout.write(f"* Pre-loading... {idx+1:3d}/{len(self.input_files):3d} ...\r")
                sys.stdout.flush()

                # Load input
                input, _ = self.load(ifile)
                input = input.double() if self.double else input
                input = input.half() if self.half else input

                # Load target
                if not self.input_only:
                    target, _ = self.load(tfile)
                    target = target.double() if self.double else target
                    target = target.half() if self.half else target

                    num_frames = int(np.min([input.shape[-1], target.shape[-1]]))
                    if input.shape[-1] != target.shape[-1]:
                        print(os.path.basename(ifile), input.shape[-1], os.path.basename(tfile),
                              target.shape[-1])
                        raise RuntimeError("Found potentially corrupt file!")

                    if target.shape[0] > 1:
                        # Load delay trajectories
                        input_peaks, output_peaks, T_delay = self.delay_analyzer[idx]
                else:
                    num_frames = int(input.shape[-1])
            else:
                input = None
                target = None
                T_delay = None
                input_peaks = None
                output_peaks = None

            # Slice file into self.length chunks minding offset.
            START_OFFSET = int(self.sync * self.fs)
            num_frames -= START_OFFSET

            self.file_examples = []  # store as list of dictionaries
            for N_chunk in range((num_frames // self.length)):
                offset = int(N_chunk * self.length) + START_OFFSET
                end = offset + self.length
                item = {
                    "idx": idx,
                    "input_file": ifile,
                    "input_audio": input[:, offset:end] if input is not None else None,
                    "offset": offset,
                    "frames": num_frames,
                    "params": params  # these are used later in create_fractional_patches()
                }
                if not self.input_only:
                    target_item = {
                        "target_file": tfile,
                        "target_audio": target[:, offset:end] if target is not None else None
                    }

                    # Delay trajectories and pulse indices
                    if self.multi_channel:

                        target_item["delay_trajectory"] = T_delay[
                            offset:end] if target is not None else None

                        # Pulse indices
                        corrected_peaks_input = None
                        corrected_peaks_output = None
                        if input_peaks is not None:
                            if offset <= int(np.max(input_peaks)):

                                # Get indices between [offset, end]
                                first_idx = np.where(input_peaks >= offset)[0][0]
                                last_idx = np.where(input_peaks <= end)[0][-1] if end <= int(
                                    np.max(input_peaks)) else input_peaks[-1]

                                # Restrict indices to segment length
                                corrected_peaks_input = input_peaks[first_idx:last_idx] - offset
                                corrected_peaks_output = output_peaks[first_idx:last_idx] - offset
                                corrected_peaks_output = corrected_peaks_output[
                                    corrected_peaks_output <= self.length]

                        target_item["input_peaks"] = corrected_peaks_input
                        target_item["output_peaks"] = corrected_peaks_output

                    item.update(target_item)
                self.file_examples.append(item)

            # Update examples
            self.examples += self.file_examples

    def create_fractional_patches(self):
        """
        Restrict examples to 'self.fraction'.
        Sample 'n_examples_per_class' examples for each unique device configuration.
        Updates self.examples accordingly.
        """
        # Get unique device configurations
        classes = set([ex['params'] for ex in self.examples])
        n_classes = len(classes)

        # Get number of examples per configuration
        fraction_examples = int(len(self.examples) * self.fraction)
        n_examples_per_class = int(fraction_examples / n_classes)
        if n_examples_per_class <= 0:
            raise ValueError(f"Fraction `{self.fraction}` set too low. No examples selected.")

        # Compute minutes of audio in total and per configuration
        n_min_total = ((self.length * n_examples_per_class * n_classes) / self.fs) / 60
        n_min_per_class = ((self.length * n_examples_per_class) / self.fs) / 60

        # Printouts
        print(f"Constructed:    {len(self.examples)} examples       Total classes:   {n_classes}")
        print(
            f"Using:          {fraction_examples} examples ({self.fraction}) Examples/class:  {n_examples_per_class}"
        )
        print(
            f"                = {n_min_total:0.2f} min      Audio per class:  {n_min_per_class:0.2f} min"
        )

        # Sample 'n_examples_per_class' examples for each configuration
        sampled_examples = []
        for config_class in classes:  # iterate over unique configurations
            # get current examples
            class_examples = [ex for ex in self.examples if ex["params"] == config_class]
            # take 'n_examples_per_class' examples for each class
            if self.shuffle:
                example_indices = np.random.randint(0,
                                                    high=len(class_examples),
                                                    size=n_examples_per_class)
            else:
                example_indices = np.arange(n_examples_per_class)
            class_examples = [class_examples[idx] for idx in example_indices]

            sampled_examples += class_examples

        # Update examples
        self.examples = sampled_examples

    #%% METHODS

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):

        offset = self.examples[idx]["offset"]
        end = offset + self.length

        # Get audio
        if self.preload:
            input = self.examples[idx]["input_audio"]
            if not self.input_only:
                target = self.examples[idx]["target_audio"]
                if self.multi_channel:
                    T_delay = self.examples[idx]["delay_trajectory"]
                    input_peaks = self.examples[idx]["input_peaks"]
                    output_peaks = self.examples[idx]["output_peaks"]
        else:
            input, _ = torchaudio.load(self.examples[idx]["input_file"],
                                       num_frames=self.length,
                                       frame_offset=offset,
                                       normalize=False)
            # Fiddle with precision
            input = input.double() if self.double else input
            input = input.half() if self.half else input

            if not self.input_only:
                target, _ = torchaudio.load(self.examples[idx]["target_file"],
                                            num_frames=self.length,
                                            frame_offset=offset,
                                            normalize=False)
                # Fiddle with precision
                target = target.double() if self.double else target
                target = target.half() if self.half else target

                if self.multi_channel:
                    input_peaks, output_peaks, T_delay = self.delay_analyzer[idx]

                    # Corrections
                    T_delay = T_delay[offset:end]

                    # Pulse indices
                    if offset <= int(np.max(input_peaks)):
                        # Get indices between [offset, end]
                        first_idx = np.where(input_peaks >= offset)[0][0]
                        last_idx = np.where(input_peaks <= end)[0][-1] if end <= int(
                            np.max(input_peaks)) else input_peaks[-1]

                        # Restrict indices to segment length
                        input_peaks = input_peaks[first_idx:last_idx] - offset
                        output_peaks = output_peaks[first_idx:last_idx] - offset
                        output_peaks = output_peaks[output_peaks <= self.length]

        # Demodulate targets
        if self.demodulate:
            target = self.delay_analyzer.demodulate(target.numpy(), input_peaks, output_peaks)
            target = torch.tensor(target)

            # Cut mean delay
            input = input[:, :-int(self.delay_analyzer.mean_delay * self.fs)]
            target = target[:, :-int(self.delay_analyzer.mean_delay * self.fs)]
            T_delay = T_delay[:-int(self.delay_analyzer.mean_delay * self.fs)]

            input_peaks = input_peaks[input_peaks <= input.shape[-1]]

            output_peaks = input_peaks  # HAX: assume everything works PERFECTLY

        # Metadata
        input_name = os.path.basename(self.examples[idx]["input_file"])
        input_name = f"{os.path.splitext(input_name)[0]}_[{offset}:{offset+self.length}]{os.path.splitext(input_name)[1]}"
        meta = {'input_name': input_name}

        if not self.input_only:
            target_name = os.path.basename(self.examples[idx]["target_file"])
            target_name = f"{os.path.splitext(target_name)[0]}_[{offset}:{offset+self.length}]{os.path.splitext(target_name)[1]}"
            meta['target_name'] = target_name
            if self.multi_channel:
                meta['delay_trajectory'] = T_delay
                if self.return_full:
                    meta['input_peaks'] = input_peaks
                    meta['output_peaks'] = output_peaks
        if self.input_only:
            return input, meta
        else:
            return input, target, meta

    def load(self, filename):
        """ Load `filename` as tensor. """
        if self.use_soundfile:
            x, fs = sf.read(filename, always_2d=True)
            x = torch.tensor(x.T)
        else:
            x, fs = torchaudio.load(filename, normalize=False)
        return x, fs

    def get_levels(self):
        """ Check maximum waveform level in `self.examples`
        """

        # Initializations
        max_level_input = 0
        n_clipped_input = 0
        max_level_target = 0
        n_clipped_target = 0

        for idx in range(len(self)):
            if self.input_only:
                input, _ = self[idx]
            else:
                input, target, _ = self[idx]

            # Input
            input_level = input.max().item()
            if input_level > max_level_input:
                max_level_input = input_level
            if np.isclose(input_level, 1.0):
                n_clipped_input += 1

            # Target
            if not self.input_only:
                target_level = target.max().item()
                if target_level > max_level_target:
                    max_level_target = target_level
                if np.isclose(target_level, 1.0):
                    n_clipped_target += 1

        print(
            f"Max input level:  {'{:0.3f}'.format(max_level_input)}, #clipped {n_clipped_input}/{len(self)}"
        )
        if not self.input_only:
            print(
                f"Max target level: {'{:0.3f}'.format(max_level_target)}, #clipped {n_clipped_target}/{len(self)}"
            )
