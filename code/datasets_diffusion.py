#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: eloimoliner, 01tot10
"""

#%% Imports

import glob
import os
import random

import numpy as np
import soundfile as sf
import torch

#%% Classes


class TapeHissdset(torch.utils.data.IterableDataset):

    def __init__(self, dset_args, overfit=False, seed=42):
        """
        torch.utils.data.IterableDataset subclass
        """
        super().__init__()

        # Attributes
        self.seg_len = int(dset_args.seg_len)
        self.fs = dset_args.fs
        self.overfit = overfit

        # Seed the random
        random.seed(seed)
        np.random.seed(seed)

        # Read files
        path = dset_args.path
        orig_p = os.getcwd()
        os.chdir(path)
        filelist = glob.glob("target*.wav")
        filelist = [os.path.join(path, f) for f in filelist]
        os.chdir(orig_p)
        assert len(
            filelist) > 0, "error in dataloading: empty or nonexistent folder"
        self.train_samples = filelist

    def __iter__(self):
        while True:
            num = random.randint(0, len(self.train_samples) - 1)
            file = self.train_samples[num]
            data, samplerate = sf.read(file)

            assert (samplerate == self.fs, "wrong sampling rate")
            data_clean = data

            # stereo to mono
            if len(data.shape) > 1:
                data_clean = np.mean(data_clean, axis=1)

            ## Normalization
            # no normalization!!
            # data_clean=data_clean/np.max(np.abs(data_clean))
            # normalize mean
            # data_clean-=np.mean(data_clean, axis=-1)

            # get 8 random batches to be a bit faster
            idx = np.random.randint(0, len(data_clean) - self.seg_len)
            segment = data_clean[idx:idx + self.seg_len]
            segment = segment.astype('float32')
            segment -= np.mean(segment, axis=-1)

            yield segment


class TapeHissTest(torch.utils.data.Dataset):

    def __init__(self,
                 dset_args,
                 fs=44100,
                 seg_len=131072,
                 num_samples=4,
                 seed=42):
        """
        torch.utils.data.Dataset subclass
        """
        super().__init__()

        # Attributes
        self.seg_len = int(seg_len)
        self.fs = fs

        # Seed the random
        random.seed(seed)
        np.random.seed(seed)

        # Read files
        path = dset_args.test.path
        print(path)
        test_file = os.path.join(path, "input_3_.wav")
        self.test_samples = []
        self.filenames = []
        self._fs = []
        for _ in range(num_samples):
            file = test_file
            self.filenames.append(os.path.basename(file))
            data, samplerate = sf.read(file)
            data = data.T
            self._fs.append(samplerate)
            if data.shape[-1] >= self.seg_len:
                idx = np.random.randint(0, data.shape[-1] - self.seg_len)
                data = data[..., idx:idx + self.seg_len]
            else:
                idx = 0
                data = np.tile(
                    data, (self.seg_len // data.shape[-1] + 1))[..., idx:idx +
                                                                self.seg_len]

            self.test_samples.append(data[...,
                                          0:self.seg_len]) # use only `seg_len`

    def __getitem__(self, idx):
        return self.test_samples[idx], self._fs[idx], self.filenames[idx]

    def __len__(self):
        return len(self.test_samples)


class ToyTrajectories(torch.utils.data.IterableDataset):

    def __init__(self, dset_args, overfit=False, seed=42):
        """
        torch.utils.data.IterableDataset subclass
        """
        super().__init__()

        # Attributes
        self.overfit = overfit
        self.seg_len = int(dset_args.seg_len)
        self.fs = dset_args.fs

        # Seed the random
        random.seed(seed)
        np.random.seed(seed)

        # Read files
        path = dset_args.path
        orig_p = os.getcwd()
        os.chdir(path)
        filelist = glob.glob("*.wav")
        filelist = [os.path.join(path, f) for f in filelist]
        os.chdir(orig_p)
        assert len(
            filelist) > 0, "error in dataloading: empty or nonexistent folder"
        self.train_samples = filelist

    def __iter__(self):
        while True:
            num = random.randint(0, len(self.train_samples) - 1)
            file = self.train_samples[num]
            data, samplerate = sf.read(file)
            assert (samplerate == self.fs, "wrong sampling rate")
            data_clean = data

            # stereo to mono
            if len(data.shape) > 1:
                data_clean = np.mean(data_clean, axis=1)

            # Normalization
            # no normalization!!
            # data_clean=data_clean/np.max(np.abs(data_clean))
            # normalize mean
            # data_clean-=np.mean(data_clean, axis=-1)

            for _ in range(8):
                # get 8 random batches to be a bit faster
                idx = np.random.randint(0, len(data_clean) - self.seg_len)
                segment = data_clean[idx:idx + self.seg_len]
                segment = segment.astype('float32')
                segment -= np.mean(segment, axis=-1)

                yield segment


class TestTrajectories(torch.utils.data.Dataset):

    def __init__(self,
                 dset_args,
                 fs=44100,
                 seg_len=131072,
                 num_samples=4,
                 seed=42):
        """
        torch.utils.data.Dataset subclass
        """
        super().__init__()

        # Attributes
        self.fs = fs
        self.seg_len = int(seg_len)

        # Seed the random
        random.seed(seed)
        np.random.seed(seed)

        # Read files
        path = dset_args.test.path
        print(path)
        orig_p = os.getcwd()
        os.chdir(path)
        filelist = glob.glob("*.wav")
        filelist = [os.path.join(path, f) for f in filelist]
        test_file = filelist[0]
        os.chdir(orig_p)

        self.test_samples = []
        self.filenames = []
        self._fs = []
        for _ in range(num_samples):
            file = test_file
            self.filenames.append(os.path.basename(file))
            data, samplerate = sf.read(file)
            data = data.T
            self._fs.append(samplerate)
            if data.shape[-1] >= self.seg_len:
                idx = np.random.randint(0, data.shape[-1] - self.seg_len)
                data = data[..., idx:idx + self.seg_len]
            else:
                idx = 0
                data = np.tile(
                    data, (self.seg_len // data.shape[-1] + 1))[..., idx:idx +
                                                                self.seg_len]

            self.test_samples.append(data[...,
                                          0:self.seg_len]) # use only `seg_len`

    def __getitem__(self, idx):
        return self.test_samples[idx], self._fs[idx], self.filenames[idx]

    def __len__(self):
        return len(self.test_samples)
