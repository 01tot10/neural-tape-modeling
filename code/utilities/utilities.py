#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 09:49:28 2023

@author: 01tot10, Alec-Wright, eloimoliner
"""

#%% Imports

import glob
import os
import re
import sys

import numpy as np
import scipy
import soundfile as sf
import torch
import torchaudio
from librosa.filters import mel as librosa_mel_fn
from torch import nn

#%% Classes


class Eloi_Prettyfier:

    def __init__(
            self,
            mode="paper_1col", #"paper_2col
            ratio=(2, 1),
    ):
        """
        Set figure style globally across scripts. Eloi's edit.

        Parameters
        ----------
        cols : int, optional
            number of columns. The default is 1.
        rows : int, optional
            number of rows. The default is 1.
        ratio : tuple, optional
            figure size ratio. The default is (2,1).
        size_multiplier : float, optional
            multiplier for figure size. The default is 1.
        target : str, optional
            figure target in ['thesis', 'paper']. The default is 'thesis'.

        """

        # FIGURE ATTRIBUTES
        #self.cols = cols
        #self.rows = rows
        #self.ratio = ratio
        #self.size_multiplier = size_multiplier
        #self.target = target

        if mode == "paper_1col":
            y_size = 3.27 * ratio[1] / ratio[0]
            self.fig_size = (3.27, y_size)
        elif mode == "paper_2col":
            y_size = 6.85 * ratio[1] / ratio[0]
            self.fig_size = (6.85, y_size)

        #self.fig_size = self.set_figuresize()

        # LINE AND TEXT ATTRIBUTES
        self.alpha = 3.0 / 4.0
        self.font_size = 9.0
        self.font_type = 'Times New Roman'

        # FREQ DOMAIN ATTRIBUTES
        self.db_lims = [-48, 18]
        self.f_lims = [20, 20e3]
        self.f_ticks = np.logspace(np.log10(10**2), np.log10(10**4),
                                   3).astype(int).tolist()

        self.line_width = 1.5


class Prettyfier:

    def __init__(self,
                 cols=1,
                 rows=1,
                 ratio=(2, 1),
                 size_multiplier=1,
                 mode='1col',
                 target='paper'):
        """
        Set figure style globally across scripts

        Parameters
        ----------
        cols : int, optional
            number of columns. The default is 1.
        rows : int, optional
            number of rows. The default is 1.
        ratio : tuple, optional
            figure size ratio. The default is (2,1).
        size_multiplier : float, optional
            multiplier for figure size. The default is 1.
        target : str, optional
            figure target in ['thesis', 'paper']. The default is 'thesis'.

        """
        # FIGURE ATTRIBUTES
        self.cols = cols
        self.rows = rows
        self.ratio = ratio
        self.size_multiplier = size_multiplier
        self.mode = mode
        self.target = target

        self.fig_size = self.set_figuresize()

        # LINE AND TEXT ATTRIBUTES
        self.line_width = 1.5
        self.alpha = 3.0 / 4.0
        self.font_size = 12.0
        self.font_type = 'Times New Roman'

        # TIME DOMAIN ATTRIBUTES
        self.y_lims = [-1.1, 1.1]

        # FREQ DOMAIN ATTRIBUTES
        self.db_lims = [-48, 18]
        self.f_lims = [20, 20e3]
        self.f_ticks = np.logspace(np.log10(10**2), np.log10(10**4),
                                   3).astype(int).tolist()

    def set_figuresize(self):
        # one cell
        x_size = 3.27 if self.mode == '1col' else 6.85 # in inches
        y_size = x_size * self.ratio[1] / self.ratio[0]

        # full grid
        x_size, y_size = x_size * self.cols * self.size_multiplier, y_size * self.rows * self.size_multiplier

        # x_size, y_size = x_size / self.cols, y_size / self.cols # normalize

        return (x_size, y_size)


class DelayAnalyzer:

    def __init__(self, data_dir, subset="train", wiggle=True):
        """
        Analyze and store delay trajectories based on input/output pulse train measurements.
        Reconstructs missing pulses if not found.
        Wiggles pulses if pulse-to-pulse difference is higher than a threshold.

        Args:
            data_dir (str): Path to the root directory of the dataset..
            subset (str, optional): Pull data either from "train", "val", "test", or "full" subsets. Defaults to "train".
            wiggle (bool, optional): Wiggle on/off (True/False). Defaults to True.
        """

        # ATTRIBUTES
        self.data_dir = data_dir
        self.subset = subset
        self.wiggle = wiggle

        # SETTINGS
        self.UPSAMPLING_METHOD = 'cubic' # TODO. Experiment with different?

        print("=" * 5, " DelayAnalyzer ", "=" * 5)
        print(f"Using:          {os.path.basename(self.data_dir)}")
        print(f"Chosen subset:  {self.subset}")
        # print(f"Upsampling method:  {self.UPSAMPLING_METHOD.capitalize()}")

        # INITIALIZATIONS
        self.multi_channel = False

        self.trajectories = []
        self.input_peaks = []
        self.input_meta = {
            "reconstruction_percentage": 0,
            "wiggle_percentage": 0
        }
        self.output_peaks = []
        self.output_meta = {
            "reconstruction_percentage": 0,
            "wiggle_percentage": 0
        }

        # Audio
        self.fs = None
        self._init_files()
        print(f"Frame-rate:     {self.fs}")
        print(
            f"Found:          {len(self.input_files), len(self.target_files)} (input, output) files"
        )

        # Trajectories
        if self.multi_channel:
            self.mean_delay = 0.0
            self.max_delay = 0.0
            self.min_delay = 1e6
            self._init_trajectories()
            print(
                f"Delay (min, mean, max)  ({'{:.0f}'.format(self.min_delay* 1000)}, {'{:.0f}'.format(self.mean_delay * 1000)}, {'{:.0f}'.format(self.max_delay * 1000)}) ms"
            )
        else:
            sys.stdout.write("Found mono files, no delay trajectory analysis!")
            sys.stdout.flush()

        print()
        print("=" * 21, "\n")

    def _init_files(self):
        """ Initialize input and output files for analysis. """

        # Inititalize search
        search_dir = re.sub('([\[\]])', '[\\1]', self.data_dir)                    # escape [ and ]
        search_string = "**" if self.subset == "full" else self.subset.capitalize(
        )

        # Input
        self.input_files = glob.glob(
            os.path.join(search_dir, search_string, "input_*.wav"))
        self.input_files.sort()

        # Target
        self.target_files = glob.glob(
            os.path.join(search_dir, search_string, "target_*.wav"))
        self.target_files.sort()

        assert len(self.input_files) > 0, "No input files found!"
        assert len(self.target_files) > 0, "No target files found!"
        assert len(self.input_files) == len(
            self.input_files), "Number of IO-files doesn't match!"

        # Set frame-rate
        md = torchaudio.info(self.input_files[0])
        self.fs = md.sample_rate

        # Test dimensionality
        input, _ = self._load(self.input_files[0])
        target, _ = self._load(self.target_files[0])

        if target.shape[0] > 1 and target.shape[0] > 1:
            self.multi_channel = True

    def _init_trajectories(self):
        """
        Initialize delay trajectories.
        Either loads from disk or runs analysis based on stored input / output.
        """

        for idx, item in enumerate(zip(self.input_files, self.target_files)):

            sys.stdout.write(
                f"* Initializing delay trajectories... {idx+1:3d}/{len(self.input_files):3d} ...\r"
            )
            sys.stdout.flush()

            # Get audio file id(s)
            (ifile, tfile) = item
            ifile_id = int(os.path.basename(ifile).split("_")[1])
            tfile_id = int(os.path.basename(tfile).split("_")[1])
            if ifile_id != tfile_id:
                raise RuntimeError(
                    f"Found non-matching file ids: {ifile_id} != {tfile_id}! Check dataset."
                )

            # Test frame-rate
            if torchaudio.info(ifile).sample_rate != self.fs:
                raise RuntimeError("Framerate not constant across dataset.")

            # Get trajectory
            trajectory_name = f"trajectory{os.path.splitext(os.path.basename(ifile).split('input')[1])[0]}.npy"
            trajectory_path = os.path.join(os.path.dirname(ifile),
                                           trajectory_name)

            if os.path.exists(trajectory_path):            # Load pre-computed
                trajectory_dict = np.load(trajectory_path,
                                          allow_pickle=True).item()

                # Collect
                self.input_peaks.append(trajectory_dict["input_peaks"])
                self.output_peaks.append(trajectory_dict["output_peaks"])
                self.trajectories.append(trajectory_dict["delay_trajectory"])

                for reconstruction_metric in self.input_meta:
                    self.input_meta[reconstruction_metric] += trajectory_dict[
                        "input_meta"][reconstruction_metric]
                    self.output_meta[reconstruction_metric] += trajectory_dict[
                        "output_meta"][reconstruction_metric]

                self.mean_delay += np.mean(trajectory_dict["delay_trajectory"])
                if np.max(trajectory_dict["delay_trajectory"]) > self.max_delay:
                    self.max_delay = np.max(trajectory_dict["delay_trajectory"])
                if np.min(trajectory_dict["delay_trajectory"]) < self.min_delay:
                    self.min_delay = np.min(trajectory_dict["delay_trajectory"])

            else: # Analyze and save

                # Load audio
                input, _ = self._load(ifile)
                target, _ = self._load(tfile)

                if target.shape[0] < 2 or target.shape[0] < 2:
                    raise RuntimeError("Input or output not multi-channel!")

                # Compute trajectory
                x_idx_pulse, y_idx_pulse, delay_trajectory, x_meta, y_meta = self.analyze_delay(
                    input[1, :], target[1, :], self.fs)

                # Collect
                self.input_peaks.append(x_idx_pulse)
                self.output_peaks.append(y_idx_pulse)
                self.trajectories.append(delay_trajectory)

                for reconstruction_metric in self.input_meta:
                    self.input_meta[reconstruction_metric] += x_meta[
                        reconstruction_metric]
                    self.output_meta[reconstruction_metric] += y_meta[
                        reconstruction_metric]

                self.mean_delay += np.mean(delay_trajectory)
                if np.max(delay_trajectory) > self.max_delay:
                    self.max_delay = np.max(delay_trajectory)
                if np.min(delay_trajectory) < self.min_delay:
                    self.min_delay = np.min(delay_trajectory)

                # Save to disk
                trajectory_dict = {
                    "input_peaks": x_idx_pulse,
                    "input_meta": x_meta,
                    "output_peaks": y_idx_pulse,
                    "output_meta": y_meta,
                    "delay_trajectory": delay_trajectory
                }
                np.save(trajectory_path, trajectory_dict)

        # Take mean of metrics
        for reconstruction_metric in self.input_meta:
            self.input_meta[reconstruction_metric] /= len(self.input_files)
            self.output_meta[reconstruction_metric] /= len(self.target_files)
        self.mean_delay /= len(self.trajectories)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index to stored trajectories.

        Returns:
            input_peaks (numpy.array):  Detected/reconstructed input pulses.
            output_peaks (numpy.array): Detected/reconstructed output pulses.
            trajectories (numpy.array): Delay trajectory.

        """
        return self.input_peaks[idx], self.output_peaks[idx], self.trajectories[
            idx]

    def analyze_delay(self, input, output, fs, demodulated=False):
        """
        Compute delay trajectories from input/output pulse trains. Based mildly on [*].
        
        [*] Kaloinen, Jussi. “Neural Modeling of the Audio Tape Echo Effect.”
            Masters Thesis, Aalto University, 2022.

        Args:
            input (numpy.array): Input pulse train.
            output (numpy.array): Output pulse train.
            fs (int): Sampling rate.
            demodulated (bool, optional): Enable fake-mode. Defaults to False.
        """
        assert isinstance(input, np.ndarray) and isinstance(
            output, np.ndarray), "Inputs should numpy arrays!"
        assert input.ndim == 1 or (input.ndim == 2 and input.shape[1]
                                   == 1), "Input shape is not right yo"

        # Find indices of pulses
        x_idx_pulse, x_meta = self._get_pulse_indices(input)
        if not demodulated:
            y_idx_pulse, y_meta = self._get_pulse_indices(output)
        else:
            y_idx_pulse = x_idx_pulse.copy()

        # match number of pulses
        x_idx_pulse = x_idx_pulse[:len(y_idx_pulse)]
        y_idx_pulse = y_idx_pulse[:len(x_idx_pulse)]

        # Compute  delay trajectories at fs = (1/pulse_period)*fs, where
        # pulse_period = int(np.mean(np.diff(x_idx_pulse))) # in samples
        T_delay_DS = (y_idx_pulse - x_idx_pulse) / fs

        # Upsample
        T_interp = scipy.interpolate.interp1d(y_idx_pulse / fs,
                                              T_delay_DS,
                                              kind=self.UPSAMPLING_METHOD,
                                              fill_value=(T_delay_DS[0],
                                                          T_delay_DS[-1]),
                                              bounds_error=False)
        t = np.arange(0, len(input) / fs, 1 / fs)
        T_delay = T_interp(t)

        if np.min(T_delay) < -1e6:
            print(
                f"min(T_delay) = {np.min(T_delay)} -> System seems non-causal, something is wrong?"
            )

        return x_idx_pulse, y_idx_pulse, T_delay, x_meta, y_meta

    def demodulate(self, output, x_idx_pulse, y_idx_pulse):
        """
        Compensate for modulating delay based on input/output pulse trains.
        Based on [*].
        
        [*] Kaloinen, Jussi. “Neural Modeling of the Audio Tape Echo Effect.”
            Masters Thesis, Aalto University, 2022.

        Args:
            input_pulse (numpy.array): Input pulse train.
            output (numpy.array): Output pulse of shape (2, N_SAMPLES)
        """
        assert isinstance(output, np.ndarray) and isinstance(
            x_idx_pulse, np.ndarray) and isinstance(
                y_idx_pulse, np.ndarray), "Arguments should numpy arrays!"
        assert x_idx_pulse.ndim == 1 or (
            x_idx_pulse.ndim == 2 and x_idx_pulse.shape[1]
            == 1), "Input pulse train is not of right shape"
        assert y_idx_pulse.ndim == 1 or (
            y_idx_pulse.ndim == 2 and y_idx_pulse.shape[1]
            == 1), "Output pulse train is not of right shape"

        # SETTINGS
        UPSAMPLING_METHOD = 'linear' # 'cubic' caused trouble with reconstruction

        # Construct "unmodulated target indices" (Eq. 57 in [*])
        period_pulse = int(np.mean(np.diff(x_idx_pulse))) # in samples
        y_hat_idx_pulse = y_idx_pulse[0] + np.arange(
            len(y_idx_pulse)) * period_pulse

        # Upsample to fs [Eq. 58]
        f = scipy.interpolate.interp1d(
            y_idx_pulse,
            y_hat_idx_pulse,
            kind=UPSAMPLING_METHOD,
            fill_value='extrapolate')   # TODO - Does this cause errors?
        t = np.arange(output.shape[-1])
        y_hat = f(t)

        # Demodulate [Eq. 59]
        g = scipy.interpolate.interp1d(y_hat,
                                       output,
                                       kind=UPSAMPLING_METHOD,
                                       fill_value=(output[:, 0], output[:, 1]),
                                       bounds_error=False)
        output_demod = g(t)

        # Remove delay between input/output
        if y_idx_pulse[0] - x_idx_pulse[0] > 0:
            SAFETY = 0
            output_demod = np.roll(output_demod,
                                   -(y_idx_pulse[0] - x_idx_pulse[0] + SAFETY),
                                   axis=1)
            output_demod[:, -(y_idx_pulse[0] - x_idx_pulse[0] +
                              SAFETY):] = np.zeros(
                                  (2, y_idx_pulse[0] - x_idx_pulse[0] + SAFETY))

        return output_demod

    def _get_pulse_indices(self, signal, rel_threshold=0.01, prominence=0.05):
        """
        Find indices of pulses from a pulse train. 
        Reconstructs missing pulses based on median pulse period.

        Args:
            signal (numpy.array): Input pulse train.
            rel_threshold (float, optional): Initial relative minimum pulse level for pulse detection. Defaults to 0.01.
            prominence (float, optional): Initial minimum prominence for pulse detection. Defaults to 0.05.

        Returns:
            pulse_indices (numpy.array): Indices of pulses.

        """
        ## SETTINGS

        # detection
        REL_THRESHOLD = rel_threshold      # required (relative) minimum level of pulses
        PROMINENCE = prominence            # required minimum pulse prominence
        PULSE_PERIOD = (1 / 100) * self.fs # Get mean period, TODO. Make dynamic!
        DISTANCE = int(0.9 *
                       PULSE_PERIOD)       # required minimum distance between pulses
        WIDTH = (None, PULSE_PERIOD / 2)   # required maximum pulse width

        # reconstruction
        MAX_REL_PERIOD = 1.5 # threshold for considering pulse missing [times period]
        WIGGLE_THRESHOLD = 5 # threshold for considering pulse wiggled [sample to sample difference] TODO. Should be dynamic...

        ## PROCESS
        CONVERGED = False
        FALLOUT_RATE = 0.5           # fallout rate of [REL_THRESHOLD, PROMINENCE] between detection iterations
        RECONSTRUCTION_THRESHOLD = 5 # maximum percentage of pulses allowed to be reconstructed

        while not CONVERGED:

            ## FIND PULSES

            # find first pulse (static threshold)
            first_idx = scipy.signal.find_peaks(np.pad(
                np.clip(signal.flatten(), a_min=0, a_max=None), (1, 1),
                'minimum'),
                                                height=np.max(signal) * 0.01,
                                                distance=DISTANCE,
                                                prominence=0.01,
                                                width=WIDTH)[0][0]
            first_idx = first_idx - 1

            # find remaining pulses (dynamic threshold)
            pulse_indices, _ = scipy.signal.find_peaks(
                np.pad(signal.flatten()[first_idx + int(PULSE_PERIOD / 2):],
                       (0, 1), 'minimum'),
                height=np.max(signal) * REL_THRESHOLD,
                distance=DISTANCE,
                prominence=PROMINENCE,
                width=WIDTH)
            pulse_indices = (pulse_indices + first_idx + int(PULSE_PERIOD / 2)
                            )                                                  # - 1

            # combine
            pulse_indices = np.concatenate(([first_idx], pulse_indices))

            # FLAG1 - Check if detected pulse period is as expected
            pulse_period = int(np.median(np.diff(pulse_indices)))

            if not np.isclose(pulse_period,
                              PULSE_PERIOD,
                              atol=PULSE_PERIOD * (MAX_REL_PERIOD - 1.0)):
                sys.stdout.write(
                    f" Failed (Pulse period {pulse_period} != {'{:.0f}'.format(PULSE_PERIOD)})! Increasing pulse detection sensitivity"
                )
                sys.stdout.flush()
                REL_THRESHOLD *= FALLOUT_RATE
                PROMINENCE *= FALLOUT_RATE
                continue

            ### RECONSTRUCTIONS

            ## MISSING PULSES

            # get indices of problematic areas
            pulse_diffs = np.diff(pulse_indices)
            problem_indices = np.where(
                pulse_diffs > MAX_REL_PERIOD * pulse_period)[0]

            # reconstruct
            total_constructed = 0
            while len(problem_indices) > 0: # Reconstruct all problem areas

                # Get current problematic area
                n = problem_indices[0]
                diff = pulse_indices[n + 1] - pulse_indices[n]

                n_constructed = 0
                while diff > MAX_REL_PERIOD * pulse_period: # Reconstruct current area

                    # Reconstruct pulse
                    new_pulse_idx = np.array(pulse_indices[n] +
                                             pulse_period).reshape(-1,)
                    pulse_indices = np.concatenate(
                        (pulse_indices[:n + 1], new_pulse_idx,
                         pulse_indices[n + 1:]))

                    # Update index and difference
                    n += 1
                    n_constructed += 1
                    diff = pulse_indices[n + 1] - pulse_indices[n]

                # Update problematic areas
                pulse_diffs = np.diff(pulse_indices)
                problem_indices = np.where(
                    pulse_diffs > MAX_REL_PERIOD * pulse_period)[0]

                total_constructed += n_constructed

            # FLAG2 - Check reconstruction percentage is below threshold
            reconstruction_percentage = (total_constructed /
                                         len(pulse_indices)) * 100

            if reconstruction_percentage > RECONSTRUCTION_THRESHOLD:
                sys.stdout.write(
                    f" Failed (Reconstructed {'{:.0f}'.format(reconstruction_percentage)}% > {RECONSTRUCTION_THRESHOLD}%)! Increasing pulse detection sensitivity"
                )
                sys.stdout.flush()
                REL_THRESHOLD *= FALLOUT_RATE
                PROMINENCE *= FALLOUT_RATE
                continue

            CONVERGED = True

            ## WIGGLY PULSES
            if self.wiggle:
                # get indices of problematic areas
                pulse_diffs = np.diff(pulse_indices, n=2)
                problem_indices = np.where(pulse_diffs >= WIGGLE_THRESHOLD)[0]
                if len(problem_indices) > 0:
                    sys.stdout.write(" Wiggling!")
                    sys.stdout.flush()

                # reconstruct
                pulse_indices[problem_indices +
                              2] = pulse_indices[problem_indices +
                                                 1] + pulse_period
                wiggle_percentage = (len(problem_indices) /
                                     len(pulse_indices)) * 100
            else:
                wiggle_percentage = 0.0

        # Construct meta
        meta = {
            "reconstruction_percentage": reconstruction_percentage,
            "wiggle_percentage": wiggle_percentage
        }

        return pulse_indices, meta

    def _load(self, filename):
        x, sr = sf.read(filename, always_2d=True)
        return x.T, sr


class TimeFreqConverter(nn.Module):
    """ Time-Frequency Converter """

    def __init__(self,
                 n_fft=2048,
                 hop_length=512,
                 win_length=2048,
                 sampling_rate=44100,
                 n_mel_channels=160,
                 mel_fmin=0.0,
                 mel_fmax=None):

        super().__init__()

        # Define Mel-basis
        mel_basis = librosa_mel_fn(sr=sampling_rate,
                                   n_fft=n_fft,
                                   n_mels=n_mel_channels,
                                   fmin=mel_fmin,
                                   fmax=mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)

        # Windowing
        window = torch.hann_window(win_length).float()
        self.register_buffer("window", window)

        # FFT Parameters
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.spectro = torchaudio.transforms.Spectrogram(n_fft=n_fft,
                                                         hop_length=n_fft // 4)

    def forward(self, audio, mel=False):
        """ Forward. """

        magnitude = self.spectro(audio).squeeze()

        if mel:
            mel_output = torch.matmul(self.mel_basis, magnitude)
            return magnitude, mel_output
        else:
            return magnitude


#%% Methods


def analyze_sweep(sweep_in,
                  sweep_out,
                  fs,
                  NFFT=2**11,
                  normalize=False,
                  N_harmonics=10):
    """
    Compute system linear frequency response + distortion components with 
    Farina's method.

    Args:
        sweep_in (numpy.ndarray): Input sweep.
        sweep_out (numpy.ndarray): Output sweep.
        fs (int): Sampling rate.
        NFFT (int, optional): FFT size. Defaults to 2**11.
        normalize (bool, optional): Normalize responses. Defaults to False.
        N_harmonics (int, optional): Number of harmonic components to compute. 

    Returns:
        responses (list): Responses wrapped in a list of dictionaries.

    """
    assert isinstance(sweep_in, np.ndarray) and isinstance(
        sweep_out, np.ndarray), "Inputs should numpy arrays!"
    assert sweep_in.ndim == 1 or (sweep_in.ndim == 2 and sweep_in.shape[1]
                                  == 1), "sweep_in.shape is not right!"
    assert sweep_out.ndim == 1 or (sweep_out.ndim == 2 and sweep_out.shape[1]
                                   == 1), "sweep_out.shape is not right!"
    sweep_in = sweep_in.flatten()
    sweep_out = sweep_out.flatten()

    # Settings
    F_LO = 1.0   # sweep start freq
    F_HI = 20000 # sweep end freq

    # construct time-array
    N_samples = len(sweep_in) # number of samples
    t_max = N_samples / fs    # maximum time instance
    t = np.linspace(0, t_max, N_samples)

    # construct inverse sweep
    sweep_in_inv = sweep_in[::-1] / np.exp(t * np.log(F_HI / F_LO) / t_max)

    # filtering
    sweep_in_conv = scipy.signal.convolve(sweep_in, sweep_in_inv)
    sweep_out_conv = scipy.signal.convolve(sweep_out, sweep_in_inv)

    # compute linear impulse response and nonlinear components

    # Initializations
    responses = []
    N_center = N_samples - 1  # location of linear impulse response
    N_length = N_samples // 8 # length of linear impulse response

    print()
    for N in range(1, N_harmonics + 1):
        sys.stdout.write(f" * Computing {N}/{N_harmonics} harmonic ...\r")
        sys.stdout.flush()

        # separation between linear response and nonlinear component
        t_delay = t_max * (np.log10(N) / np.log10(F_HI / F_LO))
        t_delay_n = int(t_delay * fs)

        # response indices
        start_index = (N_center) - t_delay_n
        indices = [start_index, start_index + N_length]

        # take responses
        response_in = sweep_in_conv[indices[0]:indices[-1]]
        response_out = sweep_out_conv[indices[0]:indices[-1]]

        # normalization
        if normalize:
            response_in = response_in / np.max(sweep_in_conv)
            response_out = response_out / np.max(sweep_in_conv)

        # calculate frequency response
        [w,
         H_in] = scipy.signal.freqz(response_in[:int(np.min([N_length, NFFT]))],
                                    1,
                                    np.min([N_length, NFFT]),
                                    whole=True)
        [_, H_out
        ] = scipy.signal.freqz(response_out[:int(np.min([N_length, NFFT]))],
                               1,
                               np.min([N_length, NFFT]),
                               whole=True)

        # frequency axis
        f = w * fs / (2 * np.pi)

        # Collect and save
        responses.append({
            'indices': indices,
            'frequencies': f,
            'impulse_response_in': response_in,
            'impulse_response_out': response_out,
            'frequency_response_in': H_in,
            'frequency_response_out': H_out
        })

        # Update response length
        N_length = int(N_length * 1 / 2)

    sys.stdout.write(f" * Computing {N}/{N_harmonics} harmonic ... Done!")
    sys.stdout.flush()

    return responses


def batch_smooth_spectrum(X, f, Noct):
    """
    Apply 1/N-octave smoothing to a frequency spectrum. Eloi's edit.
    
    Translated from
    [2] https://github.com/IoSR-Surrey/MatlabToolbox/blob/master/%2Biosr/%2Bdsp/smoothSpectrum.m
    """

    ## Smoothing

    # calculates a Gaussian function for each frequency, deriving a
    # bandwidth for that frequency

    x_oct = X.copy() # initial spectrum

    if Noct > 0: # don't bother if no smoothing

        # Take frequency indices
        first_idx = np.where(f > 0)[0][0]
        indices = np.arange(first_idx, len(f))

        for i in indices: # for each frequency bin

            # Compute freq-domain Gaussian with unity gain
            g = _gauss_f(f, f[i], Noct)
            g = np.expand_dims(g, axis=0)
            # Compute smoothed spectral coefficient
            x_oct[:, i] = np.sum(g * X)

        # Remove undershoot when X is positive
        if np.all(X >= 0):
            x_oct[x_oct < 0] = 0

    return x_oct


def smooth_spectrum(X, f, Noct):
    """
    Apply 1/N-octave smoothing to a frequency spectrum.
    
    Translated from
    [2] https://github.com/IoSR-Surrey/MatlabToolbox/blob/master/%2Biosr/%2Bdsp/smoothSpectrum.m
    """

    ## Smoothing

    # calculates a Gaussian function for each frequency, deriving a
    # bandwidth for that frequency

    x_oct = X.copy() # initial spectrum

    if Noct > 0: # don't bother if no smoothing

        # Take frequency indices
        first_idx = np.where(f > 0)[0][0]
        indices = np.arange(first_idx, len(f))

        for i in indices: # for each frequency bin

            # Compute freq-domain Gaussian with unity gain
            g = _gauss_f(f, f[i], Noct)
            # Compute smoothed spectral coefficient
            x_oct[i] = np.sum(g * X)

        # Remove undershoot when X is positive
        if np.all(X >= 0):
            x_oct[x_oct < 0] = 0

    return x_oct


def _gauss_f(f_x, F, Noct):
    """
    Calculate frequency-domain Gaussian with unity gain
    [2] https://github.com/IoSR-Surrey/MatlabToolbox/blob/master/%2Biosr/%2Bdsp/smoothSpectrum.m
    """

    sigma = (F / Noct) / np.pi                       # standard deviation
    g = np.exp(-(((f_x - F)**2) / (2 * (sigma**2)))) # Gaussian
    g = g / np.sum(g)                                # normalise magnitude

    return g


def parse_hidden_size(model_name):
    """
    Get the portion of a model name defining the hidden size

    Args:
        string (string): Model name.

    Returns:
        int: hidden size.

    """
    match = re.search('-HS\[(.\d*)\]-', model_name).group(1)
    return int(match)


def parse_model(model_name):
    """
    Get the portion of a model name defining the model type

    Args:
        string (string): Model name.

    Returns:
        str: model type.

    """
    match = model_name[:model_name.find('-')]
    return match


def parse_loss(model_name):
    """
    Get the portion of a model name defining the loss during training

    Args:
        string (string): Model name.

    Returns:
        str: loss during training.

    """
    match = re.search('\]-L\[(.*)\]-DS\[', model_name).group(1)
    return match


def nextpow2(number):
    """ Compute next power of 2. """
    return 2**(number - 1).bit_length()
