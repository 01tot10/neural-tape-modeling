#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 14:34:09 2023

@author: 01tot10
"""

#%% Imports

import time

import numpy as np
import scipy.signal
import torch
import torchaudio.functional as F
import torchaudio.transforms as T
from pedalboard import load_plugin

#%% Classes


class VSTWrapper:

    def __init__(self,
                 vst_path,
                 fs,
                 oversampling=16,
                 tape_drive=0.5,
                 tape_saturation=0.5,
                 tape_bias=0.5,
                 tape_enable=True,
                 delay_enable=False,
                 playback_loss_enable=False,
                 wow_flutter_enable=False,
                 warm_start=True,
                 bypass=False):
        """
        Wrapper class for CHOWTape VST using pedalboard.

        Args:
            vst_path (str): path to CHOWTape VST.
            fs (int): sampling rate.
            oversampling (int, optional): Oversampling factor. Defaults to 16.
            tape_drive (float, optional): Tape drive. Defaults to 0.5.
            tape_saturation (float, optional): Tape saturation. Defaults to 0.5.
            tape_bias (float, optional): Tape bias. Defaults to 0.5.
            tape_enable (bool, optional): Enable tape saturation. Defaults to True.
            delay_enable (bool, optional): Add extra delay to outputs. Defaults to False.
            playback_loss_enable (bool, optional): Enable playback loss. Defaults to False.
            wow_flutter_enable (bool, optional): Enable wow and flutter. Defaults to False.
            warm_start (bool, optional): Pad input audio with zeros before processing. Defaults to True.
            bypass (bool, optional): Bypass VST. Defaults to False.
        """

        # Global attributes
        self.vst_path = vst_path
        self.fs = fs
        self.oversampling = oversampling
        self.bypass = bypass

        # Processing attributes
        self.warm_start = warm_start
        self.warmup = 0.5 # warmup time in [s]

        # Saturation
        self.tape_saturation = tape_saturation
        self.tape_drive = tape_drive
        self.tape_bias = tape_bias

        # Delay & wobble
        self.delay_enable = delay_enable
        if delay_enable:
            self.tape_speed = 15 * 2.54e-2                        # in [m/s]
            self.tape_spacing = 3.81e-2 / 2.0                     # in [m]
            self.delay_time = self.tape_spacing / self.tape_speed # in [s]

        self.tape_enable = tape_enable
        self.playback_loss_enable = playback_loss_enable
        self.wow_flutter_enable = wow_flutter_enable

        # Initialize
        self.vst = load_plugin(vst_path)
        self._initialize()

    def _initialize(self):
        # Dynamic attributes
        self.vst.oversampling_factor = self.oversampling

        # Static attributes
        self.vst.oversampling_mode = "Linear Phase"
        self.vst.tape_mode = "NR8"
        self.vst.input_filters_on_off = False
        self.vst.tone_on_off = False

        if not self.bypass:
            self.vst.tape_on_off = self.tape_enable
            if self.tape_enable:
                self.vst.output_gain = -6
                self.vst.tape_saturation = self.tape_saturation
                self.vst.tape_drive = self.tape_drive
                self.vst.tape_bias = self.tape_bias

            self.vst.loss_on_off = self.playback_loss_enable
            self.vst.wow_flutter_on_off = self.wow_flutter_enable
            if self.wow_flutter_enable:
                self.vst.flutter_depth = 0.75
                self.vst.wow_depth = 0.75
                self.vst.wow_variance = 1.0
        else:
            self.vst.bypass = self.bypass

    def __call__(self, x):
        """
        Args:
            x : torch.tensor or numpy.array of size (BATCH_SIZE, CHANNELS, SEQUENCE_LENGTH)
        """
        assert x.ndim == 3, "Input should be 3D"

        # Flags
        to_tensor = False # Inputs and outputs are tensors

        # Fiddle with type
        if torch.is_tensor(x):
            x = x.numpy()
            to_tensor = True

        y = np.zeros(x.shape)
        for batch_idx in range(x.shape[0]):
            batch = x[batch_idx, :, :]

            if self.warm_start:
                # Process with extra padding/warmup time
                batch_padded = np.pad(batch,
                                      ((0, 0), (int(self.warmup * self.fs), 0)))
                output_padded = self.vst(batch_padded, self.fs)
                output = output_padded[:, int(self.warmup * self.fs):]
            else:
                output = self.vst(batch, self.fs)

            # Compensate for processing delay
            SAMPLE_DELAY = 2
            output = np.roll(output, SAMPLE_DELAY, axis=1)
            output[:, :SAMPLE_DELAY] = np.zeros(
                ((output.shape[0], SAMPLE_DELAY)))

            if self.delay_enable:
                output = np.roll(output, int(self.delay_time * self.fs))
                # TODO. Add zeros to start

            y[batch_idx, :, :] = output

        # Unfiddle type
        if to_tensor:
            y = torch.tensor(y)

        return y


class Tape:

    # TODO - Add method for setting batch size post initialization

    def __init__(self,
                 batch_size=1,
                 fs=int(48e3),
                 oversampling=16,
                 signal_amplitude=1e-3,
                 bias_amplitude=5e-3,
                 bias_enable=True,
                 playback_loss_enable=False,
                 FIR_order=2**7,
                 startup_enable=True,
                 delay_enable=False,
                 return_full=False,
                 return_internal=False,
                 debug=False):
        """
        Based on VA Reel-to-Reel tape recorder from [*].

        [*] Chowdhury, Jatin. “Real-Time Physical Modelling for Analog Tape Machines.”
        In Proceedings of the International Conference on Digital Audio Effects (DAFx),
        Birmingham, UK, 2019.

        Parameters
        ----------
        fs : int, optional
            sampling rate. The default is int(48e3).
        batch_size : int, optional
            number of waveforms processed in parallel. The default is 1.
        oversampling : int, optional
            oversampling factor. The default is 8.
        signal_amplitude : float, optional
            input voltage to current conversion factor. The default is 1-e3.
        bias_amplitude : float, optional
            recording bias current amplitude. The default is 5-e3.
        bias_enable : bool, optional
            enable recording bias. The default is False.
        playback_loss_enable : bool, optional
            enable playback loss filter. The default is False.
        FIR_order: int, optional
            Playback loss FIR filter order. The default is 2**7.
        startup_enable : bool, optional
            enable startup time for bias signal. The default is True.
        delay_enable : bool, optional
            enable head-to-head delay. The default is False.
        return_full : bool, optional
            return full accumulated signal. The default is False.
        return_internal : bool, optional
            return internal signals. The default is False.
        debug : bool, optional
            enable debug mode. The default is False.
        """
        # GLOBAL ATTRIBUTES
        self.debug = debug
        self.fs = fs
        self.Ts = 1 / fs
        self.return_internal = return_internal
        self.return_full = return_full

        # DATA
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.dtype = torch.float64

        # PRE AMP
        self.signal_amplitude = signal_amplitude

        # OVERSAMPLING
        self.oversampling = oversampling
        self.fs_OS = fs * oversampling
        self.Ts_OS = 1 / (fs * oversampling)
        self.resampler_OS = self._initialize_resampler(self.fs, self.fs_OS)

        # BIAS
        self.bias_enable = bias_enable
        self.bias_amplitude = bias_amplitude
        self.BIAS_FREQ = 48e3

        # RECORD HEAD
        self.REC_N = 100  # turns of wire
        self.REC_E = 0.1  # head efficiency
        self.REC_G = 6e-6 # record head gap

        # TAPE MAGNETIC PARAMETERS, Chowdhury
        # self.TAPE_Ms = 3.5e5     # magnetic saturation
        # self.TAPE_A = 22e3       # anhysteretic magnetisation
        # self.TAPE_ALPHA = 1.6e-3 # mean field parameter
        # self.TAPE_K = 27e3       # hysteresis loop width
        # self.TAPE_C = 1.7e-1     # magnetic susceptibilities

        # TAPE, Holters & Zölzer
        self.TAPE_Ms = 1.6e6     # magnetic saturation
        self.TAPE_A = 1.1e3      # anhysteretic magnetisation
        self.TAPE_ALPHA = 1.6e-3 # mean field parameter
        self.TAPE_K = 4.0e2      # hysteresis loop width
        self.TAPE_C = 1.7e-1     # magnetic susceptibilities

        # DOWNSAMPLE
        self.resampler_DS = self._initialize_resampler(self.fs_OS, self.fs)

        # DELAY
        self.delay_enable = delay_enable

        # TAPE PHYSICAL PARAMETERS
        self.TAPE_V = 7.5 * 2.54e-2 # tape speed
        self.TAPE_DELTA = 35e-6     # tape thickess

        # PLAYBACK HEAD
        self.PLAY_N = 1000            # turns of wire
        self.PLAY_E = 1.0             # head efficiency
        self.PLAY_G = 6e-6            # playback head gap
        self.PLAY_MU0 = 1.257e-6      # permeability of free space
        self.PLAY_W = 0.125 * 2.54e-2 # playback head width
        self.PLAY_D = 20e-6           # playback head spacing

        # PLAYBACK LOSSES
        self.playback_loss_enable = playback_loss_enable
        self.N_FIR = FIR_order # approximation filter order
        self._compute_filter()

        # POST AMP
        # Haxing input/output to have unity gain.
        if np.isclose(signal_amplitude, 1e-4):
            unity_term = 0.75
        elif np.isclose(signal_amplitude, 1e-3):
            unity_term = 0.525
        else:
            unity_term = 0.935
        self.POST_GAIN = unity_term * (1 / signal_amplitude) * (
            1 / self.PLAY_D)                                     # output gain correction

        # MEMORY
        self._initialize_memory()

        # STARTUP
        self.startup_enable = startup_enable
        self.T_STARTUP = 1e-2
        self.FLAG_STARTUP = True
        if startup_enable:
            self._startup()

    #%% INITIALIZATIONS

    def _initialize_memory(self):
        # Recursion
        self.H_prev = torch.zeros(self.batch_size,
                                  dtype=self.dtype,
                                  device=self.device)
        self.Hprime_prev = torch.zeros_like(self.H_prev)
        self.M_prev = torch.zeros_like(self.H_prev)

        # Bias
        self.bias_phase = 0.0

        # Waveforms
        self.I_in = torch.empty((self.batch_size, 0),
                                dtype=self.dtype,
                                device=self.device)
        self.I_in_OS = torch.empty_like(self.I_in)
        self.I_rec_OS = torch.empty_like(self.I_in)
        self.H_rec_OS = torch.empty_like(self.I_in)
        self.M_OS = torch.empty_like(self.I_in)
        self.M = torch.empty_like(self.I_in)
        self.M_del = torch.empty_like(self.I_in)
        self.V_play = torch.empty_like(self.I_in)
        self.V_out = torch.empty_like(self.I_in)

    def _initialize_resampler(self, fs_orig, fs_new):
        resampler = T.Resample(fs_orig, fs_new, dtype=self.dtype)
        return resampler

    def _compute_filter(self, return_internal=False):
        """ Playback loss filter """
        # Frequencies and wavenumbers
        f = np.linspace(0, self.fs, self.N_FIR)
        k = (2 * np.pi * f[1:int(self.N_FIR / 2)]) / self.TAPE_V

        # Compute loss
        spacing_loss = np.exp(-k * self.PLAY_D)
        thickness_loss = (1 - np.exp(-k * self.TAPE_DELTA)) / (k *
                                                               self.TAPE_DELTA)
        gap_loss = np.sin(k * self.PLAY_G / 2) / (k * self.PLAY_G / 2)
        loss = spacing_loss * thickness_loss * gap_loss

        # FIR approximation
        H = np.zeros((self.N_FIR),)
        H[0] = 1
        H[1:int(self.N_FIR / 2)] = loss
        H[int(self.N_FIR / 2):] = np.flip(H[0:int(self.N_FIR / 2)], 0)

        # Impulse response
        h = scipy.fft.ifft(H)
        h = np.abs(h) # Get rid of phase

        # Filter coefficients
        b = torch.tensor(h, device=self.device,
                         dtype=self.dtype).expand(self.batch_size, -1)
        a = torch.zeros_like(b)
        a[:, 0] = 1.0

        # Chowdhury's "roll your IDFT"
        # h2 = np.zeros(self.N_FIR)
        # for n_k in range (self.N_FIR):
        #     for k_n in range (self.N_FIR):
        #         h2[n_k] += H[k_n] * np.cos (2 * np.pi * k_n * n_k / self.N_FIR)
        #     h2[n_k] *= (1/self.N_FIR)

        if return_internal:
            return f, k, spacing_loss, thickness_loss, gap_loss, loss, H, b
        else:
            self.b = b
            self.a = a

    def _startup(self):
        """ Ramp up bias signal before further processing
        """
        # input
        t = np.arange(0, self.T_STARTUP, self.Ts)
        x = torch.zeros(t.shape, device=self.device,
                        dtype=self.dtype).expand(self.batch_size, -1)

        # Process
        _ = self(x)

    #%% DSP

    def __call__(self, V_in):
        if self.debug:
            start = time.time()
            print("Timing:         total       - relative")
            prev = start

        if V_in.shape[0] < self.batch_size:                       # batch_size < BATCH_SIZE
                                                                  # extend with zeros
            input_pad = torch.zeros(
                (self.batch_size - V_in.shape[0], V_in.shape[1]),
                device=self.device,
                dtype=self.dtype)
            V_in = torch.cat((V_in, input_pad), dim=0)
            PADDING = True
        else:
            PADDING = False

        I_in = self.H_pre(V_in)         # Pre-amplifier
        I_in_OS = self.oversample(I_in) # Oversample
        I_rec_OS = self.bias(I_in_OS)   # Bias
        H_rec_OS = self.H_rec(I_rec_OS) # Recording head

        if self.debug:
            end = time.time()
            print(
                f"Recording head: {'{:.3f}'.format(1000*(end - start))}ms     - {'{:.3f}'.format(1000*(end - prev))}ms"
            )
            prev = end

        M_OS = self.H_mag(H_rec_OS) # Magnetization

        if self.debug:
            end = time.time()
            print(
                f"Magnetization:  {'{:.3f}'.format(1000*(end - start))}ms  - {'{:.3f}'.format(1000*(end - prev))}ms"
            )
            prev = end

        M = self.downsample(M_OS)   # Downsample
        M_del = self.delay(M)       # Delay
        V_play = self.H_play(M_del) # Playback head
        V_out = self.H_post(V_play) # Post-amplifier

        # Retrieve from memory
        I_in_mem, I_in_OS_mem, I_rec_OS_mem, H_rec_OS_mem, M_OS_mem, M_mem, M_del_mem, V_play_mem, V_out_mem = self.I_in, self.I_in_OS, self.I_rec_OS, self.H_rec_OS, self.M_OS, self.M, self.M_del, self.V_play, self.V_out
        # Update memory
        self.I_in, self.I_in_OS, self.I_rec_OS, self.H_rec_OS, self.M_OS, self.M, self.M_del, self.V_play, self.V_out = I_in, I_in_OS, I_rec_OS, H_rec_OS, M_OS, M, M_del, V_play, V_out

        if self.return_full:
            I_in = torch.cat((I_in_mem, I_in), dim=1)
            I_in_OS = torch.cat((I_in_OS_mem, I_in_OS), dim=1)
            I_rec_OS = torch.cat((I_rec_OS_mem, I_rec_OS), dim=1)
            H_rec_OS = torch.cat((H_rec_OS_mem, H_rec_OS), dim=1)
            M_OS = torch.cat((M_OS_mem, M_OS), dim=1)
            M = torch.cat((M_mem, M), dim=1)
            M_del = torch.cat((M_del_mem, M_del), dim=1)
            V_play = torch.cat((V_play_mem, V_play), dim=1)
            V_out = torch.cat((V_out_mem, V_out), dim=1)

        if PADDING:                                                   # take out zeros
            V_in, V_out = V_in[:self.batch_size - input_pad.
                               shape[0], :], V_out[:self.batch_size -
                                                   input_pad.shape[0], :]
            PADDING = False
            assert not (self.return_internal), "Not implemented"

        if self.debug:
            end = time.time()
            print(
                f"Total:          {'{:.3f}'.format(1000*(end - start))}ms  - {'{:.3f}'.format(1000*(end - prev))}ms"
            )

        if self.return_internal:
            return I_in, I_in_OS, I_rec_OS, H_rec_OS, M_OS, M, M_del, V_play, V_out
        else:
            return V_out

    def H_pre(self, V_in):
        """ Pre-amplifier """
        I_in = self.signal_amplitude * V_in
        return I_in

    def oversample(self, I_in):
        """ Oversampling """
        I_in_OS = self.resampler_OS(I_in)
        return I_in_OS

    def bias(self, I_in_OS):
        """ Bias current"""
        I_rec = torch.zeros_like(I_in_OS)
        if self.bias_enable:
            # create bias
            t_bias = np.linspace(self.bias_phase,
                                 self.bias_phase + I_rec.shape[1] * self.Ts_OS,
                                 I_rec.shape[1])
            bias = np.sin(2 * np.pi * self.BIAS_FREQ * t_bias + self.bias_phase)

            # bias amplitude
            amp_bias = self.bias_amplitude * np.ones(t_bias.shape)
            if self.FLAG_STARTUP:
                # wait for BIAS_SETTLE
                BIAS_SETTLE = int(self.T_STARTUP / (2 * self.Ts_OS))
                amp_bias[:BIAS_SETTLE] = np.zeros(BIAS_SETTLE)
                # gradually increase bias for BIAS_STARTUP
                BIAS_STARTUP = BIAS_SETTLE
                amp_bias[BIAS_SETTLE:BIAS_SETTLE + BIAS_STARTUP] = np.linspace(
                    0, self.bias_amplitude, int(BIAS_STARTUP))
                # update flag
                self.FLAG_STARTUP = False
            bias *= amp_bias

            # apply
            bias = torch.tensor(bias, device=self.device, dtype=self.dtype)
            bias = bias.expand(I_in_OS.shape[0], -1)
            I_rec += I_in_OS + bias

            # increment phase
            self.bias_phase += len(t_bias) * self.Ts_OS
        else:
            I_rec = I_in_OS

        return I_rec

    def H_rec(self, I_in):
        """ Recording head """
        return (self.REC_N * self.REC_E * I_in) / self.REC_G

    def H_mag(self, H):
        """
        Magnetization
        
        H : torch.tensor of shape (B,N)
        """

        M = torch.zeros_like(H)
        for n in range(H.shape[1]): # iterate over samples
            Hn = H[:, n]

            # Time-derivative of H w/ trapezoidal rule.
            Hprime = 2 * (Hn - self.H_prev) / self.Ts_OS - self.Hprime_prev

            # 4th Order Runge-Kutta
            k1 = self.Ts_OS * self._f(self.M_prev, self.H_prev,
                                      self.Hprime_prev)

            k2 = self.Ts_OS * self._f(self.M_prev + k1 / 2,
                                      (Hn + self.H_prev) / 2,
                                      (Hprime + self.Hprime_prev) / 2)
            k3 = self.Ts_OS * self._f(self.M_prev + k2 / 2,
                                      (Hn + self.H_prev) / 2,
                                      (Hprime + self.Hprime_prev) / 2)
            k4 = self.Ts_OS * self._f(self.M_prev + k3, Hn, Hprime)

            M[:, n] = self.M_prev + k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6

            # prevent saturating tape beyond Ms
            M[:, n] = torch.clamp(M[:, n], -self.TAPE_Ms, self.TAPE_Ms)

            self.H_prev = Hn
            self.Hprime_prev = Hprime
            self.M_prev = M[:, n]

        return M

    def downsample(self, M_OS):
        """ Downsampling """
        # Process with startup time added
        M = self.resampler_DS(torch.cat((self.M_OS, M_OS), dim=1))
        M = M[:, self.M.shape[1]:]
        return M

    def delay(self, M):
        """ Delay (dummy) """
        # TODO. Add
        return M.clone()

    def H_play(self, M):
        """ Playback head """
        g = self.PLAY_N * self.PLAY_W * self.PLAY_E * self.TAPE_V * self.PLAY_MU0 * self.PLAY_G
        V = g * M
        if self.playback_loss_enable:
            V = F.lfilter(V, self.a,
                          self.b)    # TODO. Take into account filter state
        else:
            pass
        return V

    def H_post(self, x):
        """ Post-amplifier """
        y = self.POST_GAIN * x
        return y

    def return_filter(self):
        """ Get playback filters """
        return self._compute_filter(True)

    ## INTERNAL METHODS

    def _f(self, Mn, Hn, Hprimen):
        """
        Hysteretic magnetization function.
        
        Mn, Hn, Hprimen : torch.tensor of shape (B,)
        """
        Q = (Hn + self.TAPE_ALPHA * Mn) / self.TAPE_A

        # Langevin function
        LQ = Q.clone()
        indices = torch.abs(LQ) > 1e-4
        LQ[indices] = (1 / torch.tanh(LQ[indices])) - 1 / LQ[indices]
        LQ[~indices] = LQ[~indices] / 3

        # Time-derivative of LQ
        LprimeQ = LQ.clone()
        indices = torch.abs(LprimeQ) > 1e-4
        LprimeQ[indices] = 1 / (LprimeQ[indices]**2) - (
            1 / torch.tanh(LprimeQ[indices]))**2 + 1
        LprimeQ[~indices] = 1 / 3

        # Anisotropic magnetization
        M_an = self.TAPE_Ms * LQ
        M_diff = M_an - Mn

        # Deltas
        delta_S = torch.zeros_like(Hprimen)
        indices = Hprimen > 0
        delta_S[indices] = 1
        delta_S[~indices] = -1

        delta_M = torch.zeros_like(delta_S)
        indices = np.sign(delta_S) == np.sign(M_diff)
        delta_M[indices] = 1
        delta_M[~indices] = 0

        # Compute terms
        term1_nom = (1 - self.TAPE_C) * delta_M * M_diff
        term1_den = (
            1 - self.TAPE_C) * delta_S * self.TAPE_K - self.TAPE_ALPHA * M_diff
        term1 = (term1_nom / term1_den) * Hprimen
        term2 = self.TAPE_C * (self.TAPE_Ms / self.TAPE_A) * Hprimen * LprimeQ
        term3 = 1 - self.TAPE_C * self.TAPE_ALPHA * (self.TAPE_Ms /
                                                     self.TAPE_A) * LprimeQ

        # Output
        f = (term1 + term2) / term3

        return f
