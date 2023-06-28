#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Alec-Wright
"""

#%% Imports

import torch
import torch.nn.functional as F
from GreyBoxDRC import loss_funcs
from torch import nn
from utilities.utilities import TimeFreqConverter

#%% Classes


class val_loss_supervised(nn.Module):
    """ Supervised validation loss. """

    def __init__(self,
                 device='cpu',
                 spec_scales=[2048, 1024, 512, 256, 128, 64]):
        super().__init__()
        self.spec_scales = spec_scales
        self.specs = [
            TimeFreqConverter(n_fft=scale,
                              hop_length=scale // 4,
                              win_length=scale,
                              sampling_rate=44100,
                              n_mel_channels=160).to(device)
            for scale in self.spec_scales
        ]
        self.mel_spec = TimeFreqConverter(n_fft=2048,
                                          hop_length=2048 // 4,
                                          win_length=2048,
                                          sampling_rate=44100,
                                          n_mel_channels=160).to(device)

        self.ESR = loss_funcs.ESRLoss(dc_pre=False)
        self.ESRDCPre = loss_funcs.ESRLoss(dc_pre=True)
        self.MSE = nn.MSELoss()

        self.log_eps = 1e-5
        self.losses = {
            'ms_spec_loss': 0,
            'ms_log_spec_loss': 0,
            'mel_spec_loss': 0,
            'log_mel_spec_loss': 0,
            'ESR': 0,
            'MSE': 0,
            'ESRDCPre': 0
        }
        self.bst_losses = {
            'ms_spec_loss': [0, 1e9],
            'ms_log_spec_loss': [0, 1e9],
            'mel_spec_loss': [0, 1e9],
            'log_mel_spec_loss': [0, 1e9],
            'ESR': [0, 1e9],
            'MSE': [0, 1e9],
            'ESRDCPre': [0, 1e9]
        }
        self.iter_count = 0

    def forward(self, output, target):
        """ Forward. """

        losses = {
            'ms_spec_loss': 0,
            'ms_log_spec_loss': 0,
            'mel_spec_loss': 0,
            'log_mel_spec_loss': 0,
            'ESR': 0,
            'MSE': 0,
            'ESRDCPre': 0
        }

        for spec in self.specs:
            magx = spec(output, mel=False)
            magy = spec(target, mel=False)
            losses['ms_spec_loss'] += F.l1_loss(magx, magy)

            logx = torch.log10(torch.clamp(magx, self.log_eps))
            logy = torch.log10(torch.clamp(magy, self.log_eps))
            losses['ms_log_spec_loss'] += F.l1_loss(logx, logy)

        _, melx = self.mel_spec(output, mel=True)
        _, mely = self.mel_spec(target, mel=True)
        losses['mel_spec_loss'] = F.l1_loss(melx, mely)

        logmelx = torch.log10(torch.clamp(melx, min=self.log_eps))
        logmely = torch.log10(torch.clamp(mely, min=self.log_eps))
        losses['log_mel_spec_loss'] = F.l1_loss(logmelx, logmely)

        output, target = output.unsqueeze(1), target.unsqueeze(1)
        losses['ESR'] = self.ESR(output, target)
        losses['ESRDCPre'] = self.ESRDCPre(output, target)
        losses['MSE'] = self.MSE(output, target)

        return losses

    def add_loss(self, output, target):
        """ Add loss. """
        if self.iter_count == 0:
            self.losses = {
                'ms_spec_loss': 0,
                'ms_log_spec_loss': 0,
                'mel_spec_loss': 0,
                'log_mel_spec_loss': 0,
                'ESR': 0,
                'MSE': 0,
                'ESRDCPre': 0
            }

        losses = self(output, target)
        for key in losses:
            self.losses[key] += losses[key]
        self.iter_count += 1

    def end_val(self, cur_step):
        """ End validation. """
        for key in self.losses:
            loss = self.losses[key] / self.iter_count
            if loss < self.bst_losses[key][1]:
                self.bst_losses[key] = [cur_step, loss]
        self.iter_count = 0


#%% Methods


def multi_mag_loss(output,
                   target,
                   fft_sizes=(2048, 1024, 512, 256, 128, 64),
                   log=False):
    """ Multi-resolution magnitude response loss. """
    losses = {}
    total_loss = 0
    for fft in fft_sizes:
        hop = fft // 4
        losses[fft] = mag_spec_loss(output, target, fft, hop, log).item()
        total_loss += losses[fft]
    losses['total'] = total_loss
    return losses


def mag_spec_loss(output, target, fft_size=512, hop_size=128):
    """ Magnitude response loss. """

    magx = torch.abs(
        torch.stft(output,
                   n_fft=fft_size,
                   hop_length=hop_size,
                   return_complex=True))
    magy = torch.abs(
        torch.stft(target,
                   n_fft=fft_size,
                   hop_length=hop_size,
                   return_complex=True))

    return F.l1_loss(magx, magy)
