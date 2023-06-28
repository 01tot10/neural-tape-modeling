#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Alec-Wright
"""

#%% Imports

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import weight_norm
from utilities.utilities import TimeFreqConverter

#%% Classes


class MelGCrit(nn.Module):
    """Critic from MelGan"""

    def __init__(self, num_D, ndf, n_layers, downsampling_factor):
        super().__init__()
        self.n_layers = n_layers
        self.num_D = num_D
        self.model = nn.ModuleDict()
        for i in range(num_D):
            self.model[f"disc_{i}"] = NLayerDiscriminator(
                ndf,
                n_layers,
                downsampling_factor,
            )

        self.downsample = nn.AvgPool1d(4,
                                       stride=2,
                                       padding=1,
                                       count_include_pad=False)
        self.apply(weights_init)

    def forward(self, x):
        """ Forward """
        results = []
        for _, disc in self.model.items():
            results.append(disc(x))
            self.downsample(x)
        return results

    def train_crit(self, fake_ins, real_ins, optimiser):
        """ Train critic. """
        D_fake = self(fake_ins)
        D_real = self(real_ins)
        loss_D = 0
        for scale in D_fake:
            loss_D += F.relu(1 + scale[-1]).mean()
        for scale in D_real:
            loss_D += F.relu(1 - scale[-1]).mean()
        loss_D.backward()
        optimiser.step()
        return loss_D.item()

    def train_gen(self, gen_out, optimiser):
        """ Train generator. """
        D_fake = self(gen_out)
        loss_G = 0
        for scale in D_fake:
            loss_G += -scale[-1].mean()
        loss_G.backward()
        optimiser.step()
        return loss_G.item()


class NLayerDiscriminator(nn.Module):
    """Container for multiscale for MelGan critic"""

    def __init__(self, ndf, n_layers, downsampling_factor):
        super().__init__()
        model = nn.ModuleDict()

        model["layer_0"] = nn.Sequential(
            nn.ReflectionPad1d(7),
            WNConv1d(1, ndf, kernel_size=15),
            nn.LeakyReLU(0.2, True),
        )

        nf = ndf
        stride = downsampling_factor
        for n in range(1, n_layers + 1):
            nf_prev = nf
            nf = min(nf * stride, 1024)

            model["layer_%d" % n] = nn.Sequential(
                WNConv1d(
                    nf_prev,
                    nf,
                    kernel_size=stride * 10 + 1,
                    stride=stride,
                    padding=stride * 5,
                    groups=nf_prev // 4,
                ),
                nn.LeakyReLU(0.2, True),
            )

        nf = min(nf * 2, 1024)
        model["layer_%d" % (n_layers + 1)] = nn.Sequential(
            WNConv1d(nf_prev, nf, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2, True),
        )

        model["layer_%d" % (n_layers + 2)] = WNConv1d(nf,
                                                      1,
                                                      kernel_size=3,
                                                      stride=1,
                                                      padding=1)

        self.model = model

    def forward(self, x):
        """ Forward. """
        results = []
        for _, layer in self.model.items():
            x = layer(x)
            results.append(x)
        return results


class MultiSpecCrit(nn.Module):
    """Multi-scale spect crit container for spectral critic"""

    def __init__(self,
                 scales,
                 kernel_sizes,
                 hop_sizes,
                 layers,
                 chan_in,
                 chan_fac,
                 stride,
                 g_fac,
                 test_in_len,
                 tf_rep='spec',
                 log=False):
        super().__init__()
        self.scales = scales
        self.models = nn.ModuleList()

        for i in range(len(scales)):
            self.models.append(
                SpecCrit(scales[i], kernel_sizes[i], hop_sizes[i], layers,
                         chan_in, chan_fac, stride, g_fac, tf_rep, log,
                         test_in_len))

    def forward(self, x):
        """ Forward. """
        results = []
        for model in self.models:
            results.append(model(x))
        return results

    def train_crit(self, fake_ins, real_ins, optimiser):
        """ Train critic. """
        D_fake = self(fake_ins)
        D_real = self(real_ins)
        loss_D = 0
        for scale in D_fake:
            loss_D += F.relu(1 + scale).mean()
        for scale in D_real:
            loss_D += F.relu(1 - scale).mean()
        loss_D.backward()
        optimiser.step()
        return loss_D.item()

    def train_gen(self, gen_out, optimiser):
        """ Train generator. """
        D_fake = self(gen_out)
        loss_G = 0
        for scale in D_fake:
            loss_G += -scale.mean()
        loss_G.backward()
        optimiser.step()
        return loss_G.item()


class SpecCrit(nn.Module):
    """Spectral critic that takes TF repr of audio as input"""

    def __init__(self, scale, kernel_size, hop_size, layers, chan_in, chan_fac,
                 stride, g_fac, tf_rep, log, test_in_len):
        super().__init__()
        # Set number of layers  and hidden_size for network layer/s
        self.scale = scale
        self.layers = nn.ModuleList()
        self.log = log
        self.log_eps = 1e-5
        self.tf_rep = tf_rep

        self.layers += [
            TimeFreqConverter(n_fft=scale,
                              hop_length=hop_size,
                              win_length=scale,
                              sampling_rate=44100,
                              n_mel_channels=160)
        ]

        if tf_rep == 'spec':
            layer1_chan = (scale // 2) + 1
        elif tf_rep == 'mel':
            layer1_chan = 160

        self.layers += [
            WNConv1d(in_channels=layer1_chan,
                     out_channels=chan_in,
                     kernel_size=10),
            nn.LeakyReLU(0.2, True)
        ]

        for _ in range(layers - 2):
            out_channels = min(chan_in * chan_fac, 1024)
            conv_layer = [
                WNConv1d(in_channels=chan_in,
                         out_channels=out_channels,
                         kernel_size=kernel_size,
                         stride=stride,
                         groups=(out_channels) // g_fac),
                nn.LeakyReLU(0.2, True)
            ]
            self.layers += conv_layer
            chan_in = out_channels

        self.layers += [
            WNConv1d(in_channels=chan_in, out_channels=chan_in, kernel_size=5),
            nn.LeakyReLU(0.2, True)
        ]

        self.layers += [
            WNConv1d(in_channels=chan_in, out_channels=1, kernel_size=3)
        ]

        test_out = self.test_input(test_in_len)
        print(
            'Spect Disc = {}, kernel size = {}, layers = {}, output size = {},{},{} '
            .format(scale, kernel_size, layers, test_out.shape[0],
                    test_out.shape[1], test_out.shape[2]))

    def forward(self, x):
        """ Forward. """
        if self.tf_rep == 'spec':
            x = self.layers[0](x).squeeze()
        elif self.tf_rep == 'mel':
            _, x = self.layers[0](x, mel=True)
            x = x.squeeze()
        if self.log:
            x = torch.log10(torch.clamp(x, min=self.log_eps))
        for layer in self.layers[1:]:
            x = layer(x)
        return x

    def test_input(self, seq_len):
        """ Test input. """
        dummy_input = torch.randn((10, 1, seq_len))
        dummy_output = self(dummy_input)
        return dummy_output


class DilatedConvDisc(nn.Module):
    """Time domain crit based on dilated convolutions"""

    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 kernel_size=5,
                 layers=12,
                 blocks=2,
                 conv_channels=64,
                 dil_fac=2,
                 nl_func="LeakyReLU",
                 nl_params={"negative_slope": 0.2},
                 test_in_len=1):
        super(DilatedConvDisc, self).__init__()
        # Set number of layers  and hidden_size for network layer/s
        self.layers = nn.ModuleList()
        for blocks in range(blocks - 1):
            dilation = 1
            for _ in range(layers - 1):
                conv_layer = [
                    WNConv1d(in_channels=in_channels,
                             out_channels=conv_channels,
                             kernel_size=kernel_size,
                             dilation=dilation),
                    getattr(nn, nl_func)(**nl_params)
                ]
                self.layers += conv_layer
                dilation *= dil_fac
                in_channels = conv_channels
        self.layers += [
            WNConv1d(in_channels=conv_channels,
                     out_channels=out_channels,
                     kernel_size=kernel_size,
                     dilation=1)
        ]

        test_out = self.test_input(test_in_len)
        print('Dilated Conv Disc, output size = {},{},{} '.format(
            test_out.shape[0], test_out.shape[1], test_out.shape[2]))

    def forward(self, x):
        """ Forward. """
        for layer in self.layers:
            x = layer(x)
        return x

    def train_crit(self, fake_ins, real_ins, optimiser):
        """ Train critic. """
        D_fake = self(fake_ins)
        D_real = self(real_ins)
        loss_D = F.relu(1 + D_fake).mean()
        loss_D += F.relu(1 - D_real).mean()
        loss_D.backward()
        optimiser.step()
        return loss_D.item()

    def train_gen(self, gen_out, optimiser):
        """ Train generator. """
        D_fake = self(gen_out)
        loss_G = -D_fake.mean()
        loss_G.backward()
        optimiser.step()
        return loss_G.item()

    def test_input(self, seq_len):
        """ Test input. """
        dummy_input = torch.randn((10, 1, seq_len))
        dummy_output = self(dummy_input)
        return dummy_output


#%% Methods


def get_critic(critic_name, critic_pars, device, crit_lr, test_in_len):
    """ Get critic. """
    if critic_name == 'MelGanCrit':
        critic = MelGCrit(**critic_pars).to(device=device)
    elif critic_name == 'DilatedConvDisc':
        critic_pars['test_in_len'] = test_in_len
        critic = DilatedConvDisc(**critic_pars).to(device=device)
    elif critic_name == 'MultiSpecCrit':
        critic_pars['test_in_len'] = test_in_len
        critic = MultiSpecCrit(**critic_pars).to(device=device)
    optC = torch.optim.Adam(critic.parameters(), lr=crit_lr, betas=(0.5, 0.9))
    return critic, optC


# Utility functions for Weight Norm conv layers, and weight init


def WNConv1d(*args, **kwargs):
    """ WNConv1d. """
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConv2d(*args, **kwargs):
    """ WNConv2d. """
    return weight_norm(nn.Conv2d(*args, **kwargs))


def weights_init(m):
    """ Weights init. """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
