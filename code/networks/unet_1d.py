#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors:
    Eloi Moliner, Otto Mikkonen
"""

#%% Imports

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732

import torchaudio

#%% Classes


class UNet1D(nn.Module):

    def __init__(self, args, device):
        """UNet1D"""
        super(UNet1D, self).__init__()

        # Arguments
        self.args = args
        self.device = device
        self.depth = args.network.depth
        self.emb_dim = args.network.emb_dim
        self.use_norm = args.network.use_norm
        self.Ns = args.network.Ns
        self.Ss = args.network.Ss

        ## Network

        # Embedding
        self.embedding = RFF_MLP_Block(self.emb_dim).to(device)

        # Encoder
        Nin = args.network.Nin
        self.init_conv = nn.Conv1d(Nin,
                                   self.Ns[0],
                                   5,
                                   padding="same",
                                   padding_mode="zeros",
                                   bias=False)

        # Up/Down Sampling
        self.downs = nn.ModuleList([])
        self.middle = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        for i in range(self.depth):
            if i == 0:
                dim_in = self.Ns[i]
                dim_out = self.Ns[i]
            else:
                dim_in = self.Ns[i - 1]
                dim_out = self.Ns[i]

            if i < (self.depth - 1):
                self.downs.append(
                    nn.ModuleList([
                        ResnetBlock(dim_in,
                                    dim_out,
                                    self.use_norm,
                                    emb_dim=self.emb_dim,
                                    bias=False),
                        Downsample(self.Ss[i]),
                        CombinerDown("sum", 1, dim_out, bias=False)
                    ]))

            elif i == (self.depth - 1):                    # no downsampling in the last layer
                self.downs.append(
                    nn.ModuleList([
                        ResnetBlock(dim_in,
                                    dim_out,
                                    self.use_norm,
                                    bias=False,
                                    emb_dim=self.emb_dim),
                    ]))

        self.middle.append(
            nn.ModuleList([
                ResnetBlock(self.Ns[self.depth],
                            self.Ns[self.depth],
                            self.use_norm,
                            bias=False,
                            emb_dim=self.emb_dim)
            ]))
        for i in range(self.depth - 1, -1, -1):

            if i == 0:
                dim_in = self.Ns[i] * 2
                dim_out = self.Ns[i]
            else:
                dim_in = self.Ns[i] * 2
                dim_out = self.Ns[i - 1]

            if i > 0:
                self.ups.append(
                    nn.ModuleList([
                        ResnetBlock(dim_in,
                                    dim_out,
                                    use_norm=self.use_norm,
                                    bias=False,
                                    emb_dim=self.emb_dim),
                        Upsample(self.Ss[i]),
                        CombinerUp("sum", 1, dim_out, bias=False)
                    ]))

            elif i == 0:                                    # no downsampling in the last layer
                self.ups.append(
                    nn.ModuleList([
                        ResnetBlock(dim_in,
                                    dim_out,
                                    use_norm=self.use_norm,
                                    bias=False,
                                    emb_dim=self.emb_dim),
                    ]))

        # CropConcatBlock
        self.cropconcat = CropConcatBlock()

    def setup_CQT_len(self, len):
        """setup_CQT_len."""
        pass

    def forward(self, inputs, sigma):
        sigma = self.embedding(sigma)

        #print(xF.shape)
        x = inputs.unsqueeze(1)
        pyr = x
        x = self.init_conv(x)

        hs = []
        for i, modules in enumerate(self.downs):

            if i < (self.depth - 1):
                resnet, downsample, combiner = modules
                x = resnet(x, sigma)
                hs.append(x)
                x = downsample(x)
                pyr = downsample(pyr)
                x = combiner(pyr, x)

            elif i == (self.depth - 1): # no downsampling in the last layer
                (resnet,) = modules
                x = resnet(x, sigma)
                hs.append(x)

        for modules in self.middle:
            (resnet,) = modules
            x = resnet(x, sigma)

        pyr = None
        for i, modules in enumerate(self.ups):
            j = self.depth - i - 1
            if j > 0:
                resnet, upsample, combiner = modules

                skip = hs.pop()
                #print(x.shape, skip.shape)
                x = self.cropconcat(
                    x, skip
                )                    # there will be problems here, use cropping if necessary
                x = resnet(x, sigma)

                pyr = combiner(pyr, x)

                x = upsample(x)
                pyr = upsample(pyr)

            elif j == 0: # no upsampling in the last layer
                (resnet,) = modules
                skip = hs.pop()
                x = self.cropconcat(x, skip)
                         # x = torch.cat((x, hs.pop()), dim=1)
                x = resnet(x, sigma)
                pyr = combiner(pyr, x)

        #print("end ", x.shape)
        pred = pyr.squeeze(1)
        assert pred.shape == inputs.shape, "bad shapes"
        return pred


class RFF_MLP_Block(nn.Module):

    def __init__(self, emb_dim):
        """RFF_MLP_Block."""
        super().__init__()

        # Network
        self.RFF_freq = nn.Parameter(16 * torch.randn([1, 16]),
                                     requires_grad=False)
        self.MLP = nn.ModuleList([
            nn.Linear(32, emb_dim),
            nn.Linear(emb_dim, emb_dim),
        ])

    def forward(self, sigma):
        """
        Arguments:
          sigma:
              (shape: [B, 1], dtype: float32)

        Returns:
          x: embedding of sigma
              (shape: [B, 512], dtype: float32)
        """
        x = self._build_RFF_embedding(sigma)
        for layer in self.MLP:
            x = F.relu(layer(x))

        return x

    def _build_RFF_embedding(self, sigma):
        """
        Arguments:
          sigma:
              (shape: [B, 1], dtype: float32)
        Returns:
          table:
              (shape: [B, 64], dtype: float32)
        """
        freqs = self.RFF_freq
        table = 2 * np.pi * sigma * freqs
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)

        return table


class ResnetBlock(nn.Module):

    def __init__(
        self,
        dim,
        dim_out,
        use_norm=False,
        groups=8,
        emb_dim=32,
        num_dils=8,
        bias=True,
    ):
        """ResnetBlock."""
        super().__init__()

        # Arguments
        self.bias = bias
        self.use_norm = use_norm
        self.num_layers = num_dils

        # Network
        self.film = Film(dim, emb_dim, bias=bias)
        self.res_conv = nn.Conv1d(
            dim, dim_out, 1, padding_mode="zeros",
            bias=bias) if dim != dim_out else nn.Identity()
        if self.use_norm:
            self.gnorm = nn.GroupNorm(groups, dim)
        self.first_conv = nn.Sequential(nn.GELU(),
                                        nn.Conv1d(dim, dim_out, 1, bias=bias))

        self.H = nn.ModuleList()
        for i in range(self.num_layers):
            self.H.append(
                GatedResidualLayer(dim_out, 5, 2**i, bias=bias)
            )                                                   # sometimes I changed this 1,5 to 3,5. be careful!!! (in exp 80 as far as I remember)

    def forward(self, x, sigma):
        gamma, beta = self.film(sigma)

        if self.use_norm:
            x = self.gnorm(x)

        if self.bias:
            x = x * gamma + beta
        else:
            x = x * gamma # no bias

        y = self.first_conv(x)

        for h in self.H:
            y = h(y)

        return (y + self.res_conv(x)) / (2**0.5)


class Film(nn.Module):

    def __init__(self, output_dim, emb_dim, bias=True):
        """Film."""
        super().__init__()

        # Attributes
        self.bias = bias

        # Network
        if bias:
            self.output_layer = nn.Linear(emb_dim, 2 * output_dim)
        else:
            self.output_layer = nn.Linear(emb_dim, 1 * output_dim)

    def forward(self, sigma_encoding):
        sigma_encoding = self.output_layer(sigma_encoding)
        sigma_encoding = sigma_encoding.unsqueeze(
            -1)                                    # we need a secnond unsqueeze because our data is 2d [B,C,1,1]
        if self.bias:
            gamma, beta = torch.chunk(sigma_encoding, 2, dim=1)
        else:
            gamma = sigma_encoding
            beta = None

        return gamma, beta


class GatedResidualLayer(nn.Module):

    def __init__(self, dim, kernel_size, dilation, bias=True):
        """GatedResidualLayer."""
        super().__init__()

        # Network
        self.conv = nn.Conv1d(dim,
                              dim,
                              kernel_size=kernel_size,
                              dilation=dilation,
                              stride=1,
                              padding='same',
                              padding_mode='zeros',
                              bias=bias)               #freq convolution (dilated)
        self.act = nn.GELU()

    def forward(self, x):
        x = (x + self.conv(self.act(x))) / (2**0.5)

        return x


class Upsample(nn.Module):

    def __init__(self, S):
        """Upsample."""
        super().__init__()

        # Upsampler
        N = 2**12 # I use 2**12 as an arbitrary number, as we don't care about the sampling frequency of the latents
        self.resample = torchaudio.transforms.Resample(N, N * S)

    def forward(self, x):
        return self.resample(x)


class Downsample(nn.Module):

    def __init__(self, S):
        """Downsample."""
        super().__init__()

        # Downsampler
        N = 2**12 # I use 2**12 as an arbitrary number, as we don't care about the sampling frequency of the latents
        self.resample = torchaudio.transforms.Resample(N, N / S)

    def forward(self, x):
        return self.resample(x)


class CombinerUp(nn.Module):

    def __init__(self, mode, Npyr, Nx, bias=True):
        """CombinerUp."""
        super().__init__()

        # Attributes
        self.mode = mode

        # Network
        self.conv1x1 = nn.Conv1d(Nx, Npyr, 1, bias=bias)
        torch.nn.init.constant_(self.conv1x1.weight, 0)

    def forward(self, pyr, x):

        if self.mode == "sum":
            x = self.conv1x1(x)
            if pyr == None:
                return x
            else:

                return (pyr[..., 0:x.shape[-1]] + x) / (2**0.5)
        else:
            raise NotImplementedError


class CombinerDown(nn.Module):

    def __init__(self, mode, Nin, Nout, bias=True):
        """CombinerDown."""
        super().__init__()

        # Attributes
        self.mode = mode

        # Network
        self.conv1x1 = nn.Conv1d(Nin, Nout, 1, bias=bias)

    def forward(self, pyr, x):
        if self.mode == "sum":
            pyr = self.conv1x1(pyr)
            return (pyr + x) / (2**0.5)
        else:
            raise NotImplementedError


class CropConcatBlock(nn.Module):

    def forward(self, down_layer, x, **kwargs):
        x1_shape = down_layer.shape
        x2_shape = x.shape

        height_diff = (x1_shape[2] - x2_shape[2]) // 2
        down_layer_cropped = down_layer[:, :,
                                        height_diff:(x2_shape[2] + height_diff)]
        x = torch.cat((down_layer_cropped, x), 1)
        return x


#%% NOT USED

# class CropAddBlock(nn.Module):

#     def forward(self, down_layer, x, **kwargs):
#         x1_shape = down_layer.shape
#         x2_shape = x.shape

#         #print(x1_shape,x2_shape)
#         height_diff = (x1_shape[2] - x2_shape[2]) // 2

#         down_layer_cropped = down_layer[:, :,
#                                         height_diff:(x2_shape[2] + height_diff)]
#         x = torch.add(down_layer_cropped, x)
#         return x

# class FinalBlock(nn.Module):

#     def __init__(self, N0):
#         """
#         [B, T, F, N] => [B, T, F, 2]
#         Final block. Basiforwardy, a 3x3 conv. layer to map the output features to the output complex spectrogram.
#         """
#         super(FinalBlock, self).__init__()

#         # Network
#         ksize = (3, 3)
#         self.conv2 = ComplexConv1d(N0,
#                                    out_channels=1,
#                                    kernel_size=ksize,
#                                    stride=1,
#                                    padding='same',
#                                    padding_mode='zeros')

#     def forward(self, inputs):
#         pred = self.conv2(inputs)

#         return pred
