#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This work is licensed under a Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International License.
You should have received a copy of the license along with this
work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

Train diffusion-based generative model using the techniques described in the
paper "Elucidating the Design Space of Diffusion-Based Generative Models"

@author: eloimoliner, 01tot10
"""

#%% Imports

import argparse
import copy
import os
import re
import time
import warnings
from glob import glob

import numpy as np
import torch
import torchaudio

from utilities import training_stats
import utilities.training_utils as t_utils
from datasets_diffusion import TapeHissdset, ToyTrajectories
from networks.unet_1d import UNet1D
from omegaconf import OmegaConf

# False warning printed by PyTorch 1.12.
warnings.filterwarnings('ignore',
                        'Grad strides do not match bucket view strides')

#%% Classes


class EDM():
    """
    Definition of most of the diffusion parameterization, following
    (Karras et al., "Elucidating...", 2022)
    """

    def __init__(self, args):
        """
        Args:
            args (dictionary): hydra arguments
            sigma_data (float): 
        """
        self.args = args
        self.sigma_min = args.diff_params.sigma_min
        self.sigma_max = args.diff_params.sigma_max
        self.ro = args.diff_params.ro
        self.sigma_data = args.diff_params.sigma_data # depends on the training data!! precalculated variance of the dataset

        # parameters stochastic sampling
        self.Schurn = args.diff_params.Schurn
        self.Snoise = args.diff_params.Snoise

    def get_gamma(self, t):
        """
        Get the parameter gamma that defines the stochasticity of the sampler
        Args
            t (Tensor): shape: (N_steps, ) Tensor of timesteps, from which we will compute gamma
        """
        N = t.shape[0]
        gamma = torch.zeros(t.shape).to(t.device)

        # If desired, only apply stochasticity between a certain range of noises
        # Stmin is 0 by default and Stmax is a huge number by default.
        # (Unless these parameters are specified, this does nothing)
        indexes = torch.logical_and(t >= 0, t <= self.sigma_max)

        # We use Schurn=5 as the default in our experiments
        gamma[indexes] = gamma[indexes] + torch.min(
            torch.Tensor([self.Schurn / N, 2**(1 / 2) - 1]))

        return gamma

    def create_schedule(self, nb_steps):
        """
        Define the schedule of timesteps
        Args:
           nb_steps (int): Number of discretized steps
        """
        i = torch.arange(0, nb_steps + 1)
        t = (self.sigma_max**(1 / self.ro) + i / (nb_steps - 1) *
             (self.sigma_min**(1 / self.ro) - self.sigma_max**(1 / self.ro))
            )**self.ro
        t[-1] = 0
        return t

    def create_schedule_from_initial_t(self, initial_t, nb_steps):
        """
        Define the schedule of timesteps
        Args:
           nb_steps (int): Number of discretized steps
        """
        i = torch.arange(0, nb_steps + 1)
        t = (
            initial_t**(1 / self.ro) + i / (nb_steps - 1) *
            (self.sigma_min**(1 / self.ro) - initial_t**(1 / self.ro)))**self.ro
        t[-1] = 0
        return t

    def sample_ptrain_safe(self, N):
        """
        For training, getting  t according to the same criteria as sampling
        Args:
            N (int): batch size
        """
        a = torch.rand(N)
        t = (self.sigma_max**(1 / self.ro) + a *
             (self.sigma_min**(1 / self.ro) - self.sigma_max**(1 / self.ro))
            )**self.ro
        return t

    def sample_prior(self, shape, sigma):
        """
        Just sample some gaussian noise, nothing more
        Args:
            shape (tuple): shape of the noise to sample, something like (B,T)
            sigma (float): noise level of the noise
        """
        n = torch.randn(shape).to(sigma.device) * sigma
        return n

    def cskip(self, sigma):
        """
        Just one of the preconditioning parameters
        Args:
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
        
        """
        return self.sigma_data**2 * (sigma**2 + self.sigma_data**2)**-1

    def cout(self, sigma):
        """
        Just one of the preconditioning parameters
        Args:
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
        """
        return sigma * self.sigma_data * (self.sigma_data**2 + sigma**2)**(-0.5)

    def cin(self, sigma):
        """
        Just one of the preconditioning parameters
        Args:
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
        """
        return (self.sigma_data**2 + sigma**2)**(-0.5)

    def cnoise(self, sigma):
        """
        preconditioning of the noise embedding
        Args:
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
        """
        return (1 / 4) * torch.log(sigma)

    def lambda_w(self, sigma):
        """ Lambda_w. """
        return (sigma * self.sigma_data)**(-2) * (self.sigma_data**2 + sigma**2)

    def denoiser(self, xn, net, sigma):
        """
        This method does the whole denoising step,
        which implies applying the model and the preconditioning
        Args:
            x (Tensor): shape: (B,T) Intermediate noisy latent to denoise
            model (nn.Module): Model of the denoiser
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
        """
        if len(sigma.shape) == 1:
            sigma = sigma.unsqueeze(-1)
        cskip = self.cskip(sigma)
        cout = self.cout(sigma)
        cin = self.cin(sigma)
        cnoise = self.cnoise(sigma)

        return cskip * xn + cout * net(
            cin * xn, cnoise
        )                               # this will crash because of broadcasting problems, debug later!

    def prepare_train_preconditioning(self, x, sigma):
        """ prepare_train_preconditioning. """
        # Is calling the denoiser here a good idea?
        # Maybe it would be better to apply directly the preconditioning as in
        # the paper, even though Karras et al seem to do it this way in their code
        print(x.shape)
        noise = self.sample_prior(x.shape, sigma)

        cskip = self.cskip(sigma)
        cout = self.cout(sigma)
        cin = self.cin(sigma)
        cnoise = self.cnoise(sigma)

        target = (1 / cout) * (x - cskip * (x + noise))

        return cin * (x + noise), target, cnoise

    def loss_fn(self, net, x):
        """
        Loss function, which is the mean squared error between the denoised latent and the clean latent
        Args:
            net (nn.Module): Model of the denoiser
            x (Tensor): shape: (B,T) Intermediate noisy latent to denoise
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
        """
        sigma = self.sample_ptrain_safe(x.shape[0]).unsqueeze(-1).to(x.device)

        input, target, cnoise = self.prepare_train_preconditioning(x, sigma)

        print("inputs to net", input.shape, cnoise.shape)
        estimate = net(input, cnoise)

        error = estimate - target

        # here we have the chance to apply further emphasis to the error,
        # as some kind of perceptual frequency weighting could be
        return error**2, sigma


class Trainer():

    def __init__(self,
                 args,
                 dset,
                 network,
                 optimizer,
                 diff_params,
                 device='cpu'):
        """ Trainer. """
        self.args = args
        self.dset = dset
        self.network = network
        self.optimizer = optimizer
        self.diff_params = diff_params
        self.device = device

        # These are settings set by Karras. I am not sure what they do
        # np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
        torch.manual_seed(np.random.randint(1 << 31))
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.deterministic = False

        # Print model summary
        self.total_params = sum(
            p.numel() for p in self.network.parameters() if p.requires_grad)
        print("total_params: ", self.total_params / 1e6, "M")

        self.ema = copy.deepcopy(self.network).eval().requires_grad_(False)

        # Resume from checkpoint
        self.latest_checkpoint = None
        resuming = False
        if self.args.train.resume:
            if self.args.train.resume_checkpoint != "None":
                resuming = self.resume_from_checkpoint(
                    checkpoint_path=self.args.train.resume_checkpoint)
            else:
                resuming = self.resume_from_checkpoint()
            if not resuming:
                print("Could not resume from checkpoint")
                print("training from scratch")
            else:
                print("Resuming from iteration {}".format(self.it))

        if not resuming:
            self.it = 0
            self.latest_checkpoint = None

    def load_state_dict(self, state_dict):
        """ load_state_dict. """
        return t_utils.load_state_dict(state_dict,
                                       network=self.network,
                                       ema=self.ema,
                                       optimizer=self.optimizer)

    def resume_from_checkpoint(self, checkpoint_path=None, checkpoint_id=None):
        """
        Resume training from latest checkpoint available in the output directory
        """
        if checkpoint_path is not None:
            try:
                checkpoint = torch.load(checkpoint_path,
                                        map_location=self.device)
                print(checkpoint.keys())
                # If it is possible, retrieve the iteration number from the checkpoint
                try:
                    self.it = checkpoint['it']
                except:
                    self.it = 157007                                  #large number to mean that we loaded somethin, but it is arbitrary
                return self.load_state_dict(checkpoint)
            except Exception as e:
                print("Could not resume from checkpoint")
                print(e)
                print("training from scratch")
                self.it = 0
                return False
        else:
            try:
                print("trying to load a project checkpoint")
                print("checkpoint_id", checkpoint_id)
                if checkpoint_id is None:
                                                                      # find latest checkpoint_id
                    save_basename = f"{self.args.exp.exp_name}-*.pt"
                    save_name = f"{self.args.model_dir}/{save_basename}"
                    print(save_name)
                    list_weights = glob(save_name)
                    id_regex = re.compile(f"{self.args.exp.exp_name}-(\d*)\.pt")
                    list_ids = [
                        int(id_regex.search(weight_path).groups()[0])
                        for weight_path in list_weights
                    ]
                    checkpoint_id = max(list_ids)
                    print(checkpoint_id)

                checkpoint = torch.load(
                    f"{self.args.model_dir}/{self.args.exp.exp_name}-{checkpoint_id}.pt",
                    map_location=self.device)

                # If it is possible, retrieve the iteration number from the checkpoint
                try:
                    self.it = checkpoint['it']
                except:
                    self.it = 159000 #large number to mean that we loaded somethin, but it is arbitrary
                self.load_state_dict(checkpoint)
                return True
            except Exception as e:
                print(e)
                return False

    def state_dict(self):
        """ state_dict. """
        return {
            'it': self.it,
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'ema': self.ema.state_dict(),
            'args': self.args,
        }

    def save_checkpoint(self):
        """ save_checkpoint. """
        save_basename = f"{self.args.exp.exp_name}-{self.it}.pt"
        save_name = f"{self.args.model_dir}/{save_basename}"
        torch.save(self.state_dict(), save_name)
        print("saving", save_name)
        if self.args.logging.remove_last_checkpoint:
            try:
                os.remove(self.latest_checkpoint)
                print("removed last checkpoint", self.latest_checkpoint)
            except:
                print("could not remove last checkpoint",
                      self.latest_checkpoint)
        self.latest_checkpoint = save_name

    def get_batch(self):
        """ get_batch. """
        audio = next(self.dset)
        audio = torch.Tensor(audio)
        audio = audio.to(self.device).to(torch.float32)
        # Do resampling if needed
        audio = torchaudio.functional.resample(audio, self.args.dset.fs,
                                               self.args.exp.sample_rate)
        print(audio.shape, self.args.exp.seg_len)
        return audio

    def train_step(self):
        """ train_step. """
        # Train step
        it_start_time = time.time()
        self.optimizer.zero_grad()
        st_time = time.time()
        audio = self.get_batch()
        print("std", audio.std(-1))

        error, _ = self.diff_params.loss_fn(self.network, audio)
        loss = error.mean()
        loss.backward()
        # TODO: take care of the loss scaling if using mixed precision
        # do I want to call this at every round?
        # It will slow down the training. I will try it and see what happens

        if self.it <= self.args.train.lr_rampup_it:
            for g in self.optimizer.param_groups:
                #learning rate ramp up
                g['lr'] = self.args.train.lr * min(
                    self.it / max(self.args.train.lr_rampup_it, 1e-8), 1)

        if self.args.train.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(),
                                           self.args.train.max_grad_norm)

        # Update weights.
        self.optimizer.step()

        end_time = time.time()

        it_end_time = time.time()
        print("it :", self.it, "time:, ", end_time - st_time, "total_time: ",
              training_stats.report('it_time', it_end_time - it_start_time),
              "loss: ", training_stats.report('loss', loss.item()))
        # TODO: take care of the logging

    def update_ema(self):
        """Update exponential moving average of self.network weights."""

        ema_rampup = self.args.train.ema_rampup            # ema_rampup should be set to 10000 in the config file
        ema_rate = self.args.train.ema_rate                # ema_rate should be set to 0.9999 in the config file
        t = self.it * self.args.train.batch
        with torch.no_grad():
            if t < ema_rampup:
                s = np.clip(t / ema_rampup, 0.0, ema_rate)
                for dst, src in zip(self.ema.parameters(),
                                    self.network.parameters()):
                    dst.copy_(dst * s + src * (1 - s))
            else:
                for dst, src in zip(self.ema.parameters(),
                                    self.network.parameters()):
                    dst.copy_(dst * ema_rate + src * (1 - ema_rate))

    def training_loop(self):
        """ training_loop. """

        while True:
            # Accumulate gradients.
            self.train_step()
            self.update_ema()

            if self.it > 0 and self.it % self.args.train.save_interval == 0 and self.args.train.save_model:
                self.save_checkpoint()

            # Update sta
            self.it += 1

#%% Main

def _main(args, mode):
    """
    Train diffusion-based generative model using the techniques described in the
    paper "Elucidating the Design Space of Diffusion-Based Generative Models".
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global __file__

    args.exp.model_dir = args.model_dir

    if mode == "noise":
        dataset_obj = TapeHissdset(args.dset)
    elif mode == "trajectories" or mode == "toy_trajectories":
        dataset_obj = ToyTrajectories(args.dset)
    else:
        raise NotImplementedError

    diff_params = EDM(args)

    network = UNet1D(args, device).to(device)

    optimizer = torch.optim.Adam(network.parameters(),
                                 lr=args.train.lr,
                                 betas=(args.train.optimizer.beta1,
                                        args.train.optimizer.beta2),
                                 eps=args.train.optimizer.eps)

    dset = iter(
        torch.utils.data.DataLoader(dataset=dataset_obj,
                                    batch_size=args.train.batch,
                                    num_workers=args.dset.num_workers,
                                    pin_memory=True))
    trainer = Trainer(args, dset, network, optimizer, diff_params, device)
    print("trainer set up")

    # Print options.
    print()
    print('Training options:')
    print()
    print(f'Output directory:        {args.model_dir}')
    print(f'Batch size:              {args.train.batch}')
    print()

    # Train.
    trainer.training_loop()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--MODE', type=str, default=None)
    # "noise", "trajectories" or "toytrajectories"
    pags = parser.parse_args()

    if pags.MODE == "noise":
        config_path = os.path.join("../configs/", 'conf_noise.yaml')
    elif pags.MODE == "trajectories":
        config_path = os.path.join("../configs/", 'conf_trajectories.yaml')
    elif pags.MODE == "toy_trajectories":
        config_path = os.path.join("../configs/", 'conf_toytrajectories.yaml')
    else:
        raise NotImplementedError

    args = OmegaConf.load(config_path)

    _main(args, pags.MODE)

if __name__ == "__main__":
    main()
