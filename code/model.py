#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 09:46:38 2023

@author: 01tot10, Alec-Wright, eloimoliner
"""

#%% Imports

import numpy as np
import torch
import torchaudio

from networks.unet_1d import UNet1D
from utilities.utilities import nextpow2

#%% Classes


class RNN(torch.nn.Module):

    def __init__(self, input_size=1, hidden_size=8, output_size=1, skip=False):
        """
        Recurrent Neural Network:
            - Gated Recurrent Unit
            - Fully Connected output layer
        
        Args:
            input_size (int, optional): Number of input features. Defaults to 1.
            hidden_size (int, optional): Number of hidden features. Defaults to 8.
            output_size (int, optional): Number of output features. Defaults to 1.
            skip (bool, optional): Use skip connection. Defaults to False.
        """

        super(RNN, self).__init__()

        # Attributes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.skip = skip

        # Network
        self.GRU = torch.nn.GRU(input_size, hidden_size, batch_first=True)
        self.output = torch.nn.Linear(hidden_size, output_size)

        # Initializations
        self.initialize_hidden()

    def initialize_hidden(self):
        """ Initialize GRU hidden state to zeros. """
        self.hidden = None

    def detach_hidden(self):
        """ Detach GRU hidden state from computational graph """
        self.hidden = self.hidden.clone().detach()

    def warm_start(self):
        """ Process some silence to get going. """
        START_LEN = 2**10
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            x = torch.zeros((1, 1, START_LEN)).to(device)
            _ = self(x)

    def forward(self, x):
        """
        Args:
            x (torch.tensor) of shape (N_BATCHES, N_CHANNELS, N_SAMPLES)

        Returns:
            y (torch.tensor) of shape (N_BATCHES, N_CHANNELS, N_SAMPLES)
        """
        # Mind the dtype and shape
        x = x.float() if x.dtype == torch.float64 else x
        x = x.reshape(x.shape[0], x.shape[2], x.shape[1])

        if self.skip:
            skip = x
        x, self.hidden = self.GRU(x, self.hidden)
        y = self.output(x)
        if self.skip:
            y += skip

        # Mind the shape
        y = y.reshape(y.shape[0], y.shape[2], y.shape[1])
        return y

    def train_epoch(self, dataloader, loss_fcn, optimizer):
        """ Run an epoch of training

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader.
            loss_fcn (torch.nn.Module): Loss function.
            optimizer (torch.optim): Optimizer.
        """
        # Initializations
        TBPTT_INIT = 2**10
        TBPTT_LEN = 2**10
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train()

        # Run epoch
        num_batches = len(dataloader)
        epoch_loss = 0
        for _, batch in enumerate(dataloader):

            # Take current batch
            input, target, _ = batch
            if input.shape[1] > 1: # Only consider audio for training
                input, target = input[:, :1, :], target[:, :1, :]
            input, target = input.to(device), target.to(device)

            self.initialize_hidden()

            ## Truncated Back-Prop Through Time
            num_minibatches = (input.shape[2] - TBPTT_INIT) // TBPTT_LEN

            # Aggregate hidden state
            _ = self.forward(input[:, :, :TBPTT_INIT])
            self.zero_grad()

            # Run TBPTT
            minibatch_loss = 0
            sample_offset = TBPTT_INIT
            for minibatch_idx in range(num_minibatches):
                # Take current minibatch
                input_mini = input[:, :,
                                   sample_offset:sample_offset + TBPTT_LEN]
                target_mini = target[:, :,
                                     sample_offset:sample_offset + TBPTT_LEN]

                # Predict
                pred_mini = self.forward(input_mini)

                # Compute Loss
                loss = loss_fcn(pred_mini, target_mini)

                # Backprop and parameter update
                loss.backward()
                optimizer.step()

                # Reset gradient tracking
                self.detach_hidden()
                self.zero_grad()

                # Aggregate minibatch loss and increment offset
                minibatch_loss += loss.item()
                sample_offset += TBPTT_LEN

            # Collect final minibatch loss
            minibatch_loss /= num_minibatches

            # Aggregate epoch loss
            epoch_loss += minibatch_loss

        # Collect final epoch loss
        epoch_loss /= num_batches

        return epoch_loss

    def validate(self, dataloader, loss_fcn, store_examples=True):
        """ Compute loss over validation set.

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader.
            loss_fcn (torch.nn.Module): Loss function.
        """
        # Initializations
        INIT_LEN = 2**10
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval()

        # Run epoch
        num_batches = len(dataloader)
        val_loss = 0
        examples = []

        with torch.no_grad():

            for _, batch in enumerate(dataloader):

                # Take current batch
                input, target, _ = batch
                if input.shape[1] > 1: # Only consider audio for computing loss
                    input, target = input[:, :1, :], target[:, :1, :]
                input, target = input.to(device), target.to(device)

                # Aggregate hidden state
                self.initialize_hidden()
                _ = self.forward(input[:, :, :INIT_LEN])

                # Cut segments
                input = input[:, :, INIT_LEN:]
                target = target[:, :, INIT_LEN:]

                # Predict
                pred = self.forward(input)

                # Compute Loss and aggregate
                loss = loss_fcn(pred, target)
                val_loss += loss.item()

                if store_examples:
                    # Collect example
                    example = {}
                    example['input'] = input[0, 0, :]
                    example['target'] = target[0, 0, :]
                    example['prediction'] = pred[0, 0, :]
                    examples.append(example)

            # Collect final epoch loss
            val_loss /= num_batches

        return val_loss, examples

    def predict(self, input):
        """ Generate predictions. """

        # Initializations
        SEGMENT_LENGTH = 2**11
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Start processing
        num_minibatches = int(np.ceil(input.shape[-1] / SEGMENT_LENGTH))
        output = torch.empty(input.shape).to(device)

        self.initialize_hidden()
        self.warm_start()

        for idx_minibatch in range(num_minibatches):

            # Take segment
            input_mini = input[:, :, idx_minibatch *
                               SEGMENT_LENGTH:(idx_minibatch + 1) *
                               SEGMENT_LENGTH]

            # Process
            pred_mini = self.forward(input_mini)

            # Store
            output[:, :, idx_minibatch * SEGMENT_LENGTH:(idx_minibatch + 1) *
                   SEGMENT_LENGTH] = pred_mini

        return output


class TimeVaryingDelayLine(torch.nn.Module):

    def __init__(self, max_delay=40000, channels=1):
        """
        Time Varying feedforward delay line

        Args:
            max_delay (int, optional): Maximum length of delay line in samples. Defaults to 40000.
            channels (int, optional): Number of channels or audio. Defaults to 1.
        """

        super(TimeVaryingDelayLine, self).__init__()

        # Attributes
        self.max_delay = max_delay

        # Initializations
        # Buffer is initialised to batch size 1, must be initialised according to batch size
        self.buffer = torch.zeros(1, channels, max_delay)

    def forward(self, x, dt, warmup=False):
        """
        Args:
            x (torch.tensor)  of shape (N_BATCHES, N_CHANNELS, N_SAMPLES)
            dt (torch.tensor) of shape (N_BATCHES, N_CHANNELS, N_SAMPLES), where dt is number of samples of delay

        Returns:
            y (torch.tensor) of shape (N_BATCHES, N_CHANNELS, N_SAMPLES)
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Dims hack
        # x = x.permute(0, 2, 1)
        # dt = dt.permute(0, 2, 1)

        assert self.max_delay >= torch.max(dt)

        # insert buffer or zeros or previous samples at the start of x
        x_pad = torch.cat((self.buffer, x), dim=2)
        if warmup:
            self.buffer = torch.cat(
                (self.buffer[:, :, x.shape[2]:], x[:, :, -self.max_delay:]),
                dim=2)
            return x
        # unfold x into N_SAMPLES sequences of length self.max_delay + 1
        x_unfolded = x_pad.unfold(2, self.max_delay + 1, 1)

        # convert dt into samples of delay
        #delays = dt * self.max_delay
        delays = dt

        # construct vector of sample index distances from 0 to max_delay
        distances = torch.linspace(self.max_delay, 0,
                                   self.max_delay + 1).to(device)
        # for every delay time, create a vector of sample indices minus distances
        distances = torch.abs(distances - delays.unsqueeze(3))
        # subtract from 1 and apply relu, so only indices adjacent to delay length and nonzero
        distances = torch.relu(1 - distances)

        # Apply to unfolded samples, and sum, to apply linear interpolation
        y = distances * x_unfolded
        y = torch.sum(y, 3)

        # Fill buffer
        # self.buffer = x[:, :, -self.max_delay:]
        self.buffer = torch.cat(
            (self.buffer[:, :, x.shape[2]:], x[:, :, -self.max_delay:]), dim=2)

        # Dims hack
        # y = y.permute(0, 2, 1)

        return y

    def detach_buffer(self):
        """ Detach buffer from computational graph """
        self.buffer = self.buffer.clone().detach()

    def init_buffer(self, N):
        """ Init buffer with zeros
            Args:
                N (int), mini-batch size"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.buffer = torch.zeros(N, 1, self.max_delay).to(device)


class DiffDelRNN(torch.nn.Module):

    def __init__(self,
                 input_size=1,
                 hidden_size=8,
                 output_size=1,
                 skip=False,
                 max_delay=10000):
        """
        RNN + differentiable delay line.

        Args:
            input_size (int, optional): Number of input features. Defaults to 1.
            hidden_size (int, optional): Number of hidden features. Defaults to 8.
            output_size (int, optional): Number of output features. Defaults to 1.
            skip (bool, optional): Use skip connection. Defaults to False.
            max_delay (int, optional): Maximum length of delay line in samples. Defaults to 10000.
        """

        super(DiffDelRNN, self).__init__()

        # Attributes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.skip = skip

        # Network
        self.GRU = torch.nn.GRU(input_size, hidden_size, batch_first=True)
        self.output = torch.nn.Linear(hidden_size, output_size, bias=False)

        self.diffdel = TimeVaryingDelayLine(max_delay=max_delay)

        # Initializations
        self.initialize_hidden(2)

    def initialize_hidden(self, N):
        """ Initialize GRU hidden state to zeros. """
        self.hidden = None
        self.diffdel.init_buffer(N)

    def detach_hidden(self):
        """ Detach GRU hidden state from computational graph """
        self.hidden = self.hidden.clone().detach()
        self.diffdel.detach_buffer()

    def warm_start(self):
        """ Process some silence to get going. """
        START_LEN = 2**10
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            x = torch.zeros((1, 1, START_LEN)).to(device)
            d_traj = torch.zeros((1, 1, START_LEN)).to(device)

            _, __ = self(x, d_traj)

    def forward(self, x, del_traj, warmup=False):
        """
        Args:
            x (torch.tensor)        of shape (N_BATCHES, N_CHANNELS, N_SAMPLES)
            del_traj (torch.tensor) of shape (N_BATCHES, N_CHANNELS, N_SAMPLES)

        Returns:
            y (torch.tensor) of shape (N_BATCHES, N_CHANNELS, N_SAMPLES)
        """
        # Mind the dtype
        x = x.float() if x.dtype == torch.float64 else x

        ## Apply GRU
        # Mind the shape
        x = x.reshape(x.shape[0], x.shape[2],
                      x.shape[1])             # (N_BATCHES, N_SAMPLES, N_CHANNELS)

        if self.skip:
            skip = x
        x, self.hidden = self.GRU(x, self.hidden)
        y = self.output(x)
        if self.skip:
            y += skip

        # Mind the shape
        y = y.reshape(y.shape[0], y.shape[2], y.shape[1])

        # Apply delay
        pre_d = y
        y = self.diffdel(y, del_traj, warmup)

        return y, pre_d

    def train_epoch(self, dataloader, loss_fcn, optimizer):
        """ Run an epoch of training

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader.
            loss_fcn (torch.nn.Module): Loss function.
            optimizer (torch.optim): Optimizer.
        """
        # Initializations
        TBPTT_INIT = nextpow2(
            int(dataloader.dataset.delay_analyzer.max_delay *
                dataloader.dataset.fs))
        TBPTT_LEN = 2**11

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train()

        # Run epoch
        num_batches = len(dataloader)
        epoch_loss = 0
        for _, batch in enumerate(dataloader):

            # Take current batch
            input, target, meta = batch
            if input.shape[1] > 1: # Only consider audio for training
                input, target = input[:, :1, :], target[:, :1, :]

            # Take delay trajectory
            d_traj = meta['delay_trajectory'].float()
            d_traj = d_traj.unsqueeze(1) * dataloader.dataset.fs

            # To device
            input, target, d_traj = input.to(device), target.to(
                device), d_traj.to(device)

            ## Truncated Back-Prop Through Time
            num_minibatches = int(
                np.ceil((input.shape[-1] - TBPTT_INIT) // TBPTT_LEN))

            # Aggregate hidden state
            self.initialize_hidden(input.shape[0])
            _, __ = self.forward(input[:, :, :TBPTT_INIT],
                                 d_traj[:, :, :TBPTT_INIT],
                                 warmup=True)
            self.zero_grad()

            # Run TBPTT
            minibatch_loss = 0
            sample_offset = TBPTT_INIT
            for minibatch_idx in range(num_minibatches):
                # Take current minibatch
                input_mini = input[:, :,
                                   sample_offset:sample_offset + TBPTT_LEN]
                target_mini = target[:, :,
                                     sample_offset:sample_offset + TBPTT_LEN]
                d_traj_mini = d_traj[:, :,
                                     sample_offset:sample_offset + TBPTT_LEN]

                # Predict
                pred_mini, _ = self.forward(input_mini, d_traj_mini)

                # Compute Loss
                loss = loss_fcn(pred_mini, target_mini)

                # Backprop and parameter update
                loss.backward()
                optimizer.step()

                # Reset gradient tracking
                self.detach_hidden()
                self.zero_grad()

                # Aggregate minibatch loss and increment offset
                minibatch_loss += loss.item()
                sample_offset += TBPTT_LEN

            # Collect final minibatch loss
            minibatch_loss /= num_minibatches

            # Aggregate epoch loss
            epoch_loss += minibatch_loss

        # Collect final epoch loss
        epoch_loss /= num_batches

        return epoch_loss

    def validate(self, dataloader, loss_fcn, store_examples=True):
        """ Compute loss over validation set.

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader.
            loss_fcn (torch.nn.Module): Loss function.
            store_examples (bool, optional): Collect examples. Defaults to True.
        """
        # Initializations
        INIT_LEN = nextpow2(
            int(dataloader.dataset.delay_analyzer.max_delay *
                dataloader.dataset.fs))
        TBPTT_LEN = 2**11

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval()

        # Run epoch
        num_batches = len(dataloader)
        val_loss = 0
        examples = []
        with torch.no_grad():
            for _, batch in enumerate(dataloader):

                # Take current batch
                input, target, meta = batch
                if input.shape[1] > 1: # Only consider audio for training
                    input, target = input[:, :1, :], target[:, :1, :]

                # Take delay trajectory
                d_traj = meta['delay_trajectory'].float()
                d_traj = d_traj.unsqueeze(1) * dataloader.dataset.fs

                # To device
                input, target, d_traj = input.to(device), target.to(
                    device), d_traj.to(device)

                ## Truncated Back-Prop Through Time
                num_minibatches = int(
                    np.ceil((input.shape[-1] - INIT_LEN) / TBPTT_LEN))

                # Aggregate hidden state
                self.initialize_hidden(input.shape[0])
                _, __ = self.forward(input[:, :, :INIT_LEN],
                                     d_traj[:, :, :INIT_LEN],
                                     warmup=True)

                # Predict
                pred = torch.empty(target.shape).to(device)
                pre_d = torch.empty(target.shape).to(device)

                minibatch_loss = 0
                sample_offset = INIT_LEN
                for segment_idx in range(num_minibatches):
                    # Take current segments
                    input_mini = input[:, :,
                                       sample_offset:sample_offset + TBPTT_LEN]
                    target_mini = target[:, :, sample_offset:sample_offset +
                                         TBPTT_LEN]
                    d_traj_mini = d_traj[:, :, sample_offset:sample_offset +
                                         TBPTT_LEN]

                    # Predict
                    pred_mini, pre_d_mini = self.forward(
                        input_mini, d_traj_mini)

                    # Compute Loss
                    loss = loss_fcn(pred_mini, target_mini)

                    # Store
                    pred[:, :,
                         sample_offset:sample_offset + TBPTT_LEN] = pred_mini
                    pre_d[:, :,
                          sample_offset:sample_offset + TBPTT_LEN] = pre_d_mini

                    # Aggregate minibatch loss and increment offset
                    minibatch_loss += loss.item()
                    sample_offset += TBPTT_LEN

                # Collect final minibatch loss
                minibatch_loss /= num_minibatches

                # Aggregate epoch loss
                val_loss += minibatch_loss

                # Cut segments
                input = input[:, :, INIT_LEN:]
                target = target[:, :, INIT_LEN:]
                pred = pred[:, :, INIT_LEN:]
                pre_d = pre_d[:, :, INIT_LEN:]

                if store_examples:
                    # Collect example
                    example = {}
                    example['input'] = input[0, 0, :]
                    example['target'] = target[0, 0, :]
                    example['prediction'] = pred[0, 0, :]
                    example['prediction_pre_d'] = pre_d[0, 0, :]
                    examples.append(example)

            # Collect final epoch loss
            val_loss /= num_batches

        return val_loss, examples

    def predict(self, input, d_traj):
        """ Generate predictions. """

        # Initializations
        SEGMENT_LENGTH = 2**11
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Start processing
        num_minibatches = int(np.ceil(input.shape[-1] / SEGMENT_LENGTH))
        output = torch.empty(input.shape).to(device)
        output_pre_d = torch.empty(input.shape).to(device)

        self.initialize_hidden(input.shape[0])
        self.warm_start()

        for idx_minibatch in range(num_minibatches):

            # Take segment
            d_traj_mini = d_traj[:, :, idx_minibatch *
                                 SEGMENT_LENGTH:(idx_minibatch + 1) *
                                 SEGMENT_LENGTH]
            input_mini = input[:, :, idx_minibatch *
                               SEGMENT_LENGTH:(idx_minibatch + 1) *
                               SEGMENT_LENGTH]

            # Process
            pred_mini, pre_d_mini = self.forward(input_mini, d_traj_mini)

            # Store
            output[:, :, idx_minibatch * SEGMENT_LENGTH:(idx_minibatch + 1) *
                   SEGMENT_LENGTH] = pred_mini
            output_pre_d[:, :,
                         idx_minibatch * SEGMENT_LENGTH:(idx_minibatch + 1) *
                         SEGMENT_LENGTH] = pre_d_mini

        return output, output_pre_d


class DiffusionGenerator:

    def __init__(self, args, device):
        """
        Diffusion Generator.
        """
        # Load input arguments
        self.args = args
        self.device = device

        # Network
        self.network = UNet1D(args, device).to(device)

        # Load checkpoint
        checkpoint = torch.load(args.network.checkpoint, map_location=device)
        print(checkpoint.keys())
        self.network.load_state_dict(checkpoint["ema"])
        self.network.eval()

        # Set attributes
        self.sigma_min = args.diff_params.sigma_min
        self.sigma_max = args.diff_params.sigma_max
        self.ro = args.diff_params.ro
        self.sigma_data = args.diff_params.sigma_data # depends on the training data! precalculated variance of the dataset
        self.Schurn = args.diff_params.Schurn
        self.Snoise = args.diff_params.Snoise

        # parameters stochastic sampling
        self.seg_len = args.exp.seg_len
        self.sample_rate = args.exp.sample_rate
        self.out_sample_rate = args.exp.out_sample_rate

        # Set schedule
        self.schedule = self.create_schedule(self.args.diff_params.T)
        self.schedule = self.schedule.to(self.device)
        self.gamma = self.get_gamma(self.schedule).to(device)

    def create_schedule(self, nb_steps):
        """
        Define the schedule of timesteps.
        
        Args:
           nb_steps (int): Number of discretized steps
        """
        i = torch.arange(0, nb_steps + 1)
        t = (self.sigma_max**(1 / self.ro) + i / (nb_steps - 1) *
             (self.sigma_min**(1 / self.ro) - self.sigma_max**(1 / self.ro))
            )**self.ro
        t[-1] = 0
        return t

    def cskip(self, sigma):
        """
        Just one of the preconditioning parameters.
        
        Args:
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
        
        """
        return self.sigma_data**2 * (sigma**2 + self.sigma_data**2)**-1

    def cout(self, sigma):
        """
        Just one of the preconditioning parameters.
        
        Args:
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
        """
        return sigma * self.sigma_data * (self.sigma_data**2 + sigma**2)**(-0.5)

    def cin(self, sigma):
        """
        Just one of the preconditioning parameters.
        
        Args:
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
        """
        return (self.sigma_data**2 + sigma**2)**(-0.5)

    def cnoise(self, sigma):
        """
        Pre-conditioning of the noise embedding.
        
        Args:
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
        """
        return (1 / 4) * torch.log(sigma)

    def denoiser(self, xn, sigma):
        """
        This method does the whole denoising step, which implies applying the model and the preconditioning.
        
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

        return cskip * xn + cout * self.network(
            cin * xn,
            cnoise)                              # TODO - This will crash because of broadcasting problems

    def ode_update(self, x, sigma_0, sigma_1):
        """ODE update."""
        x_0_hat = self.denoiser(x, sigma_1)
        score = (x_0_hat - x) / sigma_1**2

        return x - (sigma_0 - sigma_1) * sigma_1 * score

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
        # indexes=torch.logical_and(t>self.Stmin , t<self.Stmax)

        # We use Schurn=5 as the default in our experiments
        # gamma[indexes]=gamma[indexes]+torch.min(torch.Tensor([self.Schurn/N, 2**(1/2) -1]))
        gamma = gamma + torch.min(
            torch.Tensor([self.Schurn / N, 2**(1 / 2) - 1]))

        return gamma

    def outpaint_sample(self, x_oupaint, mask):
        """Sample prior."""
        schedule = self.schedule
        schedule = torch.flip(schedule, (0,))

        z = torch.randn((1, self.seg_len), device=self.device) * schedule[-1]
        for i in range(len(schedule) - 1):
            sigma_0 = schedule[-i - 1]

            if self.gamma[i] == 0:
                sigma_1 = schedule[-i - 2]
            else:
                sigma_1 = schedule[-i - 2] + self.gamma[i] * schedule[-i - 2]
                epsilon = torch.randn(z.shape).to(self.device) * self.Snoise
                # Add extra noise
                z = z + ((sigma_1**2 - schedule[-i - 2]**2)**(1 / 2)) * epsilon

            with torch.no_grad():
                x_oupaint_n = x_oupaint + torch.randn(
                    (1, self.seg_len), device=self.device) * sigma_0
                z_masked = mask * x_oupaint_n + (1 - mask) * z
                z = self.ode_update(z_masked, sigma_1, sigma_0)

        return z

    def sample(self):
        """Sample prior."""
        schedule = self.schedule
        schedule = torch.flip(schedule, (0,))

        z = torch.randn((1, self.seg_len), device=self.device) * schedule[-1]
        for i in range(len(schedule) - 1):
            sigma_0 = schedule[-i - 1]

            if self.gamma[i] == 0:
                sigma_1 = schedule[-i - 2]
            else:
                sigma_1 = schedule[-i - 2] + self.gamma[i] * schedule[-i - 2]
                epsilon = torch.randn(z.shape).to(self.device) * self.Snoise
                # Add extra noise
                z = z + ((sigma_1**2 - schedule[-i - 2]**2)**(1 / 2)) * epsilon

            with torch.no_grad():
                z = self.ode_update(z, sigma_1, sigma_0)

        return z

    def sample_batch(self, batch_size):
        """Sample batch."""
        output = torch.zeros((batch_size, self.seg_len), device=self.device)
        for i in range(batch_size):
            output[i, :] = self.sample()

        return output

    def sample_long(self, Length):
        """Sample long."""
        Length_samples = int(Length * self.sample_rate)
        output = torch.zeros((1, Length_samples), device=self.device)

        first_sample = self.sample()
        output[..., 0:first_sample.shape[-1]] = first_sample
        if first_sample.shape[-1] > Length_samples:
            # that's it
            output = first_sample[:, :Length_samples]
        else:
            # more segments needed
            mask = torch.zeros_like(first_sample)
            samples_ov = int(self.args.outpainting.overlap * self.sample_rate)
            hop = self.seg_len - samples_ov
            mask[..., 0:samples_ov] = 1

            pointer = hop

            while (pointer + self.seg_len) < Length_samples:
                x_pred = output[..., pointer:(pointer + self.seg_len)]
                pred = self.outpaint_sample(x_pred, mask)

                output[..., pointer:(pointer + self.seg_len)] = pred

                pointer += hop

            # last chunk
            if pointer + samples_ov < Length_samples:
                x_pred = output[..., pointer::]
                x_pred = torch.cat((x_pred,
                                    torch.zeros(x_pred.shape[0],
                                                self.seg_len - x_pred.shape[-1],
                                                device=self.device)), -1)
                pred = self.outpaint_sample(x_pred, mask)

                last_samples = output[..., pointer::]
                output[..., pointer::] = pred[..., 0:last_samples.shape[-1]]

        return self.resampler(output)

    def resampler(self, x):
        """Resampler."""
        return torchaudio.functional.resample(x, self.sample_rate,
                                              self.args.exp.out_sample_rate)
