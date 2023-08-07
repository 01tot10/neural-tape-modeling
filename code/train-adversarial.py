#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Alec-Wright, 01tot10
"""

#%% Imports

import argparse
import os
import sys
import time

import critics
import evaluation
import torch
import wandb
from dataset import VADataset
from model import DiffDelRNN
from torch.utils.data import DataLoader

sys.path.append('../configs')
import AdversarialConfig

#%% Argument parser

prsr = argparse.ArgumentParser(
    description='''This is a script that runs adversarial training''')

prsr.add_argument(
    '--load_config',
    '-l',
    help=
    "Config Number, will load from AdversarialConfig.py and overwrite defaults",
    default='10')
prsr.add_argument('--DRY_RUN', action='store_true', default=False)

args = prsr.parse_args()

print("\nArguments:")
print(args)

DRY_RUN = args.DRY_RUN

#%% Main

if __name__ == '__main__':
    lc = args.load_config
    args = AdversarialConfig.main(int(args.load_config))
    seg_len, dataset_path, model_path, hid_size, lr, batch_size, tbptt_len = \
        args['segment_length'], args['dataset_path'], args['model_path'], args['hid_size'], args['gen_lr'], \
            args['batch_size'], args['tbptt_length']

    if not torch.cuda.is_available():
        print('no GPU available') # If no GPU, get small dataset
        fraction = 0.1
        device = torch.device('cpu')
    else:
        print('GPU available')
        fraction = 1.0
        device = torch.device("cuda:0")

    # Data
    dataset_path = '../audio/' + dataset_path
    print("\nDataset for training:")
    dataset_train = VADataset(dataset_path,
                              fraction=fraction,
                              length=seg_len,
                              demodulate=False,
                              preload=True,
                              double=True)
    train_d_min, train_d_max = dataset_train.min_d * dataset_train.fs - 1, dataset_train.max_d * dataset_train.fs + 1

    print("\nDataset for validation:")
    dataset_val = VADataset(dataset_path,
                            length=args['val_segment_length'],
                            fraction=fraction,
                            subset="val",
                            preload=True,
                            double=True)
    val_d_min, val_d_max = dataset_val.min_d * dataset_val.fs - 1, dataset_val.max_d * dataset_val.fs + 1

    assert dataset_train.fs == dataset_val.fs, "Sampling rates don't match"
    fs = dataset_train.fs

    # Model save info
    model_name = "UnpairedTraining_" + str(lc)
    model_path = os.path.join(model_path, model_name)
    model_running_path = os.path.join(model_path, "running")
    model_best_path = os.path.join(model_path, "best.pth")
    if not DRY_RUN:
        os.makedirs(model_path, exist_ok=True)

    # Make generator
    Generator = DiffDelRNN(hidden_size=hid_size, skip=False).to(device=device)
    optG = torch.optim.Adam(Generator.parameters(), lr=lr, betas=(0.5, 0.9))

    # Make critic
    Critic, optC = critics.get_critic(args['crit'],
                                      args['crit_pars'],
                                      device=device,
                                      crit_lr=lr,
                                      test_in_len=tbptt_len)

    wandb.init(project="neural-tape",
               mode="online" if not DRY_RUN else "disabled",
               config=args)

    start = time.time()
    crit_step = 0
    gen_step = 0
    batches_done = 0

    # Initializations
    train_init = max(int(train_d_max - train_d_min) + 2, 1024)
    d_traj_offset = int(train_d_min) - 1

    v_bs = 10
    val_trunc_len = 8192

    evaluator = evaluation.val_loss_supervised(device=device)

    for epoch in range(0, 10000):

        # Run Validation
        sys.stdout.write(f"Validation Epoch {epoch + 1}")
        val_iter = iter(
            DataLoader(dataset_val,
                       batch_size=v_bs,
                       shuffle=False,
                       num_workers=0))
        if not DRY_RUN:
            torch.save(Generator.state_dict(),
                       model_running_path + str(epoch) + '.pth')

        with torch.inference_mode():
            for batch_idx, batch in enumerate(val_iter):
                # Take current batch
                input, target, meta = batch
                d_traj = meta['delay_trajectory'].unsqueeze(1) * dataset_val.fs

                ## Truncated segments for validation
                num_minibatches = (input.shape[-1]) // tbptt_len
                num_minibatches += 1 if (
                    input.shape[-1]) % tbptt_len > 0 else num_minibatches

                if input.shape[1] > 1:                               # Only consider audio for training
                    input, target = input[:, :1, :], target[:, :1, :]
                input, target, d_traj = input.to(device), target.to(
                    device), d_traj.to(device)

                # Aggregate hidden state
                Generator.initialize_hidden(input.shape[0], val_d_max)
                pred = torch.empty(target.shape).to(device)
                pre_d = torch.empty(target.shape).to(device)

                for segments in range(num_minibatches):
                    chunk, pre_d_chunk = Generator(
                        input[:, :,
                              segments * tbptt_len:(segments + 1) * tbptt_len],
                        d_traj[:, :,
                               segments * tbptt_len:(segments + 1) * tbptt_len])
                    pred[:, :, segments * tbptt_len:(segments + 1) *
                         tbptt_len] = chunk
                    pre_d[:, :, segments * tbptt_len:(segments + 1) *
                          tbptt_len] = pre_d_chunk

                evaluator.add_loss(pred.squeeze().float(),
                                   target.squeeze().float())

                if epoch == 0:
                    wandb.log({
                        "input_" + str(batch_idx):
                            wandb.Audio(input[0, 0, :].cpu().squeeze().numpy(),
                                        caption="Input",
                                        sample_rate=44100),
                        'epoch':
                            epoch
                    })
                    wandb.log({
                        "target_" + str(batch_idx):
                            wandb.Audio(target[0, 0, :].cpu().squeeze().numpy(),
                                        caption="Target",
                                        sample_rate=44100),
                        'epoch':
                            epoch
                    })
                wandb.log({
                    "output_" + str(batch_idx):
                        wandb.Audio(pred[0, 0, :].cpu().squeeze().numpy(),
                                    caption="Output",
                                    sample_rate=44100),
                    'epoch':
                        epoch
                })
                wandb.log({
                    "output_pre_d" + str(batch_idx):
                        wandb.Audio(pre_d[0, 0, :].cpu().squeeze().numpy(),
                                    caption="Output_Pre_Delay",
                                    sample_rate=44100),
                    'epoch':
                        epoch
                })

            evaluator.end_val(epoch)
            for key in evaluator.bst_losses:
                n = 'loss/' + key
                l = evaluator.losses[key]
                wandb.log({n: l, 'epoch': epoch})
                if evaluator.bst_losses[key][0] == epoch:
                    wandb.log({'best_' + n: l, 'epoch': epoch})
                    if not DRY_RUN:
                        torch.save(Generator.state_dict(),
                                   os.path.join(model_path, key + 'best.pt'))

        sys.stdout.write(f"Training Epoch {epoch + 1}")
        inp_iter_gen = iter(
            DataLoader(dataset_train,
                       batch_size=batch_size,
                       shuffle=True,
                       num_workers=0))
        inp_iter_crit = iter(
            DataLoader(dataset_train,
                       batch_size=batch_size,
                       shuffle=True,
                       num_workers=0))
        tgt_iter_crit = iter(
            DataLoader(dataset_train,
                       batch_size=batch_size,
                       shuffle=True,
                       num_workers=0))
        for inp_gen, inp_crit, tgt_crit in zip(inp_iter_gen, inp_iter_crit,
                                               tgt_iter_crit):

            # Train Critic
            Critic.zero_grad()

            # Get Inputs for Generator, to make Fake examples for Critic
            crit_in, _, meta_crit_in = inp_crit
            d_traj_cin = meta_crit_in['delay_trajectory']
            if crit_in.shape[1] > 1:                # Only consider audio for training
                crit_in = crit_in[:, :1, :]
            d_traj_cin = (d_traj_cin.unsqueeze(1) *
                          dataset_train.fs) - d_traj_offset
            assert train_init > torch.max(
                d_traj_cin), f"{train_init} > {torch.max(d_traj_cin)}"
            assert 0 < torch.min(d_traj_cin)

            # Get Reals for Critic
            _, crit_tgt, _ = tgt_crit
            if crit_tgt.shape[1] > 1: # Only consider audio for training
                crit_tgt = crit_tgt[:, :1, :]

            crit_in, d_traj_cin, crit_tgt = crit_in.to(device), d_traj_cin.to(
                device), crit_tgt.to(device)

            # Run init data through generator
            Generator.initialize_hidden(crit_in.shape[0], train_init)
            num_minibatches = (crit_in.shape[-1] - train_init) // tbptt_len
            _ = Generator(crit_in[:, :, :train_init],
                          d_traj_cin[:, :, :train_init],
                          warmup=True)
            Generator.zero_grad()

            # Run TBPTT
            sample_offset = train_init
            for minibatch_idx in range(num_minibatches):
                # Create Fakes
                with torch.inference_mode():
                    crit_in_mini, _ = Generator(
                        crit_in[:, :, sample_offset:sample_offset + tbptt_len],
                        d_traj_cin[:, :,
                                   sample_offset:sample_offset + tbptt_len])
                target_mini = crit_tgt[:, :,
                                       sample_offset:sample_offset + tbptt_len]

                loss_crit = Critic.train_crit(crit_in_mini.float(),
                                              target_mini.float(), optC)
                wandb.log({'loss/critic': loss_crit}, step=crit_step)
                crit_step += 1

            # Train Generator
            Generator.zero_grad()

            # Get Inputs for Generator, for training generator
            gen_in, _, meta_gen_in = inp_gen
            d_traj_gin = meta_gen_in['delay_trajectory']
            if gen_in.shape[1] > 1:            # Only consider audio for training
                gen_in = gen_in[:, :1, :]
            d_traj_gin = d_traj_gin.unsqueeze(
                1) * dataset_train.fs - d_traj_offset
            assert train_init > torch.max(
                d_traj_gin), f"{train_init} > {torch.max(d_traj_gin)}"
            assert 0 < torch.min(d_traj_gin)

            gen_in, d_traj_gin = gen_in.to(device), d_traj_gin.to(device)

            # Run init data through generator
            Generator.initialize_hidden(gen_in.shape[0], train_init)
            num_minibatches = (gen_in.shape[-1] - train_init) // tbptt_len
            _ = Generator(gen_in[:, :, :train_init],
                          d_traj_gin[:, :, :train_init],
                          warmup=True)
            Generator.zero_grad()

            # Run TBPTT
            sample_offset = train_init
            for minibatch_idx in range(num_minibatches):
                # Process mini-seg
                gen_out_mini, _ = Generator(
                    gen_in[:, :, sample_offset:sample_offset + tbptt_len],
                    d_traj_gin[:, :, sample_offset:sample_offset + tbptt_len])
                loss_gen = Critic.train_gen(gen_out_mini.float(), optG)
                wandb.log({'loss/gen': loss_gen}, step=gen_step)
                gen_step += 1
                Generator.detach_hidden()
                Generator.zero_grad()
