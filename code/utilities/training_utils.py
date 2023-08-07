#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: eloimoliner, 01tot10
"""

#%% Imports

import torch
import torchaudio

#%% Methods


def resample_batch(audio, fs, fs_target, length_target=None):
    """ resample_batch. """

    device = audio.device
    B = audio.shape[0]
    # if possible resampe in a batched way
    # check if all the fs are the same and equal to 44100
    if fs_target == 22050:
        if (fs == 44100).all():
            audio = torchaudio.functional.resample(audio, 2, 1)
            return audio[:, 0:length_target] # throw away the last samples
        elif (fs == 48000).all():

            # approximate resampleint
            audio = torchaudio.functional.resample(audio, 160 * 2, 147)
            return audio[:, 0:length_target]
        else:

            # if revious is unsuccesful bccause we have examples at 441000 and
            # 48000 in the same batch,, just iterate over the batch
            proc_batch = torch.zeros((B, length_target), device=device)
            for i, (a, f_s) in enumerate(zip(
                    audio, fs)):              # I hope this will not slow down everything
                if f_s == 44100:

                    # resample by 2
                    a = torchaudio.functional.resample(a, 2, 1)
                elif f_s == 48000:
                    a = torchaudio.functional.resample(a, 160 * 2, 147)
                else:
                    print("WARNING, strange fs", f_s)

                proc_batch[i] = a[0:length_target]
            return proc_batch
    elif fs_target == 44100:
        if (fs == 44100).all():
            return audio[:, 0:length_target] # throw away the last samples
        elif (fs == 48000).all():

            # approximate resampleint
            audio = torchaudio.functional.resample(audio, 160, 147)
            return audio[:, 0:length_target]
        else:

            # if previous is unsuccesful because we have examples at 441000 and
            # 48000 in the same batch, just iterate over the batch
            B, C, L = audio.shape
            proc_batch = torch.zeros((B, C, L), device=device)
            for i, (a, f_s) in enumerate(zip(
                    audio,
                    fs.tolist())):            # I hope this will not slow down everything
                if f_s == 44100:

                    # resample by 2
                    pass
                elif f_s == 22050:
                    a = torchaudio.functional.resample(a, 1, 2)
                elif f_s == 48000:
                    a = torchaudio.functional.resample(a, 160, 147)
                else:
                    print("WARNING, strange fs", f_s)

                proc_batch[i] = a[..., 0:length_target]
            return proc_batch
    else:
        if (fs == 44100).all():
            audio = torchaudio.functional.resample(audio, 44100, fs_target)
            return audio[..., 0:length_target] # throw away the last samples
        elif (fs == 48000).all():
            print("resampling 48000 to 16000", length_target, audio.shape)

            # approximate resampleint
            audio = torchaudio.functional.resample(audio, 48000, fs_target)
            print(audio.shape)
            return audio[..., 0:length_target]
        else:

            # if previous is unsuccesful because we have examples at 441000 and
            # 48000 in the same batch, just iterate over the batch
            proc_batch = torch.zeros((B, length_target), device=device)
            for i, (a, f_s) in enumerate(zip(
                    audio, fs)):              # I hope this will not slow down everything
                if f_s == 44100:

                    # resample by 2
                    a = torchaudio.functional.resample(a, 44100, fs_target)
                elif f_s == 48000:
                    a = torchaudio.functional.resample(a, 48000, fs_target)
                else:
                    print("WARNING, strange fs", f_s)

                proc_batch[i] = a[..., 0:length_target]
            return proc_batch


def load_state_dict(state_dict,
                    network=None,
                    ema=None,
                    optimizer=None,
                    log=True):
    """
    Utility for loading state dicts for different models. 
    This function sequentially tries different strategies.
    Assuming the operations are done in place, this function will not create
    a copy of the network and optimizer (I hope)
    
    args:
        state_dict: the state dict to load
    returns:
        True if the state dict was loaded, False otherwise
    """
    if log:
        print("Loading state dict")
    if log:
        print(state_dict.keys())
    # if there
    try:
        if log:
            print("Attempt 1: trying with strict=True")
        if network is not None:
            network.load_state_dict(state_dict['network'])
        if optimizer is not None:
            optimizer.load_state_dict(state_dict['optimizer'])
        if ema is not None:
            ema.load_state_dict(state_dict['ema'])
        return True
    except Exception as e:
        if log:
            print("Could not load state dict")
            print(e)
    try:
        if log:
            print("Attempt 2: trying with strict=False")
        if network is not None:
            network.load_state_dict(state_dict['network'], strict=False)
        # we cannot load the optimizer in this setting
        if ema is not None:
            ema.load_state_dict(state_dict['ema'], strict=False)
        return True
    except Exception as e:
        if log:
            print("Could not load state dict")
            print(e)
            print("training from scratch")
    try:
        if log:
            print(
                "Attempt 3: trying with strict=False,but making sure that the shapes are fine"
            )
        if ema is not None:
            ema_state_dict = ema.state_dict()
        if network is not None:
            network_state_dict = network.state_dict()
        i = 0
        if network is not None:
            for name, param in state_dict['network'].items():
                if log:
                    print("checking", name)
                if name in network_state_dict.keys():
                    if network_state_dict[name].shape == param.shape:
                        network_state_dict[name] = param
                        if log:
                            print("assigning", name)
                        i += 1
        network.load_state_dict(network_state_dict)
        if ema is not None:
            for name, param in state_dict['ema'].items():
                if log:
                    print("checking", name)
                if name in ema_state_dict.keys():
                    if ema_state_dict[name].shape == param.shape:
                        ema_state_dict[name] = param
                        if log:
                            print("assigning", name)
                        i += 1

        ema.load_state_dict(ema_state_dict)

        if i == 0:
            if log:
                print("WARNING, no parameters were loaded")
            raise Exception("No parameters were loaded")
        elif i > 0:
            if log:
                print("loaded", i, "parameters")
            return True

    except Exception as e:
        print(e)
        print("the second strict=False failed")

    try:
        if log:
            print(
                "Attempt 4: Assuming the naming is different, with the network and ema called 'state_dict'"
            )
        if network is not None:
            network.load_state_dict(state_dict['state_dict'])
        if ema is not None:
            ema.load_state_dict(state_dict['state_dict'])
    except Exception as e:
        if log:
            print("Could not load state dict")
            print(e)
            print("training from scratch")
            print("It failed 3 times!! but not giving up")
        # print the names of the parameters in self.network

    try:
        if log:
            print(
                "Attempt 5: trying to load with different names, now model='model' and ema='ema_weights'"
            )
        if ema is not None:
            dic_ema = {}
            for (key, tensor) in zip(state_dict['model'].keys(),
                                     state_dict['ema_weights']):
                dic_ema[key] = tensor
                ema.load_state_dict(dic_ema)
            return True
    except Exception as e:
        if log:
            print(e)

    try:
        if log:
            print(
                "Attempt 6: If there is something wrong with the name of the ema parameters, we can try to load them using the names of the parameters in the model"
            )
        if ema is not None:
            dic_ema = {}
            i = 0
            for (key, tensor) in zip(state_dict['model'].keys(),
                                     state_dict['model'].values()):
                if tensor.requires_grad:
                    dic_ema[key] = state_dict['ema_weights'][i]
                    i = i + 1
                else:
                    dic_ema[key] = tensor
            ema.load_state_dict(dic_ema)
            return True
    except Exception as e:
        if log:
            print(e)

    try:
        # Assign the parameters in state_dict to self.network using a for loop
        print(
            "Attempt 7: Trying to load the parameters one by one. This is for the dance diffusion model, looking for parameters starting with 'diffusion.' or 'diffusion_ema.'"
        )
        if ema is not None:
            ema_state_dict = ema.state_dict()
        if network is not None:
            network_state_dict = ema.state_dict()
        i = 0
        if network is not None:
            for name, param in state_dict['state_dict'].items():
                print("checking", name)
                if name.startswith("diffusion."):
                    i += 1
                    name = name.replace("diffusion.", "")
                    if network_state_dict[name].shape == param.shape:
                        network_state_dict[name] = param

            network.load_state_dict(network_state_dict, strict=False)

        if ema is not None:
            for name, param in state_dict['state_dict'].items():
                if name.startswith("diffusion_ema."):
                    i += 1
                    name = name.replace("diffusion_ema.", "")
                    if ema_state_dict[name].shape == param.shape:
                        if log:
                            print(param.shape, ema.state_dict()[name].shape)
                        ema_state_dict[name] = param

            ema.load_state_dict(ema_state_dict, strict=False)

        if i == 0:
            print("WARNING, no parameters were loaded")
            raise Exception("No parameters were loaded")
        elif i > 0:
            print("loaded", i, "parameters")
            return True
    except Exception as e:
        if log:
            print(e)

    # this is for the dmae1d mddel, assuming there is only one network
    if network is not None:
        network.load_state_dict(state_dict, strict=True)
    if ema is not None:
        ema.load_state_dict(state_dict, strict=True)
    return True

    return False


def unnormalize(x, stds, args):
    """ unnormalize the STN separated audio. """
    new_std = args.exp.normalization.target_std
    if new_std == "sigma_data":
        new_std = args.diff_params.sigma_data
    x = stds * x / (new_std + 1e-8)
    return x


def normalize(xS, xT, xN, args, return_std=False):
    """ normalize the STN separated audio. """
    if args.exp.normalization.mode == "None":
        pass
    elif args.exp.normalization.mode == "residual_noise":
        # normalize the residual noise

        std = xN.std(dim=-1, keepdim=True).mean(dim=1, keepdim=True)
        new_std = args.exp.normalization.target_std

        if new_std == "sigma_data":
            new_std = args.diff_params.sigma_data

        xN = new_std * xN / (std + 1e-8)
        xS = new_std * xS / (std + 1e-8)
        xT = new_std * xT / (std + 1e-8)
    elif args.exp.normalization.mode == "residual_noise_batch":
        # normalize the residual noise per batch
        # get the std of the entire batch
        std = xN.std(dim=(0, 1, 2), unbiased=True, keepdim=False)

        new_std = args.exp.normalization.target_std

        if new_std == "sigma_data":
            new_std = args.diff_params.sigma_data

        xN = new_std * xN / (std + 1e-8)
        xS = new_std * xS / (std + 1e-8)
        xT = new_std * xT / (std + 1e-8)

    elif args.exp.normalization.mode == "all":
        std = (xN + xS + xT).std(dim=-1, keepdim=True).mean(dim=1, keepdim=True)
        new_std = args.exp.normalization.target_std
        if new_std == "sigma_data":
            new_std = args.diff_params.sigma_data
        xN = new_std * xN / (std + 1e-8)
        xS = new_std * xS / (std + 1e-8)
        xT = new_std * xT / (std + 1e-8)
    else:
        print("normalization mode not recognized")
        pass

    try:
        if return_std:
            return xS, xT, xN, std
    except Exception as e:
        print(e)
        print("warning!, std cannot be returned")
        pass

    return xS, xT, xN
