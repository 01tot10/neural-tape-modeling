def main(config=0):
    d = {}
    d['segment_length'] = 44100 * 2
    d['tbptt_length'] = 8192 * 2
    d['val_segment_length'] = 44100 * 10
    d['gen_lr'] = 0.0001
    d['val_freq'] = 1
    d['hid_size'] = 64
    d['batch_size'] = 16
    d['model_path'] = "../models/"
    """configs, critic: 0==MelGan, 1==SpectCrit, 2==MelSpectCrit, 3==DilatedConvCrit"""
    if config == 0:
        d['dataset_path'] = 'ReelToReel_Dataset_MiniPulse100_CHOWTAPE_WOWFLUTTER'
        critic = 0
    elif config == 1:
        d['dataset_path'] = 'ReelToReel_Dataset_MiniPulse100_CHOWTAPE_WOWFLUTTER'
        critic = 1
    elif config == 2:
        d['dataset_path'] = 'ReelToReel_Dataset_MiniPulse100_CHOWTAPE_WOWFLUTTER'
        critic = 2
    elif config == 3:
        d['dataset_path'] = 'ReelToReel_Dataset_MiniPulse100_CHOWTAPE_WOWFLUTTER'
        critic = 3
    elif config == 4:
        d['dataset_path'] = 'ReelToReel_Dataset_MiniPulse100_AKAI_IPS[7.5]_MAXELL'
        critic = 0
    elif config == 5:
        d['dataset_path'] = 'ReelToReel_Dataset_MiniPulse100_AKAI_IPS[7.5]_MAXELL'
        critic = 1
    elif config == 6:
        d['dataset_path'] = 'ReelToReel_Dataset_MiniPulse100_AKAI_IPS[7.5]_MAXELL'
        critic = 2
    elif config == 7:
        d['dataset_path'] = 'ReelToReel_Dataset_MiniPulse100_AKAI_IPS[7.5]_MAXELL'
        critic = 3

    elif config == 10:
        d['dataset_path'] = 'ReelToReel_Dataset_MiniPulse100_CHOWTAPE_WOWFLUTTER'
        critic = 5

    if critic == 0:
        crit = {
            'crit': 'MelGanCrit',
            'crit_pars': {
                'num_D': 3,
                'ndf': 16,
                'n_layers': 4,
                'downsampling_factor': 4
            },
            'crit_lr': 0
        }
    elif critic == 1:
        crit = {
            'crit': 'MultiSpecCrit',
            'crit_pars': {
                'scales': [128, 256, 512, 1024],
                'kernel_sizes': [21, 21, 21, 17],
                'hop_sizes': [32, 64, 128, 128],
                'layers': 4,
                'chan_in': 16,
                'chan_fac': 4,
                'stride': 1,
                'g_fac': 16,
                'tf_rep': 'spec',
                'log': True
            },
            'crit_lr': 0
        }
    elif critic == 2:
        crit = {
            'crit': 'MultiSpecCrit',
            'crit_pars': {
                'scales': [128, 256, 512, 1024],
                'kernel_sizes': [21, 21, 21, 17],
                'hop_sizes': [32, 64, 128, 128],
                'layers': 4,
                'chan_in': 16,
                'chan_fac': 4,
                'stride': 1,
                'g_fac': 16,
                'tf_rep': 'mel',
                'log': True
            },
            'crit_lr': 0
        }
    elif critic == 3:
        crit = {'crit': 'DilatedConvDisc', 'crit_pars': {}, 'crit_lr': 0}

    elif critic == 5:
        crit = {
            'crit': 'MultiSpecCrit',
            'crit_pars': {
                'scales': [512, 1024, 2048],
                'kernel_sizes': [21, 17, 7],
                'hop_sizes': [64, 64, 64],
                'layers': 4,
                'chan_in': 16,
                'chan_fac': 4,
                'stride': 1,
                'g_fac': 16,
                'tf_rep': 'spec',
                'log': True
            },
            'crit_lr': 0
        }

    d = Merge(d, crit)

    return d


def Merge(dict1, dict2):
    for i in dict2.keys():
        dict1[i] = dict2[i]
    return dict1
