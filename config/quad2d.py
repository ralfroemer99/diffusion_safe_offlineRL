import socket

from diffuser.utils import watch

#------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args

args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ## value kwargs
    ('discount', 'd'),
]

logbase = 'logs'

base = {
    'diffusion': {
        ## model
        'model': 'models.TemporalUnet',
        'diffusion': 'models.GaussianDiffusion',
        'horizon': 16,
        'n_diffusion_steps': 50,
        'action_weight': 1,         # 10 
        'loss_weights': None,
        'loss_discount': 1,
        'predict_epsilon': False,
        'dim_mults': (1, 2, 4, 8),
        'attention': False,
        # 'renderer': 'utils.MuJoCoRenderer',

        ## dataset
        'loader': 'datasets.SequenceDataset',
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': [],
        'clip_denoised': False,
        'use_padding': False,
        'use_actions': False,
        'max_path_length': 100,

        ## serialization
        'logbase': logbase,
        'prefix': 'diffusion/defaults',
        'exp_name': watch(args_to_watch),

        ## training
        'n_steps_per_epoch': 1000,  # 10000
        'loss_type': 'l2',
        'n_train_steps': 1e5,       # 1e6
        'batch_size': 32,            # 32
        'learning_rate': 2e-5,      # 2e-4
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'train_test_split': 0.9,
        'save_freq': 10000,
        'sample_freq': 20000,
        'n_saves': 5,
        'save_parallel': False,
        'n_reference': 8,
        'bucket': None,
        'device': 'cuda',
        'seed': 0,
    },

    'values': {
        'model': 'models.ValueFunction',
        'diffusion': 'models.ValueDiffusion',
        'horizon': 16,
        'n_diffusion_steps': 50,
        'dim_mults': (1, 2, 4, 8),
        # 'renderer': 'utils.MuJoCoRenderer',

        ## value-specific kwargs
        'discount': 0.99,
        'termination_penalty': 0,    # -100
        'normed': True,

        ## dataset
        'loader': 'datasets.ValueDataset',
        'normalizer': 'LimitsNormalizer',     # 'GaussianNormalizer'
        'preprocess_fns': [],
        'use_padding': False,
        'use_actions': False,
        'max_path_length': 100,

        ## serialization
        'logbase': logbase,
        'prefix': 'values/defaults',
        'exp_name': watch(args_to_watch),

        ## training
        'n_steps_per_epoch': 1000,  # 10000
        'loss_type': 'value_l2',
        'n_train_steps': 1e5,       # 200e3
        'batch_size': 32,            # 32
        'learning_rate': 2e-5,      # 2e-4
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'train_test_split': 0.9,
        'save_freq': 10000,
        'sample_freq': 0,
        'n_saves': 5,
        'save_parallel': False,
        'n_reference': 8,
        'bucket': None,
        'device': 'cuda',
        'seed': 0,
    },

    'plan': {
        'guide': 'sampling.ValueGuide',
        'policy': 'sampling.GuidedPolicy',
        'max_episode_length': 1000,
        'batch_size': 64,
        'preprocess_fns': [],
        'device': 'cuda',
        'seed': 0,

        ## sample_kwargs
        'n_guide_steps': 2,
        'scale': 1000,
        't_stopgrad': 2,
        'scale_grad_by_std': True,

        ## serialization
        'loadbase': None,
        'logbase': logbase,
        'prefix': 'plans/',
        'exp_name': watch(args_to_watch),
        'vis_freq': 100,
        'max_render': 8,

        ## diffusion model
        'horizon': 16,
        'n_diffusion_steps': 50,

        ## value function
        'discount': 0.99,

        ## loading
        'diffusion_loadpath': 'f:diffusion/defaults_H{horizon}_T{n_diffusion_steps}',
        'value_loadpath': 'f:values/defaults_H{horizon}_T{n_diffusion_steps}_d{discount}',

        'diffusion_epoch': 'best',      # 'latest'
        'value_epoch': 'best',          # 'latest'

        'verbose': False,
        'suffix': '0',
    },
}


#------------------------ overrides ------------------------#


hopper_medium_expert_v2 = {
    'plan': {
        'scale': 0.0001,
        't_stopgrad': 4,
    },
}


halfcheetah_medium_replay_v2 = halfcheetah_medium_v2 = halfcheetah_medium_expert_v2 = {
    'diffusion': {
        'horizon': 4,
        'dim_mults': (1, 4, 8),
        'attention': True,
    },
    'values': {
        'horizon': 4,
        'dim_mults': (1, 4, 8),
    },
    'plan': {
        'horizon': 4,
        'scale': 0.001,
        't_stopgrad': 4,
    },
}