from collections import namedtuple
import torch
import time
import einops
import numpy as np
import pdb

import diffuser.utils as utils
from diffuser.datasets.preprocessing import get_policy_preprocess_fn


Trajectories = namedtuple('Trajectories', 'actions observations values')


class GuidedPolicy:

    def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
        self.guide = guide
        self.diffusion_model = diffusion_model
        self.normalizer = normalizer
        self.action_dim = diffusion_model.action_dim
        self.preprocess_fn = get_policy_preprocess_fn(preprocess_fns)
        self.sample_kwargs = sample_kwargs

    def __call__(self, conditions, batch_size=1, unsafe_bounds=None, verbose=True):
        conditions = {k: self.preprocess_fn(v) for k, v in conditions.items()}
        conditions = self._format_conditions(conditions, batch_size)
        if unsafe_bounds is not None:
            unsafe_bounds = self._format_unsafe_bounds(unsafe_bounds)
            conditions.update({'unsafe_bounds': unsafe_bounds})
            conditions.update({'dims': torch.tensor([2, 4])})

        ## run reverse diffusion process
        samples = self.diffusion_model(conditions, guide=self.guide, verbose=verbose, **self.sample_kwargs)
        trajectories = utils.to_np(samples.trajectories)

        ## extract observations [ batch_size x horizon x observation_dim ]
        normed_observations = trajectories[:, :, self.action_dim:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')

        ## extract action [ batch_size x horizon x action_dim ]
        if self.action_dim > 0:
            actions = trajectories[:, :, :self.action_dim]
            actions = self.normalizer.unnormalize(actions, 'actions')

            ## extract first action
            action = actions[0, 0]
        else:
            actions = None
            action = None

        trajectories = Trajectories(actions, observations, samples.values)
        return action, trajectories

    @property
    def device(self):
        parameters = list(self.diffusion_model.parameters())
        return parameters[0].device

    def _format_conditions(self, conditions, batch_size):
        conditions = utils.apply_dict(
            self.normalizer.normalize,
            conditions,
            'observations',
        )
        conditions = utils.to_torch(conditions, dtype=torch.float32, device='cuda:0')
        conditions = utils.apply_dict(
            einops.repeat,
            conditions,
            'd -> repeat d', repeat=batch_size,
        )
        return conditions
    
    def _format_unsafe_bounds(self, unsafe_bounds):
        '''
            unsafe_bounds : dict of lists of obs_dim x 2 arrays
                { t: [ [x_min, x_max], [y_min, y_max] ] }
            unsafe_bounds_formatted : dict of (action_dim + obs_dim) x (2 * n_obs) arrays
                { t: [ x_min, x_max, y_min, y_max ] }
        '''

        unsafe_bounds_formatted = {}
        for i, _ in unsafe_bounds.items():
            unsafe_bounds_formatted[i] = np.zeros((self.action_dim + self.diffusion_model.observation_dim, 2 * len(unsafe_bounds[i])))
            for n_obs in range(len(unsafe_bounds[i])):
                unsafe_bounds_formatted[i][:self.action_dim, 2 * n_obs] = self.normalizer.normalize(unsafe_bounds[i][n_obs][:self.action_dim, 0], 'actions')
                unsafe_bounds_formatted[i][:self.action_dim, 2 * n_obs + 1] = self.normalizer.normalize(unsafe_bounds[i][n_obs][:self.action_dim, 1], 'actions')
                unsafe_bounds_formatted[i][self.action_dim:, 2 * n_obs] = self.normalizer.normalize(unsafe_bounds[i][n_obs][self.action_dim:, 0], 'observations')
                unsafe_bounds_formatted[i][self.action_dim:, 2 * n_obs + 1] = self.normalizer.normalize(unsafe_bounds[i][n_obs][self.action_dim:, 1], 'observations')
        
        unsafe_bounds_formatted = utils.to_torch(unsafe_bounds_formatted, dtype=torch.float32, device='cuda:0')
        return unsafe_bounds_formatted
    