import os
import pickle
import torch
import numpy as np
import diffuser.utils as utils
import diffuser.sampling as sampling
from envs.quad_2d import Quad2DEnv
from envs.pointmass import PointMassEnv
from diffuser.models.mlp import MLP

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

load_results = True

data_save_path = 'results/scale_batch_ablation'
with_projections = False
warmstart_steps = None

# List of arguments to pass to the script
systems_list = ['pointmass', 'quad2d']
scale_range = np.logspace(0, 4, 2)
batch_size_range = [1, 16]

for system in systems_list:
    class Parser(utils.Parser):
        dataset: str = system
        config: str = 'config.' + system
    args = Parser().parse_args('plan')

    # Store success rate and reward
    success_rate_all = np.zeros((len(batch_size_range), len(scale_range)))
    reward_mean_all = np.zeros((len(batch_size_range), len(scale_range)))
    steps_mean_all = np.zeros((len(batch_size_range), len(scale_range)))

    obs_dim = 6 if system == 'pointmass' else 9
    simulation_timesteps = 100 if system == 'pointmass' else 200

    action_dim = 2 if args.use_actions else 0
    if not args.use_actions:
        id_model = MLP(input_size=obs_dim*2, output_size=2)
        id_model.load_state_dict(torch.load('logs/' + system + '/inverse_dynamics/defaults_H' + str(args.horizon) + '_T' + str(args.n_diffusion_steps) + '_AFalse/model.pt'))
        id_model.eval()

    for idx0, batch_size in enumerate(batch_size_range):
        for idx1, scale in enumerate(scale_range):           
            load_path = data_save_path  + '/' + system + \
                '/use_actions_' + str(args.use_actions) + \
                '/projection_' + str(with_projections) + \
                '/warmstart_steps_' + str(warmstart_steps) + \
                '/scale_' + str(scale) + \
                '/n_obstacles_' + str(0) + '_' + str(0) + '_' + str(0) + '_' + str(0) + \
                '/batch_size_' + str(batch_size)
   
            with open(load_path + '/data.pkl', 'rb') as f:
                results = pickle.load(f)
            
            success_rate_all[idx0, idx1] = results['success_rate']
            reward_mean_all[idx0, idx1] = results['reward_mean']
            steps_mean_all[idx0, idx1] = results['steps_mean']

    print('-----------------------------------------------------------------------------------------------------------------')
    for _, batch_size in enumerate(batch_size_range):
        print(f'System: {args.dataset}, use actions: {args.use_actions}, BATCH_SIZE: {batch_size}, SCALES: {np.round(scale_range, 2)}, SUCCESS_RATE: {np.round(success_rate_all[_, :], 2)}, STEPS: {np.round(steps_mean_all[_, :], 2)}, REWARD: {np.round(reward_mean_all[_, :], 2)}')
    print('-----------------------------------------------------------------------------------------------------------------')
    