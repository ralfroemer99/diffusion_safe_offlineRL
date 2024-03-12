import os
import torch
import numpy as np
import diffuser.utils as utils
from envs.quad_2d import Quad2DEnv
from envs.pointmass import PointMassEnv
from diffuser.models.mlp import MLP, train_inverse_dynamics

exp = 'quad2d'  # 'pointmass' or 'quad2d

class Parser(utils.Parser):
    dataset: str = exp
    config: str = 'config.' + exp

if exp == 'pointmass':
    env = PointMassEnv(target=None, max_steps=20, epsilon=0.2, reset_target_reached=False, bonus_reward=False, 
                reset_out_of_bounds=True, theta_as_sine_cosine=True, num_episodes=1000)
else:
    env = Quad2DEnv(min_rel_thrust=0.75, max_rel_thrust=1.25, 
                max_rel_thrust_difference=0.01, target=None, max_steps=20,
                epsilon=0.5, reset_target_reached=False, bonus_reward=False, 
                reset_out_of_bounds=True, theta_as_sine_cosine=True, num_episodes=1000)
    
args = Parser().parse_args('diffusion')
dataset_config = utils.Config(
    args.loader,
    savepath=(args.savepath, 'dataset_config.pkl'),
    env=env,
    horizon=args.horizon,
    normalizer=args.normalizer,
    preprocess_fns=args.preprocess_fns,
    use_padding=args.use_padding,
    use_actions=True,
    max_path_length=args.max_path_length,
)

dataset = dataset_config()

save_path = 'logs/' + exp + '/inverse_dynamics/defaults_H' + str(args.horizon) + '_T' + str(args.n_diffusion_steps) + '_AFalse/model.pt'

# n_episodes = dataset.n_episodes
n_episodes = 100000
obs_dim = dataset.observation_dim
action_dim = dataset.action_dim

observations = []
observations_next = []
actions = []

for _ in range(n_episodes):
    termination_idx = np.where(dataset.fields['terminals'][_, :])[0][0]
    observations.append(torch.tensor(dataset.fields['normed_observations'][_, :termination_idx - 1]))
    observations_next.append(torch.tensor(dataset.fields['normed_observations'][_, 1:termination_idx]))
    actions.append(torch.tensor(dataset.fields['normed_actions'][_, :termination_idx - 1]))

dataset_id_observations = torch.cat(observations, dim=0)
dataset_id_observations_next = torch.cat(observations_next, dim=0)
dataset_id_actions = torch.cat(actions, dim=0)
model = MLP(input_size=obs_dim*2, output_size=action_dim)

train_inverse_dynamics(model, dataset_id_observations, dataset_id_observations_next, dataset_id_actions, save_path=save_path, steps=2e5)
