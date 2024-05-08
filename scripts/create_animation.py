import os
import torch
import subprocess
import pickle
import numpy as np
import diffuser.utils as utils
import diffuser.sampling as sampling
from envs.quad_2d import Quad2DEnv
from envs.pointmass import PointMassEnv
from diffuser.models.mlp import MLP

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

play_speeds = [1, 0.25]

system = 'pointmass'

with_projections = [False, True]
warm_start_steps = None

n_obstacles = [0, 5, 0, 5]        # dynamic box obstacles, static box obstacles, dynamic circle obstacles, static circle obstacles
epsilon = 0.3
# seeds = [4, 5, 6, 17, 18, 19, 22, 26, 31, 32, 33, 40, 41, 42, 43, 46, 47, 52, 54, 56, 57, 60, 64, 70, 71, 73, 75, 76, 77, 78, 80, 86, 89, 92, 94, 98]
seeds = [33, 40]

obs_dim = 6 if system == 'pointmass' else 9
simulation_timesteps = 100 if system == 'pointmass' else 200

class Parser(utils.Parser):
    dataset: str = system
    config: str = 'config.' + system

args = Parser().parse_args('plan')

## load diffusion model and value function from disk
diffusion_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.diffusion_loadpath,
    epoch=args.diffusion_epoch, seed=args.seed,
)

value_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.value_loadpath,
    epoch=args.value_epoch, seed=args.seed,
)

diffusion_losses = diffusion_experiment.losses
value_losses = value_experiment.losses

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset

value_function = value_experiment.ema
guide_config = utils.Config(args.guide, model=value_function, verbose=False)
guide = guide_config()

action_dim =2 if args.use_actions else 0
if not args.use_actions:
    id_model = MLP(input_size=obs_dim*2, output_size=2)
    id_model.load_state_dict(torch.load('logs/' + system + '/inverse_dynamics/defaults_H' + str(args.horizon) + '_T' + str(args.n_diffusion_steps) + '_AFalse/model.pt'))
    id_model.eval()

## policies are wrappers around an unconditional diffusion model and a value guide
policy_config = utils.Config(
    args.policy,
    guide=guide,                                    # guide = None        
    diffusion_model=diffusion,
    normalizer=dataset.normalizer,
    preprocess_fns=args.preprocess_fns,
    ## sampling kwargs
    scale=args.scale,                               # comment
    sample_fn=sampling.n_step_guided_p_sample,      # comment
    n_guide_steps=args.n_guide_steps,               # comment
    t_stopgrad=args.t_stopgrad,                     # comment
    scale_grad_by_std=args.scale_grad_by_std,       # comment
    verbose=False,
)
policy = policy_config()

#--------------------------------- main loop ---------------------------------#
target_reached_which = np.zeros((len(with_projections), len(seeds)))
for idx0, with_projection in enumerate(with_projections):
    for idx1, seed in enumerate(seeds):
        save_path = 'results/animation/' + system + \
                    '/use_actions_' + str(args.use_actions) + \
                    '/scale_' + str(args.scale) + '_warmstart_' + str(warm_start_steps) + \
                    '/projection_' + str(with_projection) + \
                    '/seed_' + str(seed)
        if os.path.exists(save_path):       # Delete previous results
            os.system('rm -r ' + save_path)
        else:
            os.makedirs(save_path)

        if args.dataset == 'pointmass':
            env = PointMassEnv(max_steps=simulation_timesteps, epsilon=epsilon, reset_target_reached=True, reset_out_of_bounds=True, 
                               n_moving_obstacles_box=n_obstacles[0], n_static_obstacles_box=n_obstacles[1], n_moving_obstacles_circle=n_obstacles[2], 
                               n_static_obstacles_circle=n_obstacles[3], test=True, seed=seed)
        else:
            env = Quad2DEnv(max_steps=simulation_timesteps, epsilon=epsilon, reset_target_reached=True, reset_out_of_bounds=True, 
                            n_moving_obstacles_box=n_obstacles[0], n_static_obstacles_box=n_obstacles[1], n_moving_obstacles_circle=n_obstacles[2], 
                            n_static_obstacles_circle=n_obstacles[3], test=True, seed=seed)

        #-----------------Closed-loop experiment with obstacles-----------------------#
        observations_all = np.zeros((simulation_timesteps, obs_dim))
        actions_all = np.zeros((simulation_timesteps, 2))
        reward_all = np.zeros((simulation_timesteps))

        # Reset environment
        obs = env.reset(seed=seed)
        observations_all[0, :] = obs
        for _ in range(env.max_steps):           
            # Get current state
            conditions = {0: obs}
            
            # Sample state sequence or state-action sequence
            if with_projection:
                unsafe_bounds_box, unsafe_bounds_circle = utils.compute_unsafe_regions(env.predict_obstacles(args.horizon), horizon=args.horizon, 
                                                                                          obs_dim=obs_dim, action_dim=action_dim)
            else:
                unsafe_bounds_box, unsafe_bounds_circle = None, None

            action, samples = policy(conditions=conditions, batch_size=args.batch_size, unsafe_bounds_box=unsafe_bounds_box, 
                                     unsafe_bounds_circle=unsafe_bounds_circle, verbose=False, id_model=id_model,
                                     warm_start_steps=warm_start_steps)
                            
            env.render(trajectories_to_plot=samples.observations[:, :, [0, 2]], old_path=observations_all[:_ + 1, [0, 2]], save_path=save_path)

            # Step environment
            if not args.use_actions:
                next_obs = samples.observations[0, 1, :]
                action = env.inverse_dynamics(next_obs)
            obs, reward, done, target_reached = env.step(action)

            # Log
            if _ < env.max_steps - 1:
                observations_all[_ + 1, :] = obs
            actions_all[ _, :] = action
            reward_all[_] = reward

            if target_reached == True:
                target_reached_which[idx0, idx1] = 1
            if done:
                break

        results = {'system': system,
                    'use_actions': args.use_actions,
                    'n_obstacles': n_obstacles,
                    'observations_all': observations_all,
                    'actions_all': actions_all,
                    'rewards_all': reward_all,
        }

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(save_path + '/' + system + '_seed_' + str(seed) + '.pkl', 'wb') as f:
            pickle.dump(results, f)

        for play_speed in play_speeds:
            command = 'ffmpeg -y -r ' + str(int(play_speed/env.dt)) + ' -i ' + save_path + '/screen_%03d.png -vcodec libx264 -pix_fmt yuv420p ' + save_path + '/a_video_' + str(play_speed) + '.mp4'
            subprocess.call(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        # command = "ffmpeg -r 20 -f image2 -s 1500x1500 -i results/animation/pointmass/seed_0/screen_%03d.png -vcodec libx264 -crf 25  results/animation/pointmass/seed_0/video2.mp4"
        # subprocess.call(command, shell=True)

print(target_reached_which)
if target_reached_which.shape[0] == 2:
    print('Projection changed behavior for seeds: ', seeds([np.where(target_reached_which[0] != target_reached_which[1])[0]]))
    print('Projection improved behavior for seeds: ', seeds([np.where(target_reached_which[0] < target_reached_which[1])[0]]))