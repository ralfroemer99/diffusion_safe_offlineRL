import os
import pickle
import subprocess
import time
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
save_data = False
data_save_path = 'results/data_new_env'
save_animation = False
animation_save_path = 'results/animation' if save_animation else None

# List of arguments to pass to the script
systems_list = ['pointmass', 'quad2d']
# n_obstacles_range = [[0, 5, 0, 5]]
n_obstacles_range = [[0, 5, 0, 5],
                     [2, 5, 2, 5]]
with_projections_range = [True]
# warmstart_steps_range = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, False]
warmstart_steps_range = [2, 4, 6, 8]

scale_range = [100, 10, 1]

n_trials = 100

seeds_list = None
# seeds = [4, 5, 6, 17, 18, 19, 22, 26, 31, 32, 33, 40, 41, 42, 43, 46, 47, 52, 54, 56, 57, 60, 64, 70, 71, 73, 75, 76, 77, 78, 80, 86, 89, 92, 94, 98]                                                             # pointmass, static environment   (pointmass)
# seeds = [0,  2,  4,  5,  7, 12, 13, 14, 17, 19, 22, 25, 27, 28, 30, 31, 32, 38, 42, 44, 45, 46, 47, 50, 55, 58, 59, 60, 61, 62, 66, 67, 69, 70, 73, 77, 78, 79, 81, 82, 83, 87, 93, 94, 95, 96, 99]               # pointmass, dynamic environment (pointmass)
# seeds = [0, 1, 3, 5, 8, 10, 11, 13, 14, 17, 18, 19, 20, 24, 25, 27, 31, 33, 35, 37, 38, 39, 40, 41, 42, 47, 49, 50, 52, 53, 55, 58, 61, 64, 65, 66, 68, 69, 75, 78, 80, 81, 82, 84, 88, 91, 94, 96]               # quad2d, static environment
# seeds = [1, 2, 5, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 24, 25, 26, 28, 29, 31, 33, 34, 40, 41, 47, 48, 52, 53, 54, 56, 59, 62, 65, 66, 67, 69, 72, 74, 78, 81, 82, 84, 86, 87, 88, 89, 93, 96, 97, 98, 99]   # quad2d, dynamic environment
# seeds_list = [[4, 5, 6, 17, 18, 19, 22, 26, 31, 32, 33, 40, 41, 42, 43, 46, 47, 52, 54, 56, 57, 60, 64, 70, 71, 73, 75, 76, 77, 78, 80, 86, 89, 92, 94, 98],
#               [0,  2,  4,  5,  7, 12, 13, 14, 17, 19, 22, 25, 27, 28, 30, 31, 32, 38, 42, 44, 45, 46, 47, 50, 55, 58, 59, 60, 61, 62, 66, 67, 69, 70, 73, 77, 78, 79, 81, 82, 83, 87, 93, 94, 95, 96, 99],
#               [0, 1, 3, 5, 8, 10, 11, 13, 14, 17, 18, 19, 20, 24, 25, 27, 31, 33, 35, 37, 38, 39, 40, 41, 42, 47, 49, 50, 52, 53, 55, 58, 61, 64, 65, 66, 68, 69, 75, 78, 80, 81, 82, 84, 88, 91, 94, 96],
#               [1, 2, 5, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 24, 25, 26, 28, 29, 31, 33, 34, 40, 41, 47, 48, 52, 53, 54, 56, 59, 62, 65, 66, 67, 69, 72, 74, 78, 81, 82, 84, 86, 87, 88, 89, 93, 96, 97, 98, 99]]

for system in systems_list:
    # Store success rate and reward
    success_rate_all = np.zeros((len(n_obstacles_range), len(with_projections_range), len(warmstart_steps_range)))
    # reward_mean_all = np.zeros((len(n_obstacles_range), len(with_projections_range), len(warmstart_steps_range)))

    obs_dim = 6 if system == 'pointmass' else 9
    simulation_timesteps = 100 if system == 'pointmass' else 200
    
    class Parser(utils.Parser):
        dataset: str = system
        config: str = 'config.' + system

    args = Parser().parse_args('plan')

    ## DELETE
    args.batch_size = 32        
    ## DELETE

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

    action_dim = 2 if args.use_actions else 0
    if not args.use_actions:
        id_model = MLP(input_size=obs_dim*2, output_size=2)
        id_model.load_state_dict(torch.load('logs/' + system + '/inverse_dynamics/defaults_H' + str(args.horizon) + '_T' + str(args.n_diffusion_steps) + '_AFalse/model.pt'))
        id_model.eval()

    for with_projections in with_projections_range:
        warmstart_steps_range_mod = warmstart_steps_range if with_projections else [False]

        if scale_range is None:
            scale_range = [args.scale]

        for scale in scale_range:
            for idx, n_obstacles in enumerate(n_obstacles_range):
                ## policies are wrappers around an unconditional diffusion model and a value guide
                policy_config = utils.Config(
                    args.policy,
                    guide=guide,                                    # guide = None        
                    diffusion_model=diffusion,
                    normalizer=dataset.normalizer,
                    preprocess_fns=args.preprocess_fns,
                    ## sampling kwargs
                    scale=scale,                               # comment
                    sample_fn=sampling.n_step_guided_p_sample,      # comment
                    n_guide_steps=args.n_guide_steps,               # comment
                    t_stopgrad=args.t_stopgrad,                     # comment
                    scale_grad_by_std=args.scale_grad_by_std,       # comment
                    verbose=False,
                )
                for warmstart_steps in warmstart_steps_range_mod:
                    n_reached = 0
                    reward_total = 0
                    observations_all = np.zeros((n_trials, simulation_timesteps, obs_dim))
                    actions_all = np.zeros((n_trials, simulation_timesteps, 2))
                    reward_all = np.zeros((n_trials, simulation_timesteps))
                    target_reached_all = np.zeros((n_trials))
                    is_done_all = np.zeros((n_trials, simulation_timesteps))
                    
                    #-----------------Closed-loop experiment with obstacles-----------------------#
                    if seeds_list is not None:
                        if system == 'pointmass':
                            if idx == 0:
                                trials = seeds_list[0]
                            else:
                                trials = seeds_list[1]
                        else:
                            if idx == 0:
                                trials = seeds_list[2]
                            else:
                                trials = seeds_list[3]
                    else:                    
                        trials = range(n_trials)
                    
                    for n in trials:
                        if save_animation:
                            if warmstart_steps is False:
                                save_path = animation_save_path  + '/' + system + \
                                                            '/use_actions_' + str(args.use_actions) + \
                                                            '/projection_' + str(with_projections) + \
                                                            '/no_warmstart' + \
                                                            '/scale_' + str(scale) + \
                                                            '/n_obstacles_' + str(n_obstacles[0]) + '_' + str(n_obstacles[1]) + '_' + str(n_obstacles[2]) + '_' + str(n_obstacles[3]) + \
                                                            '/seed_' + str(n)
                            else:
                                save_path = animation_save_path  + '/' + system + \
                                                                    '/use_actions_' + str(args.use_actions) + \
                                                                    '/projection_' + str(with_projections) + \
                                                                    '/warmstart_steps_' + str(warmstart_steps) + \
                                                                    '/scale_' + str(scale) + \
                                                                    '/n_obstacles_' + str(n_obstacles[0]) + '_' + str(n_obstacles[1]) + '_' + str(n_obstacles[2]) + '_' + str(n_obstacles[3]) + \
                                                                    '/seed_' + str(n)
                            if not os.path.exists(save_path):
                                os.makedirs(save_path)
                            else:
                                subprocess.call('rm -rf ' + os.path.join(save_path, '*'), shell=True)

                        if args.dataset == 'pointmass':
                            env = PointMassEnv(target=None, max_steps=simulation_timesteps, initial_state=None, epsilon=0.5, reset_target_reached=True, 
                                            bonus_reward=False, reset_out_of_bounds=True, theta_as_sine_cosine=True, num_episodes=10,
                                            n_moving_obstacles_box=n_obstacles[0], n_static_obstacles_box=n_obstacles[1], n_moving_obstacles_circle=n_obstacles[2], 
                                            n_static_obstacles_circle=n_obstacles[3], test=True, seed=n)
                        else:
                            env = Quad2DEnv(target=None, max_steps=simulation_timesteps, initial_state=None, epsilon=0.5, reset_target_reached=True, 
                                            bonus_reward=False, reset_out_of_bounds=True, theta_as_sine_cosine=True, num_episodes=10, 
                                            n_moving_obstacles_box=n_obstacles[0], n_static_obstacles_box=n_obstacles[1], n_moving_obstacles_circle=n_obstacles[2], 
                                            n_static_obstacles_circle=n_obstacles[3], min_rel_thrust=0.75, max_rel_thrust=1.25, 
                                            max_rel_thrust_difference=0.01, test=True, seed=n)

                        # Reset environment
                        obs = env.reset(seed=n)

                        observations_all[n, 0, :] = obs

                        policy = policy_config()
                        
                        for _ in range(env.max_steps):           
                            # Get current state
                            conditions = {0: obs}
                            
                            # Sample state sequence or state-action sequence
                            if with_projections:
                                unsafe_bounds_box, unsafe_bounds_circle = utils.compute_unsafe_regions(env.predict_obstacles(args.horizon), 
                                                                                                    horizon=args.horizon, 
                                                                                                    obs_dim=obs_dim, 
                                                                                                    action_dim=action_dim)
                            else:
                                unsafe_bounds_box, unsafe_bounds_circle = None, None
                                
                            warm_start = True if warmstart_steps is not False else False
                            warm_start_steps = warmstart_steps if warmstart_steps is not False else None

                            action, samples = policy(conditions=conditions, batch_size=args.batch_size, unsafe_bounds_box=unsafe_bounds_box, unsafe_bounds_circle=unsafe_bounds_circle,
                                                    warm_start_steps=warmstart_steps, verbose=False, id_model=id_model)
                            
                            if save_animation:
                                old_path = None if _ == 0 else observations_all[n][:_ + 1, [0, 2]]
                                env.render(trajectories_to_plot=samples.observations[:, :, [0, 2]], old_path=old_path, save_path=save_path)
                           
                            obs, reward, done, target_reached = env.step(action)

                            # Log
                            if _ < env.max_steps - 1:
                                observations_all[n, _ + 1, :] = obs
                            actions_all[n, _, :] = action
                            reward_all[n, _] = reward
                            is_done_all[n, _] = done

                            if target_reached == True:
                                n_reached += 1
                                target_reached_all[n] = 1

                            if done:
                                break

                        reward_total += (reward_all[n] * np.power(args.discount, np.arange(len(reward_all[n])))).sum()

                        if save_animation:
                            command = 'ffmpeg -y -r ' + str(int(1/env.dt)) + ' -i ' + save_path + '/screen_%03d.png -vcodec libx264 -pix_fmt yuv420p ' + save_path + '/a_video.mp4'
                            subprocess.call(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

                    success_rate = n_reached / n_trials
                    # reward = reward_total / n_trials
                    reward = np.mean(reward_all[reward_all != 0])

                    print(f'{args.dataset}, use actions: {args.use_actions}, with projections: {with_projections}, n_obstacles: {n_obstacles}, warmstart_steps: {warmstart_steps}, scale: {scale}, SUCCESS RATE: {success_rate}, MEAN REWARD: {round(reward, 1)}')
                    # print(f'Failed in seeds: {np.where(target_reached_all == 0)}')

                    results = {'system': system,
                                'use_actions': args.use_actions,
                                'with_projections': with_projections_range,
                                'warmstart_steps': warmstart_steps,
                                'n_obstacles': n_obstacles,
                                'trial': trials,
                                'success_rate': success_rate_all,
                                'observations_all': observations_all,
                                'actions_all': actions_all,
                                'rewards_all': reward_all,
                                'target_reached_all': target_reached_all,
                                }
                    
                    if warmstart_steps is False:
                        save_path = data_save_path  + '/' + system + \
                                                    '/use_actions_' + str(args.use_actions) + \
                                                    '/projection_' + str(with_projections) + \
                                                    '/no_warmstart' + \
                                                    '/scale_' + str(scale) + \
                                                    '/n_obstacles_' + str(n_obstacles[0]) + '_' + str(n_obstacles[1]) + '_' + str(n_obstacles[2]) + '_' + str(n_obstacles[3])
                    else:
                        save_path = data_save_path  + '/' + system + \
                                                            '/use_actions_' + str(args.use_actions) + \
                                                            '/projection_' + str(with_projections) + \
                                                            '/warmstart_steps_' + str(warmstart_steps) + \
                                                            '/scale_' + str(scale) + \
                                                            '/n_obstacles_' + str(n_obstacles[0]) + '_' + str(n_obstacles[1]) + '_' + str(n_obstacles[2]) + '_' + str(n_obstacles[3])

                    
                    if save_data:
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        with open(save_path + '/data.pkl', 'wb') as f:
                            pickle.dump(results, f)
        