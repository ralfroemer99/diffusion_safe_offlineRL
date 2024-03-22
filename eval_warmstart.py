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
save_data = True
data_save_path = 'results/data'
save_animation = False
animation_save_path = 'results/animation' if save_animation else None

# List of arguments to pass to the script
systems_list = ['quad2d']
n_obstacles_range = [[0, 5, 0, 5],
                     [2, 5, 2, 5]]
with_projections_range = [True]
warmstart_steps_range = [6, 8, 10, None]

scale_range = [1000]

n_trials = 100

seeds_list = None

for system in systems_list:
    # Store success rate and reward
    success_rate_all = np.zeros((len(n_obstacles_range), len(with_projections_range), len(warmstart_steps_range)))
    reward_mean_all = np.zeros((len(n_obstacles_range), len(with_projections_range), len(warmstart_steps_range)))
    steps_mean_all = np.zeros((len(n_obstacles_range), len(with_projections_range), len(warmstart_steps_range)))

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
    else:
        id_model = None

    for idx1, with_projections in enumerate(with_projections_range):
        warmstart_steps_range_mod = warmstart_steps_range if with_projections else [None]

        if scale_range is None:
            scale_range = [args.scale]

        for scale in scale_range:
            for idx0, n_obstacles in enumerate(n_obstacles_range):
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
                for idx2, warmstart_steps in enumerate(warmstart_steps_range_mod):
                    n_reached = 0
                    n_steps = 0
                    reward_total = 0
                    observations_all = np.zeros((n_trials, simulation_timesteps, obs_dim))
                    actions_all = np.zeros((n_trials, simulation_timesteps, 2))
                    reward_all = np.zeros((n_trials, simulation_timesteps))
                    target_reached_all = np.zeros((n_trials))
                    is_done_all = np.zeros((n_trials, simulation_timesteps))
                    
                    #-----------------Closed-loop experiment with obstacles-----------------------#
                    # if seeds_list is not None:
                    #     if system == 'pointmass':
                    #         if idx == 0:
                    #             trials = seeds_list[0]
                    #         else:
                    #             trials = seeds_list[1]
                    #     else:
                    #         if idx == 0:
                    #             trials = seeds_list[2]
                    #         else:
                    #             trials = seeds_list[3]
                    # else:                    
                    trials = range(n_trials)
                    
                    for n in trials:
                        if save_animation:
                            if warmstart_steps is None:
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
                            env = PointMassEnv(max_steps=simulation_timesteps, epsilon=0.5, reset_target_reached=True, 
                                               reset_out_of_bounds=True, n_moving_obstacles_box=n_obstacles[0], n_static_obstacles_box=n_obstacles[1], 
                                               n_moving_obstacles_circle=n_obstacles[2], n_static_obstacles_circle=n_obstacles[3], test=True, seed=n)
                        else:
                            env = Quad2DEnv(max_steps=simulation_timesteps, epsilon=0.5, reset_target_reached=True, 
                                            reset_out_of_bounds=True, n_moving_obstacles_box=n_obstacles[0], n_static_obstacles_box=n_obstacles[1], 
                                            n_moving_obstacles_circle=n_obstacles[2], n_static_obstacles_circle=n_obstacles[3], test=True, seed=n)

                        # Reset environment
                        obs = env.reset(seed=n)
                        observations_all[n, 0, :] = obs

                        policy = policy_config()
                        # warm_start_steps = warmstart_steps if warmstart_steps is not False else None

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
                                n_steps += _ + 1
                            if done:
                                n_steps += env.max_steps if target_reached != True else 0
                                break

                        reward_total += (reward_all[n] * np.power(args.discount, np.arange(len(reward_all[n])))).sum()

                        if save_animation:
                            command = 'ffmpeg -y -r ' + str(int(1/env.dt)) + ' -i ' + save_path + '/screen_%03d.png -vcodec libx264 -pix_fmt yuv420p ' + save_path + '/a_video.mp4'
                            subprocess.call(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

                    success_rate = n_reached / len(trials)
                    reward_mean = np.mean(reward_all[reward_all != 0])
                    steps_mean = n_steps / n_trials
                    reward = np.mean(reward_all[reward_all != 0])

                    success_rate_all[idx0, idx1, idx2] = success_rate
                    reward_mean_all[idx0, idx1, idx2] = reward_mean
                    steps_mean_all[idx0, idx1, idx2] = steps_mean

                    print(f'{args.dataset}, use actions: {args.use_actions}, with projections: {with_projections}, n_obstacles: {n_obstacles}, warmstart_steps: {warmstart_steps}, scale: {scale}, SUCCESS RATE: {success_rate}, MEAN STEPS: {round(steps_mean, 2)}, MEAN REWARD: {round(reward, 1)}')
                    # print(f'Failed in seeds: {np.where(target_reached_all == 0)}')

                    results = {'system': system,
                                'use_actions': args.use_actions,
                                'with_projections': with_projections,
                                'warmstart_steps': warmstart_steps,
                                'n_obstacles': n_obstacles,
                                'trial': range(n_trials),
                                'success_rate': success_rate,
                                'reward_mean': reward_mean,
                                'steps_mean': n_steps / n_trials,
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
                                    '/n_obstacles_' + str(0) + '_' + str(0) + '_' + str(0) + '_' + str(0)
                    else:
                        save_path = data_save_path  + '/' + system + \
                                    '/use_actions_' + str(args.use_actions) + \
                                    '/projection_' + str(with_projections) + \
                                    '/warmstart_steps_' + str(warmstart_steps) + \
                                    '/scale_' + str(scale) + \
                                    '/n_obstacles_' + str(0) + '_' + str(0) + '_' + str(0) + '_' + str(0)
                    
                    if save_data:
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        with open(save_path + '/data.pkl', 'wb') as f:
                            pickle.dump(results, f)
        