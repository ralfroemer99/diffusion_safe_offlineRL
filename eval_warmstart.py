import os
import pickle
import subprocess
import time
import numpy as np
import diffuser.utils as utils
import diffuser.sampling as sampling
from envs.quad_2d import Quad2DEnv
from envs.pointmass import PointMassEnv

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

data_save_path = 'results/data'
save_animation = True
animation_save_path = 'results/animation' if save_animation else None

# List of arguments to pass to the script
systems_list = ['pointmass']
# n_obstacles_range = [[0, 5],
#                      [3, 5]]
n_obstacles_range = [[0, 5]]
#                      [3, 5]]
with_projections_range = [True]
warmstart_steps_range = [4, 6, 8]


n_trials = 100
t_check_collision = 1

for system in systems_list:
    # Store success rate and reward
    success_rate_all = np.zeros((len(n_obstacles_range), len(with_projections_range), len(warmstart_steps_range)))
    reward_mean_all = np.zeros((len(n_obstacles_range), len(with_projections_range), len(warmstart_steps_range)))

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

    for idx0, with_projections in enumerate(with_projections_range):
        for idx1, n_obstacles in enumerate(n_obstacles_range):
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
            for idx2, warmstart_steps in enumerate(warmstart_steps_range):
                
                n_reached = 0
                reward_total = 0
                observations_all = np.zeros((n_trials, simulation_timesteps, obs_dim))
                actions_all = np.zeros((n_trials, simulation_timesteps, 2))
                reward_all = np.zeros((n_trials, simulation_timesteps))
                is_done_all = np.zeros((n_trials, simulation_timesteps))
                
                #-----------------Closed-loop experiment with obstacles-----------------------#
                for n in range(n_trials):
                    # if n % 10 == 0:
                    #     print(f'{args.dataset}, use actions: {args.use_actions}, with projections: {with_projections}, n_obstacles: {n_obstacles}, trial: {n}')

                    if save_animation:
                        save_path = animation_save_path  + '/' + system + \
                                                            '/n_obstacles_' + str(n_obstacles[0]) + '_' + str(n_obstacles[1]) + \
                                                            '/projection_' + str(with_projections) + \
                                                            '/warmstart_steps_' + str(warmstart_steps) + \
                                                            '/seed_' + str(n)
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        else:
                            subprocess.call('rm -rf ' + os.path.join(save_path, '*'), shell=True)

                    if args.dataset == 'pointmass':
                        env = PointMassEnv(target=None, max_steps=simulation_timesteps, initial_state=None, epsilon=0.5, reset_target_reached=True, 
                                        bonus_reward=False, reset_out_of_bounds=True, theta_as_sine_cosine=True, num_episodes=10,
                                        # n_moving_obstacles=n_obstacles[0], n_static_obstacles=n_obstacles[1])
                                        n_moving_obstacles=n_obstacles[0], n_static_obstacles=n_obstacles[1], test=True, seed=n)
                    else:
                        env = Quad2DEnv(target=None, max_steps=simulation_timesteps, initial_state=None, epsilon=0.5, reset_target_reached=True, 
                                        bonus_reward=False, reset_out_of_bounds=True, theta_as_sine_cosine=True, num_episodes=10, 
                                        n_moving_obstacles=n_obstacles[0], n_static_obstacles=n_obstacles[1], min_rel_thrust=0.75, max_rel_thrust=1.25, 
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
                            unsafe_bounds = utils.compute_unsafe_regions(env.predict_obstacles(args.horizon), horizon=args.horizon, obs_dim=obs_dim)
                            action, samples = policy(conditions=conditions, batch_size=args.batch_size, unsafe_bounds=unsafe_bounds, warm_start=True, 
                                                     warm_start_steps=warmstart_steps, verbose=False)
                        else:
                            action, samples = policy(conditions=conditions, batch_size=args.batch_size, unsafe_bounds=None, warm_start=False, verbose=False)
                        
                        if save_animation:
                            env.render(trajectories_to_plot=samples.observations[:, :, [0, 2]], save_path=save_path)

                        # Step environment
                        if not args.use_actions:
                            next_obs = samples.observations[0, 1, :]
                            action = env.inverse_dynamics(next_obs)
                        
                        obs, reward, done, target_reached = env.step(action)

                        # Log
                        if _ < env.max_steps - 1:
                            observations_all[n, _ + 1, :] = obs
                        actions_all[n, _, :] = action
                        reward_all[n, _] = reward
                        is_done_all[n, _] = done

                        if target_reached == True:
                            n_reached += 1

                        if done:
                            break

                    reward_total += (reward_all[n] * np.power(args.discount, np.arange(len(reward_all[n])))).sum()

                    if save_animation:
                        command = 'ffmpeg -y -r ' + str(int(1/env.dt)) + ' -i ' + save_path + '/screen_%03d.png -vcodec libx264 -pix_fmt yuv420p ' + save_path + '/a_video.mp4'
                        subprocess.call(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

                success_rate = n_reached / n_trials
                reward = reward_total / n_trials

                print(f'{args.dataset}, use actions: {args.use_actions}, with projections: {with_projections}, n_obstacles: {n_obstacles}, \
                      warmstart_steps: {warmstart_steps}, SUCCESS RATE: {success_rate}, MEAN REWARD: {round(reward, 1)}')

                results = {'system': system,
                            'use_actions': args.use_actions,
                            'with_projections': with_projections_range,
                            'n_obstacles': n_obstacles,
                            'success_rate': success_rate_all,
                            'reward_mean': reward_mean_all,
                            'observations_all': observations_all,
                            'actions_all': actions_all,
                            'rewards_all': reward_all,
                            }
                
                save_path = data_save_path + '/' + system + \
                                            '/n_obstacles_' + str(n_obstacles[0]) + '_' + str(n_obstacles[1]) + \
                                            '/projection_' + str(with_projections) + \
                                            '/warmstart_steps_' + str(warmstart_steps)

                # save_path = data_save_path + '/' + system + '/n_obs' + str(n_obstacles[0]) + '_' + str(n_obstacles[1]) + '/projection_' + str(with_projections) + '/warmstart_steps' + str(warmstart_steps)

                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                with open(save_path + '/data.pkl', 'wb') as f:
                    pickle.dump(results, f)
        