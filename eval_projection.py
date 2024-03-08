import os
import pickle
import time
import numpy as np
import diffuser.utils as utils
import diffuser.sampling as sampling
from envs.quad_2d import Quad2DEnv
from envs.pointmass import PointMassEnv

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

save_path = 'results/projection'

# List of arguments to pass to the script
systems_list = ['pointmass', 'quad2d']
n_obstacles_range = [[0, 0],      # dynamic obstacles, static obstacles
                     [0, 5],
                     [3, 5]]
# n_obstacles_range = [[0, 5],
#                      [3, 5]]
with_actions_range = [True]         # Has no impact
with_projections_range = [False, True]

n_trials = 100
t_check_collision = 1

for system in systems_list:
    # Store success rate and reward
    success_rate_all = np.zeros((len(n_obstacles_range), len(with_actions_range), len(with_projections_range)))
    reward_mean_all = np.zeros((len(n_obstacles_range), len(with_actions_range), len(with_projections_range)))

    obs_dim = 6 if system == 'pointmass' else 9
    simulation_timesteps = 100 if system == 'pointmass' else 200
    observations_all = np.zeros((len(n_obstacles_range), len(with_actions_range), len(with_projections_range), simulation_timesteps, obs_dim))
    actions_all = np.zeros((len(n_obstacles_range), len(with_actions_range), len(with_projections_range), simulation_timesteps, 2))
    reward_all = np.zeros((len(n_obstacles_range), len(with_actions_range), len(with_projections_range), simulation_timesteps))

    for idx2, use_actions in enumerate(with_actions_range):
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

        for idx3, with_projections in enumerate(with_projections_range):
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
                policy = policy_config()

                #-----------------Closed-loop experiment with obstacles-----------------------#
                n_reached = 0
                reward_total = 0
                for n in range(n_trials):
                    # if n % 10 == 0:
                    #     print(f'{args.dataset}, use actions: {args.use_actions}, with projections: {with_projections}, n_obstacles: {n_obstacles}, trial: {n}')

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

                    observations_all[idx1, idx2, idx3, 0, :] = obs
                    for _ in range(env.max_steps):           
                        # Get current state
                        conditions = {0: obs}
                        
                        # Sample state sequence or state-action sequence
                        if with_projections:
                            unsafe_bounds = utils.compute_unsafe_regions(env.predict_obstacles(args.horizon), horizon=args.horizon, obs_dim=obs_dim)
                            # unsafe_bounds = utils.compute_unsafe_regions(env.predict_obstacles(args.horizon), horizon=int(t_check_collision / env.dt), obs_dim=obs_dim)
                            # start_time = time.time()
                            action, samples = policy(conditions=conditions, batch_size=args.batch_size, unsafe_bounds=unsafe_bounds, verbose=False)
                            # print(f'Policy: {time.time() - start_time}')
                        else:
                            action, samples = policy(conditions=conditions, batch_size=args.batch_size, unsafe_bounds=None, verbose=False)
                        # env.render(trajectories_to_plot=samples.observations[:, :, [0, 2]])

                        # Step environment
                        if not args.use_actions:
                            next_obs = samples.observations[0, 1, :]
                            action = env.inverse_dynamics(next_obs)
                        
                        obs, reward, done, target_reached = env.step(action)

                        # Log
                        if _ < env.max_steps - 1:
                            observations_all[idx1, idx2, idx3, _ + 1, :] = obs
                        actions_all[idx1, idx2, idx3, _, :] = action
                        reward_all[idx1, idx2, idx3, _] = reward

                        if target_reached == True:
                            n_reached += 1

                        if done:
                            break

                    reward_total += (reward_all[idx1, idx2, idx3] * np.power(args.discount, np.arange(len(reward_all[idx1, idx2, idx3])))).sum()

                success_rate = n_reached / n_trials
                reward = reward_total / n_trials
                success_rate_all[idx1, idx2, idx3] = success_rate
                reward_mean_all[idx1, idx2, idx3] = reward

                print(f'{args.dataset}, use actions: {args.use_actions}, with projections: {with_projections}, n_obstacles: {n_obstacles}, SUCCESS RATE: {success_rate}, MEAN REWARD: {round(reward, 1)}')

    # print('-----------------------------------------------------------------------------------------------------------------')
    # for _, batch_size in enumerate(n_obstacles_range):
    #     print(f'{args.dataset}, use actions: {args.use_actions}, BATCH_SIZE: {batch_size}, SCALES: {np.round(scale_range, 2)}, SUCCESS_RATE: {np.round(success_rate_all[:, _], 2)}, REWARD: {np.round(reward_mean_all[:, _], 2)}')
    # print('-----------------------------------------------------------------------------------------------------------------')
    
    results = {'system': system,
            #    'env': env,
               'use_actions': args.use_actions,
               'with_projections_range': with_projections_range,
               'with_actions_range': with_actions_range,
               'n_obstacles_range': n_obstacles_range,
               'success_rate': success_rate_all,
               'reward_mean': reward_mean_all,
               'observations_all': observations_all,
               'actions_all': actions_all,
               'rewards_all': reward_all,
                }
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(save_path + '/' + system + 'use_actions' + str(args.use_actions) + '.pkl', 'wb') as f:
        pickle.dump(results, f)
