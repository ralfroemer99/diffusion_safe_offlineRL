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

data_save_path = 'results/data_new_env'
with_projections = False
warmstart_steps = None

# List of arguments to pass to the script
systems_list = ['pointmass', 'quad2d']
scale_range = np.logspace(0, 4, 5)
batch_size_range = [1, 2, 4, 8, 16]
# with_actions_range = [False, True]

n_trials = 100

for system in systems_list:
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

    # Store success rate and reward
    success_rate_all = np.zeros((len(batch_size_range), len(scale_range)))
    reward_mean_all = np.zeros((len(batch_size_range), len(scale_range)))

    obs_dim = 6 if system == 'pointmass' else 9
    simulation_timesteps = 100 if system == 'pointmass' else 200

    action_dim = 2 if args.use_actions else 0
    if not args.use_actions:
        id_model = MLP(input_size=obs_dim*2, output_size=2)
        id_model.load_state_dict(torch.load('logs/' + system + '/inverse_dynamics/defaults_H' + str(args.horizon) + '_T' + str(args.n_diffusion_steps) + '_AFalse/model.pt'))
        id_model.eval()

    for idx0, batch_size in enumerate(batch_size_range):
        for idx1, scale in enumerate(scale_range):
            ## policies are wrappers around an unconditional diffusion model and a value guide
            policy_config = utils.Config(
                args.policy,
                guide=guide,                                    # guide = None        
                diffusion_model=diffusion,
                normalizer=dataset.normalizer,
                preprocess_fns=args.preprocess_fns,
                ## sampling kwargs
                scale=scale,                            
                sample_fn=sampling.n_step_guided_p_sample,
                n_guide_steps=args.n_guide_steps,
                t_stopgrad=args.t_stopgrad,
                scale_grad_by_std=args.scale_grad_by_std,       
                verbose=False,
            )

            #-----------------Closed-loop experiment with obstacles-----------------------#
            n_reached = 0
            observations_all = np.zeros((n_trials, simulation_timesteps, obs_dim))
            actions_all = np.zeros((n_trials, simulation_timesteps, 2))
            reward_all = np.zeros((n_trials, simulation_timesteps))
            target_reached_all = np.zeros((n_trials))
            is_done_all = np.zeros((n_trials, simulation_timesteps))

            for n in range(n_trials):
                #--------------------------------- main loop ---------------------------------#
                if args.dataset == 'pointmass':
                    env = PointMassEnv(max_steps=simulation_timesteps, epsilon=0.5, reset_target_reached=True, 
                                    reset_out_of_bounds=True, test=True, seed=n)
                else:
                    env = Quad2DEnv(max_steps=simulation_timesteps, epsilon=0.5, reset_target_reached=True, 
                                    reset_out_of_bounds=True, test=True, seed=n)
                    
                # Reset environment
                obs = env.reset(seed=n)
                observations_all[n, 0, :] = obs

                policy = policy_config()
                for _ in range(env.max_steps):           
                    # Get current state
                    conditions = {0: obs}
                    
                    # Sample state sequence or state-action sequence
                    action, samples = policy(conditions=conditions, batch_size=batch_size, verbose=False, id_model=id_model)

                    # Step environment
                    # old_path = None if _ == 0 else observations_all[n][:_ + 1, [0, 2]]
                    # env.render(trajectories_to_plot=samples.observations[:, :, [0, 2]], old_path=old_path)

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

            success_rate = n_reached / n_trials
            reward_mean = np.mean(reward_all[reward_all != 0])
            success_rate_all[idx0, idx1] = success_rate
            reward_mean_all[idx0, idx1] = reward_mean

            print(f'{args.dataset}, use actions: {args.use_actions}, scale: {round(scale, 2)}, batch_size: {batch_size}, SUCCESS RATE: {success_rate}, MEAN REWARD: {round(reward_mean, 2)}')

            results = {'system': system,
                        'use_actions': args.use_actions,
                        'with_projections': with_projections,
                        'warmstart_steps': warmstart_steps,
                        'n_obstacles': [0, 0, 0, 0],
                        'trial': range(n_trials),
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
                            '/n_obstacles_' + str(0) + '_' + str(0) + '_' + str(0) + '_' + str(0)
            else:
                save_path = data_save_path  + '/' + system + \
                            '/use_actions_' + str(args.use_actions) + \
                            '/projection_' + str(with_projections) + \
                            '/warmstart_steps_' + str(warmstart_steps) + \
                            '/scale_' + str(scale) + \
                            '/n_obstacles_' + str(0) + '_' + str(0) + '_' + str(0) + '_' + str(0)

            if not os.path.exists(save_path):
                os.makedirs(save_path)
            with open(save_path + '/data.pkl', 'wb') as f:
                pickle.dump(results, f)

    print('-----------------------------------------------------------------------------------------------------------------')
    for _, batch_size in enumerate(batch_size_range):
        print(f'System: {args.dataset}, use actions: {args.use_actions}, BATCH_SIZE: {batch_size}, SCALES: {np.round(scale_range, 2)}, SUCCESS_RATE: {np.round(success_rate_all[_, :], 2)}, REWARD: {np.round(reward_mean_all[_, :], 2)}')
    print('-----------------------------------------------------------------------------------------------------------------')
    