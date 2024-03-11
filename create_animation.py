import os
import subprocess
import pickle
import numpy as np
import diffuser.utils as utils
import diffuser.sampling as sampling
from envs.quad_2d import Quad2DEnv
from envs.pointmass import PointMassEnv

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

system = 'pointmass'

with_actions = True         # Has no impact
# with_projections = [False, True]
with_projections = [True]

n_trials = 100
n_obstacles = [0, 1, 0, 5]        # dynamic box obstacles, static box obstacles, dynamic circle obstacles, static circle obstacles
epsilon = 0.2
# seeds = [0, 1]
seeds = np.arange(10, 30).tolist()

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
        save_path = 'results/animation/' + system + '/projection_' + str(with_projection) + '/seed_' + str(seed)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if args.dataset == 'pointmass':
            env = PointMassEnv(target=None, max_steps=simulation_timesteps, initial_state=None, epsilon=epsilon, reset_target_reached=True, 
                            bonus_reward=False, reset_out_of_bounds=True, theta_as_sine_cosine=True, num_episodes=10,
                            n_moving_obstacles_box=n_obstacles[0], n_static_obstacles_box=n_obstacles[1], n_moving_obstacles_circle=n_obstacles[2], 
                            n_static_obstacles_circle=n_obstacles[3], seed=seed)
        else:
            env = Quad2DEnv(target=None, max_steps=simulation_timesteps, initial_state=None, epsilon=epsilon, reset_target_reached=True, 
                            bonus_reward=False, reset_out_of_bounds=True, theta_as_sine_cosine=True, num_episodes=10, 
                            n_moving_obstacles_box=n_obstacles[0], n_static_obstacles_box=n_obstacles[1], n_moving_obstacles_circle=n_obstacles[2], 
                            n_static_obstacles_circle=n_obstacles[3], min_rel_thrust=0.75, max_rel_thrust=1.25, 
                            max_rel_thrust_difference=0.01, test=True, seed=seed)

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
                unsafe_bounds_boxes, unsafe_bounds_circles = utils.compute_unsafe_regions(env.predict_obstacles(args.horizon), horizon=args.horizon, obs_dim=obs_dim)
                # unsafe_bounds = utils.compute_unsafe_regions(env.predict_obstacles(args.horizon), horizon=int(t_check_collision / env.dt), obs_dim=obs_dim)
                # start_time = time.time()
                action, samples = policy(conditions=conditions, batch_size=args.batch_size, unsafe_bounds_box=unsafe_bounds_boxes, 
                                         unsafe_bounds_circle=unsafe_bounds_circles, verbose=False)
                # print(f'Policy: {time.time() - start_time}')
            else:
                action, samples = policy(conditions=conditions, batch_size=args.batch_size, verbose=False)

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
                    #    'env': env,
                    'use_actions': args.use_actions,
                    'n_obstacles': n_obstacles,
                    'reward_mean': reward,
                    'observations_all': observations_all,
                    'actions_all': actions_all,
                    'rewards_all': reward_all,
        }

        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        # with open(save_path + '/' + system + '_seed_' + str(seed) + '.pkl', 'wb') as f:
        #     pickle.dump(results, f)

        # command = 'ffmpeg -y -r ' + str(int(1/env.dt)) + ' -i ' + save_path + '/screen_%03d.png -vcodec libx264 -pix_fmt yuv420p ' + save_path + '/a_video.mp4'
        # subprocess.call(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        # command = "ffmpeg -r 20 -f image2 -s 1500x1500 -i results/animation/pointmass/seed_0/screen_%03d.png -vcodec libx264 -crf 25  results/animation/pointmass/seed_0/video2.mp4"
        # subprocess.call(command, shell=True)

print(target_reached_which)
if target_reached_which.shape[0] == 2:
    print('Projection changed behavior for seeds: ', np.where(target_reached_which[0] != target_reached_which[1])[0])
