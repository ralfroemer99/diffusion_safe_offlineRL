import os
import subprocess
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

# List of arguments to pass to the script
system = 'pointmass'
n_obstacles = [0, 1]        # dynamic obstacles, static obstacles
# n_obstacles_range = [[0, 5],
#                      [3, 5]]
with_actions = True         # Has no impact
with_projections = [True]
warmstart_steps_range = np.arange(1, 20)

t_check_collision = 1

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

#-----------------Closed-loop experiment with obstacles-----------------------#
if args.dataset == 'pointmass':
    env = PointMassEnv(target=None, max_steps=simulation_timesteps, initial_state=None, epsilon=0.5, reset_target_reached=True, 
                    bonus_reward=False, reset_out_of_bounds=True, theta_as_sine_cosine=True, num_episodes=10,
                    # n_moving_obstacles=n_obstacles[0], n_static_obstacles=n_obstacles[1])
                    n_moving_obstacles=n_obstacles[0], n_static_obstacles=n_obstacles[1], test=True)
else:
    env = Quad2DEnv(target=None, max_steps=simulation_timesteps, initial_state=None, epsilon=0.5, reset_target_reached=True, 
                    bonus_reward=False, reset_out_of_bounds=True, theta_as_sine_cosine=True, num_episodes=10, 
                    n_moving_obstacles=n_obstacles[0], n_static_obstacles=n_obstacles[1], min_rel_thrust=0.75, max_rel_thrust=1.25, 
                    max_rel_thrust_difference=0.01, test=True)

for with_projection in with_projections:
    for warmstart_steps in warmstart_steps_range:
        save_path = 'results/animation/' + system + '/check_projection/projection_' + str(with_projection) + '/warmstart_steps' + str(warmstart_steps)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            # Remove old images
            # command = 'rm ' + save_path + '/*.png'
            subprocess.call('rm ' + save_path + '/*.png', shell=True)

        # Reset environment
        obs = env.reset()

        # Manually define initial state, goal and obstacle
        env.state = np.array([0.2, 0.0, 0.0, 0.0])
        env.obstacles[0] = {'x': 0.0, 'y': 3.0, 'vx': 0, 'vy': 0, 'd': 0.6}
        env.target = (0.0, 4.5)

        obs = env._get_ob()

        policy = policy_config()
        for _ in range(env.max_steps):           
            # Get current state
            conditions = {0: obs}
            
            # Sample state sequence or state-action sequence
            if with_projection:
                unsafe_bounds = utils.compute_unsafe_regions(env.predict_obstacles(args.horizon), horizon=args.horizon, obs_dim=obs_dim)
                action, samples = policy(conditions=conditions, batch_size=args.batch_size, unsafe_bounds=unsafe_bounds, warm_start=True, 
                                         warm_start_steps=warmstart_steps, verbose=False)
            else:
                action, samples = policy(conditions=conditions, batch_size=args.batch_size, unsafe_bounds=None, warm_start=False, verbose=False)
            env.render(trajectories_to_plot=samples.observations[:, :, [0, 2]], save_path=save_path)

            # Step environment
            if not args.use_actions:
                next_obs = samples.observations[0, 1, :]
                action = env.inverse_dynamics(next_obs)
            
            obs, reward, done, target_reached = env.step(action)

            if done:
                env.render(save_path=save_path)
                break

        command = 'ffmpeg -y -r ' + str(int(1/env.dt)) + ' -i ' + save_path + '/screen_%03d.png -vcodec libx264 -pix_fmt yuv420p ' + save_path + '/a_video.mp4'
        subprocess.call(command, shell=True)
