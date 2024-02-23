import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import diffuser.utils as utils
import diffuser.sampling as sampling
from envs.quad_2d import Quad2DEnv

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'quad2d'
    config: str = 'config.quad2d'

args = Parser().parse_args('plan')

#-----------------------------------------------------------------------------#
#---------------------------------- loading ----------------------------------#
#-----------------------------------------------------------------------------#

## load diffusion model and value function from disk
diffusion_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.diffusion_loadpath,
    epoch=args.diffusion_epoch, seed=args.seed,
)

value_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.value_loadpath,
    epoch=args.value_epoch, seed=args.seed,
)

# utils.check_compatibility(diffusion_experiment, value_experiment)

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

#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

which_experiments = [0, 1, 0, 0]
# simulation_timesteps = 40
labels = ['x', 'y', 'theta', 'dx', 'dy', 'dtheta', 'T1', 'T2', 'reward']

batch_size = 100

#-----------------------------------------------------------------------------#
#---------Sampled open-loop trajectories for many initial conditions----------#
#-----------------------------------------------------------------------------#

if which_experiments[0]:
    env = Quad2DEnv(min_rel_thrust=0.75, max_rel_thrust=1.25, max_rel_thrust_difference=0.01, 
                    target=None, max_steps=20, initial_state=None, 
                    epsilon=0.2, reset_target_reached=False, bonus_reward=True, 
                    reset_out_of_bounds=True, theta_as_sine_cosine=True, num_episodes=10)

    fig, ax = plt.subplots(10, 10)
    n_reached = 0
    for i in range(100):
        if i % 10 == 0:
            print(f'Iteration {i}')
        obs = env.reset()

        # Condition
        conditions = {0: obs}

        # Sample open-loop plan
        action, samples = policy(conditions=conditions, batch_size=batch_size, verbose=False) 
        observations = samples.observations

        ax[i // 10, i % 10].plot(observations[0, :, 0], observations[0, :, 2])
        ax[i // 10, i % 10].plot(observations[0, 0, 0], observations[0, 0, 2], 'go')
        ax[i // 10, i % 10].plot(observations[0, 0, -2], observations[0, 0, -1], 'ro')

        # Check if goal has been reached
        if any(np.linalg.norm(observations[0, :, [0, 2]] - obs[7:9].reshape(-1, 1), axis=0) < 0.2):
            n_reached += 1

    print(f'Goal reached in {n_reached} out of 100 cases')
    plt.show()


if which_experiments[1]:
    env = Quad2DEnv(min_rel_thrust=0.75, max_rel_thrust=1.25, max_rel_thrust_difference=0.01, 
                    target=[1, 1], max_steps=20, initial_state=[0, 0, 0, 1, 0, 0], 
                    epsilon=0.2, reset_target_reached=False, bonus_reward=True, 
                    reset_out_of_bounds=True, theta_as_sine_cosine=True, num_episodes=10)

    obs = env.reset()

    # Condition
    conditions = {0: obs}

    # Sample open-loop plan
    action, samples = policy(conditions=conditions, batch_size=batch_size, verbose=False) 
    observations = samples.observations
    # actions = samples.actions
    right_direction_counter = 0
    for i in range(batch_size):
        if observations[i, -1, 0] < 0:
            right_direction_counter += 1
    print(f'Final x position has mean {observations[:, -1, 0].mean()}')
    print(f'Right direction: {right_direction_counter / batch_size * 100}%')

    # Plot open-loop plan
    fig, ax = plt.subplots(min(batch_size, 10), 9)
    fig.suptitle('Open-loop plan')
    for i in range(min(batch_size, 10)):
        ax[i, 0].plot(observations[i, :, 0], 'b')  # x
        ax[i, 1].plot(observations[i, :, 2], 'b')  # y
        ax[i, 2].plot(np.arctan2(observations[i, :, 4], observations[i, :, 5]), 'b')  # theta
        ax[i, 3].plot(observations[i, :, 1], 'b')  # dx
        ax[i, 4].plot(observations[i, :, 3], 'b')  # dy
        ax[i, 5].plot(observations[i, :, 6], 'b')  # dtheta
        # ax[i, 6].plot(actions[i, :, 0], 'b')       # T1
        # ax[i, 7].plot(actions[i, :, 1], 'b')       # T2
        ax[i, 8].plot(observations[i, :, 0], observations[i, :, 2], 'b')   # trajectory
        ax[i, 8].plot(observations[i, 0, 0], observations[i, 0, 2], 'go')  # start
        ax[i, 8].plot(obs[7], obs[8], 'ro')        # goal

        for _ in range(8):
            ax[i, _].set_ylabel(labels[_])
    
    # Compute rollout and compare
    # observations_ro = np.zeros_like(observations)
    
    # for i in range(batch_size):
    #     obs = env.reset()
    #     observations_ro[i, 0] = obs
    #     for t in range(args.horizon - 1):
    #         next_obs, reward, done, target_reached = env.step(actions[i, t])
    #         observations_ro[i, t + 1] = next_obs
        
    # for i in range(batch_size):
    #     ax[i, 0].plot(observations_ro[i, :, 0], 'r')  # x
    #     ax[i, 1].plot(observations_ro[i, :, 2], 'r')  # y
    #     ax[i, 2].plot(np.arctan2(observations_ro[i, :, 4], observations_ro[i, :, 5]), 'r')  # theta
    #     ax[i, 3].plot(observations_ro[i, :, 1], 'r')  # dx
    #     ax[i, 4].plot(observations_ro[i, :, 3], 'r')  # dy
    #     ax[i, 5].plot(observations_ro[i, :, 6], 'r')  # dtheta
    #     ax[i, 8].plot(observations_ro[i, :, 0], observations_ro[i, :, 2], 'r')   # trajectory
    
    plt.show()

#-----------------------------------------------------------------------------#
#-----------------------------Closed-loop experiment--------------------------#
#-----------------------------------------------------------------------------#

if which_experiments[2]:
    # Reset environment
    obs = env.reset()

    observations_cl = np.zeros(env.max_steps, 9)
    observations_cl[0, :] = obs
    actions_cl = np.zeros(env.max_steps, 2)
    rewards_cl = np.zeros(env.max_steps)
    for _ in range(env.max_steps - 1):
        # Get current state
        conditions = {0: env.state}
        
        # Sample action
        action, samples = policy(conditions=conditions, batch_size=batch_size, verbose=False)

        # Step environment
        obs, reward, done, target_reached = env.step(action)

        # Log
        observations_cl[_, :] = obs
        actions_cl[_, :] = action
        rewards_cl[_] = reward

        if done:
            break

    # Plot closed-loop trajectories
    fig, ax = plt.subplots(1, 9)
    ax[0].plot(observations_cl[:, 0])  # x
    ax[1].plot(observations_cl[:, 2])  # y
    ax[2].plot(np.arctan2(observations_cl[:, 4], observations_cl[:, 5]))  # theta
    ax[3].plot(observations_cl[:, 1])  # dx
    ax[4].plot(observations_cl[:, 3])  # dy
    ax[5].plot(observations_cl[:, 6])  # dtheta
    ax[6].plot(actions_cl[:, 0])       # T1
    ax[7].plot(actions_cl[:, 1])       # T2
    ax[8].plot(observations_cl[:, 0], observations_cl[:, 2])   # trajectory
    ax[8].plot(observations_cl[0, 0], observations_cl[0, 2], 'go')  # start
    ax[8].plot(observations_cl[0, 7], observations_cl[0, 8], 'ro')  # goal

    for _ in range(9):
        ax[_].set_ylabel(labels[_])

    plt.show()
