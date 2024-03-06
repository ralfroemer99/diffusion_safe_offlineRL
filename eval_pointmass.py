import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import diffuser.utils as utils
import diffuser.sampling as sampling
from envs.pointmass import PointMassEnv

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'pointmass'
    config: str = 'config.pointmass'

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

diffusion_losses = diffusion_experiment.losses
value_losses = value_experiment.losses

# Plot losses
# fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# utils.plot_losses(diffusion_losses, ax=ax[0], title='Diffusion losses')
# utils.plot_losses(value_losses, ax=ax[1], title='Value losses')
# plt.show()

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

which_experiments = [0, 0, 0, 1]
# simulation_timesteps = 40
labels = ['x', 'dx', 'y', 'dy', 'u1', 'u2', 'reward']

#-----------------------------------------------------------------------------#
#---------Sampled open-loop trajectories for many initial conditions----------#
#-----------------------------------------------------------------------------#

if which_experiments[0]:
    env = PointMassEnv(target=None, max_steps=20, initial_state=None, 
                    epsilon=0.2, reset_target_reached=False, bonus_reward=False, 
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
        action, samples = policy(conditions=conditions, batch_size=args.batch_size, verbose=False) 
        observations = samples.observations

        ax[i // 10, i % 10].plot(observations[0, :, 0], observations[0, :, 2])
        ax[i // 10, i % 10].plot(observations[0, 0, 0], observations[0, 0, 2], 'go')
        ax[i // 10, i % 10].plot(observations[0, 0, -2], observations[0, 0, -1], 'ro')

        # Check if goal has been reached
        if any(np.linalg.norm(observations[0, :, [0, 2]] - obs[4:5].reshape(-1, 1), axis=0) < 1):
            n_reached += 1

    print(f'Goal reached in {n_reached} out of 100 cases')
    plt.show()


if which_experiments[1]:
    env = PointMassEnv(target=[-1, 2], max_steps=20, initial_state=[0, 0, 0, 0], 
                       epsilon=0.2, reset_target_reached=False, bonus_reward=False, 
                       reset_out_of_bounds=True, theta_as_sine_cosine=True, num_episodes=10)

    obs = env.reset()

    # Condition
    conditions = {0: obs}

    # Sample open-loop plan
    action, samples = policy(conditions=conditions, batch_size=args.batch_size, verbose=False) 
    observations = samples.observations
    actions = samples.actions if args.use_actions else None
    right_direction_counter = 0
    for i in range(args.batch_size):
        if observations[i, -1, 0] < 0:
            right_direction_counter += 1
    print(f'Final x position has mean {observations[:, -1, 0].mean()}')
    print(f'Right direction: {right_direction_counter / args.batch_size * 100}%')

    # Plot open-loop plan
    fig, ax = plt.subplots(min(args.batch_size, 10), 7)
    fig.suptitle('Open-loop plan')
    for i in range(min(args.batch_size, 10)):
        for j in range(4):
            ax[i, j].plot(observations[i, :, j], 'b')
        if args.use_actions:
            ax[i, 4].plot(actions[i, :, 0], 'b')
            ax[i, 5].plot(actions[i, :, 1], 'b')
        ax[i, 6].plot(observations[i, :, 0], observations[i, :, 2], 'b')   # trajectory
        ax[i, 6].plot(observations[i, 0, 0], observations[i, 0, 2], 'go')  # start
        ax[i, 6].plot(obs[4], obs[5], 'ro')        # goal

        for _ in range(6):
            ax[i, _].set_ylabel(labels[_])
    
    plt.show()

#-----------------------------------------------------------------------------#
#-----------------Closed-loop experiment without obstacles--------------------#
#-----------------------------------------------------------------------------#

if which_experiments[2]:
    env = PointMassEnv(target=None, max_steps=100, initial_state=None, 
                    epsilon=0.2, reset_target_reached=True, bonus_reward=False, 
                    reset_out_of_bounds=True, theta_as_sine_cosine=True, num_episodes=10)

    n_trials = 10
    fig, ax = plt.subplots(min(n_trials, 5), 7)
    n_reached = 0
    mean_steps = 0
    for n in range(n_trials):
        # Reset environment
        obs = env.reset(seed=n)
        print('Environment %d, initial pos: %s, target: %s' % (n, [obs[0], obs[2]], env.target))

        observations_cl = np.zeros((env.max_steps, 6))
        observations_cl[0, :] = obs
        actions_cl = np.zeros((env.max_steps, 2))
        rewards_cl = np.zeros((env.max_steps))
        for _ in range(env.max_steps - 1):
            # Get current state
            conditions = {0: obs}
            
            # Sample state sequence or state-action sequence
            action, samples = policy(conditions=conditions, batch_size=args.batch_size, verbose=False)
            if _ == 0:
                observations_ol = samples.observations
                actions_ol = samples.actions if args.use_actions else None

            # Step environment
            if args.use_actions:
                # action = env.sample_action()
                obs, reward, done, target_reached = env.step(action)

            # Log
            if _ < env.max_steps - 1:
                observations_cl[_ + 1, :] = obs
            actions_cl[_, :] = action
            rewards_cl[_] = reward

            if target_reached:
                n_reached += 1
                mean_steps += _ + 1
                print(f'Environment {n} reached the goal in {_} steps, current success rate: {n_reached / (n + 1) * 100}%')

            if done:
                observations_cl = observations_cl[:_ + 1, :]
                actions_cl = actions_cl[:_ + 1, :]
                rewards_cl = rewards_cl[:_ + 1]
                break

        # Plot closed-loop trajectories
        if n < 5:
            ax_cur = ax[n] if len(ax.shape) > 1 else ax
            for i in range(4):
                ax_cur[i].plot(observations_cl[:, i])
                for j in range(min(args.batch_size, 5)):
                    ax_cur[i].plot(observations_ol[j, :, i], 'r')
            for i in range(2):
                ax_cur[i + 4].plot(actions_cl[:, i])
                if args.use_actions:
                    for j in range(min(args.batch_size, 5)):
                        ax_cur[i + 4].plot(actions_ol[j, :, i], 'r')
            ax_cur[6].plot(observations_cl[:, 0], observations_cl[:, 2], label='Closed-loop')   # trajectory
            ax_cur[6].plot(observations_cl[0, 0], observations_cl[0, 2], 'go')  # start
            ax_cur[6].plot(observations_cl[0, 4], observations_cl[0, 5], 'ro')  # goal
            for j in range(min(args.batch_size, 5)):
                ax_cur[6].plot(observations_ol[j, :, 0], observations_ol[j, :, 2], color='r')
            ax_cur[6].set_xlim(-5, 5)
            ax_cur[6].set_ylim(-5, 5)
            ax_cur[6].legend()

            for _ in range(6):
                ax_cur[_].set_ylabel(labels[_])

    if n_reached > 0:
        print(f'Goal reached in {n_reached} out of {n_trials} cases, success rate: {n_reached / n_trials * 100}%, mean steps: {mean_steps / n_reached}')
    else:
        print(f'Goal not reached in any of the {n_trials} cases')
    print(f'Scale: {args.scale}')

    plt.show()

#-----------------------------------------------------------------------------#
#-----------------Closed-loop experiment with obstacles-----------------------#
#-----------------------------------------------------------------------------#
if which_experiments[3]:
    env = PointMassEnv(target=None, max_steps=100, initial_state=None, epsilon=0.5, reset_target_reached=True, 
                       bonus_reward=False, reset_out_of_bounds=True, theta_as_sine_cosine=True, num_episodes=10,
                       n_moving_obstacles=0, n_static_obstacles=0)

    n_trials = 100
    fig, ax = plt.subplots(min(n_trials, 5), 7)
    n_reached = 0
    mean_steps = 0
    reward_total = 0
    for n in range(n_trials):
        # Reset environment
        obs = env.reset(seed=n)
        # print('Env %d, initial pos: %s, target: %s' % (n, [obs[0], obs[2]], env.target))

        observations_cl = np.zeros((env.max_steps, 6))
        observations_cl[0, :] = obs
        actions_cl = np.zeros((env.max_steps, 2))
        rewards_cl = np.zeros((env.max_steps))
        for _ in range(env.max_steps - 1):           
            # Get current state
            conditions = {0: obs}
            
            # Sample state sequence or state-action sequence
            unsafe_bounds = utils.compute_unsafe_regions(env.predict_obstacles(args.horizon), horizon=args.horizon)
            action, samples = policy(conditions=conditions, batch_size=args.batch_size, unsafe_bounds=None, verbose=False)
            if _ == 0:
                observations_ol = samples.observations
                actions_ol = samples.actions if args.use_actions else None

            # env.render(trajectories_to_plot=samples.observations[:, :, [0, 2]])

            # Step environment

            if not args.use_actions:
                next_obs = samples.observations[0, 1, :]
                action = env.inverse_dynamics(next_obs)
                # action = env.sample_action()
            
            obs, reward, done, target_reached = env.step(action)

            # Log
            if _ < env.max_steps - 1:
                observations_cl[_ + 1, :] = obs
            actions_cl[_, :] = action
            rewards_cl[_] = reward

            if target_reached == True:
                n_reached += 1
                mean_steps += _ + 1
                # print(f'Env {n}: REACHED in {_} steps, current success rate: {n_reached / (n + 1) * 100}%')

            # if target_reached == -1:
            #     print(f'Env {n}: COLLISION in {_} steps')

            if done:
                observations_cl = observations_cl[:_ + 1, :]
                actions_cl = actions_cl[:_ + 1, :]
                rewards_cl = rewards_cl[:_ + 1]
                reward_total += (rewards_cl * np.power(args.discount, np.arange(len(rewards_cl)))).sum()
                break

        # Plot closed-loop trajectories
        if n < 5:
            ax_cur = ax[n] if len(ax.shape) > 1 else ax
            for i in range(4):
                ax_cur[i].plot(observations_cl[:, i])
                for j in range(min(args.batch_size, 5)):
                    ax_cur[i].plot(observations_ol[j, :, i], 'r')
            for i in range(2):
                ax_cur[i + 4].plot(actions_cl[:, i])
                if args.use_actions:
                    for j in range(min(args.batch_size, 5)):
                        ax_cur[i + 4].plot(actions_ol[j, :, i], 'r')
            ax_cur[6].plot(observations_cl[:, 0], observations_cl[:, 2], label='Closed-loop')   # trajectory
            ax_cur[6].plot(observations_cl[0, 0], observations_cl[0, 2], 'go')  # start
            ax_cur[6].plot(observations_cl[0, 4], observations_cl[0, 5], 'ro')  # goal
            for j in range(min(args.batch_size, 5)):
                ax_cur[6].plot(observations_ol[j, :, 0], observations_ol[j, :, 2], color='r')
            ax_cur[6].set_xlim(-5, 5)
            ax_cur[6].set_ylim(-5, 5)
            ax_cur[6].legend()

            for _ in range(6):
                ax_cur[_].set_ylabel(labels[_])
    if n_reached > 0:
        print(f'Goal reached in {n_reached} out of {n_trials} cases, success rate: {n_reached / n_trials * 100}%, mean steps: {mean_steps / n_reached}, mean reward: {reward_total / n_trials}')
    else:
        print(f'Goal not reached in any of the {n_trials} cases')

    # plt.show()
