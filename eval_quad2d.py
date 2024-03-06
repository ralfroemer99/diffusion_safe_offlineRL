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
labels = ['x', 'dx', 'y', 'dy', 'theta', 'dtheta', 'T1', 'T2', 'reward']

batch_size = 100

#-----------------------------------------------------------------------------#
#---------Sampled open-loop trajectories for many initial conditions----------#
#-----------------------------------------------------------------------------#

if which_experiments[0]:
    env = Quad2DEnv(min_rel_thrust=0.75, max_rel_thrust=1.25, max_rel_thrust_difference=0.01, 
                    target=None, max_steps=20, initial_state=None, 
                    epsilon=0.5, reset_target_reached=False, bonus_reward=True, 
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
                    epsilon=0.5, reset_target_reached=False, bonus_reward=True, 
                    reset_out_of_bounds=True, theta_as_sine_cosine=True, num_episodes=10)

    obs = env.reset()

    # Condition
    conditions = {0: obs}

    # Sample open-loop plan
    action, samples = policy(conditions=conditions, batch_size=batch_size, verbose=False) 
    observations = samples.observations
    actions = samples.actions if args.use_actions else None
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
        for j in range(4):
            ax[i, j].plot(observations[i, :, j], 'b')       
        ax[i, 4].plot(np.arctan2(observations[i, :, 4], observations[i, :, 5]), 'b')  # theta
        ax[i, 5].plot(observations[i, :, 6], 'b')  # dtheta
        if args.use_actions:
            ax[i, 6].plot(actions[i, :, 0], 'b')       # T1
            ax[i, 7].plot(actions[i, :, 1], 'b')       # T2
        ax[i, 8].plot(observations[i, :, 0], observations[i, :, 2], 'b')   # trajectory
        ax[i, 8].plot(observations[i, 0, 0], observations[i, 0, 2], 'go')  # start
        ax[i, 8].plot(obs[7], obs[8], 'ro')        # goal

        for _ in range(8):
            ax[i, _].set_ylabel(labels[_])
    
    plt.show()

#-----------------------------------------------------------------------------#
#----------------Closed-loop experiment without obstacles---------------------#
#-----------------------------------------------------------------------------#

if which_experiments[2]:
    env = Quad2DEnv(min_rel_thrust=0.75, max_rel_thrust=1.25, max_rel_thrust_difference=0.01, 
                    target=None, max_steps=200, initial_state=None, 
                    epsilon=0.5, reset_target_reached=False, bonus_reward=True, 
                    reset_out_of_bounds=True, theta_as_sine_cosine=True, num_episodes=10)

    n_trials = 5
    fig, ax = plt.subplots(min(n_trials, 5), 9)
    n_reached = 0
    for n in range(n_trials):
        # Reset environment
        obs = env.reset()
        print('Environment %d, initial pos: %s, target: %s' % (n, [obs[0], obs[2]], env.target))

        observations_cl = np.zeros((env.max_steps, 9))
        observations_cl[0, :] = obs
        actions_cl = np.zeros((env.max_steps, 2))
        rewards_cl = np.zeros((env.max_steps))
        for _ in range(env.max_steps - 1):
            # Get current state
            conditions = {0: obs}
            
            # Sample state sequence or state-action sequence
            action, samples = policy(conditions=conditions, batch_size=batch_size, verbose=False)
            if _ == 0:
                observations_ol = samples.observations
                actions_ol = samples.actions if args.use_actions else None

            # Step environment
            if args.use_actions:
                obs, reward, done, target_reached = env.step(action)

            # Log
            if _ < env.max_steps - 1:
                observations_cl[_ + 1, :] = obs
            actions_cl[_, :] = action
            rewards_cl[_] = reward

            if target_reached:
                n_reached += 1

            if done:
                observations_cl = observations_cl[:_ + 1, :]
                actions_cl = actions_cl[:_ + 1, :]
                rewards_cl = rewards_cl[:_ + 1]
                break

        # Plot closed-loop trajectories
        if n < 5:
            ax_cur = ax[n] if len(ax.shape) > 1 else ax
            for i in range(4):
                ax_cur[i].plot(observations_cl[:, i])                                   # x, y, dx, dy
                for j in range(5):
                    ax_cur[i].plot(observations_ol[j, :, i], 'r')
            ax_cur[4].plot(np.arctan2(observations_cl[:, 4], observations_cl[:, 5]))    # theta
            ax_cur[5].plot(observations_cl[:, 6])                                       # dtheta
            for j in range(5):
                ax_cur[4].plot(np.arctan2(observations_ol[j, :, 4], observations_ol[j, :, 5]), 'r')
                ax_cur[5].plot(observations_ol[j, :, 6], 'r')
            
            for i in range(2):
                ax_cur[i + 6].plot(actions_cl[:, i])
                if args.use_actions:
                    for j in range(5):
                        ax_cur[i + 6].plot(actions_ol[j, :, i], 'r')
            ax_cur[8].plot(observations_cl[:, 0], observations_cl[:, 2], label='Closed-loop')   # trajectory
            ax_cur[8].plot(observations_cl[0, 0], observations_cl[0, 2], 'go')  # start
            ax_cur[8].plot(observations_cl[0, 7], observations_cl[0, 8], 'ro')  # goal
            for j in range(5):
                ax_cur[8].plot(observations_ol[j, :, 0], observations_ol[j, :, 2], color='r')
            ax_cur[8].set_xlim(-5, 5)
            ax_cur[8].set_ylim(-5, 5)
            ax_cur[8].legend()

            for _ in range(8):
                ax_cur[_].set_ylabel(labels[_])

    print(f'Goal reached in {n_reached} out of {n_trials} cases, success rate: {n_reached / n_trials * 100}%')

    plt.show()

#-----------------------------------------------------------------------------#
#-----------------Closed-loop experiment with obstacles-----------------------#
#-----------------------------------------------------------------------------#
if which_experiments[3]:
    batch_size = 10

    env = Quad2DEnv(min_rel_thrust=0.75, max_rel_thrust=1.25, max_rel_thrust_difference=0.01, target=None, max_steps=200, 
                    initial_state=None, epsilon=0.5, reset_target_reached=True, bonus_reward=False, reset_out_of_bounds=True, 
                    theta_as_sine_cosine=True, num_episodes=10, n_moving_obstacles=0, n_static_obstacles=0, test=True)

    n_trials = 100
    fig, ax = plt.subplots(min(n_trials, 5), 9)
    n_reached = 0
    mean_steps = 0
    reward_total = 0
    for n in range(n_trials):
        # Reset environment
        obs = env.reset(seed=n)
        # print('Env %d, initial pos: %s, target: %s' % (n, [obs[0], obs[2]], env.target))

        observations_cl = np.zeros((env.max_steps, 9))
        observations_cl[0, :] = obs
        actions_cl = np.zeros((env.max_steps, 2))
        rewards_cl = np.zeros((env.max_steps))
        for _ in range(env.max_steps - 1):           
            # Get current state
            conditions = {0: obs}
            
            # Sample state sequence or state-action sequence
            unsafe_bounds = utils.compute_unsafe_regions(env.predict_obstacles(args.horizon), horizon=args.horizon)
            action, samples = policy(conditions=conditions, batch_size=batch_size, unsafe_bounds=None, verbose=False)
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
                ax_cur[i].plot(observations_cl[:, i])                        # x, dx, y, dy
                for j in range(min(batch_size, 5)):
                    ax_cur[i].plot(observations_ol[j, :, i], 'r')
            
            ax_cur[4].plot(np.arctan2(observations_cl[:, 4], observations_cl[:, 5]))    # theta
            ax_cur[5].plot(observations_cl[:, 6])                                       # dtheta
            for j in range(min(batch_size, 5)):
                ax_cur[4].plot(np.arctan2(observations_ol[j, :, 4], observations_ol[j, :, 5]), 'r')
                ax_cur[5].plot(observations_ol[j, :, 6], 'r')
            
            for i in range(2):
                ax_cur[i + 6].plot(actions_cl[:, i])
                if args.use_actions:
                    for j in range(min(batch_size, 5)):
                        ax_cur[i + 6].plot(actions_ol[j, :, i], 'r')
            ax_cur[8].plot(observations_cl[:, 0], observations_cl[:, 2], label='Closed-loop')   # trajectory
            ax_cur[8].plot(observations_cl[0, 0], observations_cl[0, 2], 'go')  # start
            ax_cur[8].plot(observations_cl[0, 7], observations_cl[0, 8], 'ro')  # goal
            for j in range(min(batch_size, 5)):
                ax_cur[8].plot(observations_ol[j, :, 0], observations_ol[j, :, 2], color='r')
            ax_cur[8].set_xlim(-5, 5)
            ax_cur[8].set_ylim(-5, 5)
            ax_cur[8].legend()

            for _ in range(8):
                ax_cur[_].set_ylabel(labels[_])
    if n_reached > 0:
        print(f'Goal reached in {n_reached} out of {n_trials} cases, success rate: {n_reached / n_trials * 100}%, mean steps: {mean_steps / n_reached}, mean reward: {reward_total / n_trials}')
        print(f'Use actions: {args.use_actions}, scale: {args.scale}')
    else:
        print(f'Goal not reached in any of the {n_trials} cases')

    # plt.show()