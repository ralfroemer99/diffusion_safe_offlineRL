import numpy as np
import os
import torch
import argparse
import matplotlib.pyplot as plt
from utils.logger import Logger
from envs.quad_2d import Quad2DEnv

parser = argparse.ArgumentParser()

parser.add_argument('--num-episodes-trainset', type=int, default=100,
                    help='Number of episodes for creating the trainset.')
parser.add_argument('--num-episodes-testset', type=int, default=30,
                    help='Number of episodes for creating the testset.')
parser.add_argument('--num-steps', type=int, default=16,
                    help='Number of steps in each episode.')
parser.add_argument('--state-dim', type=int, default=9,
                    help='State dimension.')
parser.add_argument('--action-dim', type=int, default=2,
                    help='Action dimension.')
parser.add_argument('--min_rel_thrust', type=float, default=0.75,
                    help='Maximum total thrust for the propellers.')
parser.add_argument('--max_rel_thrust', type=float, default=1.25,
                    help='Maximum total thrust for the propellers.')
parser.add_argument('--max_rel_thrust_difference', type=float, default=0.01,
                    help='Maximum difference between the propeller thrusts.')
parser.add_argument('--epsilon', type=float, default=0.3,
                    help='Tolerance for reaching the target.')
parser.add_argument('--test', default=True,
                    help='Generate training or testing dataset.')
parser.add_argument('--training-dataset', type=str, default='quad2d_train.pkl',
                    help='Training dataset.')
parser.add_argument('--testing-dataset', type=str, default='quad2d_test.pkl',
                    help='Testing dataset.')
parser.add_argument('--seed', type=int, default=1,
                    help='Random seed (default: 1).')
parser.add_argument('--render', default=False,
                    help='Render environment.')

args = parser.parse_args()

test = args.test
if test:
    data_file_name = args.testing_dataset
    num_episodes = args.num_episodes_testset
else:
    data_file_name = args.training_dataset
    num_episodes = args.num_episodes_trainset
seed = args.seed
max_steps = args.num_steps
env = Quad2DEnv(min_rel_thrust=args.min_rel_thrust, max_rel_thrust=args.max_rel_thrust, 
                max_rel_thrust_difference=args.max_rel_thrust_difference, target=None, max_steps=max_steps,
                epsilon=args.epsilon, reset_target_reached=False, bonus_reward=False)

state_dim = args.state_dim
action_dim = args.action_dim

# Make directory for saving the datasets
directory = os.path.dirname(os.path.abspath(__file__))
folder = os.path.join(directory + '/data/')
logger = Logger(folder)
if not os.path.exists(folder):
    os.makedirs(folder)

# Set seeds
env.action_space.seed(seed)
np.random.seed(seed)

trajectories_all = []
for episode in range(num_episodes):
    state = env.reset()
    trajectory = np.zeros((max_steps, state_dim + action_dim + 2))
    for step in range(max_steps):
        action = env.sample_action()
        # Ensure that actions are not further apart than
        next_state, reward, done, target_reached = env.step(action)
        if args.render:
            env.render()
        logger.obslog((state, action, reward, next_state, done, 0))     # Last: dummy CBF value
        trajectory[step, :state_dim] = state
        trajectory[step, state_dim:state_dim + action_dim] = action
        trajectory[step, -2] = reward
        trajectory[step, -1] = done
        state = next_state
        if done:
            break
    trajectories_all.append(trajectory)

logger.save_obslog(filename=data_file_name)

# Plot the first ten trajectories
fig, ax = plt.subplots(10, 11)
labels = ['x', 'y', 'theta', 'dx', 'dy', 'dtheta', 'T1', 'T2', 'reward', 'done']
for i in range(10):
    ax[i, 0].plot(trajectories_all[i][:, 0])
    ax[i, 1].plot(trajectories_all[i][:, 2])
    ax[i, 2].plot(np.arctan2(trajectories_all[i][:, 4], trajectories_all[i][:, 5]))
    ax[i, 3].plot(trajectories_all[i][:, 1])
    ax[i, 4].plot(trajectories_all[i][:, 3])
    ax[i, 5].plot(trajectories_all[i][:, 6])
    ax[i, 6].plot(trajectories_all[i][:, 9])
    ax[i, 7].plot(trajectories_all[i][:, 10])
    ax[i, 8].plot(trajectories_all[i][:, 11])
    ax[i, 9].plot(trajectories_all[i][:, 12])  
    ax[i, 10].plot(trajectories_all[i][:, 0], trajectories_all[i][:, 2])
    ax[i, 10].plot(trajectories_all[i][0, 0], trajectories_all[i][0, 2], 'ro')
    ax[i, 10].plot(trajectories_all[i][0, 7], trajectories_all[i][0, 8], 'go')
    # ax[i, 10].plot(trajectories_all[i][-1, 0], trajectories_all[i][-1, 2], 'rx')
    for j in range(10):
        ax[i, j].set_ylabel(labels[j])
    ax[i, 10].set_ylabel('y')
    ax[i, 10].set_xlabel('x')
    ax[i, 10].set_xlim(-5, 5)
    ax[i, 10].set_ylim(-5, 5)
plt.show()
