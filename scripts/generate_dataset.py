import os
import pickle
import numpy as np
from envs.quad_2d import Quad2DEnv
from envs.pointmass import PointMassEnv
import matplotlib.pyplot as plt

# Environment
robot = 'quad2d'        # 'quad2d' or 'pointmass'

# Create Environment
if robot == 'quad2d':
    env = Quad2DEnv(min_rel_thrust=0.75, max_rel_thrust=1.25, 
                    max_rel_thrust_difference=0.01, target=None, max_steps=40,
                    epsilon=0.2, reset_target_reached=False, bonus_reward=False, 
                    reset_out_of_bounds=True, theta_as_sine_cosine=True, num_episodes=100000)
    file_name = 'quad2d_dataset.pkl'
else:
    env = PointMassEnv(target=None, max_steps=20, epsilon=0.2, reset_target_reached=False, bonus_reward=False, 
                reset_out_of_bounds=True, theta_as_sine_cosine=True, num_episodes=100000)
    file_name = 'pointmass_dataset.pkl'
horizon = 16
# env.seed(0)

dataset = env.make_dataset()

if not os.path.exists('data'):
    # If not, create it
    os.makedirs('data')
# Save file
with open('data/' + file_name, 'wb') as f:
    pickle.dump(dataset, f)

# Load file
with open('data/quad2d_dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

theta = np.arctan2(dataset['observations'][:, 4], dataset['observations'][:, 5])

# print('x min: %f, x max: %f, x mean: %f' % np.mean(dataset['observations'][:, 0]))
print('x min: %f, x max: %f, x mean: %f' % (np.min(dataset['observations'][:, 0]), np.max(dataset['observations'][:, 0]), np.mean(dataset['observations'][:, 0])))
print('y min: %f, y max: %f, y mean: %f' % (np.min(dataset['observations'][:, 2]), np.max(dataset['observations'][:, 2]), np.mean(dataset['observations'][:, 2])))
print('theta min: %f, theta max: %f, theta mean: %f' % (np.min(theta), np.max(theta), np.mean(theta)))
print('dx min: %f, dx max: %f, dx mean: %f' % (np.min(dataset['observations'][:, 1]), np.max(dataset['observations'][:, 1]), np.mean(dataset['observations'][:, 1])))
print('dy min: %f, dy max: %f, dy mean: %f' % (np.min(dataset['observations'][:, 3]), np.max(dataset['observations'][:, 3]), np.mean(dataset['observations'][:, 3])))
print('dtheta min: %f, dtheta max: %f, dtheta mean: %f' % (np.min(dataset['observations'][:, 6]), np.max(dataset['observations'][:, 6]), np.mean(dataset['observations'][:, 6])))

# Visualize dataset
fig, ax = plt.subplots(3, 1, figsize=(10, 10))
ax[0].scatter(dataset['observations'][:, 0], dataset['observations'][:, 2])
ax[1].scatter(dataset['observations'][:, 1], dataset['observations'][:, 3])
ax[2].scatter(dataset['actions'][:, 0], dataset['actions'][:, 1])
plt.show()

# Visualize 100 position trajectories
fig, ax = plt.subplots(10, 10, figsize=(10, 10))
index_start = 0
indices_end = [i for i, x in enumerate(dataset['terminals']) if x == 1]
i = 0
which_traj = 0
while i < 100:
    index_end = indices_end[which_traj]
    if index_end - index_start < horizon:
        which_traj += 1
        index_start = index_end + 1
        continue

    observations = dataset['observations'][index_start:index_start + horizon]
    index_start = index_end + 1

    # Plot observations
    ax[i // 10, i % 10].plot(observations[:, 0], observations[:, 2])

    i += 1

plt.show()

# Visualize all states of 10 trajectories
n_plot = 10
fig, ax = plt.subplots(n_plot, 10, figsize=(10, 10))
labels = ['x', 'y', 'theta', 'dx', 'dy', 'dtheta', 'T1', 'T2', 'reward']

index_start = 0
indices_end = [i for i, x in enumerate(dataset['terminals']) if x == 1]
i = 0
which_traj = 0
while i < n_plot:
    index_end = indices_end[which_traj]
    if index_end - index_start < horizon:
        which_traj += 1
        index_start = index_end + 1
        continue

    # observations = dataset['observations'][index_start:index_end + 1]
    # actions = dataset['actions'][index_start:index_end + 1]
    # rewards = dataset['rewards'][index_start:index_end + 1]
    observations = dataset['observations'][index_start:index_start + horizon]
    actions = dataset['actions'][index_start:index_start + horizon]
    rewards = dataset['rewards'][index_start:index_start + horizon]
    index_start = index_end + 1

    # Plot observations
    ax[i, 0].plot(observations[:, 0])
    ax[i, 1].plot(observations[:, 2])
    ax[i, 2].plot(np.arctan2(observations[:, 4], observations[:, 5]))
    ax[i, 3].plot(observations[:, 1])
    ax[i, 4].plot(observations[:, 3])
    ax[i, 5].plot(observations[:, 6])
    ax[i, 6].plot(actions[:, 0])
    ax[i, 7].plot(actions[:, 1])
    ax[i, 8].plot(rewards)
    ax[i, 9].plot(observations[:, 0], observations[:, 2])
    ax[i, 9].plot(observations[0, 0], observations[0, 2], 'go')
    ax[i, 9].plot(observations[0, 7], observations[0, 8], 'ro')
    for _ in range(9):
        ax[i, _].set_ylabel(labels[_])
    i += 1
    

plt.show()