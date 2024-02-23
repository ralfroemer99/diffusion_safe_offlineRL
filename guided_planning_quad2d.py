import numpy as np
import os
import torch
import argparse
from utils.logger import Logger
from envs.quad_2d import Quad2DEnv
from utils.guided_policy import ValueGuide, CbfGuide
from diffuser.models.temporal import TemporalUnet, ValueFunction
from utils.models import GaussianDiffusion as ProbDiffusion
from utils.models import CbfDiffusion
from utils.guided_policy import n_step_doubleguided_p_sample
from utils.guided_policy import n_step_guided_p_sample
import diffuser.utils as utils
import einops
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib as mpl
mpl.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
import copy
import time

parser = argparse.ArgumentParser()

parser.add_argument('--num-targets', type=int, default=1,
                    help='Number of targets.')
parser.add_argument('--num-steps', type=int, default=20,
                    help='Number of steps.')
parser.add_argument('--batch-size', type=int, default=64,
                    help='Batch size (number of generated planning).')
parser.add_argument('--state-dim', type=int, default=9,
                    help='Number of episodes.')
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
parser.add_argument('--target-location', type=float, default=(1.5, 1.5),
                    help='Location of the obstacle.')
parser.add_argument('--sequence-length', type=int, default=16,
                    help='Sequence length.')
parser.add_argument('--diffusion-steps', type=int, default=20,
                    help='Number of diffusion steps.')
parser.add_argument('--use-attention', default=True,
                    help='Use attention layer in temporal U-net.')
parser.add_argument('--value-only', default=True,
                    help='Use only value function model for guided planning '
                         'or the combination of value and cbf classifier.')
parser.add_argument('--observation-dim-w', type=int, default=84,
                    help='Width of the input measurements (RGB images).')
parser.add_argument('--observation-dim-h', type=int, default=84,
                    help='Height of the input measurements (RGB images).')
parser.add_argument('--save-traj', default=False,
                    help='Generate training or testing dataset.')
parser.add_argument('--trajectory-dataset', type=str, default='reacher_trajectories.pkl',
                    help='Training dataset.')
parser.add_argument('--seed', type=int, default=1,
                    help='Random seed (default: 1).')
parser.add_argument('--experiment', type=str, default='Quad2D',
                    help='Experiment.')
parser.add_argument('--model-type', type=str, default='ProbDiffusion',
                    help='Model type.')
parser.add_argument('--num-classes', type=int, default=2,
                    help='Number of classes (safe and unsafe).')


args = parser.parse_args()


def save_frames_as_mp4(frames, filename='reacher_animation.mp4', folder='results/', idx=0):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)

    save_dir = folder + 'videos/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    FFwriter = animation.FFMpegWriter(fps=10)
    anim.save(save_dir + str(idx) + '_' + filename, writer=FFwriter)

def save_frames_as_png(frames, filename='snapshots', folder='results/'):
    save_dir = folder + 'videos/snapshots'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i in range(len(frames)):
        plt.imshow(frames[i])
        plt.savefig(save_dir + '/' + filename + '_' + str(i) + '.png')

save_traj = args.save_traj
if save_traj:
    data_file_name = args.trajectory_dataset
max_steps = args.num_steps
seed = args.seed

value_only = False
env = Quad2DEnv(min_rel_thrust=args.min_rel_thrust, max_rel_thrust=args.max_rel_thrust, 
                max_rel_thrust_difference=args.max_rel_thrust_difference, target=None, max_steps=max_steps,
                epsilon=args.epsilon, reset_target_reached=False, bonus_reward=False)

state_dim = args.state_dim
action_dim = args.action_dim

directory = os.path.dirname(os.path.abspath(__file__))
folder = os.path.join(directory + '/data/')
logger = Logger(folder)

# Set seeds
env.action_space.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

horizon = args.sequence_length  # planning horizon
observation_dim = state_dim
transition_dim = observation_dim + action_dim
nr_diffusion_steps = args.diffusion_steps
use_attention = args.use_attention

exp = args.experiment
mtype = args.model_type
save_pth_dir = directory + '/results/' + str(exp) + '/' + str(mtype)
if not os.path.exists(save_pth_dir):
    os.makedirs(save_pth_dir)

model = TemporalUnet(horizon, transition_dim, dim=32, dim_mults=(1, 2, 4, 8), attention=use_attention)

diffusion = ProbDiffusion(model, horizon, observation_dim, action_dim, n_timesteps=nr_diffusion_steps,
                        loss_type='l2', clip_denoised=False, predict_epsilon=False,
                        action_weight=1.0, loss_discount=1.0, loss_weights=None)

ema_diffusion = copy.deepcopy(diffusion).cuda()
checkpoint = torch.load(save_pth_dir + '/ProbDiff_Model_best.pth')
diffusion.model.load_state_dict(checkpoint['model'])
ema_diffusion.model.load_state_dict(checkpoint['ema_model'])

value_model = ValueFunction(horizon, transition_dim, dim=32, dim_mults=(1, 2, 4, 8), attention=use_attention, final_sigmoid=False)
value_diffusion = ValueGuide(value_model)
ema_value_diffusion = copy.deepcopy(value_diffusion).cuda()
checkpoint = torch.load(save_pth_dir + '/ValueDiff_Model_best.pth')
value_diffusion.model.load_state_dict(checkpoint['model'])
ema_value_diffusion.model.load_state_dict(checkpoint['ema_model'])

batch_size = args.batch_size
x_start = np.zeros((batch_size, horizon, transition_dim))
nr_targets = args.num_targets

cond = {}
frames = []
save_gif = False
successes = 0
failures = 0
safety_violations = 0
total_rews = []
nr_steps = []

diffusion.eval()
value_diffusion.eval()
ema_diffusion.eval()
ema_value_diffusion.eval()

trajectories_all = []
start = time.time()
for episode in range(nr_targets):
    state = env.reset()

    # frames.append(env.render(mode='rgb_array'))
    print('Episode: ', episode)
    total_rew = 0
    trajectory = np.zeros((max_steps, state_dim + action_dim + 2))
    for step in range(max_steps):
        print('Step: ', step)
        cond = {0: torch.from_numpy(state).cuda()}
        cond = utils.to_torch(cond, dtype=torch.float32, device='cuda:0')
        cond = utils.apply_dict(
               einops.repeat,
               cond,
               'd -> repeat d', repeat=batch_size,
               )
        samples = ema_diffusion(cond, guide=ema_value_diffusion,
                            sample_fn=n_step_guided_p_sample)
        action = samples.trajectories[0, 0, 0:action_dim].cpu().numpy()
        value = samples.values.cpu().numpy()

        # Plot open-loop plan
        if step == 0:
            fig, ax = plt.subplots(1, 9)
            labels = ['x', 'y', 'theta', 'dx', 'dy', 'dtheta', 'tau1', 'tau2']
            ax[0].plot(samples.trajectories[0, :, action_dim].cpu().numpy())
            ax[1].plot(samples.trajectories[0, :, action_dim + 2].cpu().numpy())
            ax[2].plot(np.arctan2(samples.trajectories[0, :, action_dim + 4].cpu().numpy(), 
                                samples.trajectories[0, :, action_dim + 5].cpu().numpy()))
            ax[3].plot(samples.trajectories[0, :, action_dim + 1].cpu().numpy())
            ax[4].plot(samples.trajectories[0, :, action_dim + 3].cpu().numpy())
            ax[5].plot(samples.trajectories[0, :, action_dim + 6].cpu().numpy())
            ax[6].plot(samples.trajectories[0, :, 0].cpu().numpy())
            ax[7].plot(samples.trajectories[0, :, 1].cpu().numpy())
            ax[8].plot(samples.trajectories[0, :, action_dim].cpu().numpy(), 
                       samples.trajectories[0, :, action_dim + 2].cpu().numpy())
            ax[8].plot(samples.trajectories[0, 0, action_dim].cpu().numpy(), 
                       samples.trajectories[0, 0, action_dim + 2].cpu().numpy(), 'go')
            ax[8].plot(samples.trajectories[0, 0, action_dim + 7].cpu().numpy(),
                          samples.trajectories[0, 0, action_dim + 8].cpu().numpy(), 'ro')
            for i in range(8):
                ax[i].set_ylabel(labels[i])
            plt.show()

        # print("value", value[0])
        # print("action", action)
        next_state, reward, done, target_reached = env.step(action)
        # print("reward", reward)
        total_rew += reward
        if save_gif:
            frames.append(env.render(mode='rgb_array'))
        if save_traj:
            logger.obslog((state, action, reward, next_state, done))
        # env.render()
            
        trajectory[step, :state_dim] = state
        trajectory[step, state_dim:state_dim + action_dim] = action
        trajectory[step, -2] = reward
        trajectory[step, -1] = done
        state = next_state

        if done:
            if target_reached:
                successes += 1
            else:
                failures += 1
            print("total reward", total_rew)
            print("Successful episodes:", successes)
            print("Unsuccessful episodes:", failures)
            total_rews.append(total_rew)
            nr_steps.append(step)
            # save_frames_as_mp4(frames, folder=save_pth_dir + '/', idx=episode)
            # save_frames_as_png(frames, folder=save_pth_dir + '/')
            # frames = []
            trajectories_all.append(trajectory)
            break

end = time.time()
print("Total time:", end - start)

print("Successful episodes:", successes)
print("Unsuccessful episodes:", failures)
print("Total number of constrain violations:", safety_violations)
total_rews = np.array(total_rews)
print("Average reward:", total_rews.mean())
print("Std reward:", total_rews.std())
nr_steps = np.array(nr_steps)
print("Average nr steps:", nr_steps.mean())
print("Std nr steps:", nr_steps.std())

if save_traj:
    logger.save_obslog(filename=data_file_name)

# Plot the first ten trajectories
fig, ax = plt.subplots(nr_targets, 11)
ax = np.reshape(ax, (1, -1))
labels = ['x', 'y', 'theta', 'dx', 'dy', 'dtheta', 'T1', 'T2', 'reward', 'done']
for i in range(nr_targets):
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