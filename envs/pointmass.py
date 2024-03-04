import gym
import pickle
import numpy as np
from typing import Optional
from numpy import cos, pi, sin
from gym import core, spaces
from gym.error import DependencyNotInstalled
import math
import random


class PointMassEnv(core.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}

    dt = 0.1

    MASS = 1        # [kg]  Mass of the quadrotor

    MAX_X = 5       # [m]   Maximum and minimum values of the x position
    MAX_Y = 5       # [m]   Maximum and minimum values of the y position
    MAX_VEL_X = 5   # [m/s] Maximum velocity in x direction
    MAX_VEL_Y = 5   # [m/s] Maximum velocity in y direction

    MAX_ACC = 5        # Bounds on the tangential velocities of the wheels

    acc_noise_max = 0.0

    SCREEN_DIM = 500

    #: use dynamics equations from the nips paper or the book
    # state = [x, dx, y, dy]
    # action = [ddx, ddy]
    # observation = [x, dx, y, dy, x_des, y_des]

    def __init__(self, target=None, max_steps=100, num_episodes=1000, epsilon=0.2, reset_target_reached=False, 
                 reset_out_of_bounds=False, bonus_reward=False, initial_state=None, theta_as_sine_cosine=True, 
                 n_moving_obstacles=0, n_static_obstacles=0, reward='squared_distance'):
        self.screen = None
        self.clock = None
        self.isopen = True
        self.name = "PointMassEnv"
        self.theta_as_sine_cosine = theta_as_sine_cosine
        
        # Observation space bounds
        obs_high = np.array(
            [self.MAX_X, self.MAX_VEL_X, self.MAX_Y, self.MAX_VEL_Y, self.MAX_X, self.MAX_Y], dtype=np.float32
        )
        obs_low = -obs_high

        # State space bounds
        state_high = np.array(
            [self.MAX_X, self.MAX_VEL_X, self.MAX_Y, self.MAX_VEL_Y], dtype=np.float32
        )
        state_low = -state_high

        # Action space bounds
        action_low = np.array([-self.MAX_ACC, -self.MAX_ACC], dtype=np.float32)
        action_high = np.array([self.MAX_ACC, self.MAX_ACC], dtype=np.float32)

        self.state_space = spaces.Box(low=state_low, high=state_high, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)
        self.state = None

        if target:
            self.target = target
            self.random_target = False
        else:
            is_valid = False
            while not is_valid:
                self.target = self._sample_target()
                is_valid = self._check_target(self.target)
            self.random_target = True
        self.max_steps = max_steps
        self._max_episode_steps = max_steps      # For compatibility with diffuser
        self.num_episodes = num_episodes
        self.timestep = 0
        self.epsilon = epsilon
        self.reset_target_reached = reset_target_reached
        self.reset_out_of_bounds = reset_out_of_bounds
        self.reward = reward
        self.bonus_reward = bonus_reward
        self.initial_state = initial_state
        self.n_moving_obstacles = n_moving_obstacles
        self.n_static_obstacles = n_static_obstacles
        self.n_obstacles = n_moving_obstacles + n_static_obstacles

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None
    ):
        super().reset(seed=seed)
        
        if self.n_obstacles > 0:
            self._generate_obstacles()

        is_valid = False
        if not self.random_target:
            self.state = self.initial_state
        else:
            while not is_valid:
                self.state = self.state_space.sample()
                # self.state = self.np_random.uniform(low=-np.pi, high=np.pi, size=(4,)).astype(np.float32)
                is_valid = self._check_initial_pos(self.state) and self._check_initial_vel(self.state)
        self.timestep = 0
        if self.random_target:
            is_valid = False
            while not is_valid:
                self.target = self._sample_target()
                is_valid = self._check_target(self.target)
        self.target_reached = False

        if not return_info:
            return self._get_ob()
        else:
            return self._get_ob(), {}

    def _sample_target(self, seed=None):
        # Random x/y target position in [-self.MAX_X/Y, self.MAX_X/Y]
        x = 2 * self.MAX_X * (self.np_random.rand() - 0.5)
        y = 2 * self.MAX_Y * (self.np_random.rand() - 0.5)

        return (x, y)

    def _check_target(self, target):
        # Check if the target is at least epsilon away from the initial position
        if self.state is not None:
            p = self._get_coordinates(self.state)
            distance = np.linalg.norm(np.array(p) - np.array(target))
            if distance <= self.epsilon:
                return False
        return True

    def _check_initial_pos(self, state):
        if self.target is not None:
            p = self._get_coordinates(state)
            distance = np.linalg.norm(np.array(p) - np.array(self.target))
            if distance <= self.epsilon:
                return False
        return True
    
    def _check_initial_vel(self, state):
        # Ensure that the agent can avoid going out of bounds with maximum acceleration
        if (state[1] > 0) & (state[0] + 0.5 * state[1] ** 2 / self.MAX_ACC > self.MAX_X) or \
            (state[1] < 0) & (state[0] - 0.5 * state[1] ** 2 / self.MAX_ACC < -self.MAX_X) or \
            (state[3] > 0) & (state[2] + 0.5 * state[3] ** 2 / self.MAX_ACC > self.MAX_Y) or \
            (state[3] < 0) & (state[2] - 0.5 * state[3] ** 2 / self.MAX_ACC < -self.MAX_Y):
            return False
        return True     
    
    def _generate_obstacles(self):
        self.obstacles = []

        for _ in range(self.n_moving_obstacles):
            d = 0.2 + self.np_random.rand() * (1 - 0.2)         # Create a square obstacle with a random initial position and velocity

            x = (2 * self.MAX_X - d) * (self.np_random.rand() - 0.5)    # Random x/y initial position in [-self.MAX_X/Y + d/2, self.MAX_X/Y - d/2]
            y = (2 * self.MAX_Y - d) * (self.np_random.rand() - 0.5)

            vx = 2 * self.MAX_VEL_X * (self.np_random.rand() - 0.5)     # Random x/y velocity in [-self.MAX_VEL_X/Y, self.MAX_VEL_X/Y]
            vy = 2 * self.MAX_VEL_Y * (self.np_random.rand() - 0.5)

            self.obstacles.append({'x': x, 'y': y, 'vx': vx, 'vy': vy, 'd': d})

        for _ in range(self.n_static_obstacles):
            d = 0.2 + self.np_random.rand() * (1 - 0.2)

            x = (2 * self.MAX_X - d) * (self.np_random.rand() - 0.5)
            y = (2 * self.MAX_Y - d) * (self.np_random.rand() - 0.5)

            self.obstacles.append({'x': x, 'y': y, 'vx': 0, 'vy': 0, 'd': d})

    def step(self, a):
        s = self.state
        assert s is not None, "Call reset before using AcrobotEnv object."
        acc = a

        #add torque limit
        acc = np.clip(acc, a_min=-self.MAX_ACC, a_max=self.MAX_ACC)

        # Add noise to the force action
        if self.acc_noise_max > 0:
            acc += self.np_random.uniform(
                -self.acc_noise_max, self.acc_noise_max
            )

        # Now, augment the state with our force action so it can be passed to _dsdt
        s_augmented = np.append(s, acc)

        ns = rk4(self._dsdt, s_augmented, [0, self.dt])

        ns[0] = bound(ns[0], -self.MAX_X, self.MAX_X)
        ns[1] = bound(ns[1], -self.MAX_VEL_X, self.MAX_VEL_X)
        ns[2] = bound(ns[2], -self.MAX_Y, self.MAX_Y)
        ns[3] = bound(ns[3], -self.MAX_VEL_Y, self.MAX_VEL_Y)

        self.prev_state = self.state
        self.state = ns

        # Move obstacles
        if self.n_moving_obstacles > 0:
            for i in range(self.n_moving_obstacles):
                obstacle = self.obstacles[i]
                possible_new_x = obstacle['x'] + obstacle['vx'] * self.dt
                possible_new_y = obstacle['y'] + obstacle['vy'] * self.dt
                if possible_new_x <= -self.MAX_X + obstacle['d'] / 2 or possible_new_x >= self.MAX_X - obstacle['d'] / 2:
                    obstacle['vx'] *= -1
                if possible_new_y <= -self.MAX_Y + obstacle['d'] / 2 or possible_new_y >= self.MAX_Y - obstacle['d'] / 2:
                    obstacle['vy'] *= -1

                obstacle['x'] += obstacle['vx'] * self.dt
                obstacle['y'] += obstacle['vy'] * self.dt

        done = self._is_done()
        reward = self._get_reward()

        self.timestep += 1
        return (self._get_ob(), reward, done, self.target_reached)
    
    def sample_action(self):
        a = self.action_space.sample()
        return a

    def inverse_dynamics(self, s_next):
        m = self.MASS

        # Unpack the state
        s = self.state
        dx, dy = s[1], s[3]
        dx_next, dy_next = s_next[1], s_next[3]

        # Get accelerations
        ddx = (dx_next - dx) / self.dt
        ddy = (dy_next - dy) / self.dt

        # Solve for the forces
        a1 = ddx / m
        a2 = ddy / m

        return (a1, a2)
    
    def get_obstacles(self):
        return self.obstacles if self.n_obstacles > 0 else None
    
    def predict_obstacles(self, horizon):
        '''
            Predict the future positions of the moving obstacles for the given horizon.
            return {0: [{'x': x, 'y': y, 'vx': vx, 'vy': vy, 'd': d}, ...], 1: [...], ..., horizon: [...]}
        '''

        if self.n_obstacles == 0:
            return None

        predictions = {}
        for i in range(horizon):
            predictions[i] = []
            for j in range(self.n_obstacles):       
                if i == 0:
                    predictions[i].append({'x': self.obstacles[j]['x'], 'y': self.obstacles[j]['y'], 'vx': self.obstacles[j]['vx'], 'vy': self.obstacles[j]['vy'], 'd': self.obstacles[j]['d']})
                    continue
                
                x = predictions[i - 1][j]['x']
                y = predictions[i - 1][j]['y']

                vx = predictions[i - 1][j]['vx']
                vy = predictions[i - 1][j]['vy']
                
                possible_x = x + vx * self.dt
                possible_y = y + vy * self.dt

                if possible_x <= -self.MAX_X + self.obstacles[j]['d'] / 2 or possible_x >= self.MAX_X - self.obstacles[j]['d'] / 2:
                    vx *= -1
                if possible_y <= -self.MAX_Y + self.obstacles[j]['d'] / 2 or possible_y >= self.MAX_Y - self.obstacles[j]['d'] / 2:
                    vy *= -1

                predictions[i].append({'x': x + vx * self.dt, 'y': y + vy * self.dt, 'vx': vx, 'vy': vy, 'd': self.obstacles[j]['d']})

        return predictions

    def make_dataset(self):
        dataset = {}
        keys = ['observations', 'actions', 'rewards', 'terminals', 'timeouts']

        dataset = {key: [] for key in keys}
        
        episode = 0
        while episode < self.num_episodes:
          
            state = self.reset(seed=episode)

            dataset_episode = {key: [] for key in keys}

            for step in range(self.max_steps):
                action = self.sample_action()
                next_state, reward, done, target_reached = self.step(action)
                
                dataset_episode['observations'].append(state)
                dataset_episode['actions'].append(action)
                dataset_episode['rewards'].append([reward])
                dataset_episode['terminals'].append([done])
                dataset_episode['timeouts'].append([0 if step < self.max_steps - 1 else 1])

                state = next_state
                if done:
                    break
            
            if len(dataset_episode['rewards']) < 16:
                continue
            
            episode += 1

            if episode % 2000 == 0:
                print("Generated training episode %d of %d" % (episode, self.num_episodes))

            for key in keys:
                dataset[key].extend(dataset_episode[key])

        # Convert lists to numpy arrays
        for key in dataset.keys():
            dataset[key] = np.array(dataset[key])

        print("Dataset shape: ", dataset['observations'].shape)
        # print("Observation limits: ", np.min(dataset['observations'], axis=0), np.max(dataset['observations'], axis=0))
        # print("Action limits: ", np.min(dataset['actions'], axis=0), np.max(dataset['actions'], axis=0))

        return dataset
    
    def get_dataset(self):
        path = 'data/pointmass_dataset.pkl'
        # Check if there is a file at the specified path
        try:
            with open(path, 'rb') as f:
                dataset = pickle.load(f)
        except FileNotFoundError:
            dataset = self.make_dataset()
            with open(path, 'wb') as f:
                pickle.dump(dataset, f)

        return dataset

    def _get_ob(self):
        s = self.state
        assert s is not None, "Call reset before using PointmassEnv object."
        return np.array(
            [s[0], s[1], s[2], s[3], self.target[0], self.target[1]], dtype=np.float32
        )

    def _get_coordinates(self, state):
        p = [state[0], state[2]]
        return p

    def _get_distance_to_target(self):
        p = self._get_coordinates(self.state)
        distance = np.linalg.norm(np.array(p) - np.array(self.target))
        return distance

    def _get_reward(self):
        distance = self._get_distance_to_target()
        reward = -distance ** 2 if self.reward == 'squared_distance' else -distance
        # reward = -distance
        if distance <= self.epsilon:
            if self.bonus_reward:
                reward += 1000.0
        return reward

    def _is_done(self):   
        if self.reset_target_reached:
            distance = self._get_distance_to_target()
            if distance <= self.epsilon:
                self.target_reached = True
                return True
        
        if self.reset_out_of_bounds:
            if self.state[0] <= -self.MAX_X + 1e-4 or self.state[0] >= self.MAX_X - 1e-4 or \
                self.state[1] <= -self.MAX_VEL_X + 1e-4 or self.state[1] >= self.MAX_VEL_X - 1e-4 or \
                self.state[2] <= -self.MAX_Y + 1e-4 or self.state[2] >= self.MAX_Y - 1e-4 or \
                self.state[3] <= -self.MAX_VEL_Y + 1e-4 or self.state[3] >= self.MAX_VEL_Y - 1e-4:
                # print("Out of bounds!!!")
                return True
            
        if self.n_obstacles > 0:
            for i in range(self.n_obstacles):
                obstacle = self.obstacles[i]
                if self.state[0] >= obstacle['x'] - obstacle['d'] / 2 and \
                    self.state[0] <= obstacle['x'] + obstacle['d'] / 2 and \
                    self.state[2] >= obstacle['y'] - obstacle['d'] / 2 and \
                    self.state[2] <= obstacle['y'] + obstacle['d'] / 2:
                    # print("Collision!!!")
                    self.target_reached = -1
                    return True

        if self.timestep == self.max_steps - 1:
            # print("Timeout!!!")
            return True
        
        return False    

    def _dsdt(self, s_augmented):
        # Get parameters
        m = self.MASS

        # Unpack the state and action
        a1 = s_augmented[-2]
        a2 = s_augmented[-1]
        s = s_augmented[:-1]
        _, dx, _, dy = s[0], s[1], s[2], s[3]

        ddx = 1 / m * a1
        ddy = 1 / m * a2

        return (dx, ddx, dy, ddy, 0.0, 0.0)
    
    def render(self, trajectories_to_plot=None):
        try:
            import pygame
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.SCREEN_DIM, self.SCREEN_DIM))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((self.SCREEN_DIM, self.SCREEN_DIM))
        self.surf.fill((255, 255, 255))
        s = self.state

        bound = self.MAX_X + 0.2
        scale = self.SCREEN_DIM / (bound * 2)
        offset = self.SCREEN_DIM / 2

        if s is None:
            return None

        # Plot the point mass as a dot
        pygame.draw.circle(self.surf, (64, 64, 255), (int(scale * s[0] + offset), int(scale * s[2] + offset)), int(scale * 0.1))

        # Plot the target position as a dot
        pygame.draw.circle(self.surf, (0, 0, 0), (int(scale * self.target[0] + offset), int(scale * self.target[1] + offset)), int(scale * 0.1))

        # Plot the obstacles as squares
        for obstacle in self.obstacles:
            left, top = int(scale * (obstacle['x'] - obstacle['d'] / 2) + offset), int(scale * (obstacle['y'] - obstacle['d'] / 2) + offset)
            pygame.draw.rect(self.surf, (64, 255, 64), pygame.Rect(left, top, scale * obstacle['d'], scale * obstacle['d']))

        # Plot the trajectories
        if trajectories_to_plot is not None:
            for _ in range(trajectories_to_plot.shape[0]):
                traj = trajectories_to_plot[_]
                traj = scale * traj + offset
                pygame.draw.lines(self.surf, (64, 64, 255), False, list(map(tuple, traj.tolist())), 2)
        
        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        
        pygame.event.pump()
        self.clock.tick(self.metadata["render_fps"])
        pygame.display.flip()

        return self.isopen

def wrap(x, m, M):
    """Wraps ``x`` so m <= x <= M; but unlike ``bound()`` which
    truncates, ``wrap()`` wraps x around the coordinate system defined by m,M.\n
    For example, m = -180, M = 180 (degrees), x = 360 --> returns 0.

    Args:
        x: a scalar
        m: minimum possible value in range
        M: maximum possible value in range

    Returns:
        x: a scalar, wrapped
    """
    diff = M - m
    while x > M:
        x = x - diff
    while x < m:
        x = x + diff
    return x

def bound(x, m, M=None):
    """Either have m as scalar, so bound(x,m,M) which returns m <= x <= M *OR*
    have m as length 2 vector, bound(x,m, <IGNORED>) returns m[0] <= x <= m[1].

    Args:
        x: scalar
        m: The lower bound
        M: The upper bound

    Returns:
        x: scalar, bound between min (m) and Max (M)
    """
    if M is None:
        M = m[1]
        m = m[0]
    # bound x between min (m) and Max (M)
    return min(max(x, m), M)


def rk4(derivs, y0, t):
    """
    Integrate 1-D or N-D system of ODEs using 4-th order Runge-Kutta.

    Example for 2D system:

        >>> def derivs(x):
        ...     d1 =  x[0] + 2*x[1]
        ...     d2 =  -3*x[0] + 4*x[1]
        ...     return d1, d2

        >>> dt = 0.0005
        >>> t = np.arange(0.0, 2.0, dt)
        >>> y0 = (1,2)
        >>> yout = rk4(derivs, y0, t)

    Args:
        derivs: the derivative of the system and has the signature ``dy = derivs(yi)``
        y0: initial state vector
        t: sample times

    Returns:
        yout: Runge-Kutta approximation of the ODE
    """

    try:
        Ny = len(y0)
    except TypeError:
        yout = np.zeros((len(t),), np.float_)
    else:
        yout = np.zeros((len(t), Ny), np.float_)

    yout[0] = y0[:Ny]

    for i in np.arange(len(t) - 1):

        this = t[i]
        dt = t[i + 1] - this
        dt2 = dt / 2.0
        y0 = yout[i]

        k1 = np.asarray(derivs(y0))
        k2 = np.asarray(derivs(y0 + dt2 * k1))
        k3 = np.asarray(derivs(y0 + dt2 * k2))
        k4 = np.asarray(derivs(y0 + dt * k3))
        yout[i + 1] = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    # We only care about the final timestep and we cleave off action value which will be zero
    return yout[-1][:4]