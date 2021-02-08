
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import gym
from gym import spaces
from gym import wrappers
from utils.monitor import Monitor
import random


class DQN(nn.Module):
    def __init__(self, obs_dim, num_actions, q_fun_neurons, dueling=False, belief_module=None):
        super(DQN, self).__init__()
        self.num_actions = num_actions
        self.obs_dim = obs_dim

        self.fc1_val = nn.Linear(in_features=obs_dim, out_features=q_fun_neurons)
        self.fc2_val = nn.Linear(in_features=q_fun_neurons, out_features=q_fun_neurons)
        self.fc3_val = nn.Linear(in_features=q_fun_neurons, out_features=1)

        self.dueling = dueling
        if dueling:
            self.fc1_adv = nn.Linear(in_features=obs_dim, out_features=q_fun_neurons)
            self.fc2_adv = nn.Linear(in_features=q_fun_neurons, out_features=q_fun_neurons)
            self.fc3_adv = nn.Linear(in_features=q_fun_neurons, out_features=num_actions)

        self.relu = nn.ReLU()

    def forward(self, x):
        val = self.relu(self.fc1_val(x))
        val = self.relu(self.fc2_val(val))
        val = self.fc3_val(val).expand(x.size(0), self.num_actions)
        
        if self.dueling:
            adv = self.relu(self.fc1_adv(x))
            adv = self.relu(self.fc2_adv(adv))
            adv = self.fc3_adv(adv)
            val = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.num_actions)

        return val




class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """ Linear decay of the threshold for epsilon greedy exploration.

        Args:
            schedule_timesteps (int): Number of timesteps for linear decay.
            initial_p (float): Initial threshold.
            final_p (float): Final threshold.
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p            = final_p
        self.initial_p          = initial_p

    def value(self, t):
        """ See Schedule.value"""
        fraction  = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


def set_global_seeds(i):
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(i) 
    np.random.seed(i)
    random.seed(i)

def wrap_env(env, seed, double_dqn, dueling_dqn, clip_reward=False, atari=False):
    """ Wrap the environment in a Monitor wrapper.
    If the environment is an Atari environment, Deepmind's wrapper is added.
    """
    set_global_seeds(seed)
    env.seed(seed)

    expt_dir = 'tmp'
    env = Monitor(env, expt_dir, force=True, video_callable=False)
    if atari:
        env = wrap_deepmind_ram(env, clip_reward=clip_reward)
    return env

def get_wrapper_by_name(env, classname):
    currentenv = env
    while True:
        if classname in currentenv.__class__.__name__:
            return currentenv
        elif isinstance(env, gym.Wrapper):
            currentenv = currentenv.env
        else:
            raise ValueError("Couldn't find wrapper named %s"%classname)


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env=None, noop_max=30):
        """ Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        super(NoopResetEnv, self).__init__(env)
        self.noop_max = noop_max
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def _reset(self):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset()
        noops = np.random.randint(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, _, _ = self.env.step(0)
        return obs

class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        """ Take action on reset for environments that are fixed until firing."""
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self):
        self.env.reset()
        obs, _, _, _ = self.env.step(1)
        obs, _, _, _ = self.env.step(2)
        return obs

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env=None):
        """ Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        super(EpisodicLifeEnv, self).__init__(env)
        self.lives = 0
        self.was_real_done  = True
        self.was_real_reset = False

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert somtimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset()
            self.was_real_reset = True
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
            self.was_real_reset = False
        self.lives = self.env.unwrapped.ale.lives()
        return obs

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4, delay=0):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=2)
        self._skip       = skip
        self._act_buffer = deque(maxlen=delay)

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs[0])
            total_reward += np.sum(reward)
            if done:
                break

        self._act_buffer.append(action)

        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        output = (max_frame, np.stack(self._act_buffer))

        return output, total_reward, done, info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        self._act_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs[0])
        for a in obs[1]:
            self._act_buffer.append(a)
        return obs

class ProcessFrame84(gym.Wrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1))

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return _process_frame84(obs), reward, done, info

    def reset(self):
        return _process_frame84(self.env.reset())

class ClippedRewardsWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, np.sign(reward), done, info

def wrap_deepmind_ram(env, clip_reward=False):
    env = EpisodicLifeEnv(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4, delay=env.delay.current)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    if clip_reward: 
        env = ClippedRewardsWrapper(env)
    return env
