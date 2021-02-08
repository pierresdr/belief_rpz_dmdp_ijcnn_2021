import numpy as np
import random
import torch
from utils.DDQNCore import combined_shape
from numpy.core.numeric import greater_equal, equal
from utils.DQNBuffer import sample_n_unique



class PredBuffer:
    def __init__(self, state_dim, obs_dim, size, delay):
        """Experience replay buffer for belief network.

        Args: 
            size (int): Maximum number of transitions to store in the buffer.
            state_dim (int): State space dimension.
            obs_dim (int): Extended state space dimension.
            delay (int): Maximum value of the delay.
        """
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.delay = delay
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.state_buf = np.zeros(combined_shape(size, state_dim), dtype=np.float32)
        self.mask_buf = np.zeros(combined_shape(size, delay), dtype=np.bool)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, state):
        """Store observations in the buffer.
        """
        self.obs_buf[self.ptr] = obs
        self.mask_buf[np.arange(self.ptr-self.delay+1,self.ptr+1)] = greater_equal.outer(np.arange(self.delay-1,-1,-1),np.arange(self.delay))
        self.state_buf[self.ptr] = state[0].reshape(-1)

        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def get_pred_data(self, batch_size):
        """Sample a batch of data for the belief network training.
        """
        idx = np.random.choice(np.arange(self.size), size=batch_size, replace=False) 

        # Mask for states which spread over another episode.
        mask = self.mask_buf[idx]

        # Create the hidden state to be predicted 
        hidden_states = np.stack([np.roll(self.state_buf, -i, axis=0) for i in range(self.delay)], 1)
        hidden_states = hidden_states[idx]

        return self.obs_buf[idx], hidden_states[mask], mask


class ReplayBuffer(object):
    def __init__(self, size, size_pred_buf, frame_history_len, act_dim, state_dim, obs_dim, continuous_state=True):
        """Experience replay buffer for DQN.

        Args: 
            size (int): Maximum number of transitions to store in the buffer.
            size_pred_buf (int): Maximum number of data to store in the prediction/belief buffer.
            act_dim (int): Action space dimension.
            state_dim (int): State space dimension.
            obs_dim (int): Extended state space dimension.
            delay (int): Maximum value of the delay.
            frame_history_len (int): Nomber of consecutive frames in the state for Atari envs.
        """
        self.size = size
        self.frame_history_len = frame_history_len
        self.continuous_state = continuous_state

        self.next_idx      = 0
        self.num_in_buffer = 0
        self.act_dim = act_dim

        self.obs      = None
        self.action   = None
        self.reward   = None
        self.done     = None

        # Recover the maximum delay
        self.max_delay = int((obs_dim - state_dim)/act_dim)

        # Create the prediction/belief buffer
        self.pred_buf = PredBuffer(state_dim, obs_dim, size_pred_buf, self.max_delay)

    def can_sample(self, batch_size):
        """Checks if the buffer contains enough data to sample.
        """
        return batch_size + 1 <= self.num_in_buffer

    def _encode_sample(self, idxes):
        obs_batch      = np.concatenate([self._encode_observation(idx)[None] for idx in idxes], 0)
        act_batch      = self.action[idxes]
        rew_batch      = self.reward[idxes]
        next_obs_batch = np.concatenate([self._encode_observation(idx + 1)[None] for idx in idxes], 0)
        done_mask      = np.array([1.0 if self.done[idx] else 0.0 for idx in idxes], dtype=np.float32)

        return obs_batch, act_batch, rew_batch, next_obs_batch, done_mask


    def sample(self, batch_size):
        """Sample data for DQN training.

        Args:
            batch_size (int): Size of the batch to sample.

        Returns:
            obs_batch (np.array): Batch of observations.
            act_batch (np.array): Corresponding batch of actions.
            rew_batch (np.array): Corresponding batch of rewards.
            next_obs_batch (np.array): Corresponding batch of next observations.
            done_mask (np.array): Corresponding batch of done booleans.
        """
        assert self.can_sample(batch_size)
        idxes = sample_n_unique(lambda: random.randint(0, self.num_in_buffer - 2), batch_size)
        return self._encode_sample(idxes)

    def encode_recent_observation(self):
        """Encodes the last stored observation in the buffer.
        
        Returns:
            observation (np.array): Encoded observation.
        """
        assert self.num_in_buffer > 0
        return self._encode_observation((self.next_idx - 1) % self.size)

    def _encode_observation(self, idx):
        """For Atari environments, the observation is built as consecutive
        frames in order to preserve Markovianity. 
        """
        end_idx   = idx + 1 # make noninclusive
        start_idx = end_idx - self.frame_history_len
        # this checks if we are using low-dimensional observations, such as RAM
        # state, in which case we just directly return the latest RAM.
        if len(self.obs.shape) == 2:
            return self.obs[end_idx-1]
        # if there weren't enough frames ever in the buffer for context
        if start_idx < 0 and self.num_in_buffer != self.size:
            start_idx = 0
        for idx in range(start_idx, end_idx - 1):
            if self.done[idx % self.size]:
                start_idx = idx + 1
        missing_context = self.frame_history_len - (end_idx - start_idx)
        # if zero padding is needed for missing context
        # or we are on the boundry of the buffer
        if start_idx < 0 or missing_context > 0:
            frames = [np.zeros_like(self.obs[0]) for _ in range(missing_context)]
            for idx in range(start_idx, end_idx):
                frames.append(self.obs[idx % self.size])
            return np.concatenate(frames, 0) # c, h, w instead of h, w c
        else:
            # this optimization has potential to saves about 30% compute time \o/
            # c, h, w instead of h, w c
            img_h, img_w = self.obs.shape[2], self.obs.shape[3]
            return self.obs[start_idx:end_idx].reshape(-1, img_h, img_w)

    def store_observation(self, observation):
        """Store a new observation in the buffer.

        Args: 
            observation (np.array): Data to store.

        Returns:
            idx (int): Index in the buffer.
        """
        if self.obs is None:
            if self.continuous_state:
                self.obs = np.empty([self.size] + list(observation.shape), dtype=np.float32)
            else:
                self.obs = np.empty([self.size] + list(observation.shape), dtype=np.int32)
            self.action   = np.empty([self.size],                     dtype=np.int32)
            self.reward   = np.empty([self.size],                     dtype=np.float32)
            self.done     = np.empty([self.size],                     dtype=np.bool)
        self.obs[self.next_idx] = observation

        ret = self.next_idx
        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)

        return ret

    def store_pred(self, obs, info):
        """Store observation in the prediction/belief buffer.

        Args: 
            obs (np.array): The observation to store.
            info (np.array): Further information in case of stochastic delay.
        """
        self.pred_buf.store(obs, info[1])

    def store_effect(self, idx, action, reward, done):
        """Store effects of action.

        Args:
            idx (int): Index in buffer.
            action (int): Action performed.
            reward: (float): Reward collected.
            done (bool): Episode is done.
        """
        self.action[idx] = action
        self.reward[idx] = reward
        self.done[idx]   = done

    def sample_pred_data(self, batch_size):
        return self.pred_buf.get_pred_data(batch_size)
