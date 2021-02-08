import numpy as np
import random
import torch

def sample_n_unique(sampling_f, n):
    """Helper function. Given a function `sampling_f` that returns
    comparable objects, sample n such unique objects.
    """
    res = []
    while len(res) < n:
        candidate = sampling_f()
        if candidate not in res:
            res.append(candidate)
    return res

class ReplayBuffer(object):
    def __init__(self, size, frame_history_len, act_dim, continuous_state=True):
        """Experience replay buffer for DQN.

        Args: 
            size (int): Maximum number of transitions to store in the buffer.
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
            self.action   = np.empty([self.size], dtype=np.int32)
            self.reward   = np.empty([self.size], dtype=np.float32)
            self.done     = np.empty([self.size], dtype=np.bool)
        self.obs[self.next_idx] = observation

        ret = self.next_idx
        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)

        return ret

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