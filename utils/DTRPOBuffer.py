import torch
import numpy as np
from utils import DTRPOCore as Core
from numpy.core.numeric import greater_equal, equal


# -------------------------------------------------------------------------------------------------------------
# -----------------------------       Deterministic buffer      -----------------------------------------------
# -------------------------------------------------------------------------------------------------------------


class PredBuffer:
    def __init__(self, state_dim, obs_dim, size, batch_size, delay):
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.delay = delay
        self.batch_size = batch_size
        self.obs_buf = np.zeros(Core.combined_shape(size, obs_dim), dtype=np.float32)
        self.state_buf = np.zeros(Core.combined_shape(size, state_dim), dtype=np.float32)
        self.mask_buf = np.zeros(Core.combined_shape(size, delay), dtype=np.bool)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, state):
        self.obs_buf[self.ptr] = obs
        self.mask_buf[np.arange(self.ptr-self.delay+1,self.ptr+1)] = greater_equal.outer(np.arange(self.delay-1,-1,-1),np.arange(self.delay))
        self.state_buf[self.ptr] = state[0].reshape(-1)

        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def get_pred_data(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, to feed the prediction network.
        """
        idx = np.random.choice(np.arange(self.size), size=self.batch_size, replace=False) 
        # Create hidden states
        mask = self.mask_buf[idx]
        hidden_states = np.stack([np.roll(self.state_buf, -i, axis=0) for i in range(self.delay)], 1)
        hidden_states = hidden_states[idx]

        return self.obs_buf[idx], hidden_states[mask], mask


class GAEBufferDeter:
    """
    A buffer for storing trajectories experienced by a TRPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, state_dim, act_dim, act_rpz_dim, size_pred_buf, batch_size_pred, size, gamma=0.99,
                 lam=0.95):
        self.state_dim = state_dim
        self.max_delay = int((obs_dim - state_dim)/act_rpz_dim)
        self.act_rpz_dim = act_rpz_dim
        self.obs_buf = np.zeros(Core.combined_shape(size, obs_dim), dtype=np.float32)
        self.state_buf = np.zeros(Core.combined_shape(size, state_dim), dtype=np.float32)
        self.act_buf = np.zeros(Core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(Core.combined_shape(size, 1), dtype=np.bool)
        self.index_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.pred_buf = PredBuffer(state_dim, obs_dim, size_pred_buf, batch_size_pred, self.max_delay)

    def store(self, obs, act, rew, val, done, info, logp, index, pretrain=False):
        """Store last agent-environment interaction in the buffer.
        """
        if not pretrain:
            assert self.ptr < self.max_size  # buffer has to have room so you can store
            self.obs_buf[self.ptr] = obs
            self.act_buf[self.ptr] = act
            self.val_buf[self.ptr] = val
            self.logp_buf[self.ptr] = logp
            self.index_buf[self.ptr] = index
            self.done_buf[self.ptr] = np.array([done])
            self.rew_buf[self.ptr] = np.sum(rew)
            self.pred_buf.store(obs, info[1])
            self.ptr += 1
        else:
            self.pred_buf.store(obs, info[1])

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = Core.discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = Core.discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean = np.average(self.adv_buf)
        adv_std = np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf, index=self.index_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}

    def get_pred_data(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, to feed the prediction network.
        """
        extended_states, hidden_states, mask = self.pred_buf.get_pred_data()
        # if reset:
        #     self.ptr, self.path_start_idx = 0, 0
        data = dict(extended_states=torch.as_tensor(extended_states, dtype=torch.float32), 
                    hidden_states=torch.as_tensor(hidden_states, dtype=torch.float32),
                    mask=mask)
        return data

    def reset(self):
        self.ptr, self.path_start_idx = 0, 0





# -------------------------------------------------------------------------------------------------------------
# -----------------------------       Stochastic buffer      --------------------------------------------------
# -------------------------------------------------------------------------------------------------------------


class PredBufferStoch:
    def __init__(self, state_dim, obs_dim, size, batch_size, max_delay):
        # self.init_delay = 0 
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.max_delay = max_delay
        self.batch_size = batch_size
        self.obs_buf = np.zeros(Core.combined_shape(size, obs_dim), dtype=np.float32)
        self.state_buf = np.zeros((size, max_delay, state_dim), dtype=np.float32)
        self.mask_buf = np.zeros(Core.combined_shape(size, max_delay), dtype=np.bool)
        self.delay = np.zeros(size, dtype=np.int32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, info, delay, done):
        self.delay[self.ptr] = delay
        self.obs_buf[self.ptr] = obs
        self.mask_buf[self.ptr] = np.append(np.ones(delay), np.zeros(self.max_delay-delay))
        if not done:
            # Create a mask which prevent from computing the loss on states that have never actually been observed
            self.mask_buf[np.arange(self.ptr-self.max_delay+1, self.ptr+1)] *= greater_equal.outer(
                    np.arange(self.max_delay - 1, -1, -1), np.arange(self.max_delay))

        self.state_buf[self.ptr] = np.roll(self.state_buf[self.ptr-1], 1, axis=0)

        if info[0] > 0:
            for i, s in enumerate(info[1]):
                # The same state is used for the prediction in the previous 'delay' extended state
                self.state_buf[self.ptr+np.arange(-delay + i + 1, 1), np.arange(delay-i)] = s

        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def get_pred_data(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, to feed the prediction network.
        """
        idx = np.random.choice(np.arange(self.size), size=self.batch_size, replace=False) 
        # Create hidden states
        mask = self.mask_buf[idx]
        hidden_states = self.state_buf[idx]
        hidden_states = np.concatenate([np.flip(hidden_states[i][mask[i]],1) for i in range(self.batch_size)])

        return self.obs_buf[idx], hidden_states, mask


class GAEBufferStoch:
    """
    A buffer for storing trajectories experienced by a TRPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, state_dim, act_dim, act_rpz_dim, size_pred_buf, batch_size_pred, size, gamma=0.99,
                 lam=0.95):
        self.state_dim = state_dim
        self.max_delay = int((obs_dim - state_dim)/act_rpz_dim)
        self.act_rpz_dim = act_rpz_dim
        self.obs_buf = np.zeros(Core.combined_shape(size, obs_dim), dtype=np.float32)
        self.state_buf = np.zeros(Core.combined_shape(size, state_dim), dtype=np.float32)
        self.act_buf = np.zeros(Core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size+self.max_delay, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(Core.combined_shape(size, 1), dtype=np.bool)
        self.index_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.pred_buf = PredBufferStoch(state_dim, obs_dim, size_pred_buf, batch_size_pred, self.max_delay)
        self.init_delay = None

    def store(self, obs, act, rew, val, done, info, logp, index, pretrain=False):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        delay = int((len(obs) - self.state_dim)/self.act_rpz_dim)
        obs = np.append(obs, np.zeros((self.max_delay-delay)*self.act_rpz_dim))
        if not done or self.ptr == 0:
            self.init_delay = delay
        if not pretrain:
            assert self.ptr < self.max_size  # buffer has to have room so you can store
            self.obs_buf[self.ptr] = obs
            self.act_buf[self.ptr] = act
            self.val_buf[self.ptr] = val
            self.logp_buf[self.ptr] = logp
            self.index_buf[self.ptr] = index
            self.done_buf[self.ptr] = np.array([done])
            self.pred_buf.store(obs, info, delay, done)
            if info[0] > 0:
                # warning, info contains informations about the future of the obs that is stored
                idx = np.arange(self.init_delay+self.ptr-delay, self.init_delay+self.ptr-delay+info[0])
                self.rew_buf[idx] = rew
            self.ptr += 1
        else:
            self.pred_buf.store(obs, info, delay, done)

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = Core.discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = Core.discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        self.pred = np.zeros(self.max_size, dtype=np.bool)
        # the next two lines implement the advantage normalization trick
        adv_mean = np.average(self.adv_buf)
        adv_std = np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf, index=self.index_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}

    def get_pred_data(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, to feed the prediction network.
        """
        extended_states, hidden_states, mask = self.pred_buf.get_pred_data()
        data = dict(extended_states=torch.as_tensor(extended_states, dtype=torch.float32), 
                    hidden_states=torch.as_tensor(hidden_states, dtype=torch.float32),
                    mask=mask)
        return data

    def reset(self):
        self.ptr, self.path_start_idx = 0, 0