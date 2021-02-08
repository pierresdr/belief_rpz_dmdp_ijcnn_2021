import math
import numpy as np
import scipy.signal
import torch
import torch.nn as nn
from gym.spaces import Box, Discrete
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from utils.various import get_space_dim
from utils.belief_module import BeliefModuleDeter, BeliefModuleStoch


def flat_grads(grads):
    return torch.cat([grad.contiguous().view(-1) for grad in grads])


def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params


def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


def conjugate_gradients(Avp, b, nsteps, residual_tol=1e-10):
    x = torch.zeros(b.size())
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for i in range(nsteps):
        _Avp = Avp(p)
        alpha = rdotr / torch.dot(p, _Avp)
        x += alpha * p
        r -= alpha * _Avp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)  # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)  # Critical to ensure v has right shape.


class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, pi_hidden_sizes=(64, 64), v_hidden_sizes=(64, 64),
                 activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], pi_hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, pi_hidden_sizes, activation)
            # self.pi = MLPCategoricalActor(obs_dim, 1, pi_hidden_sizes, activation)

        # build value function
        self.v = MLPCritic(obs_dim, v_hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]


class TRNGaussianActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, encoder):
        super().__init__()
        self.encoder = encoder
        self.pi = MLPGaussianActor(obs_dim, act_dim, hidden_sizes, activation)

    def _distribution(self, obs):
        encoded_obs = self.encoder(obs, policy=True).squeeze()
        pi_input = encoded_obs.detach()
        mu = self.pi.mu_net(pi_input)
        std = torch.exp(self.pi.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)



class TRNActorCritic(nn.Module):
    def __init__(self, obs_dim, action_space, state_space, enc_dim=128, enc_heads=2, enc_ff=8, enc_l=1, dropout=0.0,
                 enc_rescaling=False, enc_causal=False, pi_hidden_sizes=(64, 64), v_hidden_sizes=(64, 64),
                 activation=nn.Tanh, hidden_dim=8, pred_to_pi=False, lstm=False, n_layers=3, hidden_size=16,
                 n_blocks_maf=5, hidden_dim_maf=16, use_belief=True, conv=False, only_last_belief=False):
        super().__init__()

        # Encoder Builder
        pi_in_dim = hidden_dim
        if use_belief:
            self.enc = BeliefModuleStoch(state_space, action_space, encoder_dim=enc_dim, encoder_heads=enc_heads, encoder_ff_hid=enc_ff, 
                    encoder_layers=enc_l, hidden_size=hidden_size, num_layers=n_layers, hidden_dim=hidden_dim, 
                    n_blocks_maf=n_blocks_maf, hidden_dim_maf=hidden_dim_maf, dropout=dropout, 
                    rescaling=enc_rescaling, causal=enc_causal, conv=conv, lstm=lstm, only_last_belief=only_last_belief)
        else:
            self.enc = BeliefModuleDeter(state_space, action_space, encoder_dim=enc_dim, encoder_heads=enc_heads, encoder_ff_hid=enc_ff, 
                    encoder_layers=enc_l, hidden_size=hidden_size, num_layers=n_layers, hidden_dim=hidden_dim, 
                    dropout=dropout, rescaling=enc_rescaling, causal=enc_causal, conv=conv, lstm=lstm, only_last_belief=only_last_belief,
                    pred_to_pi=pred_to_pi)
            if pred_to_pi:
                pi_in_dim = get_space_dim(state_space)

        # Policy builder (Depends on action space)
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(pi_in_dim, action_space.shape[0], pi_hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(pi_in_dim, action_space.n, pi_hidden_sizes, activation)

        # Value Function Builder
        self.v = MLPCritic(pi_in_dim, v_hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            # if obs.shape[1]==self.state_dim:
            #     enc_obs = to_tensor(s_t)
            # else:
            enc_obs = self.enc(obs).detach()
            pi = self.pi._distribution(enc_obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(enc_obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]
