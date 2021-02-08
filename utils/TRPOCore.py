import numpy as np
import scipy.signal
import torch
import torch.nn as nn
from gym.spaces import Box, Discrete
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from utils.various import get_space_dim 


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

    def __init__(self, obs_dim, state_dim, act_dim, hidden_sizes, activation, conv=False):
        super().__init__()
        if conv:
            self.state_dim = state_dim
            self.dimension_input = (int(np.sqrt(state_dim)), int(np.sqrt(state_dim)))
            conv_layers = [nn.Conv2d(in_channels=1, out_channels=4, kernel_size=2, padding=1),
                    nn.Conv2d(in_channels=4, out_channels=4, kernel_size=2, padding=1),
                    nn.MaxPool2d(kernel_size=3)]
            self.conv = nn.Sequential(*conv_layers)
            obs_dim = obs_dim + get_encoder_dim() - state_dim
            self.input_mapping = self.mapping_conv
        else:
            self.input_mapping = self.mapping_id
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        obs = self.input_mapping(obs)
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def mapping_id(self, x):
        return x

    def mapping_conv(self, x):
        if len(x.size())==1:
            x = x.reshape(1,-1)
        state = x[:,:self.state_dim].reshape(x.size(0), 1, 
                self.dimension_input[0], self.dimension_input[1])
        action = x[:,self.state_dim:]

        x = self.conv(state).reshape(x.size(0), -1)
        x = torch.cat((x, action),dim=1)
        return x.squeeze()

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


def get_encoder_dim(conv_param=None):
    return 16

class MLPCritic(nn.Module):

    def __init__(self, obs_dim, state_dim, hidden_sizes, activation, conv=False):
        super().__init__()
        if conv:
            self.state_dim = state_dim
            self.dimension_input = (int(np.sqrt(state_dim)), int(np.sqrt(state_dim)))
            conv_layers = [nn.Conv2d(in_channels=1, out_channels=4, kernel_size=2, padding=1),
                    nn.Conv2d(in_channels=4, out_channels=4, kernel_size=2, padding=1),
                    nn.MaxPool2d(kernel_size=3)]
            self.conv = nn.Sequential(*conv_layers)
            obs_dim = obs_dim + get_encoder_dim() - state_dim
            self.input_mapping = self.mapping_conv
        else:
            self.input_mapping = self.mapping_id
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def mapping_id(self, x):
        return x

    def mapping_conv(self, x):
        if len(x.size())==1:
            x = x.reshape(1,-1)
        state = x[:,:self.state_dim].reshape(x.size(0), 1, 
                self.dimension_input[0], self.dimension_input[1])
        action = x[:,self.state_dim:]

        x = self.conv(state).reshape(x.size(0), -1)
        x = torch.cat((x, action),dim=1)
        return x.squeeze()

    def forward(self, obs):
        obs = self.input_mapping(obs)
        return torch.squeeze(self.v_net(obs), -1)  # Critical to ensure v has right shape.


class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, state_space, pi_hidden_sizes=(64, 64), v_hidden_sizes=(64, 64),
                 activation=nn.Tanh(), conv=False):
        super().__init__()

        obs_dim = get_space_dim(observation_space)
        state_dim = get_space_dim(state_space)

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], pi_hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, state_dim, action_space.n, pi_hidden_sizes, activation, conv=conv)

        # build value function
        self.v = MLPCritic(obs_dim, state_dim, v_hidden_sizes, activation, conv=conv)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]