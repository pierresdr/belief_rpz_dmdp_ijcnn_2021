
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import gym
from gym import spaces
from gym import wrappers
from utils.monitor import Monitor
import random
from utils.various import *
from utils.belief_module import BeliefModuleDeter, BeliefModuleStoch
from utils.DQNCore import DQN, LinearSchedule, wrap_env, get_wrapper_by_name



def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

class DDQN(nn.Module):
    def __init__(self, obs_dim, state_space, action_space, q_fun_neurons, dueling=False, 
                enc_dim=128, enc_heads=2, enc_ff=8, enc_l=1, dropout=0.0, enc_rescaling=False, 
                enc_causal=False, hidden_dim=8, pred_to_pi=False, lstm=False, n_layers=3, hidden_size=16,
                n_blocks_maf=5, hidden_dim_maf=16, stoch_env=False, conv=False, only_last_belief=False):

        super(DDQN, self).__init__()

        dqn_input_dim = hidden_dim
        if stoch_env:
            self.enc = BeliefModuleStoch(state_space, action_space, encoder_dim=enc_dim, encoder_heads=enc_heads, encoder_ff_hid=enc_ff, 
                    encoder_layers=enc_l, hidden_size=hidden_size, num_layers=n_layers, hidden_dim=hidden_dim, 
                    n_blocks_maf=n_blocks_maf, hidden_dim_maf=hidden_dim_maf, dropout=dropout, 
                    rescaling=enc_rescaling, causal=enc_causal, conv=conv, lstm=lstm, only_last_belief=only_last_belief)
        else:
            self.enc = BeliefModuleDeter(state_space, action_space, encoder_dim=enc_dim, encoder_heads=enc_heads, encoder_ff_hid=enc_ff, 
                    encoder_layers=enc_l, hidden_size=hidden_size, num_layers=n_layers, hidden_dim=hidden_dim, 
                    n_blocks_maf=n_blocks_maf, hidden_dim_maf=hidden_dim_maf, dropout=dropout, 
                    rescaling=enc_rescaling, causal=enc_causal, conv=conv, lstm=lstm, only_last_belief=only_last_belief,
                    pred_to_pi=pred_to_pi)
            if pred_to_pi:
                dqn_input_dim = get_space_dim(state_space)

        self.dqn = DQN(dqn_input_dim, get_space_dim(action_space), q_fun_neurons, dueling=dueling)

    def forward(self, x):
        with torch.no_grad():
            x = self.enc(x)
        return self.dqn(x)

