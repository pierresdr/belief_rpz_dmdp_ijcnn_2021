import math
import numpy as np
import scipy.signal
import torch
import torch.nn as nn
from gym.spaces import Box, Discrete
from utils.various import get_space_dim
from utils.flow import MAF





class InputMapping(nn.Module):
    """Input mapping for the belief network. Takes a couple of state action as input and returns 
    a quantity that will be processed by the encoder.

    Args:
        state_space (gym.space): The env state space.
        action_space (gym.space): The env action space.
        encoder_dim (int): Size required for the encoder input.
        rescaling (bool): Rescale the inputs to [-1,1].
        conv (bool): Process the input with a convolution.
    """
    def __init__(self, state_space, action_space, encoder_dim, rescaling=False, conv=False):
        super(InputMapping, self).__init__()

        # Access env dimensions
        self.state_dim = get_space_dim(state_space)
        self.action_dim = get_space_dim(action_space)
        self.rescaling = rescaling

        # Setup rescaling factor if rescale option is active
        if self.rescaling:
            self.low_state_value = state_space.low
            self.high_state_value = state_space.high
            self.low_action_value = action_space.low
            self.high_action_value = action_space.high
            self.action_scaling = torch.from_numpy(np.max((self.high_action_value, -self.low_action_value), 0)).reshape(1, 1, -1)
            self.state_scaling = torch.from_numpy(np.max((self.high_state_value, -self.low_state_value), 0)).reshape(1, 1, -1)

        # The option to use convolutions is not implemented
        if conv:
            raise NotImplementedError
        else:
            self.linear = nn.Linear(self.state_dim + self.action_dim, encoder_dim, bias=True)
            self.input_mapping = self.mapping_lin

        self.relu = nn.ReLU()

    def mapping_conv(self, extended_states):
        raise NotImplementedError

    def mapping_lin(self, extended_states):
        # Form pairs of (last_observed_state, action) for all the actions in the extended state
        state = extended_states[:, :self.state_dim].reshape(extended_states.size(0), 1, self.state_dim)
        action = extended_states[:, self.state_dim:].reshape(extended_states.size(0), -1, self.action_dim)
        if self.rescaling:
            state = state / self.state_scaling
            action = action / self.action_scaling
        x = state.repeat(1, action.size(1), 1)
        x = torch.cat((x, action), dim=2)
        
        x = self.linear(x)
        x = self.relu(x)

        return x

    def forward(self, x):
        return self.input_mapping(x)

        



class TransformerEncoder(nn.Module):
    """A tranformer encoder for the belief network.
    Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems. 2017.

    Args: 
        state_space (gym.space): The env state space.
        action_space (gym.space): The env action space.
        encoder_dim (int): Size required for the encoder input.
        rescaling (bool): Rescale the inputs to [-1,1].
        conv (bool): Process the input with a convolution.
        encoder_heads (int): Number of heads of the encoder.
        encoder_ff_hid (int): Number of neurons for the feed-forward mapping.
        encoder_layers (int): Number of layers.
        hidden_dim (int): Output dimension of the encoder.
        causal (bool): Use a causal transformer mapping.
        dropout (float): Dropout percentage.
        output_activation (str): Name of the output activation function.
    """
    def __init__(self, state_space, action_space, encoder_dim, encoder_heads, encoder_ff_hid, encoder_layers=1,
                 hidden_dim=8, dropout=0.0, rescaling=False, causal=False, conv=False, output_activation='nn.ReLU'):
        super(TransformerEncoder, self).__init__()


        # Encoder Properties
        self.encoder_dim = encoder_dim
        self.encoder_ff_hid = encoder_ff_hid
        self.encoder_layers = encoder_layers
        self.encoder_heads = encoder_heads
        self.dropout = dropout
        self.hidden_dim = hidden_dim

        # Encoder Options
        self.rescaling = rescaling
        self.causal = causal
        self.input_mask = None

        # Input Mapping
        self.input_mapping = InputMapping(state_space, action_space, encoder_dim, rescaling=rescaling, conv=conv)

        # Positional Encoding & Normalization Layer
        self.positional_encoding = PositionalEncoding(encoder_dim, dropout=0.0, max_len=50)
        self.norm_layer = nn.LayerNorm(self.encoder_dim, eps=1e-5)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(self.encoder_dim, nhead=encoder_heads,
                                                   dim_feedforward=encoder_ff_hid,
                                                   dropout=dropout, activation='relu')
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=encoder_layers, norm=None)
        self.output_mapping = nn.Linear(encoder_dim, hidden_dim, bias=True)
        self.output_activation = eval(output_activation+'()')


    @staticmethod
    def _generate_square_subsequent_mask(sz):
        """Generate a mask to ensure causality.
        """
        mask = torch.tril(torch.ones(sz, sz))
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, extended_states):
        x = self.input_mapping(extended_states)
        if self.causal:
            device = x.device
            if self.input_mask is None or self.input_mask.size(0) != x.size(1):
                self.input_mask = self._generate_square_subsequent_mask(x.size(1)).to(device)

        
        x = self.positional_encoding(x.transpose(0, 1))
        x = self.norm_layer(x)

        
        # Encoder Self-Attention
        encoded_state = self.encoder(x, mask=self.input_mask)
        encoded_state = self.output_mapping(encoded_state.transpose(0, 1))
        encoded_state = self.output_activation(encoded_state)
        
        return encoded_state






class LSTMEncoder(nn.Module):
    """A tranformer encoder for the belief network.
    Hochreiter, Sepp, and Jürgen Schmidhuber. "Long short-term memory." Neural computation 9.8 (1997): 1735-1780.
    
    Args: 
        state_space (gym.space): The env state space.
        action_space (gym.space): The env action space.
        rescaling (bool): Rescale the inputs to [-1,1].
        conv (bool): Process the input with a convolution.
        hidden_size (int): Hidden neuron size of LSTM.
        num_layers (int): Number of layers.
        hidden_dim (int): Output dimension of the encoder.
        dropout (float): Dropout percentage.
        output_activation (str): Name of the output activation function.
    """
    def __init__(self, state_space, action_space, hidden_size, num_layers, hidden_dim=8, rescaling=False, 
                dropout=0.1, output_activation='nn.ReLU'):
        super(ReconstructionLSTMStoch, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rescaling = rescaling

        # Input Mapping
        self.input_mapping = InputMapping(state_space, action_space, encoder_dim, rescaling=rescaling, conv=conv)
        
        # LSTM network
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers,
                            bias=True, batch_first=True, dropout=dropout)

        # Output network
        self.output_mapping = nn.Linear(hidden_size, hidden_dim, bias=True)
        self.output_activation = eval(output_activation+'()')

    def forward(self, x):
        x = self.input_mapping(x)

        hidden = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        cell = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        x, _ = self.lstm(x, (hidden, cell))
        x = self.output_mapping(x)
        x = self.output_activation(x)
        return x




class BeliefModuleStoch(nn.Module):
    """Belief module to predict belief over future states given an extended state as input.
    This belief module works on stochastic environments.

    Args: 
        state_space (gym.space): The env state space.
        action_space (gym.space): The env action space.
        rescaling (bool): Rescale the inputs to [-1,1].
        conv (bool): Process the input with a convolution.
        hidden_size (int): Hidden neuron size of LSTM.
        num_layers (int): Number of layers.
        hidden_dim (int): Output dimension of the encoder.
        dropout (float): Dropout percentage.
        output_activation (str): Name of the output activation function.
        encoder_dim (int): Size required for the encoder input.
        encoder_heads (int): Number of heads of the encoder.
        encoder_ff_hid (int): Number of neurons for the feed-forward mapping.
        encoder_layers (int): Number of layers.
        causal (bool): Use a causal transformer mapping.
        lstm (bool): If true, the encoder is an LSTM, otherwise a transformer.
        only_last_belief (bool): Predict only the last belief.
        n_blocks_maf (int): Number of blocks of MAF network.
        hidden_dim_maf (int): Hidden dimension in the MAF network.
    """   
    def __init__(self, state_space, action_space, encoder_dim, encoder_heads, encoder_ff_hid, encoder_layers=1,
                 hidden_size=32, num_layers=3, hidden_dim=8, n_blocks_maf=5, hidden_dim_maf=16, dropout=0.0,
                 rescaling=False, causal=False, conv=False, lstm=False, only_last_belief=False):
        super(BeliefModuleStoch, self).__init__()

        # Encoder Properties
        self.state_dim = get_space_dim(state_space)
        self.input_mask = None
        self.only_last_belief = only_last_belief

        if lstm:
            self.encoder = LSTMEncoder(state_space, action_space, hidden_size, num_layers, hidden_dim, rescaling,
                                       dropout, output_activation='nn.Tanh')
        else:
            self.encoder = TransformerEncoder(state_space, action_space, encoder_dim, encoder_heads, encoder_ff_hid,
                                              encoder_layers, hidden_dim, dropout, rescaling, causal, conv, 
                                              output_activation='nn.Tanh')

        # Prediction Network
        self.maf_proba = MAF(n_blocks=n_blocks_maf, input_dim=self.state_dim, hidden_dim=hidden_dim_maf,
                             cond_dim=hidden_dim)

    def log_probs(self, extended_states, hidden_states, mask):
        """Returns the log probability of the hidden states given the extended states.
        """
        encoded_state = self.encoder(extended_states)
        states = torch.zeros(encoded_state.size(0), encoded_state.size(1), self.state_dim).to(encoded_state.device)
        states[mask] = hidden_states

        u, log_jacob = self.maf_proba(states, encoded_state)
        log_probs = (-0.5 * u.pow(2) - 0.5 * math.log(2 * math.pi)).sum(
            -1, keepdim=True)

        # update the loss for higher wieghts on later belief 
        if self.only_last_belief:
            importance = torch.zeros(log_probs.size(0),log_probs.size(1),1).to(encoded_state.device)
            importance[:, -1, :] = 1
            log_jacob = log_jacob*importance
        return u[mask], (log_probs + log_jacob).sum(-1, keepdim=True)[mask]

    def get_cond(self, extended_states):
        return self.encoder(extended_states)

    def forward(self, extended_states):
        encoded_state = self.encoder(extended_states)

        return encoded_state[:, -1, :].squeeze(1)



class BeliefModuleDeter(nn.Module):
    """Belief module to predict belief over future states given an extended state as input.
    This module works on deterministic environments.

    Args: 
        state_space (gym.space): The env state space.
        action_space (gym.space): The env action space.
        rescaling (bool): Rescale the inputs to [-1,1].
        conv (bool): Process the input with a convolution.
        hidden_size (int): Hidden neuron size of LSTM.
        num_layers (int): Number of layers.
        hidden_dim (int): Output dimension of the encoder.
        dropout (float): Dropout percentage.
        output_activation (str): Name of the output activation function.
        encoder_dim (int): Size required for the encoder input.
        encoder_heads (int): Number of heads of the encoder.
        encoder_ff_hid (int): Number of neurons for the feed-forward mapping.
        encoder_layers (int): Number of layers.
        causal (bool): Use a causal transformer mapping.
        lstm (bool): If true, the encoder is an LSTM, otherwise a transformer.
        only_last_belief (bool): Predict only the last belief.
        pred_to_pi (bool): Use the prediction rather than the encoded state as input to the policy module.
    """    
    def __init__(self, state_space, action_space, encoder_dim, encoder_heads, encoder_ff_hid, encoder_layers=1,
                 hidden_size=32, num_layers=3, hidden_dim=8, dropout=0.0, rescaling=False, 
                 causal=False, conv=False, lstm=False, only_last_belief=False, pred_to_pi=False):
        super(BeliefModuleDeter, self).__init__()

        # Encoder Properties
        self.state_dim = get_space_dim(state_space)
        self.input_mask = None
        self.only_last_belief = only_last_belief
        self.pred_to_pi = pred_to_pi
        self.rescaling = rescaling

        if lstm:
            self.encoder = LSTMEncoder(state_space, action_space, hidden_size, num_layers, hidden_dim, rescaling,
                                       dropout)
        else:
            self.encoder = TransformerEncoder(state_space, action_space, encoder_dim, encoder_heads, encoder_ff_hid,
                                              encoder_layers, hidden_dim, dropout, rescaling, causal, conv)

        self.output_mapping = nn.Linear(hidden_dim, self.state_dim, bias=True)

    def encode_to_predict(self, encoded_state):
        pred = self.output_mapping(encoded_state)

        if self.rescaling:
            pred = pred * self.encoder.input_mapping.state_scaling
        return pred

    def predict(self, extended_states):
        encoded_state = self.encoder(extended_states)
        return self.encode_to_predict(encoded_state)

    def forward(self, extended_states):
        encoded_state = self.encoder(extended_states)

        if not self.pred_to_pi:
            return encoded_state[:, -1, :].squeeze(1)

        pred = self.encode_to_predict(encoded_state)
        return pred[:, -1, :].squeeze(1)





class PositionalEncoding(nn.Module):
    """
    Adapted from:
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000, append=False):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.append = append
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(1, d_model+1, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)

        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        if self.append:
            x = torch.cat((x,self.pe[:x.size(0), :].repeat(1,x.size(1),1)),2)
        else:
            x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
