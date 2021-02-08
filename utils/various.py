import os, datetime
import torch
import numpy as np
from prettytable import PrettyTable
from torch.autograd import Variable
from gym.spaces.dict import Dict 
from gym.spaces.discrete import Discrete
import operator
from functools import reduce
from torch.nn.utils.rnn import pad_sequence

USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor


def prRed(prt): print("\033[91m {}\033[00m" .format(prt))
def prGreen(prt): print("\033[92m {}\033[00m" .format(prt))
def prYellow(prt): print("\033[93m {}\033[00m" .format(prt))
def prLightPurple(prt): print("\033[94m {}\033[00m" .format(prt))
def prPurple(prt): print("\033[95m {}\033[00m" .format(prt))
def prCyan(prt): print("\033[96m {}\033[00m" .format(prt))
def prLightGray(prt): print("\033[97m {}\033[00m" .format(prt))
def prBlack(prt): print("\033[98m {}\033[00m" .format(prt))


def prod(factors):
    return reduce(operator.mul, factors, 1)


def to_numpy(var):
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()


def to_tensor(ndarray, requires_grad=False, dtype=FLOAT):
    return Variable(torch.from_numpy(ndarray), requires_grad=requires_grad).type(dtype)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

# Old version, see below
# def get_output_folder(parent_dir, env_name):
#     """Return save folder.
#     Assumes folders in the parent_dir have suffix -run{run
#     number}. Finds the highest run number and sets the output folder
#     to that number + 1. This is just convenient so that if you run the
#     same script multiple times tensorboard can plot all of the results
#     on the same plots with different names.
#     Parameters
#     ----------
#     parent_dir: str
#       Path of the directory containing all experiment runs.
#     env_name: str
#       Name of the environment used for folder naming
#     Returns
#     -------
#     parent_dir/run_dir
#       Path to this run's save directory.
#     """
#     os.makedirs(parent_dir, exist_ok=True)
#     experiment_id = 0
#     for folder_name in os.listdir(parent_dir):
#         if not os.path.isdir(os.path.join(parent_dir, folder_name)):
#             continue
#         try:
#             folder_name = int(folder_name.split('-run')[-1])
#             if folder_name > experiment_id:
#                 experiment_id = folder_name
#         except:
#             pass
#     experiment_id += 1

#     parent_dir = os.path.join(parent_dir, env_name)
#     parent_dir = parent_dir + '-run{}'.format(experiment_id)
#     os.makedirs(parent_dir, exist_ok=True)
#     return parent_dir


def get_output_folder(parent_dir, env_name):
    """Return save folder.
    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.
    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.
    env_name: str
      Name of the environment used for folder naming
    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-' + datetime.datetime.now().strftime("%y-%m-%d_%H_%M_%f")
    os.makedirs(parent_dir, exist_ok=True)
    return parent_dir


def network_print(net):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in net.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    prLightPurple(table)
    # prLightPurple(f"Total Trainable Parameters: {total_params}")
    return total_params


def get_space_dim(space):
    if type(space) == Dict:
        return sum([get_space_dim(v) for k, v in space.spaces.items()])
    elif type(space) == Discrete:
        return space.n
    else:
        return prod(space.shape)


def hidden_state_matrix(states, max_delay, cum_obs=None):
    """ Returns a matrix of states where row correspond to 
    consecutive times of the MDP and columns to the states 
    that are to be predicted by the network up to an extended
    state length of max_length.
    """
    if cum_obs is None:
        states = torch.cat((states, torch.zeros(max_delay, states.size(1))))
        return pad_sequence([states[i:i+max_delay]
                             for i in range(0, len(states)-max_delay)], batch_first=True)
    else:
        states = torch.cat((states, torch.zeros(len(cum_obs)+max_delay, 3)))
        return pad_sequence([states[cum_obs[i]:cum_obs[i]+max_delay]
                             for i in range(len(cum_obs))], batch_first=True)


def buffer_to_matrix(samples, state_dim, stochastic_delays=False, action_dim=1):
    """ Read data from the buffer and align extended states to real states
    thanks to the use of a mask.
    """
    # the number of observed states trajectory['cum_obs'][-1] should be one 
    # more than the number of states used for the loss cumputation:
    # sum([len(s) for s in trajectory['states']])
    if stochastic_delays:
        extended_states = pad_sequence([s for trajectory in samples
                                        for s in trajectory['extended_states']], batch_first=True)
        max_delay = extended_states.shape[1]-state_dim
        states = torch.cat([hidden_state_matrix(torch.cat(trajectory['states'], 0), max_delay, trajectory['cum_obs'][1:])
                            for trajectory in samples])
        len_traj = [sum([len(s) for s in trajectory['states']]) for trajectory in samples]
        mask = pad_sequence([torch.ones(max(0, len(s)-state_dim - max(0, s_i-len_traj[t_i]+max_delay+1)), dtype=torch.bool)
                             for (t_i, trajectory) in enumerate(samples)
                             for (s_i, s) in enumerate(trajectory['extended_states'])], batch_first=True)
        # the following is necessary in case the maximum length extended state is one of the last states 
        # and it doesnt appear in the mask and thus isnt used for pad_sequence 
        if mask.size(1) != states.size(1):
            mask = torch.cat((mask, torch.zeros(mask.size(0), states.size(1)-mask.size(1), dtype=torch.bool)), dim=1)
    else:
        extended_states = torch.cat([torch.stack(trajectory['extended_states']) for trajectory in samples])
        max_delay = int((extended_states.shape[1]-state_dim)/action_dim)
        len_traj = [sum([len(s) for s in trajectory['states']]) for trajectory in samples]
        mask = pad_sequence([torch.ones(max(0, max_delay - max(0, s_i - len_traj[t_i] + max_delay + 1)), dtype=torch.bool)
                             for (t_i, trajectory) in enumerate(samples)
                             for (s_i, s) in enumerate(trajectory['extended_states'])], batch_first=True)
        states = torch.cat([hidden_state_matrix(torch.cat(trajectory['states'], 0), max_delay)
                            for trajectory in samples])
        if mask.size(1) != states.size(1):
            mask = torch.cat((mask, torch.zeros(mask.size(0), states.size(1)-mask.size(1), dtype=torch.bool)), dim=1)
    
    return extended_states, mask, states



class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)



    
def get_model_path(save_dir, epoch=None):
    if epoch is None:
        files_name = os.listdir(save_dir)
        files_name.remove('model_parameters.txt')
        models = list(filter(lambda n: "model" in n,  files_name))
        if len(models)==1:
            load_path = os.path.join(save_dir, models[0])
        else: 
            temp = np.array([int(name.replace('model_','').replace('.pt','')) for name in iter(models)])
            load_path = os.path.join(save_dir, 'model_'+str(max(temp))+'.pt')
    else:
        load_path = os.path.join(save_dir, 'model_'+str(epoch)+'.pt')
        
    return load_path