from collections import defaultdict
import numpy as np
import warnings
import torch
import sys
from utils.torch import get_device
from utils.normalized_env import NormalizedEnv
from torch.nn.utils.rnn import pad_sequence
from utils.various import get_space_dim


# [Reference] https://github.com/ajlangley/trpo-pytorch


class Simulator:
    def __init__(self, env, policy, n_trajectories, trajectory_len, delay=0, stochastic_delays=False, seed=None, **env_args):
        self.env = np.asarray([env(delay=delay, stochastic_delays=stochastic_delays) for _ in range(n_trajectories)])
        self.action_space = self.env[0].action_space
        self.policy = policy
        self.n_trajectories = n_trajectories
        self.trajectory_len = trajectory_len
        self.device = get_device()

        # If a Seed is given, seed the Envs and Action Space for Delay Reproducibility
        if seed is not None:
            for env in self.env:
                env.action_space.seed(seed)






class SinglePathSimulatorStoch(Simulator):
    def __init__(self, env_name, policy, n_trajectories, trajectory_len, delay=0, stochastic_delays=False, **env_args):
        Simulator.__init__(self, env_name, policy, n_trajectories, trajectory_len, delay, **env_args)


    # A method that runs different trajectories to collect samples
    def sample_trajectories(self):

        self.policy.eval()

        with torch.no_grad():
            # Initialize the Trajectories and their Done variables
            memory = np.asarray([defaultdict(list) for _ in range(self.n_trajectories)])
            done = [False] * self.n_trajectories
            for trajectory in memory:
                trajectory['done'] = False

            # Set the Environment Reset State as the initial State of each Trajectory
            for env, trajectory in zip(self.env, memory):
                state = env.reset()
                temp = torch.tensor([i%env.action_space.n==state[1][i//env.action_space.n] for i in range(env.action_space.n*len(state[1]))]).float()
                state = torch.cat((torch.tensor(state[0]).float().reshape(-1), temp))
                trajectory['extended_states'].append(state)
                trajectory['cum_obs'].append(0)

            # Trajectory Loop
            step = 0
            while not np.all(done):
                # Select which trajectories must continue (i.e. they didn't reach a final State)
                continue_mask = [i for i, trajectory in enumerate(memory) if not trajectory['done']]
                trajs_to_update = [trajectory for trajectory in memory if not trajectory['done']]
                continuing_envs = [env for i, env in enumerate(self.env) if i in continue_mask]

                # Prepare a Tensor containing the last State of each Trajectory
                states_lengths = [len(trajectory['extended_states'][-1]) for trajectory in trajs_to_update]
                policy_input = pad_sequence([trajectory['extended_states'][-1].clone().detach().to(self.device)
                                            for trajectory in trajs_to_update],batch_first=True)

                # For each Trajectory, retrieve the Action Distribution from the Policy given the last State
                # action_dists = self.policy(policy_input)

                # Sample an Action for each Trajectory from the Action Distribution
                # actions = action_dists.sample()

                # Torch NAN Error:
                # if torch.isnan(actions).any():
                    # warnings.warn('NAN Action Detected')
                    # sys.exit()

                # Clamp the Action to match Normalized Enviroment input
                # actions = torch.clamp(actions, -1.0, 1.0)
                # actions = actions.cpu()

                # For each Trajectory execute the sampled Action
                for env, trajectory in zip(continuing_envs, trajs_to_update):
                    action = env.action_space.sample()
                    state, reward, done, info = env.step(action)

                    reward = torch.tensor(reward, dtype=torch.float)

                    trajectory['actions'].append(action)
                    trajectory['rewards'].append(reward)
                    trajectory['done'] = done
                    # trajectory['hidden_states'].append(torch.tensor(env.hidden_obs).float())
                    if info[0]>0:
                        temp = torch.stack([torch.tensor(s).reshape(-1) for s in info[1]])
                        trajectory['states'].append(temp.float())
                    trajectory['cum_obs'].append(trajectory['cum_obs'][-1]+info[0])
                    
                    # Force Done = True if the Trajectory Length has been reached
                    if step == self.trajectory_len - 1:
                        trajectory['done'] = True

                    # If the Trajectory is not finished, append the last State
                    if not trajectory['done']:
                        temp = torch.tensor([i%env.action_space.n==state[1][i//env.action_space.n] for i in range(env.action_space.n*len(state[1]))]).float()
                        state = torch.cat((torch.tensor(state[0]).float().reshape(-1), temp))
                        trajectory['extended_states'].append(state)

                # Update Done Vector used to decide which Trajectory must continue
                done = [trajectory['done'] for trajectory in memory]
                step += 1

        return memory
