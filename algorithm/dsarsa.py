from utils.various import prGreen, prYellow
from datetime import datetime as dt
from datetime import timedelta
from matplotlib import pyplot as plt
import torch
import numpy as np
import os


class DSARSA:

    def __init__(self, env, seed=0, delay=3, epochs=200, steps=12500, max_steps=2500, lam=1.0, gamma=0.99, lr=0.3, e=0.3,
                 s_space=10, a_space=3, train_render=False, train_render_ep=1, save_dir='./output/sarsa'):

        # Seed
        self.seed = seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Gym Environment
        self.env = env
        self.env.action_space.seed(seed)
        self.delay = delay
        self.state_dim = self.env.state_space.shape[0]

        # Training Parameters
        self.epochs = epochs
        self.steps = steps
        self.max_steps = max_steps

        # SARSA Parameters
        self.lam = lam
        self.gamma = gamma
        self.lr = lr
        self.e = e

        # Discretization Setup
        self.s_space = s_space
        self.high_s = self.env.state_space.high
        self.low_s = self.env.state_space.low
        self.a_space = a_space
        self.actions_index = np.arange(0, self.a_space, 1, dtype=np.int32)
        self.high_a = self.env.action_space.high
        self.low_a = self.env.action_space.low
        self.a_step = (self.high_a-self.low_a)/(self.a_space-1)
        self.actions = np.arange(self.low_a, self.high_a + self.a_step, self.a_step)
        self.probs = [1 / self.a_space for _ in range(self.a_space)]

        # Q and eligibility traces
        self.eligibility = np.zeros((self.s_space ** self.state_dim, self.a_space))
        self.Q = np.zeros((self.s_space ** self.state_dim, self.a_space))

        # Results
        self.epoch = 0
        self.avg_reward = []
        self.std_reward = []
        self.avg_length = []
        self.timings = []
        self.elapsed_time = timedelta(0)
        self.save_dir = save_dir
        self.train_render = train_render
        self.train_render_ep = train_render_ep

    def discretize_s(self, s):
        s = s[0]
        s = np.floor((s-self.low_s) / (self.high_s-self.low_s) * (self.s_space-1))
        s = np.sum([s[i] * (self.s_space ** i) for i in range(self.state_dim)])
        return int(s)

    def discretize_a(self, s):
        for i in range(len(s[1])):
            s[1][i] = self.actions[np.abs(np.array(self.actions) - s[1][i]).argmin()]
        return s

    def index_a(self, a):
        a = np.where(self.actions == a)
        return a[0][0]

    def train(self):
        start_time = dt.now()
        for epoch in range(self.epochs):
            ep_rewards = []
            ep_lengths = []
            self.epoch = epoch + 1
            s = self.env.reset()
            disc_steps = self.delay
            a = self.sample_a(s, e_greedy=True)

            ep_ret = 0
            ep_len = 0
            for step in range(self.steps):
                # Next Step in the Environment to produce "s, a, r, s, a"
                next_s, r, d, _ = self.env.step([self.actions[a]])

                ep_ret += r.sum()
                ep_len += 1

                next_a = self.sample_a(next_s, e_greedy=True)

                # The Actions that the environment randomly samples to construct the first augmented state of the
                # episode are not discretized, we need to associate them to the nearest discrete action in the Agent
                # each time an episode starts for "delay" steps.
                if disc_steps >= 0:
                    s = self.discretize_a(s)
                    next_s = self.discretize_a(next_s)
                    disc_steps -= 1

                # Retrieve the index of the actual action that has been executed in the environment in this step,
                # rather than the action chosen by the agent this step. This action is the first action in the
                # augmented state after the state.
                a_act = self.index_a(s[1][0])
                next_a_act = self.index_a(next_s[1][0])

                # Update
                self.update(s, a_act, r.sum(), next_s, next_a_act)

                s = next_s
                a = next_a

                if self.train_render and (self.epoch % self.train_render_ep == 0):
                    self.env.render()

                timeout = ep_len == self.max_steps
                terminal = d or timeout
                epoch_ended = step == self.steps - 1

                if terminal or epoch_ended:
                    if epoch_ended and not terminal:
                        prGreen('\tWarning: trajectory cut off by epoch at %d steps.' % ep_len)
                    if terminal:
                        ep_rewards.append(ep_ret)
                        ep_lengths.append(ep_len)
                        s = self.env.reset()
                        disc_steps = self.delay
                        a = self.sample_a(s, e_greedy=True)
                        ep_len = 0
                        ep_ret = 0

            self.avg_reward.append(np.average(ep_rewards))
            self.std_reward.append(np.std(ep_rewards))
            self.avg_length.append(np.average(ep_lengths))
            self.eligibility = np.zeros((self.s_space ** self.state_dim, self.a_space))

            self.elapsed_time = dt.now() - start_time
            self.timings.append(self.elapsed_time)
            self.print_update()
            self.save_session()
            self.save_results()

    def update(self, s, a, r, next_s, next_a):
        # State Discretization
        s = self.discretize_s(s)
        next_s = self.discretize_s(next_s)

        # Update Eligibility Traces
        self.eligibility = self.gamma * self.lam * self.eligibility
        self.eligibility[s, a] = 1

        # Compute TD Error
        delta_t = r + self.gamma*self.Q[next_s, next_a] - self.Q[s, a]

        try:
            self.Q = self.Q + self.lr*delta_t*self.eligibility
        except RuntimeWarning:
            print(self.Q)
            print(self.lr)
            print(delta_t)
            print(self.eligibility)
            return

    def sample_a(self, s, e_greedy=True):
        if e_greedy and np.random.rand() < self.e:
            action = np.random.choice(self.actions_index, p=self.probs)
        else:
            s = self.discretize_s(s)
            action = np.argmax(self.Q[s, :])
        return action

    def print_update(self):
        update_message = '[EPOCH]: {0}\t[AVG. REWARD]: {1:.4f}\t[EP. LENGTH]: {2}\t[ELAPSED TIME]: {3}'
        elapsed_time_str = ''.join(str(self.elapsed_time).split('.')[0])
        format_args = (self.epoch, self.avg_reward[-1], self.avg_length[-1], elapsed_time_str)
        prYellow(update_message.format(*format_args))

    def save_session(self):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        save_path = os.path.join(self.save_dir, 'model.pt')

        ckpt = {'q_table': self.Q,
                'avg_reward': self.avg_reward,
                'std_reward': self.std_reward,
                'actions': self.actions,
                'actions_index': self.actions_index,
                'epoch': self.epoch,
                'elapsed_time': self.elapsed_time,
                'timings': self.timings
                }

        torch.save(ckpt, save_path)

    def load_session(self):
        load_path = os.path.join(self.save_dir, 'model.pt')
        ckpt = torch.load(load_path)

        self.Q = ckpt['q_table']
        self.actions = ckpt['actions']
        self.actions_index = ckpt['actions_index']
        self.avg_reward = ckpt['avg_reward']
        self.std_reward = ckpt['std_reward']
        self.epoch = ckpt['epoch']
        self.timings = ckpt['timings']
        self.elapsed_time = ckpt['elapsed_time']

    def save_results(self):
        x = range(0, self.epoch, 1)
        self.errorbar_plot(x, self.avg_reward, xlabel='Epochs', ylabel='Average Reward', filename='reward',
                           error=self.std_reward)
        self.scatter_plot(x, self.avg_reward, xlabel='Epochs', ylabel='Average Reward', filename='reward_scatter')

    def scatter_plot(self, x, y, xlabel='Epochs', ylabel='Average Reward', filename='reward'):
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        ax.scatter(x, y)
        plt.savefig(os.path.join(self.save_dir, filename + '.png'))
        plt.close(fig)

    def errorbar_plot(self, x, y, xlabel='Epochs', ylabel='Average Reward', filename='reward', error=None):
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        ax.errorbar(x, y, yerr=error, fmt='-o')
        plt.savefig(os.path.join(self.save_dir, filename + '.png'))
        plt.close(fig)

    def test(self, test_episodes=10, max_steps=250):
        self.load_session()
        episode = 0

        reward = []
        # Test Loop: For Each Episode
        while episode < test_episodes:
            s = self.env.reset()

            step = 0
            ep_ret = 0
            while step < max_steps:
                s = self.discretize_s(s)
                a = np.argmax(self.Q[s, :])

                next_s, r, d, _ = self.env.step([self.actions[a]])
                s = next_s

                # self.env.render()

                ep_ret += r.sum()
                step += 1

                if d:
                    break

            episode += 1

            # Print Episode Result
            update_message = '[EPISODE]: {0}\t[EPISODE REWARD]: {1:.4f}'
            format_args = (episode, ep_ret)
            prYellow(update_message.format(*format_args))

            # Append Episode Reward
            reward.append(ep_ret)

        # Save test results
        save_path = os.path.join(self.save_dir, 'test_result.pt')
        ckpt = {
            'seed': self.seed,
            'reward': reward
        }
        torch.save(ckpt, save_path)

        self.env.close()
