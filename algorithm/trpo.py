import warnings
import numpy as np
import torch.distributions
import utils.TRPOCore as Core
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.optim import Adam
from utils.TRPOBuffer import GAEBuffer
from datetime import datetime as dt
from datetime import timedelta
from utils.various import *

EPS = 1e-8


class TRPO:

    def __init__(self, env, actor_critic=Core.MLPActorCritic, ac_kwargs=dict(), seed=0, steps_per_epoch=4000,
                 epochs=50, gamma=0.99, delta=0.01, vf_lr=1e-3, train_v_iters=80, damping_coeff=0.1, cg_iters=10,
                 backtrack_iters=30, backtrack_coeff=0.8, lam=0.97, max_ep_len=1000, save_dir=None, save_period=1,
                 stoch_env=False, memoryless=False):
        """
        Trust Region Policy Optimization
        Schulman, John, et al. "Trust region policy optimization." International conference on machine learning. 2015.

        Args:
            env (gym.env): Env where the DTRPO is trained.
            actor_critic (class): Actor-Critic algorithm.
            ac_kwargs (dict): Any kwargs appropriate for the actor_critic.
            seed (int): Seed for random number generators.
            steps_per_epoch (int): Number of steps of interaction (state-action pairs)
                for the agent and the environment in each epoch.
            epochs (int): Number of epochs of interaction (equivalent to
                number of policy updates) to perform.
            gamma (float): Discount factor. (Always between 0 and 1.)
            delta (float): KL-divergence limit for TRPO / NPG update.
                (Should be small for stability. Values like 0.01, 0.05.)
            vf_lr (float): Learning rate for value function optimizer.
            train_v_iters (int): Number of gradient descent steps to take on
                value function per epoch.
            damping_coeff (float): Artifact for numerical stability, should be
                smallish. Adjusts Hessian-vector product calculation:
                .. math:: Hv \\rightarrow (\\alpha I + H)v
                where :math:`\\alpha` is the damping coefficient.
                Probably don't play with this hyperparameter.
            cg_iters (int): Number of iterations of conjugate gradient to perform.
                Increasing this will lead to a more accurate approximation
                to :math:`H^{-1} g`, and possibly slightly-improved performance,
                but at the cost of slowing things down.
                Also probably don't play with this hyperparameter.
            backtrack_iters (int): Maximum number of steps allowed in the
                backtracking line search. Since the line search usually doesn't
                backtrack, and usually only steps back once when it does, this
                hyperparameter doesn't often matter.
            backtrack_coeff (float): How far back to step during backtracking line
                search. (Always between 0 and 1, usually above 0.5.)
            lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
                close to 1.)
            max_ep_len (int): Maximum length of trajectory / episode / rollout.
            save_dir (str): Path to the folder where results are saved.
        """

        # Seed
        self.seed = seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Environment Initialization
        self.env = env
        # 1. Memoryless Agent: Aware only of the last observed state.
        self.memoryless = memoryless
        if self.memoryless:
            self.observation_space = self.env.state_space
        else:
            self.observation_space = self.env.observation_space
        # 2. Set the rest of the parameters
        self.env.action_space.seed(seed)
        self.obs_dim = get_space_dim(self.observation_space)
        self.act_dim = get_space_dim(self.env.action_space)
        self.stoch_env = stoch_env

        # Actor-Critic Module
        self.ac = actor_critic(self.observation_space, self.env.action_space, self.env.state_space, **ac_kwargs)

        # Value Function Optimizer
        self.vf_lr = vf_lr
        self.train_v_iters = train_v_iters
        self.vf_optimizer = Adam(self.ac.v.parameters(), lr=self.vf_lr)

        # Count variables
        var_counts = tuple(Core.count_vars(module) for module in [self.ac.pi, self.ac.v])
        prLightPurple('Number of parameters: \t pi: %d, \t v: %d\n' % var_counts)

        # Set up experience buffer
        self.gamma = gamma
        self.lam = lam
        self.steps_per_epoch = steps_per_epoch
        # action is set to 1, ok for all tested envs, otherwise problem with discrete action spaces size
        self.buf = GAEBuffer(self.obs_dim, self.env.action_space.shape, self.steps_per_epoch, self.gamma, self.lam)

        # Other TRPO Parameters
        self.epochs = epochs
        self.max_ep_len = max_ep_len
        self.delta = delta
        self.damping_coeff = damping_coeff
        self.cg_iters = cg_iters
        self.backtrack_iters = backtrack_iters
        self.backtrack_coeff = backtrack_coeff

        # Results and Save Variables
        self.save_dir = save_dir
        self.save_period = save_period
        self.epoch = 0
        self.elapsed_time = timedelta(0)
        self.avg_reward = []
        self.std_reward = []
        self.avg_length = []
        self.v_losses = []
        self.timings = []

    def compute_loss_pi(self, data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        # Policy loss
        _, logp = self.ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        loss_pi = -(ratio * adv).mean()
        return loss_pi

    def compute_loss_v(self, data):
        obs, ret = data['obs'], data['ret']
        return ((self.ac.v(obs) - ret) ** 2).mean()

    def compute_kl(self, data, old_pi):
        obs, act = data['obs'], data['act']
        pi, _ = self.ac.pi(obs, act)
        kl_loss = torch.distributions.kl_divergence(pi, old_pi).mean()
        return kl_loss

    @torch.no_grad()
    def compute_kl_loss_pi(self, data, old_pi):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        # Policy loss
        pi, logp = self.ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        loss_pi = -(ratio * adv).mean()
        kl_loss = torch.distributions.kl_divergence(pi, old_pi).mean()
        return loss_pi, kl_loss

    def hessian_vector_product(self, data, old_pi, v):
        kl = self.compute_kl(data, old_pi)

        grads = torch.autograd.grad(kl, self.ac.pi.parameters(), create_graph=True)
        flat_grad_kl = Core.flat_grads(grads)

        kl_v = (flat_grad_kl * v).sum()
        grads = torch.autograd.grad(kl_v, self.ac.pi.parameters())
        flat_grad_grad_kl = Core.flat_grads(grads)

        return flat_grad_grad_kl + v * self.damping_coeff

    def update(self):
        data = self.buf.get()
        # Policy Update
        self.update_pi(data)
        # Value Function Update
        v_loss = self.update_v(data)
        self.v_losses.append(v_loss)

    def update_pi(self, data):
        # Compute algorithm pi distribution
        obs, act = data['obs'], data['act']
        with torch.no_grad():
            old_pi, _ = self.ac.pi(obs, act)

        pi_loss = self.compute_loss_pi(data)
        pi_l_old = pi_loss.item()
        # v_l_old = self.compute_loss_v(data).item()

        grads = Core.flat_grads(torch.autograd.grad(pi_loss, self.ac.pi.parameters()))

        # Core calculations for TRPO
        Hx = lambda v: self.hessian_vector_product(data, old_pi, v)
        x = Core.conjugate_gradients(Hx, grads, self.cg_iters)

        alpha = torch.sqrt(2 * self.delta / (torch.matmul(x, Hx(x)) + EPS))

        old_params = Core.get_flat_params_from(self.ac.pi)

        def set_and_eval(step):
            new_params = old_params - alpha * x * step
            Core.set_flat_params_to(self.ac.pi, new_params)
            loss_pi, kl_loss = self.compute_kl_loss_pi(data, old_pi)
            return kl_loss.item(), loss_pi.item()

        # TRPO augments npg with backtracking line search, hard kl
        for j in range(self.backtrack_iters):
            kl, pi_l_new = set_and_eval(step=self.backtrack_coeff ** j)

            if kl <= self.delta and pi_l_new <= pi_l_old:
                prGreen('\tAccepting new params at step %d of line search.' % j)
                break

            if j == self.backtrack_iters - 1:
                prRed('\tLine search failed! Keeping algorithm params.')
                kl, pi_l_new = set_and_eval(step=0.)

    def update_v(self, data):
        # Value Function Learning
        loss_v = None
        for i in range(self.train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_v = self.compute_loss_v(data)
            loss_v.backward()
            self.vf_optimizer.step()
        return loss_v.detach().item()

    def format_o(self, o):
        if self.stoch_env:
            temp_o = torch.tensor([i % self.act_dim == o[1][i//self.act_dim]
                                   for i in range(self.act_dim*len(o[1]))]).float()
            o = torch.cat((torch.tensor(o[0]), temp_o.reshape(-1)))
        elif self.memoryless:
            o = torch.tensor(o[0])
        else:
            o = torch.cat((torch.tensor(o[0]), torch.tensor(o[1]).reshape(-1)))
        return o

    def train(self):
        # Prepare for interaction with environment
        start_time = dt.now()
        o, ep_ret, ep_len = self.env.reset(), 0, 0
        o = self.format_o(o)

        # Main loop: collect experience in env and update/log each epoch
        for epoch in range(1, self.epochs + 1):
            self.epoch = epoch

            ep_rewards = []
            ep_lengths = []
            episode = 0
            for t in range(self.steps_per_epoch):
                a, v, logp = self.ac.step(torch.as_tensor(o, dtype=torch.float32))

                next_o, r, d, _ = self.env.step(a)
                next_o = self.format_o(next_o)

                ep_ret += np.sum(r)
                ep_len += 1

                # save and log
                self.buf.store(o, a, np.sum(r), v, logp, episode)

                # Update obs (critical!)
                o = next_o

                timeout = ep_len == self.max_ep_len
                terminal = d or timeout
                epoch_ended = t == self.steps_per_epoch - 1

                if terminal or epoch_ended:
                    # If Epoch ended before Trajectory could end
                    if epoch_ended and not terminal:
                        prGreen('\tWarning: trajectory cut off by epoch at %d steps.' % ep_len)
                    # If trajectory didn't reach terminal state, bootstrap value target
                    if timeout or epoch_ended:
                        _, v, _ = self.ac.step(torch.as_tensor(o, dtype=torch.float32))
                    # If the Trajectory ended by its own, set State Value to 0
                    else:
                        v = 0
                    # Let the Buffer adjust itself when a Trajectory has ended
                    self.buf.finish_path(v)
                    if terminal:
                        # Only print EpRet and EpLen if trajectory finished
                        ep_rewards.append(ep_ret)
                        ep_lengths.append(ep_len)
                    o, ep_ret, ep_len = self.env.reset(), 0, 0
                    o = self.format_o(o)
                    episode += 1

            # Perform TRPO update at the end of the Epoch
            self.update()

            # Gather Epoch results and print them
            self.elapsed_time = dt.now() - start_time
            self.avg_reward.append(np.average(ep_rewards))
            self.std_reward.append(np.std(ep_rewards))
            self.avg_length.append(np.average(ep_lengths))
            self.timings.append(self.elapsed_time)
            self.print_update()

            # Save Epoch Results each "save_period" epochs
            if epoch % self.save_period == 0:
                self.save_session()

            self.save_results()

    def print_update(self):
        update_message = '[EPOCH]: {0}\t[AVG. REWARD]: {1:.4f}\t[V LOSS]: {2:.4f}\t[ELAPSED TIME]: {3}'
        elapsed_time_str = ''.join(str(self.elapsed_time).split('.')[0])
        format_args = (self.epoch, self.avg_reward[-1], self.v_losses[-1], elapsed_time_str)
        prYellow(update_message.format(*format_args))

    def save_session(self):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        save_path = os.path.join(self.save_dir, 'model_' + str(self.epoch) + '.pt')

        ckpt = {'policy_state_dict': self.ac.pi.state_dict(),
                'value_state_dict': self.ac.v.state_dict(),
                'avg_reward': self.avg_reward,
                'std_reward': self.std_reward,
                'v_losses': self.v_losses,
                'epoch': self.epoch,
                'elapsed_time': self.elapsed_time,
                'timings': self.timings
                }

        torch.save(ckpt, save_path)

    def load_session(self, test_epoch=0):
        if test_epoch != 0:
            model = 'model_' + str(test_epoch) + '.pt'
        else:
            model = 'model.pt'

        load_path = os.path.join(self.save_dir, model)
        ckpt = torch.load(load_path)

        self.ac.pi.load_state_dict(ckpt['policy_state_dict'])
        self.ac.v.load_state_dict(ckpt['value_state_dict'])
        self.avg_reward = ckpt['avg_reward']
        self.std_reward = ckpt['std_reward']
        self.v_losses = ckpt['v_losses']
        self.epoch = ckpt['epoch']
        self.timings = ckpt['timings']
        self.elapsed_time = ckpt['elapsed_time']

    def save_results(self):
        x = range(0, self.epoch, 1)
        self.errorbar_plot(x, self.avg_reward, xlabel='Epochs', ylabel='Average Reward', filename='reward',
                           error=self.std_reward)
        self.scatter_plot(x, self.avg_reward, xlabel='Epochs', ylabel='Average Reward', filename='reward_scatter')
        self.scatter_plot(x, self.v_losses, xlabel='Epochs', ylabel='Value Loss', filename='vloss_scatter')

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

    def test(self, test_epoch=2000, test_episodes=10, max_steps=250):
        self.load_session(test_epoch)
        episode = 0

        reward = []
        # Test Loop: For each Episode
        while episode < test_episodes:
            self.ac.pi.eval()
            o = self.env.reset()
            o = self.format_o(o)

            # For each Step
            step = 0
            ep_ret = 0.0
            while step < max_steps:
                a, _, _ = self.ac.step(torch.as_tensor(o, dtype=torch.float32))
                next_o, r, d, _ = self.env.step(a)
                o = self.format_o(next_o)
                # self.env.render()
                ep_ret += np.sum(r)
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
