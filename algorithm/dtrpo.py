import warnings
import numpy as np
import torch.distributions
import utils.DTRPOCore as Core
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.nn import MSELoss, L1Loss
from utils.DTRPOBuffer import GAEBufferDeter, GAEBufferStoch
from datetime import datetime as dt
from datetime import timedelta
from utils.various import *




EPS = 1e-8


class DTRPO:

    def __init__(self, env, actor_critic=Core.TRNActorCritic, ac_kwargs=dict(), seed=0, steps_per_epoch=4000,
                 epochs=50, gamma=0.99, delta=0.01, vf_lr=1e-3, train_v_iters=80, damping_coeff=0.1, cg_iters=10,
                 backtrack_iters=30, backtrack_coeff=0.8, lam=0.97, max_ep_len=1000, save_dir=None,
                 enc_lr=0.001, maf_lr=0.001, train_enc_iters=1, pretrain_epochs=50, pretrain_steps=5000, 
                 enc_loss='mse', size_pred_buf=4000, batch_size_pred=4000, train_continue=False,
                 save_period=1, epochs_belief_training=50, use_belief=True,):
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
            enc_lr (float): Learning rate of the encoder in the belief module.
            maf_lr (float): Learning rate of the MAF netowrk in the belief module.
            train_enc_iters (int): Ratio of encoder to MAF network trainings.
            pretrain_epochs (int): Number of epochs to train belief module before TRPO 
                training fires.
            pretrain_steps (int): Number of steps per epoch in the pretraining epochs.
            enc_loss (str): Name of the loss to use for the belief module (in the deterministic case).
            use_belief (bool): Whether the env is stochastic or not.
            size_pred_buf (int): Size of the belief module replay buffer.
            batch_size_pred (int): Size of the batch for belief gradient estimation.
            train_continue (bool): Whether to load a previous model and continue training.
        """

        # Seed
        self.seed = seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Environment
        self.env = env
        self.env.action_space.seed(seed)
        self.obs_dim = get_space_dim(self.env.observation_space)
        self.act_dim = get_space_dim(self.env.action_space)
        self.state_dim = get_space_dim(self.env.state_space)
        self.use_belief = use_belief

        # Actor-Critic Module
        self.ac = actor_critic(self.obs_dim, self.env.action_space, self.env.state_space, use_belief=use_belief, **ac_kwargs)

        # Value Function Optimizer
        self.vf_lr = vf_lr
        self.train_v_iters = train_v_iters
        self.vf_optimizer = Adam(self.ac.v.parameters(), lr=self.vf_lr)

        # Encoder Optimizer
        self.pretrain_epochs = pretrain_epochs
        self.epochs_belief_training = epochs_belief_training
        self.pretrain_steps = pretrain_steps
        self.train_enc_iters = train_enc_iters
        self.enc_lr = enc_lr
        self.maf_lr = maf_lr
        if use_belief:
            self.update_enc = self.update_enc_stoch
            self.enc_optimizer = Adam(self.ac.enc.encoder.parameters(), lr=self.enc_lr)
            self.maf_optimizer = Adam(self.ac.enc.maf_proba.parameters(), lr=self.maf_lr)
        else:
            self.update_enc = self.update_enc_deter
            self.enc_optimizer = Adam(self.ac.enc.parameters(), lr=self.enc_lr)
        if enc_loss == 'mse':
            self.ENCLoss = MSELoss(reduction='mean')
        elif enc_loss == 'mae':
            self.ENCLoss = L1Loss(reduction='mean')

        # Count variables
        var_counts = tuple(Core.count_vars(module) for module in [self.ac.pi, self.ac.v, self.ac.enc])
        prLightPurple('Number of parameters: \t pi: %d, \t v: %d, \t enc: %d\n' % var_counts)

        # Set up experience buffer
        self.gamma = gamma
        self.lam = lam
        self.steps_per_epoch = steps_per_epoch
        if self.env.stochastic_delays:
            self.buf = GAEBufferStoch(self.obs_dim, self.state_dim, self.env.action_space.shape, self.act_dim, size_pred_buf, batch_size_pred,
                                      self.steps_per_epoch, self.gamma, self.lam)
        else:
            # action is set to 1, ok for all tested envs, otherwise problem with discrete action spaces size
            self.buf = GAEBufferDeter(self.obs_dim, self.state_dim, self.env.action_space.shape, self.act_dim, size_pred_buf, batch_size_pred,
                                      self.steps_per_epoch, self.gamma, self.lam)

        # Other TRPO Parameters
        self.epochs = epochs
        self.max_ep_len = max_ep_len
        self.delta = delta
        self.damping_coeff = damping_coeff
        self.cg_iters = cg_iters
        self.backtrack_iters = backtrack_iters
        self.backtrack_coeff = backtrack_coeff
        self.train_continue = train_continue

        # Results and Save Variables
        self.save_dir = save_dir
        self.save_period = save_period
        self.epoch = 0
        self.elapsed_time = timedelta(0)
        self.timings = []
        self.avg_reward = []
        self.std_reward = []
        self.avg_length = []
        self.enc_losses = []
        self.v_losses = []

        if self.use_belief:
            self.compute_loss_enc = self.compute_loss_enc_stoch
        else:
            self.compute_loss_enc = self.compute_loss_enc_deter

        self.log_dir = os.path.join(self.save_dir,'log')

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

    def compute_loss_enc_deter(self, data):
        """Compute the loss of the belief module for deterministic env.
        """
        obs, states, mask = data['extended_states'], data['hidden_states'], data['mask']
        preds = self.ac.enc.predict(obs)
        return self.ENCLoss(preds[mask], states)

    def compute_loss_enc_stoch(self, data):
        """Compute the loss of the belief stochastic for deterministic env.
        """
        obs, states, mask = data['extended_states'], data['hidden_states'], data['mask']
        u, log_probs = self.ac.enc.log_probs(obs, states, torch.from_numpy(mask))
        if self.epoch % self.save_period == 0:
            self.save_noise(u)
            self.save_proba(log_probs)
            # self.save_belief(obs)
            # self.save_hidden_state(obs)
        return -log_probs.mean()

    def save_proba(self, log_probs):
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        ax.hist(torch.exp(log_probs).detach().numpy())
        plt.savefig(os.path.join(self.save_dir,str(self.epoch)+'_proba.png'))
        plt.close(fig)

    def save_hidden_state(self, obs):
        num_samples = min(obs.size(0), 100)
        with torch.no_grad():
            obs = self.ac.enc(obs).detach()
        obs = obs[:num_samples]
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        ax.hist(obs.detach().numpy())
        plt.savefig(os.path.join(self.save_dir, str(self.epoch)+ '_hidden_state.png'))
        plt.close(fig)

    def save_belief(self, obs):
        num_samples = min(obs.size(0),100)
        with torch.no_grad():
                cond = self.ac.enc.get_cond(obs).detach()
        cond = cond[:num_samples]
        samples = self.ac.enc.maf_proba.sample(num_samples=num_samples, cond_inputs=cond)
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        ax.hist(samples[:, -1, :].detach().numpy())
        plt.savefig(os.path.join(self.save_dir ,str(self.epoch)+'_belief.png'))
        plt.close(fig)

    def save_noise(self, u):
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        u = torch.cat((u, torch.normal(torch.zeros(u.size(0))).reshape(-1, 1)), 1)
        ax.hist(u.detach().numpy(), range=(-4, 4))
        plt.savefig(os.path.join(self.save_dir, str(self.epoch)+ '_noise.png'))
        plt.close(fig)

    def save_obs_density(self, obs):
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        ax.hist(obs.detach().numpy())
        plt.savefig(os.path.join(self.save_dir, str(self.epoch)+ '_encoded_state.png'))
        plt.close(fig)

    def compute_kl(self, data, old_pi):
        obs, act = data['obs'], data['act']
        pi, _ = self.ac.pi(obs, act)
        kl_loss = torch.distributions.kl_divergence(pi, old_pi).mean()
        return kl_loss

    # @torch.no_grad()
    # def compute_kl_reg_for_belief(self, data, old_pi):
    #     obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
    #     # Policy loss
    #     pi, logp = self.ac.pi(obs, act)
    #     kl_loss = torch.distributions.kl_divergence(pi, old_pi).mean()
    #     return loss_pi, kl_loss

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

    def update(self, pretrain=False, stop_belief_training=False):
        # If Pretraining then optimize only the Encoder
        if pretrain:
            # Encoder Update
            enc_loss = self.update_enc()
            self.enc_losses.append(enc_loss)
            # Value Function Loss Compatibility
            self.v_losses.append(np.nan)
            self.buf.reset()
        # Stop training belief module
        elif stop_belief_training:
            self.enc_losses.append(np.nan)
            data = self.buf.get()
            obs = data['obs']
            with torch.no_grad():
                obs = self.ac.enc(obs).detach()
            data['obs'] = obs
            # Policy Update
            self.update_pi(data)
            # Value Function Update
            v_loss = self.update_v(data)
            self.v_losses.append(v_loss)
        # Else optimize Policy and Value Function
        else:
            enc_loss = self.update_enc()
            self.enc_losses.append(enc_loss)
            # Extract Encoder Prediction once for all the Networks that needs it
            data = self.buf.get()
            obs = data['obs']
            with torch.no_grad():
                obs = self.ac.enc(obs).detach()
            data['obs'] = obs
            # Policy Update
            self.update_pi(data)
            # Value Function Update
            v_loss = self.update_v(data)
            self.v_losses.append(v_loss)

    def update_pi(self, data):
        # Compute old pi distribution
        obs, act = data['obs'], data['act']
        with torch.no_grad():
            old_pi, _ = self.ac.pi(obs, act)

        pi_loss = self.compute_loss_pi(data)
        pi_l_old = pi_loss.item()

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
                prRed('\tLine search failed! Keeping old params.')
                kl, pi_l_new = set_and_eval(step=0.)

    def update_enc_deter(self):
        # Encoder Learning
        loss_enc = None
        self.ac.enc.train()
        for i in range(self.train_enc_iters):
            self.enc_optimizer.zero_grad()
            data = self.buf.get_pred_data()
            loss_enc = self.compute_loss_enc(data)
            loss_enc.backward(retain_graph=True)
            self.enc_optimizer.step()
        self.ac.enc.eval()
        return loss_enc.detach().item()

    def update_enc_stoch(self):
        # Encoder Learning
        loss_enc = None
        self.ac.enc.train()
        for i in range(self.train_enc_iters):
            self.enc_optimizer.zero_grad()
            self.maf_optimizer.zero_grad()
            data = self.buf.get_pred_data()
            loss_enc = self.compute_loss_enc(data)
            # kl_reg = compute_kl_reg_for_belief(self, data, old_pi)
            # if i == self.train_enc_iters-1:
            loss_enc.backward(retain_graph=True)
            self.enc_optimizer.step()
            self.maf_optimizer.step()
            # else:
            #     loss_enc.backward()
            #     self.enc_optimizer.step()
                # self.maf_optimizer.step()
        self.ac.enc.eval()
        # for p in self.ac.enc.maf_proba.parameters():
        #             p.data.clamp_(-1,1)
        return loss_enc.detach().item()

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
        if isinstance(self.env.action_space, Discrete):
            # Create one hot encoding of the actions contained in the state
            temp_o = torch.tensor([i % self.act_dim == o[1][i // self.act_dim]
                                   for i in range(self.act_dim * len(o[1]))]).float()

            o = torch.cat((torch.tensor(o[0]), temp_o.reshape(-1)))
        else:
            o = torch.cat((torch.tensor(o[0]), torch.tensor(o[1].astype(float)).reshape(-1)))
        return o

    def train(self):
        # Load previous training final data in order to continue from there
        if self.train_continue:
            self.load_session()

        # Prepare for interaction with environment for the first Episode:
        # Start recording timings, reset the environment to get s_0
        start_time = dt.now()
        o, ep_ret, ep_len = self.env.reset(), 0, 0
        o = self.format_o(o)

        # Set the Variable that will stop Belief Module Training to False
        stop_belief_training = False

        # ---- TRAINING LOOP ----
        for epoch in range(1, self.epochs + 1):
            self.epoch = epoch

            # If Pre-Training of Belief Module is (still) active:
            if self.epoch < self.pretrain_epochs:
                pretrain = True
                max_epoch_steps = self.pretrain_steps
            else:
                pretrain = False
                max_epoch_steps = self.steps_per_epoch

            # If Belief Module completed its training:
            if self.epoch > self.epochs_belief_training:
                stop_belief_training = True

            # Reset Episodes variables
            ep_rewards = []
            ep_lengths = []
            episode = 0

            for t in range(max_epoch_steps):
                # Select a new action
                a, v, logp = self.ac.step(torch.as_tensor(o, dtype=torch.float32).unsqueeze(dim=0))

                # Execute the action
                next_o, r, d, info = self.env.step(a.reshape(-1))
                next_o = self.format_o(next_o)
                ep_ret += np.sum(r)
                ep_len += 1

                # Save the visited transition in to the Buffer
                self.buf.store(o, a, np.sum(r), v, not d, info, logp, episode, pretrain=pretrain)
                o = next_o

                # Is the episode terminated and why?
                timeout = ep_len == self.max_ep_len
                terminal = d or timeout
                epoch_ended = t == self.steps_per_epoch - 1

                if terminal or epoch_ended:
                    # If Epoch ended before Episode could end
                    if epoch_ended and not terminal:
                        prGreen('\tWarning: trajectory cut off by epoch at %d steps.' % ep_len)
                    # If Episode didn't reach terminal state, bootstrap value target of the reached state
                    if timeout or epoch_ended:
                        _, v, _ = self.ac.step(torch.as_tensor(o, dtype=torch.float32).unsqueeze(dim=0))
                    # If the Trajectory ended by its own, set State Value to 0
                    else:
                        v = 0
                    # Let the Buffer adjust itself when an Episode has ended
                    self.buf.finish_path(v)
                    if terminal:
                        # Only print EpRet and EpLen if Episode finished (otherwise they're not indicative)
                        ep_rewards.append(ep_ret)
                        ep_lengths.append(ep_len)

                    # Prepare for interaction with environment for the next episode:
                    # Start recording timings, reset the environment to get s_0
                    o, ep_ret, ep_len = self.env.reset(), 0, 0
                    o = self.format_o(o)

                    # Update Episode counting variable
                    episode += 1

            # Perform TRPO update at the end of the Epoch
            self.update(pretrain=pretrain, stop_belief_training=stop_belief_training)

            # Record Timings
            self.elapsed_time = dt.now() - start_time

            # Gather Epoch results and print them
            self.avg_reward.append(np.average(ep_rewards))
            self.std_reward.append(np.std(ep_rewards))
            self.avg_length.append(np.average(ep_lengths))
            self.timings.append(self.elapsed_time)
            self.print_update()

            # Save Epoch Results each "save_period" epochs
            if epoch % self.save_period == 0:
                self.save_session()

            # Plot all the data of this Epoch
            self.save_results()

    def print_update(self):
        update_message = '[EPOCH]: {0}\t[AVG. REWARD]: {1:.4f}\t[ENC. LOSS]: {2:.4f}\t[V LOSS]: {3:.4f}\t[ELAPSED TIME]: {4}'
        elapsed_time_str = ''.join(str(self.elapsed_time).split('.')[0])
        format_args = (self.epoch, self.avg_reward[-1], self.enc_losses[-1], self.v_losses[-1], elapsed_time_str)
        prYellow(update_message.format(*format_args))

    def save_session(self):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        save_path = os.path.join(self.save_dir, 'model_'+str(self.epoch)+'.pt')

        ckpt = {'policy_state_dict': self.ac.pi.state_dict(),
                'value_state_dict': self.ac.v.state_dict(),
                'encoder_state_dict': self.ac.enc.state_dict(),
                'avg_reward': self.avg_reward,
                'std_reward': self.std_reward,
                'enc_losses': self.enc_losses,
                'v_losses': self.v_losses,
                'epoch': self.epoch,
                'timings': self.timings,
                'elapsed_time': self.elapsed_time
                }

        torch.save(ckpt, save_path)

    def load_session(self, epoch=None):
        if epoch is None:
            files_name = os.listdir(self.save_dir)
            files_name.remove('model_parameters.txt')
            models = list(filter(lambda n: "model" in n,  files_name))
            if len(models)==1:
                load_path = os.path.join(self.save_dir, 'model.pt')
            else: 
                temp = np.array([int(name.replace('model_','').replace('.pt','')) for name in iter(models)])
                load_path = os.path.join(self.save_dir, 'model_'+str(max(temp)-1)+'.pt')
        else:
            load_path = os.path.join(self.save_dir, 'model_'+str(epoch)+'.pt')
        
        ckpt = torch.load(load_path)

        self.ac.pi.load_state_dict(ckpt['policy_state_dict'])
        self.ac.v.load_state_dict(ckpt['value_state_dict'])
        self.ac.enc.load_state_dict(ckpt['encoder_state_dict'])
        self.avg_reward = ckpt['avg_reward']
        self.std_reward = ckpt['std_reward']
        self.enc_losses = ckpt['enc_losses']
        self.v_losses = ckpt['v_losses']
        self.epoch = ckpt['epoch']
        self.timings = ckpt['timings']
        self.elapsed_time = ckpt['elapsed_time']

    def save_results(self):
        x = range(0, self.epoch, 1)
        self.errorbar_plot(x, self.avg_reward, xlabel='Epochs', ylabel='Average Reward', filename='reward',
                           error=self.std_reward)
        self.scatter_plot(x, self.avg_reward, xlabel='Epochs', ylabel='Average Reward', filename='reward_scatter')
        self.scatter_plot(x, self.enc_losses, xlabel='Epochs', ylabel='Encoder Loss', filename='encloss_scatter')
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

    def test(self, test_episodes=10, max_steps=250, epoch=None, test=None):
        # Load the Model to train:
        self.load_session(epoch)

        # Set Episodes variables
        episode = 0
        reward = []

        # ---- TESTING LOOP ----
        while episode < test_episodes:
            # Policy Evaluation mode
            self.ac.pi.eval()

            # Reset the Environment for this Episode
            o = self.env.reset()
            o = self.format_o(o)

            # For each Step
            step = 0
            ep_ret = 0.0
            while step < max_steps:
                # Select an Action
                a, _, _ = self.ac.step(torch.as_tensor(o, dtype=torch.float32).unsqueeze(dim=0))
                # Execute the Action
                next_o, r, d, _ = self.env.step(a.reshape(-1))
                o = self.format_o(next_o)
                #self.env.render()
                ep_ret += np.sum(r)
                step += 1
                if d:
                    break

            # Update Episode counting variable
            episode += 1

            # Print Episode Result
            update_message = '[EPISODE]: {0}\t[EPISODE REWARD]: {1:.4f}'
            format_args = (episode, ep_ret)
            prYellow(update_message.format(*format_args))

            # Append Episode Reward
            reward.append(ep_ret)

        # Save test results
        save_path = None
        if test is not None:
            save_path = os.path.join(self.save_dir, 'test_result_' + test + '.pt')
        elif epoch is not None:
            save_path = os.path.join(self.save_dir, 'test_result_' + str(epoch) + '.pt')
        elif self.env.stochastic_delays:
            os.path.join(self.save_dir, 'test_result_' + str(self.env.delay.p) + '.pt')
        else:
            save_path = os.path.join(self.save_dir, 'test_result.pt')

        ckpt = {
            'seed': self.seed,
            'reward': reward
        }
        torch.save(ckpt, save_path)

        self.env.close()
