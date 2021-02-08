import sys
import gym.spaces
import itertools
from collections import namedtuple
from utils.DQNBuffer import *
from utils.DQNCore import *
from utils.logger import Logger
from utils.various import *
import torch.nn.functional as F
import utils.plots as plots

# Adapted from:
# https://github.com/transedward/pytorch-dqn

# Check if cuda is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dqn_learning(env, env_id, q_func, q_fun_neurons, optimizer_spec, exploration=LinearSchedule(1000000, 0.1),
        stopping_criterion=None, replay_buffer_size=1000000, batch_size=32, gamma=0.99, learning_starts=10000,
        learning_freq=4, frame_history_len=4, target_update_freq=10000, double_dqn=False, dueling_dqn=False,
        save_dir='output', max_traj_steps=250, log_training=False):
    """ Train a Deep Q-learning Agent.
    Mnih, Volodymyr, et al. "Playing atari with deep reinforcement learning." arXiv preprint arXiv:1312.5602 (2013).

    Args:
        env (gym.Env): Env to train on.
        atari_env_id (string): If the env is in the Atari suite, name of the Atari game.
        q_func (torch.nn.Module): Function to approximate the Q-function.
        optimizer_spec (OptimizerSpec): Optimizer constructor and kwargs.
        exploration (function): Schedule for the threshold for epsilon greedy exploration.
        stopping_criterion (function): Checks whether the training should stop given the iteration.
        replay_buffer_size (int): Size of DQN's replay buffer.
        batch_size (int): Size of the batch in the update of DQN.
        gamma (float): Discount factor.
        learning_starts (int): Length of initial random exploration.
        learning_freq (int): Ratio of env interaction to DQN update.
        frame_history_len (int): Number of frames taken as input to DQN (if need be).
        target_update_freq (int): Ratio of Q network to target Q network updates.
        double_dqn (bool): Apply double DQN.
        dueling_dqn (bool): Apply dueling DQN.
        save_dir (float): Path to save directory.
        log_training (bool): Save log information during training.
    """

    assert type(env.observation_space['last_obs']) == gym.spaces.Box
    assert type(env.action_space) == gym.spaces.Discrete

    # Initialize the logger
    if log_training:
        logger = Logger(os.path.join(save_dir, 'logs'))

    # Access env dimensions
    num_actions = env.action_space.n
    obs_dim = get_space_dim(env.observation_space)
    
    # Set Q target and Q 
    Q = q_func(obs_dim, num_actions, q_fun_neurons, dueling=dueling_dqn).to(device)
    Q_target = q_func(obs_dim, num_actions, q_fun_neurons, dueling=dueling_dqn).to(device)

    # Initialize optimizers and loss function
    optimizer = optimizer_spec.constructor(Q.parameters(), **optimizer_spec.kwargs)
    loss_f = nn.MSELoss()

    # Create the replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len, num_actions, continuous_state=True)

    # Initialization of variables
    num_param_updates = 0
    mean_episode_reward      = -float('nan')
    best_mean_episode_reward = -float('inf')
    LOG_EVERY_N_STEPS = 1000
    SAVE_MODEL_EVERY_N_STEPS = 10000
    exploration_values = []
    avg_rewards = []
    std_rewards = []
    last_rewards = []

    # Initialization of the env
    last_obs = env.reset()
    # Create one hot encoding of the actions contained in the state
    temp_obs = torch.tensor([i % num_actions == last_obs[1][i // num_actions]
                                for i in range(num_actions * len(last_obs[1]))]).float()
    last_obs = np.concatenate((torch.tensor(last_obs[0]).float().reshape(-1), temp_obs))
    
    # ---- TRAINING LOOP ----
    for t in itertools.count():
        if stopping_criterion is not None and stopping_criterion(env, t):
            break

        # Store transition
        last_stored_frame_idx = replay_buffer.store_observation(last_obs)
        observations = replay_buffer.encode_recent_observation()

        # Q-learning action selection
        if t < learning_starts:
            action = np.random.randint(num_actions)
        else:
            # Epsilon greedy exploration
            sample = random.random()
            threshold = exploration.value(t)
            if sample > threshold:
                obs = torch.from_numpy(observations).unsqueeze(0).to(device)
                q_value_all_actions = Q(obs).cpu()
                action = ((q_value_all_actions).data.max(1)[1])[0]
            else:
                action = torch.IntTensor([[np.random.randint(num_actions)]])[0][0]

        # Env step
        obs, reward, done, info = env.step(action)
        temp_obs = torch.tensor([i % num_actions == obs[1][i // num_actions]
                                for i in range(num_actions * len(obs[1]))]).float()
        obs = np.concatenate((torch.tensor(obs[0]).float().reshape(-1), temp_obs))

        # Store effect of action
        replay_buffer.store_effect(last_stored_frame_idx, action, np.sum(reward), done)

        # Reset env if done
        if done:
            obs = env.reset()
            temp_obs = torch.tensor([i % num_actions == obs[1][i // num_actions]
                                for i in range(num_actions * len(obs[1]))]).float()
            obs = np.concatenate((torch.tensor(obs[0]).float().reshape(-1), temp_obs))

        # Update last_obs
        last_obs = obs

        # ---- Q UPDATE ----
        if (t > learning_starts and
                t % learning_freq == 0 and
                replay_buffer.can_sample(batch_size)):

            # sample transition batch from replay memory
            # done_mask = 1 if next state is end of episode
            obs_t, act_t, rew_t, obs_tp1, done_mask = replay_buffer.sample(batch_size)
            obs_t = torch.from_numpy(obs_t).to(device)
            act_t = torch.from_numpy(act_t).type(torch.int64).to(device)
            rew_t = torch.from_numpy(rew_t).to(device)
            obs_tp1 = torch.from_numpy(obs_tp1).to(device)
            done_mask = torch.from_numpy(done_mask).to(device)

            # input batches to networks
            # get the Q values for current observations (Q(s,a, theta_i))
            q_values = Q(obs_t)
            q_s_a = q_values.gather(1, act_t.unsqueeze(1))
            q_s_a = q_s_a.squeeze()

            # Double DQN
            if double_dqn:
                # Get q network's selected action
                q_tp1_values = Q(obs_tp1).detach()
                _, a_prime = q_tp1_values.max(1)

                # Compute target q network's q values
                q_target_tp1_values = Q_target(obs_tp1).detach()
                q_target_s_a_prime = q_target_tp1_values.gather(1, a_prime.unsqueeze(1))
                q_target_s_a_prime = q_target_s_a_prime.squeeze()

                # Compute Bellman error
                q_target_s_a_prime = (1 - done_mask) * q_target_s_a_prime 
                target = rew_t + gamma * q_target_s_a_prime
            else:
                # Compute q network's q value
                q_tp1_values = Q_target(obs_tp1).detach()
                q_s_a_prime, a_prime = q_tp1_values.max(1)

                # Compute Bellman error
                q_s_a_prime = (1 - done_mask) * q_s_a_prime 
                target = rew_t + gamma * q_s_a_prime

            # Compute the loss and backpropagate
            loss = F.smooth_l1_loss(target, q_s_a)
            optimizer.zero_grad()
            loss.backward()

            # Update q network
            optimizer.step()
            num_param_updates += 1

            # Update target q network
            if num_param_updates % target_update_freq == 0:
                Q_target.load_state_dict(Q.state_dict())

            # Add gradients to the logger
            if log_training and t % LOG_EVERY_N_STEPS == 0:
                for tag, value in Q.named_parameters():
                    tag = tag.replace('.', '/')
                    logger.histo_summary(tag, to_np(value), t+1)
                    logger.histo_summary(tag+'/grad', to_np(value.grad), t+1)

        # ---- SAVING ----
        # Save model epoch's parameters
        if t % SAVE_MODEL_EVERY_N_STEPS == 0:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            model_save_path = os.path.join(save_dir, 'model_'+str(t)+'.pt')
            torch.save(Q.state_dict(), model_save_path)

        # Compute model epoch's performances
        episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
        if len(episode_rewards) > 0:
            mean_episode_reward = np.mean(episode_rewards[-100:])
            std_episode_reward = np.std(episode_rewards[-100:])
            best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)

            exploration_values.append(exploration.value(t))
            avg_rewards.append(mean_episode_reward)
            std_rewards.append(std_episode_reward)
            last_rewards.append(episode_rewards[-1])

        # Save and print model epoch's performances
        if t % LOG_EVERY_N_STEPS == 0:
            print("---------------------------------")
            print("Timestep %d" % (t,))
            print("learning started? %d" % (t > learning_starts))
            print("mean reward (100 episodes) %f" % mean_episode_reward)
            print("best mean reward %f" % best_mean_episode_reward)
            print("episodes %d" % len(episode_rewards))
            print("exploration %f" % exploration.value(t))
            print("learning_rate %f" % optimizer_spec.kwargs['lr'])
            sys.stdout.flush()

            save_session(save_dir, exploration_values, avg_rewards, 
                    std_rewards, last_rewards, np.arange(t)>learning_starts)

            # TensorBoard logging
            if log_training:
                info = {
                    'learning_started': (t > learning_starts),
                    'num_episodes': len(episode_rewards),
                    'exploration': exploration.value(t),
                    'learning_rate': optimizer_spec.kwargs['lr'],
                }

                for tag, value in info.items():
                    logger.scalar_summary(tag, value, t+1)

                if len(episode_rewards) > 0:
                    info = {
                        'last_episode_rewards': episode_rewards[-1],
                    }

                    for tag, value in info.items():
                        logger.scalar_summary(tag, value, t+1)

                if (best_mean_episode_reward != -float('inf')):
                    info = {
                        'mean_episode_reward_last_100': mean_episode_reward,
                        'best_mean_episode_reward': best_mean_episode_reward
                    }

                    for tag, value in info.items():
                        logger.scalar_summary(tag, value, t+1)


def save_session(save_dir, exploration_values, avg_rewards, std_rewards, last_rewards, learning_started):
    """Saves performances of DQN as plots and torch.save.
    """
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    save_path = os.path.join(save_dir, 'stats.pt')

    ckpt = {
        'learning_started': learning_started,
        'last_rewards': last_rewards,
        'std_rewards': std_rewards,
        'avg_rewards': avg_rewards,
        'exploration_values': exploration_values
    }

    torch.save(ckpt, save_path)

    # plots
    # x = range(0, len(avg_rewards), 1)
    # plots.errorbar_plot(x, avg_rewards, xlabel='Epochs', ylabel='Average Reward', filename='reward', error=std_rewards,
    #               save_dir=save_dir)

    # plots
    x = range(0, len(avg_rewards), 1)
    plots.errorbar_plot(x, avg_rewards, xlabel='Epochs', ylabel='Average Reward', filename='reward',
                        error=std_rewards, save_dir=save_dir)


def dqn_testing(env, env_id, q_func, q_fun_neurons, batch_size=32, gamma=0.99, frame_history_len=4, 
            double_dqn=False, dueling_dqn=False, save_dir='output', test_episodes=10, max_steps=250, 
            epoch_to_load=None):
    """Tests a Deep Q-learning Agent over several episodes.

    Args:
        env (gym.Env): Env to train on.
        atari_env_id (string): If the env is in the Atari suite, name of the Atari game.
        q_func (torch.nn.Module): Function to approximate the Q-function.
        optimizer_spec (OptimizerSpec): Optimizer constructor and kwargs.
        exploration (function): Schedule for the threshold for epsilon greedy exploration.
        stopping_criterion (function): Checks whether the training should stop given the iteration.
        replay_buffer_size (int): Size of DQN's replay buffer.
        batch_size (int): Size of the batch in the update of DQN.
        gamma (float): Discount factor.
        learning_starts (int): Length of initial random exploration.
        learning_freq (int): Ratio of env interaction to DQN update.
        frame_history_len (int): Number of frames taken as input to DQN (if need be).
        target_update_freq (int): Ratio of Q network to target Q network updates.
        double_dqn (bool): Apply double DQN.
        dueling_dqn (bool): Apply dueling DQN.
        save_dir (float): Path to save directory.
        log_training (bool): Save log information during training.
        train_enc_iters (int): Ratio of encoder network to MAF network updates. 
        belief_module (dict): Parameters for the belief module.
        enc_lr (float): Learning rate of the encoder network.
        maf_lr (float): Learning rate of the MAF network.
        batch_size_pred (int): Size of the batch for belief module training.
        size_pred_buf (int): Size of the belief module's replay buffer.
    """

    # Access env dimensions
    num_actions = env.action_space.n
    obs_dim = get_space_dim(env.observation_space)
    
    # Load model parameters
    load_path = get_model_path(save_dir, epoch_to_load)
    ckpt = torch.load(load_path)

    # Set Q with saved parameters
    Q = q_func(obs_dim, num_actions, q_fun_neurons,
            dueling=dueling_dqn).to(device)
    Q.load_state_dict(ckpt)
    Q.eval()


    # Initialization of variables
    episode = 0
    reward = []


    while episode < test_episodes:
        o = env.reset()
        # Create one hot encoding of the actions contained in the state
        temp_o = torch.tensor([i % num_actions == o[1][i // num_actions]
                                for i in range(num_actions * len(o[1]))]).float()
        o = torch.cat((torch.tensor(o[0]).float().reshape(-1), temp_o))


        # Test trajectory
        step = 0
        ep_ret = 0.0

        # ---- TESTING LOOP ----
        while step < max_steps:
            
            # Select and play action
            q_values = Q(o.reshape(1,-1))
            a = ((q_values).data.max(1)[1])[0]
            next_o, r, d, _ = env.step(a)

            # Create one hot encoding of the actions contained in the state
            temp_o = torch.tensor([i % num_actions == next_o[1][i//num_actions]
                                    for i in range(num_actions*len(next_o[1]))]).float()
            next_o = torch.cat((torch.tensor(next_o[0]).float().reshape(-1), temp_o))

            # Render test
            env.render()

            # store outcome
            o = next_o
            ep_ret += np.sum(r)
            step += 1

            # Stop episode if done
            if d:
                break

        episode += 1

        # Print episode result
        update_message = '[EPISODE]: {0}\t[EPISODE REWARD]: {1:.4f}'
        format_args = (episode, ep_ret)
        prYellow(update_message.format(*format_args))

        # Append episode reward
        reward.append(ep_ret)

    # Save test results
    if epoch_to_load is not None: 
        save_path = os.path.join(save_dir, 'test_result_'+str(epoch_to_load)+'.pt')
    else:
        save_path = os.path.join(save_dir, 'test_result.pt')
    torch.save(reward, save_path)

    env.close()

