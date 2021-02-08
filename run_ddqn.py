import json
import argparse
import torch.nn as nn
from importlib import import_module
from algorithm.ddqn import ddqn_learning, ddqn_testing
from utils.various import *
from utils.DDQNCore import *
from collections import namedtuple
import torch.optim as optim
from utils.delays import DelayWrapper
import gym_puddle 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trust Region Policy Optimization (PyTorch)')

    # General Arguments for Training and Testing TRPO
    parser.add_argument('--mode', default='test', type=str, choices=['train', 'test'])
    parser.add_argument('--atari_env', default='freeway', type=str, help='Name of the Atari env.')
    parser.add_argument('--env', default='PuddleWorld', type=str)
    parser.add_argument('--clip_reward', action='store_true', help='Clip to {-1,0,1} the env reward.')

    parser.add_argument('--seed', '-s', type=int, default=0, help='Seed for Reproducibility purposes.')
    parser.add_argument('--delay', type=int, default=3, help='Number of Delay Steps for the Environment.')
    parser.add_argument('--stochastic_delays', action='store_true', help='Use stochastic delays.')
    parser.add_argument('--max_delay', default=50, type=int, help='Maximum delay of the environment.')
    parser.add_argument('--delay_proba', type=float, default=0.7, help='Probability of observation for the delay process.')

   # Train Specific Arguments
    parser.add_argument('--log_training', action='store_true', help='Log performances during training.')

    # Test Specific Arguments
    parser.add_argument('--test_episodes', type=int, default=10, help='Number of Test Episodes.')
    parser.add_argument('--test_steps', type=int, default=250, help='Number of Steps per Test Episode.')
    parser.add_argument('--epoch_load', type=str, default=None, help='Epoch to load.')

    # DQN Arguments
    parser.add_argument('--max_timesteps', type=int, default=1000000, help='Maximum number of steps in the wrapped env.')
    parser.add_argument('--dueling_dqn', action='store_true', help='Use dueling dqn.')
    parser.add_argument('--double_dqn', action='store_true', help='Use double dqn.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--replay_buffer_size', type=int, default=1000000, help='Replay buffer size.')
    parser.add_argument('--frame_history_len', type=int, default=4, help='Frame history length.')
    parser.add_argument('--target_update_freq', type=int, default=10000, help='Traget network update frequency.')
    parser.add_argument('--gamma', type=float, default=0.99, help='Gamma in the q update.')
    parser.add_argument('--learning_freq', type=int, default=4, help='Learning frequency.')
    parser.add_argument('--learning_rate', type=float, default=0.00025, help='Learning rate for DQN s RMSProp.')
    parser.add_argument('--alpha', type=float, default=0.95, help='Alpha for DQN s RMSProp.')
    parser.add_argument('--epsilon', type=float, default=0.01, help='Epsilon for DQN s RMSProp.')
    parser.add_argument('--learning_starts', type=int, default=50000, help='Length of warm-up.')
    parser.add_argument('--max_traj_steps', type=int, default=250, help='Length of warm-up.')

    # DQN 
    parser.add_argument('--q_fun_neurons', type=int, default=128, help='Number of neurons per layer in the q function.')

    # Train Transformer Encoder Specific Arguments
    parser.add_argument('--pretrain_epochs', type=int, default=50, help='Epochs for Encoder Pre-Training.')
    parser.add_argument('--pretrain_steps', type=int, default=5000, help='Epochs for Encoder Pre-Training.')
    parser.add_argument('--train_enc_iters', type=int, default=1, help='Encoder Adam Optimizer iterations per epoch.')
    parser.add_argument('--size_pred_buf', type=int, default=1000000, help='Size of the prediction buffer.')
    parser.add_argument('--batch_size_pred', type=int, default=32, help='Batch size for the prediction training.')

    # Transformer Encoder Specific Arguments
    parser.add_argument('--enc_lr', type=float, default=5e-3, help='Encoder Adam Optimizer Learning Rate.')
    parser.add_argument('--maf_lr', type=float, default=5e-3, help='Encoder MAF Adam Optimizer Learning Rate.')
    parser.add_argument('--enc_dim', type=int, default=64, help='Encoder Dimension.')
    parser.add_argument('--enc_heads', type=int, default=2, help='Encoder heads for Multi-Attention.')
    parser.add_argument('--enc_l', type=int, default=1, help='Encoder number of layers.')
    parser.add_argument('--enc_ff', type=int, default=8, help='Encoder FeedForward layer dimension.')
    parser.add_argument('--enc_rescaling', action='store_true', help='Whether activate State Rescaling or not.')
    parser.add_argument('--enc_causal', action='store_true', help='Whether using a Causal Enc. or Standard Enc.')
    parser.add_argument('--enc_pred_to_pi', action='store_true', help='Whether feeding Pi with Prediction or Encoded State.')
    parser.add_argument('--only_last_belief', action='store_true', help='Learn only the last belief distribution.')

    # Masked Autoregressive Flow Specific Arguments
    parser.add_argument('--n_blocks_maf', default=5, type=int, help='Number of MAF Layers')
    parser.add_argument('--hidden_dim', default=8, type=int, help='Number of Encoder Layers')
    parser.add_argument('--hidden_dim_maf', default=16, type=int, help='Number of Encoder Layers')

    # Folder Management Arguments
    parser.add_argument('--save_dir', default='./output/ddqn', type=str, help='Output folder for the Trained Model')
    args = parser.parse_args()
    
    
   # ---- ENV INITIALIZATION ----
    env = gym.make(args.env + '-v0')
    if args.mode == 'train':
        env = wrap_env(env, args.seed, args.double_dqn, args.dueling_dqn, clip_reward=args.clip_reward, atari=args.env=='AtariDelayEnv')
    
    # Add the delay wrapper
    env = DelayWrapper(env, delay=args.delay, stochastic_delays=args.stochastic_delays, p_delay=args.delay_proba, max_delay=args.max_delay)

     # Check if the env is stochastic
    stoch_envs = ['PuddleWorld']
    if args.env in stoch_envs:
        stoch_MDP = True
    else: 
        stoch_MDP = False


    # ---- TRAINING SPEC ----
    # Stopping criterion for the training loop
    def stopping_criterion(env, t):
            # notice that here t is the number of steps of the wrapped env,
            # which is different from the number of steps in the underlying env
            return get_wrapper_by_name(env, "Monitor").get_total_steps() >= args.max_timesteps

    # Network parameters optimizer
    OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])
    optimizer = OptimizerSpec(
        constructor=optim.RMSprop,
        kwargs=dict(lr=args.learning_rate, alpha=args.alpha, eps=args.epsilon)
    )

    

    # ---- TRAIN MODE ---- #
    EXPLORATION_SCHEDULE = LinearSchedule(1000000, 0.1)
    if args.mode == 'train':
        # Create output folder and save training parameters
        args.save_dir = get_output_folder(os.path.join(args.save_dir, args.env+'-Results'), args.env)
        with open(os.path.join(args.save_dir, 'model_parameters.txt'), 'w') as text_file:
            json.dump(args.__dict__, text_file, indent=2)

        # Initialization of the belief module
        belief_module = dict(
                enc_dim=args.enc_dim,  enc_heads=args.enc_heads, enc_ff=args.enc_ff, enc_l=args.enc_l,
                enc_rescaling=args.enc_rescaling, enc_causal=args.enc_causal, pred_to_pi=args.enc_pred_to_pi,
                hidden_dim=args.hidden_dim, n_blocks_maf=args.n_blocks_maf, hidden_dim_maf=args.hidden_dim_maf,
                only_last_belief=args.only_last_belief, stoch_env=stoch_MDP
        )

        ddqn_learning(
            env=env,
            atari_env_id=args.atari_env,
            q_func=DDQN,
            q_fun_neurons=args.q_fun_neurons,
            optimizer_spec=optimizer,
            exploration=EXPLORATION_SCHEDULE,
            stopping_criterion=stopping_criterion,
            replay_buffer_size=args.replay_buffer_size,
            batch_size=args.batch_size,
            gamma=args.gamma,
            learning_starts=args.learning_starts,
            learning_freq=args.learning_freq,
            frame_history_len=args.frame_history_len,
            target_update_freq=args.target_update_freq,
            double_dqn=args.double_dqn,
            dueling_dqn=args.dueling_dqn,
            belief_module=belief_module,
            save_dir=args.save_dir,
            max_traj_steps=args.max_traj_steps,
            train_enc_iters=args.train_enc_iters,
            enc_lr=args.enc_lr,
            maf_lr=args.maf_lr,
            batch_size_pred=args.batch_size_pred,
            size_pred_buf=args.size_pred_buf
        )

    # ---- TEST MODE ---- #
    elif args.mode == 'test':
        # Recover parameters of the trained model
        args.save_model = next(filter(lambda x: '.pt' in x, os.listdir(args.save_dir)))
        model_path = os.path.join(args.save_dir, args.save_model)
        load_parameters = os.path.join(args.save_dir, 'model_parameters.txt')
        with open(load_parameters) as text_file:
            file_args = json.load(text_file)


        # Initialization of the belief module
        belief_module = dict(
                enc_dim=file_args['enc_dim'],  enc_heads=file_args['enc_heads'], enc_ff=file_args['enc_ff'], enc_l=file_args['enc_l'],
                enc_rescaling=file_args['enc_rescaling'], enc_causal=file_args['enc_causal'], pred_to_pi=file_args['enc_pred_to_pi'],
                hidden_dim=file_args['hidden_dim'], n_blocks_maf=file_args['n_blocks_maf'], hidden_dim_maf=file_args['hidden_dim_maf'],
                only_last_belief=file_args['only_last_belief'], stoch_env=stoch_MDP
        )

        ddqn_testing(
            env=env,
            atari_env_id=args.atari_env,
            q_func=DDQN,
            q_fun_neurons=file_args['q_fun_neurons'],
            batch_size=args.batch_size,
            frame_history_len=file_args['frame_history_len'],
            double_dqn=file_args['double_dqn'],
            dueling_dqn=file_args['dueling_dqn'],
            belief_module=belief_module,
            save_dir=args.save_dir,
            test_episodes=args.test_episodes,
            max_steps=args.test_steps
        )









