import json, argparse, os
import gym, gym_puddle
from algorithm.dqn import dqn_learning, dqn_testing
# from utils.various import *
from utils.DQNCore import DQN, wrap_env, LinearSchedule
from collections import namedtuple
import torch.optim as optim
from utils.delays import DelayWrapper
from utils.various import get_output_folder

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trust Region Policy Optimization (PyTorch)')

    # General Arguments for Training and Testing TRPO
    parser.add_argument('--mode', default='test', type=str, choices=['train', 'test'])
    parser.add_argument('--atari_env', default='freeway', type=str, help='Name of the Atari env, in case \
                                                                            the env is from the suite Atari.')
    parser.add_argument('--env', default='PuddleWorld', type=str)
    parser.add_argument('--clip_reward', action='store_true', help='Clip to {-1,0,1} the env reward.')
    parser.add_argument('--seed', '-s', type=int, default=0, help='Seed for Reproducibility purposes.')
    parser.add_argument('--delay', type=int, default=3, help='Number of Delay Steps for the Environment.')
    parser.add_argument('--stochastic_delays', action='store_true', help='Whether to use stochastic delays.')
    parser.add_argument('--max_delay', default=50, type=int, help='Maximum delay of the environment.')
    parser.add_argument('--delay_proba', type=float, default=0.7, help='Probability of observation for the \
                                                                            stochastic delay process.')

    # Train Specific Arguments
    parser.add_argument('--log_training', action='store_true', help='Log performances during training.')

    # Test Specific Arguments
    parser.add_argument('--test_episodes', type=int, default=10, help='Number of Test Episodes.')
    parser.add_argument('--test_steps', type=int, default=250, help='Number of Steps per Test Episode.')
    parser.add_argument('--epoch_load', type=str, default=None, help='Saved model\'s epoch to load.')

    # DQN Module Arguments
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

    # Folder Management Arguments
    parser.add_argument('--save_dir', default='./output/dqn', type=str, help='Output folder for the Trained Model')
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
        # The number of steps of the wrapped environment can differ from the unwrapped env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= args.max_timesteps

    # Network parameters optimizer
    OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])
    optimizer = OptimizerSpec(
        constructor=optim.RMSprop,
        kwargs=dict(lr=args.learning_rate, alpha=args.alpha, eps=args.epsilon)
    )

    # ---- TRAIN MODE ---- #
    if args.mode == 'train':
        # Create output folder and save training parameters
        args.save_dir = get_output_folder(os.path.join(args.save_dir, args.env+'-Results'), args.env)
        with open(os.path.join(args.save_dir, 'model_parameters.txt'), 'w') as text_file:
            json.dump(args.__dict__, text_file, indent=2)

        dqn_learning(
            env=env,
            env_id=args.atari_env,
            q_func=DQN,
            q_fun_neurons=args.q_fun_neurons,
            optimizer_spec=optimizer,
            exploration=LinearSchedule(1000000, 0.1),
            replay_buffer_size=args.replay_buffer_size,
            batch_size=args.batch_size,
            gamma=args.gamma,
            learning_starts=args.learning_starts,
            learning_freq=args.learning_freq,
            frame_history_len=args.frame_history_len,
            target_update_freq=args.target_update_freq,
            double_dqn=args.double_dqn,
            dueling_dqn=args.dueling_dqn,
            save_dir=args.save_dir,
            max_traj_steps=args.max_traj_steps,
            log_training=args.log_training,
        )

    # ---- TEST MODE ---- #
    elif args.mode == 'test':
        # Recover parameters of the trained model
        args.save_model = next(filter(lambda x: '.pt' in x, os.listdir(args.save_dir)))
        model_path = os.path.join(args.save_dir, args.save_model)
        load_parameters = os.path.join(args.save_dir, 'model_parameters.txt')
        with open(load_parameters) as text_file:
            file_args = json.load(text_file)

        dqn_testing(
            env=env,
            env_id=args.atari_env,
            q_func=DQN,
            q_fun_neurons=file_args['q_fun_neurons'],
            batch_size=args.batch_size,
            frame_history_len=file_args['frame_history_len'],
            double_dqn=file_args['double_dqn'],
            dueling_dqn=file_args['dueling_dqn'],
            save_dir=args.save_dir,
            test_episodes=args.test_episodes,
            max_steps=args.test_steps
        )









