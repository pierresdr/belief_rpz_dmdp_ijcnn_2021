from algorithm.sarsa import SARSA
from algorithm.dsarsa import DSARSA
from utils.various import *
from utils.delays import DelayWrapper
import os
import json
import gym


def launch_sarsa(args, seed):
    # Environment Initialization
    # ---- ENV INITIALIZATION ----
    env = gym.make(args.env + '-v0')
    env.seed(seed)

    # Set Environment episode Max Length
    if args.mode == 'train':
        env._max_episode_steps = args.max_ep_len
    else:
        env._max_episode_steps = args.test_steps

    # Add the delay wrapper
    env = DelayWrapper(env, delay=args.delay)

    # Method Initialization + Folder Initialization
    if args.dsarsa:
        sarsa = DSARSA
        if args.mode == 'train':
            args.save_dir = './output/dsarsa'
    else:
        sarsa = SARSA
        if args.mode == 'train':
            args.save_dir = './output/sarsa'

    # ---- TRAIN MODE ---- #
    if args.mode == 'train':
        args.save_dir = get_output_folder(os.path.join(args.save_dir, args.env + '-Results'), args.env)
        with open(os.path.join(args.save_dir, 'model_parameters.txt'), 'w') as text_file:
            json.dump(args.__dict__, text_file, indent=2)

        agent = sarsa(env, seed=seed, delay=args.delay, epochs=args.epochs, steps=args.steps_per_epoch,
                      max_steps=args.max_ep_len, lam=args.lam, gamma=args.gamma, lr=args.lr, e=args.e,
                      s_space=args.s_space, a_space=args.a_space, save_dir=args.save_dir,
                      train_render=args.train_render, train_render_ep=args.train_render_ep)

        agent.train()

    # ---- TEST MODE ---- #
    elif args.mode == 'test':
        args.save_model = next(filter(lambda x: '.pt' in x, os.listdir(args.save_dir)))
        model_path = os.path.join(args.save_dir, args.save_model)
        load_parameters = os.path.join(args.save_dir, 'model_parameters.txt')
        with open(load_parameters) as text_file:
            file_args = json.load(text_file)

        agent = sarsa(env, delay=args.delay, s_space=file_args['s_space'], a_space=file_args['a_space'], save_dir=args.save_dir)

        agent.test(test_episodes=args.test_episodes, max_steps=args.test_steps)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Trust Region Policy Optimization (PyTorch)')

    # General Arguments for Training and Testing TRPO
    parser.add_argument('--mode', default='train', type=str, choices=['train', 'test'])
    parser.add_argument('--env', default='Pendulum', type=str)

    parser.add_argument('--delay', type=int, default=30, help='Number of Delay Steps for the Environment.')
    parser.add_argument('--seeds', nargs='+', type=int, default=0, help='Seed for Reproducibility purposes.')
    parser.add_argument('--curr_seed', type=int, default=0, help='Seed of the current run for parameter saving.')
    parser.add_argument('--train_render', action='store_true', help='Whether render the Env during training or not.')
    parser.add_argument('--train_render_ep', type=int, default=1, help='Which episodes render the env during training.')

    # Train Specific Arguments
    parser.add_argument('--steps_per_epoch', type=int, default=12500, help='Number of Steps per Epoch.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of Epochs of Training.')
    parser.add_argument('--max_ep_len', type=int, default=2500, help='Max Number of Steps per Episode')

    # Test Specific Arguments
    parser.add_argument('--test_episodes', type=int, default=10, help='Number of Test Episodes.')
    parser.add_argument('--test_steps', type=int, default=250, help='Number of Steps per Test Episode.')

    # SARSA Specific Arguments
    parser.add_argument('--dsarsa',  action='store_true', help='Whether to use DSARSA or SARSA.')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount Factor.')
    parser.add_argument('--lam', type=float, default=0.9, help='Eligibility Traces Factor.')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning Rate.')
    parser.add_argument('--e', type=float, default=0.2, help='E-Greedy Policy Parameter.')

    # Discretization Specific Arguments
    parser.add_argument('--s_space', type=int, default=10, help='State Space Discretization Grid Dimension.')
    parser.add_argument('--a_space', type=int, default=3, help='Action Space Discretization.')

    # Folder Management Arguments
    parser.add_argument('--save_dir', default='./output/sarsa', type=str, help='Output folder for the Trained Model')
    args = parser.parse_args()

    for i in args.seeds:
        print('Launching Seed: ' + str(i))
        args.curr_seed = i
        launch_sarsa(args, i)
