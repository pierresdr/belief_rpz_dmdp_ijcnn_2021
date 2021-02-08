import json, argparse
import gym#, gym_puddle
from importlib import import_module
from utils import TRPOCore as Core
from algorithm.trpo import TRPO
import torch.nn as nn
from utils.various import *
from utils.delays import DelayWrapper
from utils.stochastic_wrapper import StochActionWrapper


def launch_trpo(args, seed):
    # ---- ENV INITIALIZATION ----
    env = gym.make(args.env)
    env.seed(seed)

    # Add stochasticity wrapper
    if args.force_stoch_env:
        env = StochActionWrapper(env, distrib='Gaussian', param=args.stoch_mdp_param, seed=seed)

    # Add the delay wrapper
    env = DelayWrapper(env, delay=args.delay)

    # ---- TRAIN MODE ----
    if args.mode == 'train':
        # Create output folder and save training parameters
        save_dir = get_output_folder(args.save_dir, args.env)
        with open(os.path.join(save_dir, 'model_parameters.txt'), 'w') as text_file:
            json.dump(args.__dict__, text_file, indent=2)

        # Policy module parameters
        ac_kwargs = dict(
            pi_hidden_sizes=[args.pi_hid] * args.pi_l,
            v_hidden_sizes=[args.v_hid] * args.v_l,
            conv=args.convolutions,
            activation=eval(args.pi_activation)
        )

        trpo = TRPO(env, actor_critic=Core.MLPActorCritic, ac_kwargs=ac_kwargs, seed=seed,
                    steps_per_epoch=args.steps_per_epoch, epochs=args.epochs, gamma=args.gamma, delta=args.delta,
                    vf_lr=args.vf_lr, train_v_iters=args.v_iters, damping_coeff=args.damping_coeff,
                    cg_iters=args.cg_iters, backtrack_iters=args.backtrack_iters, backtrack_coeff=args.backtrack_coeff,
                    lam=args.lam, max_ep_len=args.max_ep_len, save_dir=save_dir, save_period=args.save_period,
                    memoryless=args.memoryless)

        trpo.train()

    # ---- TEST MODE ---- #
    elif args.mode == 'test':
        # Recover parameters of the trained model
        args.save_model = next(filter(lambda x: '.pt' in x, os.listdir(args.save_dir)))
        model_path = os.path.join(args.save_dir, args.save_model)
        load_parameters = os.path.join(args.save_dir, 'model_parameters.txt')
        with open(load_parameters) as text_file:
            file_args = json.load(text_file)

        # Policy module parameters
        ac_kwargs = dict(
            pi_hidden_sizes=[file_args['pi_hid']] * file_args['pi_l'],
            v_hidden_sizes=[file_args['v_hid']] * file_args['v_l'],
            conv=file_args['convolutions'],
            activation=eval(file_args['pi_activation'])
        )

        trpo = TRPO(env, actor_critic=Core.MLPActorCritic, ac_kwargs=ac_kwargs, seed=seed,
                    save_dir=args.save_dir, memoryless=file_args['memoryless'])

        trpo.test(test_epoch=args.test_epoch, test_episodes=args.test_episodes, max_steps=args.test_steps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trust Region Policy Optimization (PyTorch)')

    # General Arguments for Training and Testing TRPO
    parser.add_argument('--mode', default='train', type=str, choices=['train', 'test'])
    parser.add_argument('--env', default='Pendulum-v0', type=str)

    parser.add_argument('--delay', type=int, default=3, help='Number of Delay Steps for the Environment.')
    parser.add_argument('--seeds', '-s', nargs='+', type=int, default=0, help='Seed for Reproducibility purposes.')
    parser.add_argument('--curr_seed', type=int, default=0, help='Seed of the current run for parameter saving.')
    parser.add_argument('--train_render', action='store_true', help='Whether render the Env during training or not.')
    parser.add_argument('--train_render_ep', type=int, default=1, help='Which episodes render the env during training.')
    parser.add_argument('--force_stoch_env', action='store_true', help='Force the env to be stochastic.')
    parser.add_argument('--stoch_mdp_param', type=float, default=1, help='Depending on the stochasticity of the action, for Gaussian, param is the std.')
    parser.add_argument('--memoryless', action='store_true', help='Force Memoryless learning regardless of delay.')

    # Train Specific Arguments
    parser.add_argument('--steps_per_epoch', type=int, default=5000, help='Number of Steps per Epoch.')
    parser.add_argument('--epochs', type=int, default=500, help='Number of Epochs of Training.')
    parser.add_argument('--max_ep_len', type=int, default=250, help='Max Number of Steps per Episode')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount Factor.')
    parser.add_argument('--delta', type=float, default=0.001, help='TRPO Max KL Divergence.')

    # Test Specific Arguments
    parser.add_argument('--test_episodes', type=int, default=10, help='Number of Test Episodes.')
    parser.add_argument('--test_steps', type=int, default=250, help='Number of Steps per Test Episode.')
    parser.add_argument('--test_epoch', type=int, default=0, help='Epoch to be loaded for the Test.')

    # Value Function Specific Arguments
    parser.add_argument('--v_hid', type=int, default=64, help='Number of Neurons in each Hidden Layers.')
    parser.add_argument('--v_l', type=int, default=1, help='Number of Hidden Layers in each Network.')
    parser.add_argument('--vf_lr', type=float, default=0.01, help='Value Function Adam Learning Rate.')
    parser.add_argument('--v_iters', type=int, default=3, help='Value Function number of Iterations per Epoch.')

    # Convolution pre-processing
    parser.add_argument('--convolutions', action='store_true', help='Whether to pre-process input with a convolution.')

    # Policy Network Specific Arguments
    parser.add_argument('--pi_activation', default='nn.ReLU', type=str)
    parser.add_argument('--pi_hid', type=int, default=64, help='Number of Neurons in each Hidden Layers.')
    parser.add_argument('--pi_l', type=int, default=2, help='Number of Hidden Layers in each Network.')
    parser.add_argument('--damping_coeff', type=float, default=0.1, help='Numerical stability for Hessian Product')
    parser.add_argument('--cg_iters', type=int, default=10, help='CG Iterations for Hessian Product')
    parser.add_argument('--backtrack_iters', type=int, default=10, help='Max Backtracking Iterations for Line Search')
    parser.add_argument('--backtrack_coeff', type=float, default=0.8, help='Distance for each Backtracking Iteration')

    # Generalized Advantage Estimation Specific Arguments
    parser.add_argument('--lam', type=float, default=0.97, help='Lambda Coefficient for GAE.')

    # Folder Management Arguments
    parser.add_argument('--save_dir', default='./output/trpo', type=str, help='Output folder for the Trained Model')
    parser.add_argument('--save_period', type=int, default=1, help='Save models learned parameters every save_period.')
    args = parser.parse_args()

    for i in list(args.seeds):
        print('Launching Seed: ' + str(i))
        args.curr_seed = i
        launch_trpo(args, i)
