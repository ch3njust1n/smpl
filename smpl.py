#! /home/ubuntu/anaconda2/envs/smpl/bin/python

'''
	Justin Chen
	6.19.17

	Simultaneous Multi-Party Learning (SMPL)
	[sim-puh l]

	Hyper-parallel distributed training for deep neural networks
'''

from distributed.parameter_server import ParameterServer
from multiprocessing import Process, cpu_count
from random import random
import time, argparse, signal, os


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Default host address')
    parser.add_argument('--port', type=int, default=9888, help='Port number for GradientServer')
    parser.add_argument('--async_global', type=bool, default=True, help='Set for globally asynchronous training (default: True)')
    parser.add_argument('--async_mid', type=bool, default=True, help='Set for asynchronous training within hyperedges (default: True)')
    parser.add_argument('--async_local', type=bool, default=False, help='Set for asynchronous training on each peer (default: True)')
    parser.add_argument('--batch_size', type=int, default=16, help='Data batch size (default: 16)')
    parser.add_argument('--cuda', type=str2bool, default=False, help='Enables CUDA training (default: False)')
    parser.add_argument('--data', '-d', type=str, default='mnist', help='Data directory')
    parser.add_argument('--dev', '-v', type=str2bool, default=True, help='Development mode will fix random \
                        seed and keep session objects for analysis (default: True)')
    parser.add_argument('--drop_last', type=bool, default=False, help='True if last batch should be dropped if \
                        the dataset is not divisible by the batch size (default: False)')
    parser.add_argument('--ds_host', type=str, default='128.31.26.25', help='Data server host address')
    parser.add_argument('--ds_port', type=int, default=9888, help='Data server port')
    parser.add_argument('--epsilon', '-x', type=percent, default=0.3, help='Chance of selecting a \
                        random set model during parametere synchronization. (default: 0.3)')
    parser.add_argument('--eth', type=str, default='ens3', help='Peers\' ethernet interface (default: ens3)')
    parser.add_argument('--flush', '-f', type=str2bool, default=True, help='Clear all parameters from previous \
                        sessions')
    parser.add_argument('--hyperepochs', '-e', type=int, default=1, help='Total number of hyperepochs \
                        across all cliques for this peer (default: 1)')
    parser.add_argument('--local_parallel', '-l', type=local_parallel, default='hogwild!', 
                        help='Hogwild!, Divergent Exploration, or SGD (default: Hogwild!)')
    parser.add_argument('--learning_rate', '-lr', type=int, default=1e-3, help='Learning rate e.g i = 10^(-i)')
    parser.add_argument('--log_freq', type=int, default=100, help='Frequency for logging training')
    parser.add_argument('--name', '-n', type=str, default='MNIST', help='Name of experiment')
    parser.add_argument('--party', '-p', type=str, default='party.json', help='Name of party configuration file.')
    parser.add_argument('--regular', '-r', default=3, help='Maximum number of simultaneous hyperedges at \
                        any given time (default: 3)')
    parser.add_argument('--save', '-s', type=str, default='model/save', 
                        help='Directory to save trained model parameters to')
    parser.add_argument('--seed', type=int, default=-1, help='Random seed for dev only!')
    parser.add_argument('--shuffle', type=bool, default=True, help='True if data should be shuffled (default: True)')
    parser.add_argument('--sparsity', type=percent, default=0.5, help='Parameter sharing sparsification level (default: 0.0)')
    parser.add_argument('--uniform', '-u', type=edge_size, default=2, help='Hyperedge size (default: 2)')
    parser.add_argument('--variety', type=int, default=1, 
                        help='Minimum number of new members required in order to enter into a new clique. \
                        Prevents perfectly overlapping with current sessions.')
    args = parser.parse_args()

    # Launch parameter server
    try:
        start = time.time()
        ps = ParameterServer(args)

        signal.signal(signal.SIGINT, ps.force_stop)
        print('time (second): ', time.time() - start)
    except KeyboardInterrupt:
        pass


def str2bool(s):
    if s.lower() in ('y', 'yes', 't', 'true', '1'):
        return True
    elif s.lower() in ('n', 'no', 'f', 'false', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Error: Boolean value expected: {}'.format(s))


def percent(value):

    if value < 0 or value > 100:
        raise argparse.ArgumentTypeError('Error: Value must be in the range [0,100]')

    return value


def edge_size(size):

    if type(size) != int:
        raise argparse.ArgumentTypeError('Error: Clique size must be an integer: {}'.format(size))

    if size < 0:
        raise argparse.ArgumentTypeError('Error: Clique size must be non-negative: {}'.format(size))

    return size


def local_parallel(strategy):
    s = strategy.lower()
    if s in ('hogwild', 'hogwild!'):
        return 'hogwild'
    elif s in ('dex', 'divex', 'divergent', 'divergentex'):
        return 'dex'
    elif s in ('sgd', 'stochasticgd', 'vanilla'):
        return 'sgd'
    else:
        raise argparse.ArgumentTypeError('Error: Unsupported local parallelization technique: {}'.format(s))


def strategy(strategy):
    s = strategy.lower()
    if s in ('random', 'rand', 'r'):
        return 'random'
    elif s in ('best', 'b'):
        return 'best'
    elif s in ('worst', 'w'):
        return 'worst'
    else:
        raise argparse.ArgumentTypeError('Error: Invalid hyperedge formation strategy: {}'.format(s))


if __name__ == '__main__':
    main()