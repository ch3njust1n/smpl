'''
	Justin Chen
	6.19.17

	Simultaneous Multi-Party Learning (SMPL)
	[sim-puh l]

	Hyper-parallel distributed training for deep neural networks

	Boston University 
	Hariri Institute for Computing and 
    Computational Sciences & Engineering
'''

from distributed.parameter_server import ParameterServer
from multiprocessing import Process, cpu_count
from random import random
import time, argparse, signal


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Default host address')
    parser.add_argument('--port', type=int, default=9888, help='Port number for GradientServer')
    parser.add_argument('--cuda', type=str2bool, default=False, help='Enables CUDA training')
    parser.add_argument('--clique', '-c', type=int, default=3, help='Clique size')
    parser.add_argument('--data', '-d', type=str, default='mnist', help='Data directory')
    parser.add_argument('--epochs', '-e', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--flush', '-f', type=str2bool, default=True, help='Clear all parameters from previous sessions')
    parser.add_argument('--local_parallel', '-l', type=verify_local_parallel, default='hogwild', 
                        help='Hogwild! or Divergent Exploration')
    parser.add_argument('--lr', '-r', type=int, default=2, help='Learning rate e.g i = 10^(-i)')
    parser.add_argument('--log_freq', type=int, default=100, help='Frequency for logging training')
    parser.add_argument('--max', '-m', default=cpu_count(), help='Maximum number of simultaneous cliques')
    parser.add_argument('--name', '-n', type=str, default='MNIST', help='Name of experiment')
    parser.add_argument('--party', '-p', type=str, default='party.json', help='Name of party configuration file.')
    parser.add_argument('--save', '-s', type=str, default='model/save', 
                        help='Directory to save trained model parameters to')
    parser.add_argument('--scale', type=int, default=16,
                        help='Power of ten for scaling gradients to control preserving accuracy. Max=16 floating point.')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for dev only!')
    parser.add_argument('--strategy', type=verify_hyperedge, default='rand', help='Clique formation strategy')
    parser.add_argument('--variety', type=int, default=1, 
                        help='Minimum number of new members required in order to enter into a new clique. \
                        Prevents perfectly overlapping with current sessions.')
    args = parser.parse_args()


    # Distribute data across peers


    # Launch parameter server
    try:
        start = time.time()
        ParameterServer(args).listen()

        signal.signal(signal.SIGINT, gs.force_stop)
        print 'time (second): ', time.time() - start
    except KeyboardInterrupt:
        pass


def str2bool(s):
    if s.lower() in ('y', 'yes', 't', 'true', '1'):
        return True
    elif s.lower() in ('n', 'no', 'f', 'false', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')


def verify_local_parallel(strategy):
    s = strategy.lower()
    if s in ('hogwild', 'hogwild!'):
        return 'hogwild'
    elif s in ('dex', 'divex', 'divergent', 'divergentex'):
        return 'dex'
    else:
        raise argparse.ArgumentTypeError('Unsupported local parallelization technique: {}'.format(s))


def verify_hyperedge(strategy):
    s = strategy.lower()
    if s in ('random', 'rand', 'r'):
        return 'random'
    elif s in ('best', 'b'):
        return 'best'
    elif s in ('worst', 'w'):
        return 'worst'
    else:
        raise argparse.ArgumentTypeError('Invalid hyperedge formation strategy: {}'.format(s))


if __name__ == '__main__':
    main()