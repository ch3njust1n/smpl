'''
	Justin Chen
	6.19.17

	Module for handling multi-party training

	Boston University 
	Hariri Institute for Computing and 
    Computational Sciences & Engineering
'''

import sys
sys.path.insert(0, 'data')

import os, torch, logging, utils, json
import datetime as dt
import parameter_tools as pt
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from numpy import sum, zeros, unravel_index
from itertools import count
from random import random
from math import ceil


torch.manual_seed(1)


class Train(object):

    def __init__(self, sess_id, network, epochs, lr, dataset, cache, parallel, average):
        # Init global parameter state variables
        self.network = network
        self.accuracy = 0.0
        self.cache = cache
        self.average = average

        # Training settings
        self.sess_id = sess_id
        self.mpc = False
        self.save = 'model/save'
        self.log_interval = 100
        self.epochs = epochs
        self.iterations = 3000
        self.batch_size = 64
        self.momentum = 0.9
        self.lr = lr
        self.sparsity = 0.25
        self.scale = 16
        self.optimizer = self.network.optimizer(self.network.parameters(), lr=self.lr)
        self.parallel = parallel

        # Load data
        self.train_loader, self.test_loader = dataset.load_data(self.batch_size)
        self.dataset_size = len(self.train_loader)

        logging.basicConfig(filename='gradient.log', level=logging.DEBUG)


    '''
    Output: network  (Network) - reference to model
            validate (list) - list of validation accuracies
            losses   (list) - list of training losses
    '''            
    def train(self):
        losses = []
        validations = []
        pid = 0

        pt.to_cuda(self.network, cuda=self.cuda)
        total = len(self.train_loader.dataset)
        sess = json.loads(self.cache.get(self.sess_id))

        for epoch in range(1, self.epochs + 1):
            pid = os.getpid()
            self.network.train()

            for batch_idx, (data, target) in enumerate(self.train_loader):
                batch_size = len(data)
                data = Variable(pt.to_cuda(data, cuda=self.cuda))
                target = Variable(pt.to_cuda(target, cuda=self.cuda))

                self.optimizer.zero_grad()
                loss = self.network.loss(self.network(data), target)
                loss.backward()
                losses.append(loss.data.tolist()[0])

                # Average gradients across peers
                average(self.sess_id, self.network)

                self.optimizer.step()

                if batch_idx % self.log_interval == 0:
                    print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        pid, epoch, batch_idx * batch_size, total,
                        100. * batch_idx / len(self.train_loader), loss.data[0]))

            self.accuracy = self.validate()
            validations.append(self.accuracy)

        # Divergent Exploration parallelization strategy
        if self.parallel == 'dex':           
            # should create a local best and just replace it if it's better than the existing one else 
            # exit and discard computation
            if sess['val'][-1] < validations[-1]:
                sess['parameters'] = self.network.get_parameters(tolist=True)
                sess['pid'] = pid
                sess['val'] = validations
                sess['losses'] = losses
                self.cache.set(self.sess_id, json.dumps(sess))


    '''
    Output: acc (float) - validation accuracy
    '''
    def validate(self):
        self.network.eval()
        test_loss = 0
        correct = 0
        acc_diff = 1.0

        for data, target in self.test_loader:
            data = pt.to_cuda(data, cuda=self.cuda)
            target = pt.to_cuda(target, cuda=self.cuda)
            data, target = Variable(data, volatile=True), Variable(target)
            output = self.network(data)
            test_loss += self.network.loss(output, target, size_average=False).data[0]

            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()

        total = len(self.test_loader) * self.batch_size if task == 'avg' else len(self.test_loader.dataset)
        test_loss /= total
        acc = 100. * correct / total
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, total, acc))

        return acc

