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

import os, utils, json
import datetime as dt
import parameter_tools as pt
import torch.optim as optim
from torch.autograd import Variable
from numpy import sum, zeros, unravel_index
from itertools import count
from random import random
from math import ceil
from trainer import DistributedTrainer


class Train(DistributedTrainer):

    '''
    Train is a user-defined class that describes the behavior of during hyperedge training.

    Input: config (dict) Dictionary containing training settings that's passed to the super class
    '''
    def __init__(self, config):
        super(Train, self).__init__(*config)

        # Training settings
        self.batch_size = 64
        self.epochs = 5
        self.log_interval = 100
        self.lr = 1e-3
        self.momentum = 0.9
        self.optimizer = self.network.optimizer(self.network.parameters(), lr=self.lr)
        self.save = 'model/save'
        self.load_data()


    '''
    Implements one hyperedge training session

    Save the following outputs to the the ParameterServer cache
    network  (Network) Reference to model
    validate (list)    List of validation accuracies
    losses   (list)    List of training losses
    '''            
    def train(self):
        pid = os.getpid()
        pt.to_cuda(self.network, cuda=self.cuda)

        for ep in range(0, self.epochs):
            self.network.train()

            for batch_idx, (data, target) in enumerate(self.train_loader):
                batch_size = len(data)
                self.total += batch_size
                data = Variable(pt.to_cuda(data, cuda=self.cuda))
                target = Variable(pt.to_cuda(target, cuda=self.cuda))

                self.optimizer.zero_grad()
                loss = self.network.loss(self.network(data), target)
                loss.backward()
                self.losses.append(loss.data.tolist()[0])

                if ep+1 % self.epoch == 0:
                    if batch_idx % self.log_interval == 0:
                        print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            pid, ep, batch_idx * batch_size, self.train_size,
                            100. * batch_idx / self.train_size, loss.data[0]))

                if ep == self.epochs-1 and batch_idx == self.train_size-1:
                    # Average gradients across peers
                    # and only take one gradient step per hyperedge
                    self.allreduce()
                    self.optimizer.step()

                self.validations.append(self.validate())

        # save hyperedge
        self.cache()


    '''
    Output: acc (float) - validation accuracy
    '''
    def validate(self):
        self.network.eval()
        test_loss = 0
        correct = 0
        acc_diff = 1.0

        for data, target in self.val_loader:
            data = pt.to_cuda(data, cuda=self.cuda)
            target = pt.to_cuda(target, cuda=self.cuda)
            data, target = Variable(data, volatile=True), Variable(target)
            output = self.network(data)
            test_loss += self.network.loss(output, target, size_average=False).data[0]

            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()

        total = self.val_size * self.batch_size if task == 'avg' else self.val_size
        test_loss /= total
        acc = 100. * correct / total
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, total, acc))

        return acc

