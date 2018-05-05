'''
	Justin Chen
	6.19.17

	Module for handling multi-party training
'''

import sys
sys.path.insert(0, 'data')

import os, utils
import datetime as dt
import parameter_tools as pt
import torch.optim as optim
from torch.autograd import Variable
from numpy import sum, zeros, unravel_index
from itertools import count
from random import random
from math import ceil
from trainer import DistributedTrainer, DevTrainer


# class Train(DistributedTrainer):
class Train(DevTrainer):

    '''
    Train is a user-defined class that describes the behavior of during hyperedge training.

    Input: config (dict) Dictionary containing training settings that's passed to the super class
    '''
    def __init__(self, config):
        super(Train, self).__init__(*config)

        # Training settings
        self.batch_size = 16
        self.epochs = 1
        self.log_interval = 100
        self.lr = 1
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
        
        for ep in range(0, self.epochs):
            self.network.train()
            ep_loss = 0

            for batch_idx, (data, target) in enumerate(self.train_loader):
                batch_size = len(data)
                data = Variable(pt.to_cuda(data, cuda=self.cuda))
                target = Variable(pt.to_cuda(target, cuda=self.cuda))

                self.optimizer.zero_grad()
                loss = self.network.loss(self.network(data), target)
                loss.backward()
                ep_loss += loss.data.tolist()[0]

                self.optimizer.step()
                self.validations.append(self.validate())
                break
            self.log(self.pid, ep, loss, batch_idx, batch_size)
            self.ep_losses.append(ep_loss/self.num_train_batches)
            
            # Must call to share train size and 
            # validation size with parameter server
            self.share()
