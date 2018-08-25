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
from torch import equal
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


    '''
    Implements one hyperedge training session

    Save the following outputs to the the ParameterServer cache
    network  (Network) Reference to model
    validate (list)    List of validation accuracies
    losses   (list)    List of training losses
    '''            
    def train(self):
        batch_idx = 0
        val = 0
        for ep in range(0, self.epochs):
            self.network.train()
            ep_loss = 0

            for batch_idx, (data, target) in enumerate(self.train_loader):
                batch_size = len(data)
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                loss = self.network.loss(self.network(data), target)
                loss.backward()
                ep_loss += loss.item()
                self.optimizer.step()

                if batch_idx % self.log_interval == 0:
                    self.log_epoch(self.pid, ep, loss, batch_idx, batch_size)

            self.validations.append(super(DevTrainer, self).validate())
            self.ep_losses.append(ep_loss/self.num_train_batches)

        # Must call to share train size and 
        # validation size with parameter server
        self.share()
