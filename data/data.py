'''
	Justin Chen

	6.19.17

	Boston University 
	Hariri Institute for Computing and 
    Computational Sciences & Engineering
'''
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.animation as animation
from matplotlib import style
from torch import utils, stack, LongTensor
from torchvision import datasets, transforms
import os, multiprocessing as mp


'''
Class for handling state of shared dataset across processes. The dataset will only be loaded once
and pulled into memory throughout training and each process will be able to point to this object
and get its own DataLoader instance so that the user only has to worry about defining the batch size
in train.Train.
'''
class SMPLData(object):
    def __init__(self, dataset, cuda=False, shares=0, index=0, logger=None):
        self.logger = logger
        self.dataset = dataset
        self.cuda = cuda
        self.shares = shares
        self.index = index
        self.train, self.test = self.select(dataset) #these should be iterators


    def select(self, dataset):
        if dataset.lower() == 'mnist':
            return self.load_mnist()
        elif dataset.lower() == 'emnist':
            return self.load_emnist()


    def load_data(self, batch_size, num_workers=mp.cpu_count(), pin_memory=True, shuffle=True):
        # kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory} if self.cuda else {}

        # train = utils.data.DataLoader(self.train, batch_size=batch_size, shuffle=shuffle, **kwargs)
        # test = utils.data.DataLoader(self.test, batch_size=batch_size, shuffle=shuffle, **kwargs)

        # return train, test
        pass


    '''
    Extract appropriate partition of data
    e.g. If the data is paritioned into 32 samples each, then peer.id=0 gets samples [0,31]
    '''
    def partition(self, data):
        size = int(len(data)/self.shares)
        self.start_indx = self.index * size
        share_size = size + 1 if self.start_indx + size + 1 == len(data) else size
        self.end_indx = self.start_indx + size

        samples = []
        labels = []

        for i in range(self.start_indx, self.end_indx):
            samples.append(data[i][0])
            labels.append(data[i][1])

        return stack(samples), LongTensor(labels)


    def load_emnist(self):
        train = datasets.EMNIST('../data/emnist', train=True, download=True,
                        transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))]))

        test = datasets.EMNIST('../data/emnist', train=False, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))]))

        return self.partition(train), self.partition(test)


    def load_mnist(self):
        train = datasets.MNIST('../data/mnist', train=True, download=True,
                        transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))]))

        test = datasets.MNIST('../data/mnist', train=False, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))]))

        return self.partition(train), self.partition(test)


def get_cmap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)


'''
Visualize a batch of MNIST digits
Input: samples (torch.FloatTensor) Tensor of batch of MNIST images
       targets (torch.FloatTensor) Tensor of batch of MNIST labels
'''
def visualize(samples, labels):
    # first index is sample in batch
    batch_size = len(samples)
    for s in range(batch_size):
        pixels = samples[s][0].numpy().reshape((28, 28))
        plt.title('Label: {label}'.format(label=label[s]))
        plt.imshow(pixels, cmap='gray')
        plt.show()


def visualize_2D(x_data, y_data=[], title='', x_label='', y_label=''):
    fig = plt.figure()
    plot = fig.add_subplot(111)

    if len(y_data) == 0:
        plot.plot(x_data)
    else:
        plot.plot(x_data, y_data)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def subplots_2D(num_plots, title, x_data_list=[], y_data_list=[], subplt_titles=[], x_label=[], y_label=[]):
    f, axarr = plt.subplots(num_plots, sharex=False)
    subplt_titles[0] = ' '.join([title, subplt_titles[0]])

    if len(x_data_list) == 0:
        for i, p in enumerate(axarr):
            p.set_title(subplt_titles[i])
            p.set_xlabel(x_label[i])
            p.set_ylabel(y_label[i])
            p.plot(y_data_list[i])
    else:
        for i, p in enumerate(axarr):
            p.set_title(subplt_titles[i])
            p.set_xlabel(x_label[i])
            p.set_ylabel(y_label[i])
            p.plot(x_data_list[i], y_data_list[i])

    plt.tight_layout()
    plt.show()
