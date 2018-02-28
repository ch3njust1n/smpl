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
import matplotlib.animation as animation
from matplotlib import style
from torch import utils
from torchvision import datasets, transforms
import logging, os


'''
Class for handling state of shared dataset across processes. The dataset will only be loaded once
and pulled into memory throughout training and each process will be able to point to this object
and get its own DataLoader instance so that the user only has to worry about defining the batch size
in train.Train.
'''
class SMPLData(object):
    def __init__(self, cuda):
        logging.basicConfig(filename='gradient.log', level=logging.DEBUG)

        # Could implement a selection mechanism for datasets later
        self.train, self.test = self.load_mnist()


    def load_data(self, batch_size, num_workers=1, pin_memory=True, shuffle=True):
        kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory} if self.cuda else {}

        train = utils.data.DataLoader(self.train, batch_size=batch_size, shuffle=shuffle, **kwargs)
        test = utils.data.DataLoader(self.test, batch_size=batch_size, shuffle=shuffle, **kwargs)

        return train, test


    def load_mnist(self):
        train = datasets.MNIST('../data', train=True, download=True,
                        transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))]))

        test = datasets.MNIST('../data', train=False, 
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))]))

        return train, test


def get_cmap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)


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
