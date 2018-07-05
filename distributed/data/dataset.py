'''
    Justin Chen
    7.3.18

    Module for loading data
'''
import sys, os
sys.path.insert(0, 'distributed')
from threading import Thread
from parameter_channel import ParameterChannel
from torchvision import transforms
import torchvision.datasets as datasets
from torch.utils.data import TensorDataset, DataLoader

class Dataset(object):
    def __init__(self, cuda, batch_size, dataset, host='', port=-1):
        self.batch_size = batch_size
        self.cuda = cuda
        self.host = host
        self.port = port
        self.mode = 1
        self.name = dataset

        if len(host) > 0 and port != -1:
            ds_config = {"alias": "dataserver", "host": self.host, "port": self.port, "id": 8}
            self.pc = ParameterChannel(ds_config)
            self.mode = 0

        self.load_devset()


    '''
    Load dataset
    Input: mode (int) If 0, request data from DataServer. Else 1, load local dataset
    '''
    def load_data(self):
        data = None
        if self.mode == 0:
            ok, train_data = self.pc.send(self.host, self.port, {"api": "get_trainset", 
                                          "args":[self.me['alias'], self.batch_size]})
            ok, val_data = self.pc.send(self.host, self.port, {"api": "get_valset", 
            												    "args":[self.me['alias'], self.batch_size]})
            ok, test_data = self.pc.send(self.host, self.port, {"api": "get_testset", 
            													   "args":[self.me['alias'], self.batch_size]})
        elif self.mode == 1:
        	# self.load_local_data()
            self.load_devset()
        else:
            raise Exception('Invalid training mode')


    '''
    Load MNIST dataset - mainly for benchmarking and development
    '''
    def load_devset(self, data_dir='../data'):
        kwargs = {'num_workers': os.cpu_count(), 'pin_memory': True} if self.cuda else {}
        self.train_loader = DataLoader(datasets.MNIST(data_dir, train=True, download=True,
                                       transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))])),
                                       batch_size=self.batch_size, shuffle=True, **kwargs)
        self.val_loader = DataLoader(datasets.MNIST(data_dir, train=False, download=True,
                                       transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))])),
                                       batch_size=self.batch_size, shuffle=True, **kwargs)

