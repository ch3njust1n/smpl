'''
	Justin Chen
	3.26.18

	Module for abstracting trainer classes

	Boston University 
	Hariri Institute for Computing and 
    Computational Sciences & Engineering
'''
from torch import manual_seed
import torchvision.datasets as datasets
from torch.utils.data import TensorDataset, DataLoader
from multiprocessing import cpu_count
import json

class DistributedTrainer(object):
    def __init__(self, sess_id, network, allreduce, train_path, val_path, batch_size=1, 
    			 cache=None, cuda=False, parallel=False, seed=-1, drop_last=False, shuffle=True):

    	self.allreduce  = allreduce
    	self.batch_size = batch_size
        self.cache      = cache
        self.cuda		= cuda
        self.drop_last  = drop_last
        self.network    = network
        self.parallel   = parallel
        self.seed       = seed
        self.sess_id    = sess_id
        self.shuffle    = shuffle
        self.train_path = train_path
        self.total      = 0
        self.val_path   = val_path

        if self.seed != -1:
            manual_seed(self.seed)

        self.losses 	  = []
        self.validations  = []


    '''
    Load train and validation sets
    '''
    def load_data(self):
    	# Load data
    	self.train_loader = DataLoader(datasets.ImageFolder(self.train_path), batch_size=self.batch_size, 
    								   num_workers=cpu_count(), drop_last=self.drop_last, shuffle=self.shuffle)
    	self.val_loader   = DataLoader(datasets.ImageFolder(self.val_path), batch_size=self.batch_size, 
        							   num_workers=cpu_count(), drop_last=self.drop_last, shuffle=self.shuffle)
    	self.train_size   = len(self.train_loader)
    	self.val_size     = len(self.val_loader)


    '''
    Aggregate, average, and update gradients

    Output: Gradients averaged across peers in this hyperedge
    '''
    def allreduce(self):
        self.allreduce(self.sess_id, self.network, self.total)
        sess = json.loads(self.cache.get(self.sess_id))
        self.network.update_parameters(sess['parameters'], gradients=True)
     

    '''
    Save hyperedge training variables
    '''
    def cache(self):
    	sess = json.loads(self.cache.get(self.sess_id))
    	sess["parameters"] = [x.data.tolist() for x in self.network.parameters()]
    	sess["accuracy"] = self.validations[-1]
    	sess["accuracies"] = self.validations
    	sess["losses"] = self.losses
    	sess["train_size"] = self.train_size
    	sess["val_size"] = self.val_size

    	self.cache.set(self.sess_id, json.dumps(sess))
