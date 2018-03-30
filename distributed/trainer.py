'''
	Justin Chen
	3.26.18

	Module for abstracting trainer classes

	Boston University 
	Hariri Institute for Computing and 
    Computational Sciences & Engineering
'''

from torch import manual_seed
from torchvision import transforms
import torchvision.datasets as datasets
from torch.utils.data import TensorDataset, DataLoader
from multiprocessing import cpu_count, current_process
import parameter_tools as pt
import json, os, logging


class Trainer(object):
    def __init__(self, batch_size, cuda, data, drop_last, network, shuffle, seed=-1):

        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)
        handler = logging.FileHandler(os.path.join(os.getcwd(),'logs/train.log'), mode='w')
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        self.batch_size   = batch_size
        self.cuda         = cuda
        self.data         = data
        self.drop_last    = drop_last
        self.log_interval = 1
        self.losses       = []
        self.network      = network
        self.seed         = seed
        self.shuffle      = shuffle
        self.train_loader = None
        self.train_path   = os.path.join(data, 'train')
        self.train_size   = 0
        self.val_loader   = None
        self.val_path     = os.path.join(data, 'val')
        self.val_size     = 0
        self.validations  = []

        if self.seed != -1:
            manual_seed(self.seed)

        pt.to_cuda(self.network, cuda=self.cuda)


    '''
    Load train and validation sets. This function does not preprocess any data.
    All data is assumed to have already been preprocessed.
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


    '''
    Input: pid        (int)
           ep         (int)
           loss       (tensor)
           batch_idx  ()
           batch_size ()
    '''
    def log(self, pid, ep, loss, batch_idx, batch_size):
        if batch_idx % self.log_interval == 0:
            print('{}\tepoch: {} [{}/{} ({:.0f}%)]\tloss: {:.6f}'.format(
                  pid, ep, batch_idx * batch_size, self.train_size,
                  100. * batch_idx / self.train_size, loss.data[0]))



class DistributedTrainer(Trainer):
    def __init__(self, network, sess_id, data, batch_size=1, cuda=False, 
                 drop_last=False, shuffle=True, seed=-1):
        super(DistributedTrainer, self).__init__(batch_size, cuda, data, drop_last, 
                                                 network, shuffle, seed)

        self.pid         = current_process().pid
        self.sess_id     = sess_id
        self.total_val   = 0
        self.total_train = 0 


'''
Trainer for development only. Loads MNIST dataset on every worker.
'''
class DevTrainer(DistributedTrainer):
    def __init__(self, network, sess_id, data, batch_size=1, cuda=False, drop_last=False, shuffle=True, seed=-1):
        super(DevTrainer, self).__init__(network, sess_id, data, batch_size, cuda, drop_last, 
                                         shuffle, seed)


    def load_data(self):
        kwargs = {'num_workers': cpu_count(), 'pin_memory': True} if self.cuda else {}
        self.train_loader = DataLoader(datasets.MNIST('../data', train=True, download=True,
                                       transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))])),
                                       batch_size=self.batch_size, shuffle=True, **kwargs)
