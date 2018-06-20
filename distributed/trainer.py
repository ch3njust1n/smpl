'''
	Justin Chen
	3.26.18

	Module for abstracting trainer classes
'''

from torchvision import transforms
from torch.autograd import Variable
import torchvision.datasets as datasets
from torch.utils.data import TensorDataset, DataLoader
from multiprocessing import current_process
import parameter_tools as pt
import os, logging, ujson, psutil


class Trainer(object):
    def __init__(self, network, data, batch_size=1, cuda=False, drop_last=False, shuffle=True, seed=-1, log=None):
        super(Trainer, self).__init__()
        self.batch_size        = batch_size
        self.cuda              = cuda
        self.data              = data
        self.drop_last         = drop_last
        self.ep_losses         = []
        self.epochs            = 1
        self.log_interval      = 1
        self.log               = log
        self.network           = network
        self.num_train_batches = 0
        self.num_val_batches   = 0
        self.seed              = seed
        self.shuffle           = shuffle
        self.train_loader      = None
        self.train_path        = os.path.join(data, 'train')
        self.train_size        = 0
        self.val_loader        = None
        self.val_path          = os.path.join(data, 'val')
        self.val_size          = 0
        self.validations       = []

        pt.to_cuda(self.network, cuda=self.cuda)


    '''
    Load train and validation sets. This function does not preprocess any data.
    All data is assumed to have already been preprocessed.
    '''
    def load_data(self):
        # Load data
        self.train_loader = DataLoader(datasets.ImageFolder(self.train_path), batch_size=self.batch_size, 
                                       num_workers=os.cpu_count(), drop_last=self.drop_last, shuffle=self.shuffle)
        self.val_loader   = DataLoader(datasets.ImageFolder(self.val_path), batch_size=self.batch_size, 
                                       num_workers=os.cpu_count(), drop_last=self.drop_last, shuffle=self.shuffle)

        self.num_train_batches = len(self.train_loader)
        self.num_val_batches   = len(self.val_loader)
        self.train_size        = self.num_train_batches*self.batch_size
        self.val_size          = self.num_val_batches*self.batch_size


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

        test_loss /= self.val_size
        acc = 100. * correct / self.val_size
        self.log.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, self.val_size, acc))

        return acc


    '''
    Log epoch information

    Input: pid        (int)    Process id
           ep         (int)    Current epoch
           loss       (tensor) Loss value of epoch ep
           batch_idx  (int)    Batch index
           batch_size (int)    Batch size
    '''
    def log_epoch(self, pid, ep, loss, batch_idx, batch_size):
        self.log.info('pid: {}\t cpu: {}%\tepoch: {} [{}/{} ({:.0f}%)]\tloss: {:.6f}'.format(
                      pid, psutil.cpu_percent(), ep, batch_idx * batch_size, self.train_size,
                      100. * batch_idx * batch_size / self.train_size, loss.data[0]))



class DistributedTrainer(Trainer):
    def __init__(self, sess_id, cache, network, data, batch_size, cuda=False, drop_last=False, shuffle=True, seed=-1, log=None):
        super(DistributedTrainer, self).__init__(network, data, batch_size=batch_size, cuda=cuda, 
                                                 drop_last=drop_last, shuffle=shuffle, seed=seed, log=log)

        self.pid     = current_process().pid
        self.sess_id = sess_id
        self.cache   = cache


    '''
    cache train set and validation set sizes, and final validation accuracy with parameter server
    '''
    def share(self):
        self.log.debug('train.share() sess_id:{}'.format(self.sess_id))
        sess = ujson.loads(self.cache.get(self.sess_id))
        sess['ep_losses'] = self.ep_losses
        sess['train_size'] = self.train_size
        sess['val_size'] = self.val_size
        sess['accuracy'] = self.validations[-1]
        sess['train_batches'] = self.num_train_batches
        sess['val_batches'] = self.num_val_batches
        self.cache.set(self.sess_id, ujson.dumps(sess))


'''
Trainer for development only. Loads MNIST dataset on every worker.
'''
class DevTrainer(DistributedTrainer):
    def __init__(self, log, sess_id, cache, network, data, batch_size=1, cuda=False, drop_last=False, 
                 shuffle=True, seed=-1):
        super(DevTrainer, self).__init__(sess_id, cache, network, data, batch_size, cuda, drop_last, shuffle, seed, log)
        self.total_val   = 0
        self.total_train = 0


    def load_data(self):
        kwargs = {'num_workers': os.cpu_count(), 'pin_memory': True} if self.cuda else {}
        self.train_loader = DataLoader(datasets.MNIST('../data', train=True, download=True,
                                       transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))])),
                                       batch_size=self.batch_size, shuffle=True, **kwargs)
        self.val_loader = DataLoader(datasets.MNIST('../data', train=False, download=True,
                                       transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))])),
                                       batch_size=self.batch_size, shuffle=True, **kwargs)
        self.num_train_batches = len(self.train_loader)
        self.num_val_batches   = len(self.val_loader)
        self.train_size        = self.num_train_batches*self.batch_size
        self.val_size          = self.num_val_batches*self.batch_size
