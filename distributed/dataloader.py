'''
Justin Chen
Custom Dataloader to handle arbitrary partitions of datasets and to define custom behavior each epoch
3.4.2018
'''

import random
from torch import rand, stack, LongTensor, manual_seed

class DistributedDataLoader(object):
    '''
    samples (list) list of sample tensors
    labels (list) list of label tensors
    '''
    def __init__(self, dataset, batch_size=1, shuffle=True):
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.length = len(self.dataset)
        self.batches = self.length/self.batch_size
        self.reset()


    '''
    Reset iterator index, shuffle dataset, and create mini-batches
    '''
    def reset(self):
        self.batched = []
        self.current = 0
        random.shuffle(self.dataset) if self.shuffle else None
        batch = []

        # create mini-batches of size self.batch_size
        for b in range(self.batches):
            start = b*self.batch_size
            end = start+self.batch_size
            batch = self.dataset[start:] if end+1 == self.length else self.dataset[start:end]
            self.batched.append(batch)


    def __iter__(self):
        return self


    '''
    Get next mini-batch
    '''
    def next(self):
        if self.current >= self.batches:
            self.reset()
            raise StopIteration
        else:
            batch = self.batched[self.current]
            self.current += 1
            return batch


manual_seed(2)
samples = [rand(1,2) for i in range(10)]
labels = [i for i in range(10)]
dataset = zip(samples, labels)

print 'dataset type: '+str(type(dataset))
ddl = DistributedDataLoader(dataset, batch_size=2, shuffle=True)
epochs = 3

for e in range(epochs):
    print 'epoch %d'%e
    for i, (sample, label) in enumerate(ddl):
        print '\tbatch %d' % i
        # print sample
        # print label
