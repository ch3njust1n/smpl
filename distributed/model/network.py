'''
	Justin Chen
	6.19.17

	Baseline feedfoward, convolutional, and recurrent networks

    Instructions: Define your network architecture here!
'''
# import sys, os
# sys.path.insert(1, os.path.join(sys.path[0], '..'))

from .. import utils
from .. import parameter_tools as pt
from torch import FloatTensor, LongTensor, zeros, stack, sparse, Size
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.multiprocessing import Process, cpu_count
import logging, os


class Network(nn.Module):
    def __init__(self, seed=0, log=None):
        super(Network, self).__init__()
        self.optimizer = optim.SGD
        self.log = log


    '''
    Gather either the network parameters or gradients into the specified format

    Input: tolist (bool, optional)
           reference (bool, optional)
           grads (bool, optional)
    Output: parameters/gradients (list)
    '''
    def _weights(self, tolist=False, reference=False, grads=False):
        if tolist:
            return [x.grad.data.tolist() if grads else x.data.tolist() for x in self.parameters()]
        else:
            if reference:
                # tensors in list are passed by reference
                return [x.grad.data if grads else x.data for x in self.parameters()]
            else:
                parameters = []

                for x in self.parameters():
                    params = x.grad.data if grads else x.data
                    parameters.append(zeros(params.size()).copy_(params))

                return parameters


    '''
    Wrapper function for getting network parameters

    Input: tolist (bool, optional)
           reference (bool, optional)
    Output: parameters (list)
    '''
    def get_parameters(self, tolist=False, reference=False):
        return self._weights(tolist, reference, grads=False)


    '''
    Wrapper function for getting network gradients

    Input: tolist (bool, optional)
           reference (bool, optional)
    Output: gradients (list)
    '''
    def get_gradients(self, tolist=False, reference=False):
        return self._weights(tolist, reference, grads=True)


    '''
    Updates this network's parameters. Assumes consistent architectures.

    Input: params    (list) - list of lists or a list of tensors
           gradients (bool, optional)
    '''
    def update_parameters(self, params, gradients=False):
        self.log.debug('update_parameters params:{}'.format(params))
        if type(params) != list:
            raise Exeception('InvalidTypeException: Expected a list of lists or a list of torch.FloatTensors')

        if len(params) > 0:
            if type(params[0]) == nn.parameter.Parameter:
                self.log.debug('UP0')
                for update, model in zip(params, self.parameters()):
                    if gradients:
                        model.grad.data = update.data
                    else:
                        model.data = update.data
            elif type(params[0]) == FloatTensor:
                self.log.debug('UP1')
                for update, model in zip(params, self.parameters()):
                    if gradients:
                        model.grad.data = update
                    else:
                        model.data = update
            elif type(params[0]) == list:
                self.log.debug('UP2')
                for update, model in zip(params, self.parameters()):
                    if gradients:
                        model.data = pt.list_to_tensor(update)
                    else:
                        model.data = pt.list_to_tensor(update)
            else:
                self.log.debug('UP Err')
                raise Exception('error: parameter type not supported <'+str(type(params))+'>')


    '''
    Returns network gradients

    Input:  tolist (bool, optional) If true, convert gradients to a nested list
            spares (bool) If true, sparsify gradients 
    Output: (list) List of gradients
    '''
    def gradients(self, tolist=False):
        if tolist:
            return [x.grad.data.tolist() for x in self.parameters()]
        else:  
            return [zeros(x.grad.data.size()).copy_(x.grad.data) for x in self.parameters()]


    '''
    Take the difference between the parameters in this network and a given network. 
    The given network should be an older version of this network.

    Input:  network   Parameters which can be in the form of a Network object, list of tensors, or nested
                      list of lists
            sparsify  (bool) If True, return sparse parameters with their corresponding coordinates as a nested 
                             list of lists
            tolist    (bool) If True, return parameters as a nested list of lists
    Output: gradients (list) A list of torch.FloatTensors representing the difference between the network parameters
    '''
    def multistep_grad(self, network, sparsify=False, tolist=False):
        # MAY NEED TO MAKE b IN EACH OF THESES CASES A TENSOR e.g. b.data-a instead of b-a
        self.log.debug('MSG network:{}\n ThisParam:{}'.format(network, str(self.get_parameters(reference=True))))
        
        gradients = []

        if isinstance(network, Network):
            self.log.debug('MSG 0')
            gradients = [b-a for (b, a) in zip(sself.get_parameters(reference=True), network.parameters())]
        elif isinstance(network, list) and len(network) > 0:
            if isinstance(network[0], FloatTensor):
                self.log.debug('MSG 1')
                gradients = [b-a for (b, a) in zip(self.get_parameters(reference=True), network)]
            elif isinstance(network[0], list):
                self.log.debug('MSG 2')
                gradients = [b-FloatTensor(a) for (b, a) in zip(self.get_parameters(reference=True), network)]
        else:
            self.log.debug('MSG 3')
            raise Exception('Error: {} type not supported'.format(type(network)))

        return pt.largest_k(gradients) if sparsify else [g.data.tolist() for g in gradients] if tolist else gradients


    '''
    Get a list of coordinate-gradient pairs
    Output: grads (list) List of coorindate-gradient pairs across all layers of the network
    '''
    def get_sparse_gradients(self):
        grads = []
        for x in self.parameters():
            grads.extend(pt.largest_k(x.grad.data))
        return grads


    '''
    This function adds gradients to specific coordinates in layer index.
    This should only be used after back-propagation.

    Input:  index (int) Integer index into the network.parameters() 
                    e.g. 0 
            coords (list) Nested list containing lists of coordinate gradient pairs
                    e.g. [[[0, 0, 0], -0.4013189971446991], [[0, 0, 1], 0.4981425702571869]]
            avg (int, optional)
    '''
    def add_coordinates(self, index, coords, avg=1):
        
        # get corresponding parameters
        params = [p for p in self.parameters()]
        p = params[index].data

        cd = []
        gd = []
        
        # extract coordinate-gradient pairs and combine gradients at the same coordiante
        for c in coords:
            point = c[0][1:]
            c[1] /= avg
            
            if point in cd:
                gd[cd.index(point)] += (c[1])
            else:
                cd.append(point)
                gd.append(c[1])

        # create coordinate/index tensor i, and value tensor v
        i = LongTensor(cd)
        v = FloatTensor(gd)
        s = list(p.size())

        # ensure that size has two coordinates e.g. prevent cases like (2L,)
        if len(s) == 1:
            s.append(1)

        # update parameters with gradients at particular coordinates
        grads = sparse.FloatTensor(i.t(), v, Size(s)).to_dense()
        # params[index].grad.data += grads # Commenting out for now. Refer to issue #48


    '''
    Update parameters across the network in parallel.
    This should only be used after back-propagation.

    TODO: 1.  Multiply by learning rate, and add to model 
              Refer to Large Scale Distributed Deep Networks, Dean et al, 2012

    Input: coords (list) Nested list of lists containing coordinate-gradient pairs from parameter_tools.largest_k()
            e.g. [[[0, 0, 0], 0.23776602745056152], [[0, 0, 1], -0.09021180123090744], [[1, 0, 0], 0.10222198069095612]]
           avg (int, optional)
    '''
    def add_batched_coordinates(self, coords, avg=1):
        self.log.debug('adding coords: {}'.format(coords))
        num_procs = cpu_count()
        self.share_memory()
        processes = []

        # sort and bin into layers
        params_coords = {}
        sorted_coords = sorted(coords, key=lambda x: x[0][0])

        for l in sorted_coords:
            self.log.debug('what is l? {}'.format(l))
            layer = l[0][0]
            self.log.debug('layer:{}'.format(layer))
            if layer[0] in params_coords:
                params_coords[layer[0]].append(l)
            else:
                params_coords[layer[0]] = l

        # update parameters in parallel
        for k in params_coords.keys():
            p = Process(target=self.add_coordinates, args=(k, params_coords[k], avg,))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()


'''
Network used for development only.
Train on MNIST
'''
class DevNet(Network):
    def __init__(self):
        super(DevNet, self).__init__()
        self.loss = F.nll_loss
        self.fc1 = nn.Linear(784, 10)
        self.fc2 = nn.Linear(10, 10)


    def forward(self, x):
        x = x.view(-1, 784)
        x = F.sigmoid(self.fc1(x))

        return F.log_softmax(self.fc2(x))


class DevNeuron(Network):
    def __init__(self, seed, log):
        super(DevNeuron, self).__init__(seed=seed, log=log)
        self.loss = F.nll_loss
        self.fc1 = nn.Linear(2, 1)


    def forward(self, x):
        pass