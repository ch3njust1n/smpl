'''
	Justin Chen
	6.19.17

	Baseline feedfoward, convolutional, and recurrent networks

    Instructions: Define your network architecture here!

	Boston University 
	Hariri Institute for Computing and 
    Computational Sciences & Engineering
'''
# import sys, os
# sys.path.insert(1, os.path.join(sys.path[0], '..'))

from .. import parameter_tools as pt
from torch import FloatTensor, LongTensor, zeros, stack, sparse, Size
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.multiprocessing import Process, cpu_count


class Network(nn.Module):
    def __init__(self):
        self.optimizer = optim.SGD


    '''
    Gather either the network parameters or gradients into the specified format

    Input: tolist (bool, optional)
           reference (bool, optional)
           grads (bool, optional)
    Output: parameters/gradients (list)
    '''
    def get_parameters(self, tolist=False, reference=False, grads=False):
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
    Updates this network's parameters. Assumes consistent architectures.

    Input: params    (list) - list of lists or a list of tensors
           gradients (bool, optional)
    '''
    def update_parameters(self, params, gradients=False):
        if type(params) != list:
            raise Exeception('InvalidTypeException: Expected a list of lists or a list of torch.FloatTensors')

        if len(params) > 0:
            if type(params[0]) == nn.parameter.Parameter:
                for update, model in zip(params, self.parameters()):
                    if gradients:
                        model.grad.data = update.data
                    else:
                        model.data = update.data
            elif type(params[0]) == FloatTensor:
                for update, model in zip(params, self.parameters()):
                    if gradients:
                        model.grad.data = update
                    else:
                        model.data = update
            elif type(params[0]) == list:
                for update, model in zip(params, self.parameters()):
                    if gradients:
                        model.data = pt.list_to_tensor(update)
                    else:
                        model.data = pt.list_to_tensor(update)
            else:
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

    Input:  network  Parameters, which can be in the form of a Network object, list of tensors, or nested
                     list of lists
    Output: gradient A list of tensors representing the difference between the network parameters
    '''
    def multistep_grad(self, network, sparsify=False):
        # MAY NEED TO MAKE b IN EACH OF THESES CASES A TENSOR e.g. b.data-a instead of b-a
        if isinstance(network, Network):
            return [b-a for (b, a) in zip(self.parameters(), network.parameters())]
        elif isinstance(network, list) and len(network) > 0:
            if isinstance(network[0], FloatTensor):
                return [b-a for (b, a) in zip(self.parameters(), network)]
            elif isinstance(network[0], list):
                return [b-FloatTensor(a) for (b, a) in zip(self.parameters(), network)]
        else:
            raise Exception('Error: {} type not supported'.format(type(network)))



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
        params[index].grad.data += grads


    '''
    Update parameters across the network in parallel.
    This should only be used after back-propagation.

    Input: coords (list) Nested list of lists containing coordinate-gradient pairs from parameter_tools.largest_k()
            e.g. [[[0, 0, 0], 0.23776602745056152], [[0, 0, 1], -0.09021180123090744], [[1, 0, 0], 0.10222198069095612]]
           avg (int, optional)
    '''
    def add_batched_coordinates(self, coords, avg=1):
        num_procs = cpu_count()
        self.share_memory()
        processes = []

        # sort and bin into layers
        params_coords = {}
        sorted_coords = sorted(coords, key=lambda x: x[0][0])

        for l in sorted_coords:
            layer = l[0][0]
            if layer in params_coords:
                params_coords[layer].append(l)
            else:
                params_coords[layer] = [l]

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