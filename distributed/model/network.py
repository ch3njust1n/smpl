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
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp


class Network(nn.Module):
    def __init__(self):
        self.network = None


    '''
    Gather either the network parameters or gradients into the specified format

    Input: tolist (bool), reference (bool), grads (bool)
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
                    parameters.append(torch.zeros(params.size()).copy_(params))

                return parameters


    '''
    Assumes consistent architectures
    Input: params (list) - list of lists or a list of tensors
    '''
    def update_parameters(self, params, gradients=False):
        if type(params) != list:
            raise Exeception('InvalidTypeException: Expected a list of lists or a list of torch.FloatTensors')

        if len(params) > 0:
            if type(params[0]) == torch.nn.parameter.Parameter:
                for update, model in zip(params, self.parameters()):
                    if gradients:
                        model.grad.data = update.data
                    else:
                        model.data = update.data
            elif type(params[0]) == torch.FloatTensor:
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

    Input: tolist (bool) If true, convert gradients to a nested list
           spares (bool) If true, sparsify gradients 
    Output: (list) List of gradients
    '''
    def get_gradients(self, tolist=False):
        if tolist:
            return [x.grad.data.tolist() for x in self.parameters()]
        else:  
            return [torch.zeros(x.grad.data.size()).copy_(x.grad.data) for x in self.parameters()]


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
        i = torch.LongTensor(cd)
        v = torch.FloatTensor(gd)
        s = list(p.size())

        # ensure that size has two coordinates e.g. prevent cases like (2L,)
        if len(s) == 1:
            s.append(1)

        # update parameters with gradients at particular coordinates
        grads = torch.sparse.FloatTensor(i.t(), v, torch.Size(s)).to_dense()
        params[index].grad.data += grads


    '''
    Update parameters across the network in parallel.
    This should only be used after back-propagation.

    Input: coords (list) Nested list of lists containing coordinate-gradient pairs from parameter_tools.largest_k()
            e.g. [[[0, 0, 0], 0.23776602745056152], [[0, 0, 1], -0.09021180123090744], [[1, 0, 0], 0.10222198069095612]]
    '''
    def add_batched_coordinates(self, coords, avg=1):
        num_procs = mp.cpu_count()
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
            p = mp.Process(target=self.add_coordinates, args=(k, params_coords[k], avg,))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()


class TestNet(Network):
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(784,200)
        self.fc2 = nn.Linear(200,47)


class NeuralNetwork(Network):
    def __init__(self):
        super(Network, self).__init__()
        prob_drop = 0.2
        self.loss = F.nll_loss
        self.optimizer = optim.Adam
        self.fc1 = nn.Linear(784, 200)
        torch.nn.init.normal(self.fc1.weight)
        torch.nn.init.normal(self.fc1.bias)
        self.do1 = nn.Dropout(p=prob_drop)
        self.fc2 = nn.Linear(200, 100)
        torch.nn.init.normal(self.fc2.weight)
        torch.nn.init.normal(self.fc2.bias)
        self.do2 = nn.Dropout(p=prob_drop)
        self.fc3 = nn.Linear(100, 60)
        torch.nn.init.normal(self.fc3.weight)
        torch.nn.init.normal(self.fc3.bias)
        self.do3 = nn.Dropout(p=prob_drop)
        self.fc4 = nn.Linear(60, 30)
        torch.nn.init.normal(self.fc4.weight)
        torch.nn.init.normal(self.fc4.bias)
        self.do4 = nn.Dropout(p=prob_drop)
        self.fc5 = nn.Linear(30, 10)
        torch.nn.init.normal(self.fc5.weight)
        torch.nn.init.normal(self.fc5.bias)


    def forward(self, x):
        x = x.view(-1, 784)
        x = self.do1(F.sigmoid(self.fc1(x)))
        x = self.do2(F.sigmoid(self.fc2(x)))
        x = self.do3(F.sigmoid(self.fc3(x)))
        x = self.do4(F.sigmoid(self.fc4(x)))

        return F.log_softmax(self.fc5(x))


class Convolution(nn.Module):
    def __init__(self):
        super(Convolution, self).__init__()
        self.loss = F.nll_loss
        self.optimizer = optim.Adam
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.training = True

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 1)
        self.loss = F.nll_loss
        self.optimizer = optim.Adam

    def forward(self, input, future=0):
        outputs = []
        h_t = Variable(torch.zeros(input.size(0), 51).double(), requires_grad=False)
        c_t = Variable(torch.zeros(input.size(0), 51).double(), requires_grad=False)
        h_t2 = Variable(torch.zeros(input.size(0), 1).double(), requires_grad=False)
        c_t2 = Variable(torch.zeros(input.size(0), 1).double(), requires_grad=False)

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(c_t, (h_t2, c_t2))
            outputs += [c_t2]
        for i in range(future):
            h_t, c_t = self.lstm1(c_t2, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(c_t, (h_t2, c_t2))
            outputs += [c_t2]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs
