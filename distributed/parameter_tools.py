'''
    Justin Chen
    2.7.18

    Module for manipulating network parameters

    Boston University
    Hariri Institute for Computing and 
    Computational Sciences & Engineering
'''
import torch
import numpy as np


'''
Converts tensor to a CUDA tensor

Input: tensor (PyTorch tensor)
Output: (torch.cuda.FloatTensor)
'''
def to_cuda(tensor, cuda=False):
    return tensor.cuda() if cuda else tensor


'''
Converts a single nested list representing a matrix to a PyTorch FloatTensor

Input: data (list)
Output: (torch.FloatTensor)
'''
def list_to_tensor(data, cuda=False):
    return to_cuda(torch.FloatTensor(x), cuda)


'''
Converts an entire list of nested lists representing a matrices to a list of PyTorch tensors

Input: data (list) List of tensors
Output: (list) List of PyTorch torch.FloatTensor
'''
def all_to_tensor(data):
    return [list_to_tensor(l) for l in data]


'''
Computes the difference between the original and updated list of tensors

Input: original (list) List of tensors
       updated (list) List of tensors
Output: (list)
'''
def param_delta(self, original, updated):
    return [x-y for (x, y) in zip(original, updated)]


'''
Extracts the largest k values across all tensors in the given list

Input: data    (list) List of PyTorch FloatTensors
       k       (numeric) Value indicating amount of parameters to keep
       percent (bool) Indicates if k is a percent or an integer
       zeros   (bool) If False, ignores all parameters equal to zero

Output: (list) List of lists containing coordinates-parameter pairs
        e.g. [[[0, 0, 0], -0.43671706318855286], [[0, 0, 1], -0.4151779115200043], [[1, 0, 0], 0.19337968528270721]]
'''
def largest_k(tensors, k=0, percent=True, zeros=True):
    # flatten all tensors into vectors and then concat and find top-k
    flat = []
    dims = []
    size = []
    coords = []

    if k == 0:
        return []
    elif k < 0:
        raise Exception('k must be >= 0')

    for t in tensors:
        n = torch.numel(t)
        flat.append(t.view(n))
        d = [x for x in t.size()]

        if len(d) == 1:
            d.append(1L)
        dims.append(tuple(d))
        size.append(n)

    merged = torch.cat(flat)
    total = torch.numel(merged)

    use_cuda = merged.is_cuda

    if percent and k > 0 and k <= 1:
        k = int(math.ceil(k * total))
    elif k <= total:
        k = int(k)
    else:
        raise Exception('k must be in range [0, %d]\ntotal parameters: %d' % (total, total))

    top, coord_1d = torch.topk(torch.abs(merged), k)
    top = torch.gather(merged, 0, to_cuda(torch.LongTensor(coord_1d.tolist()), use_cuda))
    coord_1d = sorted([(t, c) for (t, c) in zip(top, coord_1d)], key=lambda x: x[1])

    # partition topk into bins corresponding to original matrices
    bound = 0
    for i, (s, d) in enumerate(zip(size, dims)):
        b = []
        grad = []
        lower = bound
        bound += s
        j = 0

        for j, (t, c) in enumerate(coord_1d):
            if c < bound:
                b.append(c - lower)
                grad.append(t)

                if j + 1 == len(coord_1d):
                    coord_1d = []
            else:
                coord_1d = coord_1d[j:]
                break

        if len(b) > 0:

            axises = list(np.unravel_index(b, d))
            axises.append(grad)

            for point in zip(*axises[:]):
                g = point[-1]
                if zeros or (not zeros and g > 0):
                    c = list(point[:-1])
                    c.insert(0, i)
                    coords.append([c, g])
                
    return coords


'''
Determines if a given input is a PyTorch vector

Input: tensor (torch.FloatTensor)
Output: (bool)
'''
def is_vector(tensor):
    return len(tensor.size()) == 1


'''
Makes a tensor given specific coordinates and values

Input: coord_list (list), grads_list (list), layer (nn.Layer)
Output: (torch.sparse.FloatTensor)
'''
def make_tensor(coord_list, grads_list, layer):
    # create sparse tensors
    coord = torch.LongTensor(coord_list)
    grads = torch.FloatTensor(grads_list)

    return torch.sparse.FloatTensor(coord.t(), grads, layer.size()).to_dense()

'''
Directly replace local gradients with corresponding server gradients

Input:
Output:
'''
def insert_parameters(self, grad, selected):

    return grad

