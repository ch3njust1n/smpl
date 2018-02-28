import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from network import TestNet as nn
import redis
import torch

model = nn()

# print model.state_dict()

print 'before:'
print [m.data for m in model.parameters()]


cache = redis.StrictRedis(host='localhost', port=6379, db=0)

print 'mod'
mod = [(1+t.data).tolist() for t in model.parameters()]
print mod

cache.set('test', mod)
params = eval(cache.get('test'))
print params
print type(params)
print 'converting to tensors:'
params = [torch.FloatTensor(x) for x in params]
print params
print 'type of params:'
print str(type(params[0]))

for a,b in zip(params, model.parameters()):
	b.data=a

print 'after:'
p = [m for m in model.parameters()]
print 'api:'
print model.get_parameters()
print type(p[0])
print type(torch.zeros(3,3)) == torch.FloatTensor

