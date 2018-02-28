'''
Current issues:
- torch.nn.parameters.Parameters -> torch.FloatTensor?
'''


import torch, redis, logging, parameter_tools as pt
import torch.nn as nn
import torch.multiprocessing as mp

logging.basicConfig(filename='gradient.log', level=logging.DEBUG)
cache = redis.StrictRedis(host='localhost', port=6379, db=0)


'''
Assumes given coordinates and corresponding gradients
e.g. [[[0, 1, 0], 0.8878482580184937], [[0, 2, 0], 0.7528610229492188], [[1, 2, 0], 0.46069568395614624]]
Coordinate meaning: 
[a, b, c] -> a = layer (a%2 == 0) or biases (a%2 != 0), b = row, c = column
e.g. [1, 2, 0]
'''

class TestNet(nn.Module):
	def __init__(self):
		super(TestNet, self).__init__()
		self.fc1 = nn.Linear(3,2)
		self.fc1 = nn.Linear(2,1)

	def add_coordinates(self, index, coords):
		params = [p.data for p in self.parameters()]

		cd = []
		gd = []
		p = params[index]

		for c in coords:
			point = c[0][1:]
			if point in cd:
				gd[cd.index(point)] += c[1]
			else:
				cd.append(point)
				gd.append(c[1])

		i = torch.LongTensor(cd)
		v = torch.FloatTensor(gd)
		s = list(p.size())


		if len(s) == 1:
			s.append(1)

		grads = torch.sparse.FloatTensor(i.t(), v, torch.Size(s)).to_dense()
		print 'param[%d] before: '%index
		print p
		p += grads
		print 'param[%d] after: '%index
		print p

	def add_batched_coordinates(self, coords):
		num_procs = mp.cpu_count()
		self.share_memory()
		processes = []

		# before
		print 'before:'
		params = [p for p in self.parameters()]
		print params

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
			# self.add_coordinates(k, params_coords[k])
			p = mp.Process(target=self.add_coordinates, args=(k, params_coords[k],))
			p.start()
			processes.append(p)

		for p in processes:
			p.join()

		# after
		print 'after:'
		updated = [p for p in self.parameters()]

		# can remove return statement once this works
		return updated

network = TestNet()
pm = []
kp = []
for t in network.parameters():
	pm.append(t.data.tolist())
	kp.append(t.data.float())

cache.set('test', pm)
params = pt.largest_k(kp, k=3)
print '\nlargest_k:'
print params
print '\nextended:'
params.extend(params)
print params

print '\nadd_batched_coordinates:'
print network.add_batched_coordinates(params)
