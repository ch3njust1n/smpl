# from torch import utils as pyt_utils
# from torchvision import datasets, transforms

# cuda = False
# kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
# train = datasets.MNIST('../data', train=True, download=False, 
# 					   transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))

# test  = datasets.MNIST('../data', train=True, download=False, 
# 					   transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))

# batch_size = 2
# train_loader = pyt_utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, **kwargs)
# print 'train.batch_size 2?: '
# print train_loader.batch_size

# batch_size = 4
# train_loader = pyt_utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, **kwargs)
# print 'train.batch_size 4?: '
# print train_loader.batch_size


from data import SMPLData
ds = SMPLData()
train, test = ds.load_data(64)

for batch_idx, (data, target) in enumerate(train):
	print batch_idx
	print len(data)
	print len(target)
	raw_input('...')