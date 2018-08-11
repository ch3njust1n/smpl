'''
    Justin Chen
    7.20.18

    Module for handle Secure Multi-Party Computation sessions
'''
import numpy as np
from os import urandom
from torch import FloatTensor


class Session(object):
	def __init__(self):
		self.otp = None
		self.phase = 0


	'''
	Generate a one-time pad

	Input:  shape (Tuple)
	Output: pad   (FloatTensor)
	'''
	def get_one_time_pad(self, shape):
		for d in range(len(shape)):
			torch.FloatTensor(np.array(map(ord, os.urandom(10))))


	'''
	Split secrete share

	Input:
	Output:
	'''
	def split(self, x):
		pass


	'''
	Encode with one-time pad
	'''
	def encode(self, x):
		pass


	'''
	Remove one-time pad

	Input:
	Output:
	'''
	def decode(self, x):
		pass