# dataset init
from importlib import import_module
import os
import numpy as np
import torch
from torch.utils import data
from torch.utils.data import DataLoader

class Data(object):
	def __init__(self, args):
		self.args = args
		loader = import_module("dataset.%s" % args.dataset)
		path = os.path.join(args.data_path, args.dataset)
		train_set, test_set = loader.get_datasets(path, args.batch_size)

		self.train_loader = DataLoader(train_set,
									   batch_size = args.batch_size,
									   num_workers = args.workers,
									   shuffle = True,
									   pin_memory = True)
		self.test_loader = DataLoader(test_set,
									  batch_size = 256,
									  num_workers = args.workers,
									  shuffle = False,
									  pin_memory = True)

		self.train_set = train_set
		self.test_set = test_set

		self.num_classes = {'mnist':10,
							'cifar10':10,
							'tsb':2}
		self.img_size = {'mnist':32,
						 'cifar10':32,
						 'tsb':1}