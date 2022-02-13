from utils import strlist_to_list
from collections import OrderedDict
from math import ceil
from pdb import set_trace as st
import torch.nn as nn
import torch

class Layer:
	def __init__(self, name, size, layer_index, layer_type=None):
		self.name = name
		self.size = []
		for x in size:
			self.size.append(x)
		self.layer_index = layer_index
		self.layer_type = layer_type

class MetaPruner:
	def __init__(self, model, args):
		self.model = model
		self.args = args

		self.learnable_layers = (nn.Conv2d, nn.Linear)
		self.layers = OrderedDict()
		self.all_layers = []
		self._register_layers()

		self.kept_wg = {}
		self.pruned_wg = {}

		self.get_pr()

	def _register_layers(self):
		ix = -1
		self._max_len_name = 0
		layer_shape = {}

		for name, module in self.model.named_modules():
			self.all_layers += [name]
			if isinstance(module, self.learnable_layers):
				ix += 1
				layer_shape[name] = [ix, module.weight.size()]
				self._max_len_name = max(self._max_len_name, len(name))

				size = module.weight.size()
				self.layers[name] = Layer(name, size, ix, layer_type=module.__class__.__name__)

	def _get_layer_pr(self, name):
		layer_index = self.layers[name].layer_index
		stage_pr = strlist_to_list(self.args.stage_pr, ttype=float)
		pr = stage_pr[layer_index]

		return pr

	def get_pr(self):
		self.pr = {}
		get_layer_pr = self._get_layer_pr
		for name, module in self.model.named_modules():
			if isinstance(module, self.learnable_layers):
				self.pr[name] = get_layer_pr(name)



