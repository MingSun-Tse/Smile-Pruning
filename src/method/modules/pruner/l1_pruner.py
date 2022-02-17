from utils import strlist_to_list
from collections import OrderedDict
from math import ceil
from pdb import set_trace as st
import torch.nn as nn
import torch
from .meta_pruner import MetaPruner

class l1_pruner(MetaPruner):
	def __init__(self, model, args):
		super(l1_pruner, self).__init__(model, args)

	def prune(self):
		self._get_kept_wg_mag()
		self._prune_and_build_new_model()
		return self.model

	def _get_kept_wg_mag(self):
		for name, module in self.model.named_modules():
			if isinstance(module, self.learnable_layers):
				shape = module.weight.data
				score = module.weight.abs().flatten()

				self.pruned_wg[name] = self._pick_pruned(score, self.pr[name], self.args.pick_pruned, name)
				self.kept_wg[name] = list(set(range(len(score))) - set(self.pruned_wg[name]))

	def _pick_pruned(self, w_abs, pr, mode, name=None):
		if pr == 0:
			return []
		w_abs_list = w_abs
		n_wg = len(w_abs_list)
		n_pruned = min(ceil(pr * n_wg), n_wg - 1) # avoid pruning all
		if mode == 'rand':
			out = np.random.permutation(n_wg)[:n_pruned]
		elif mode == 'min':
			out = w_abs_list.sort()[1][:n_pruned]
			out = out.data.cpu().numpy()

		return out

	def _prune_and_build_new_model(self):
		self._get_mask()
		return

	def _get_mask(self):
		self.mask = {}
		for name, module in self.model.named_modules():
			if isinstance(module, self.learnable_layers):
				mask = torch.ones_like(module.weight.data).cuda().flatten()
				pruned = self.pruned_wg[name]
				mask[pruned] = 0
				self.mask[name] = mask.view_as(module.weight.data)


