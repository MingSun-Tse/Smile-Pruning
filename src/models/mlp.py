from utils import *
import torch
import torch.nn.functional as F

class fc(nn.Module):
	def __init__(self, args):
		super(fc, self).__init__()
		# create model
		hidden_act = act_dict(args.mlp_hiddenAct)
		final_act = act_dict(args.mlp_finalAct)
		layers_width = strlist_to_list(args.mlp_layers, int)

		net = []
		for i in range(len(layers_width) - 2):
			net.append(nn.Linear(layers_width[i], layers_width[i+1]))
			net.append(hidden_act)

		net.append(nn.Linear(layers_width[i+1], layers_width[i+2]))
		net.append(final_act)

		self.net = nn.ModuleList(net)

		# ib 
		self.feats_per_epo = []
		self.feats_per_epo_test = []
		self.crt_feats = None

		self.plot_pre_act = args.mlp_plotPreAct
		self.plot_layers_indexs = []
		for i, layer in enumerate(self.net):
			if hasattr(layer, 'weight') == self.plot_pre_act:
				self.plot_layers_indexs.append(i)

		self.reset()

	def forward(self, x):
		x = x.view(x.size(0), -1)
		crt_feat = x
		plot_layers_index = 0

		for i, layer in enumerate(self.net):
			crt_feat = layer(crt_feat)

			if i in self.plot_layers_indexs:
				self.add_info(plot_layers_index, crt_feat.detach())
				plot_layers_index += 1

		return crt_feat

	def next_epoch(self, mode):
		if mode == 'train':
			self.feats_per_epo.append(self.crt_feats)
		elif mode == 'test':
			self.feats_per_epo_test.append(self.crt_feats)
		else:
			raise NotImplementedError

		self.reset()

	def add_info(self, layer_index, feat):
		if self.crt_feats[layer_index] is None:
			self.crt_feats[layer_index] = feat
		else:
			self.crt_feats[layer_index] = torch.cat((self.crt_feats[layer_index], feat), axis = 0)

	def reset(self):
		self.crt_feats = [None for _ in range(len(self.plot_layers_indexs))]

	def remove_saved_feats(self, mode):
		if mode == 'train':
			self.feats_per_epo = []
		elif mode == 'test':
			self.feats_per_epo_test = []
		else:
			raise NotImplementedError



def mlp(args):
	return fc(args)

