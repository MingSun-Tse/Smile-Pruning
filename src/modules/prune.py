from utils import *

class Layer:
	def __init__(self, name, size, layer_index, layer_type=None):
		self.name = name
		self.size = []
		for x in size:
			self.size.append(x)
		self.layer_index = layer_index
		self.layer_type = layer_type

class metapruner:
	def __init__(self, model, args, logger,):
		self.model = model
		self.args = args
		self.logger = logger

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
		pr = self.args.stage_pr[layer_index]

		return pr

	def get_pr(self):
		self.pr = {}
		get_layer_pr = self._get_layer_pr
		for name, module in self.model.named_modules():
			if isinstance(module, self.learnable_layers):
				self.pr[name] = get_layer_pr(name)

class mag_pruner(metapruner):
	def __init__(self, model, args, logger):
		super(mag_pruner, self).__init__(model, args, logger)

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
			if isinstance(m, self.learnable_layers):
				mask = torch.ones_like(module.weight.data).cuda().flatten()
				pruned = self.pruned_wg[name]
				mask[pruned] = 0
				self.mask[name] = mask.view_as(module.weight.data)

def prune(loader, train_loader, test_loader, criterion, model, args,
			 train_index, train_loss_list, train_acc1_list,
			 test_index, test_loss_list, test_acc1_list,
			 grad_mean, grad_std, grad_abs_mean, grad_abs_std):
	pruner = prune_dict[args.prune_method]



