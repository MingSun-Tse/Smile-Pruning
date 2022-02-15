from method.modules import module_dict
from utils import strlist_to_list

def lth(model, loader, criterion, args, logger):
	lth_pipe = '[train, prune, reinit, train]'
	pipeline = strlist_to_list(lth_pipe, str)

	for each_module_str in pipeline:
		each_module = module_dict[each_module_str]
		each_module(model, loader, criterion, args, logger)
		
		# update config for each module