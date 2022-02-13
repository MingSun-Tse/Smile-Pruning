# modules init
from .vanilla_train import vanilla_train
from .prune import prune

module_dict = {
	'vanilla_train': vanilla_train,
	'prune': prune,
}