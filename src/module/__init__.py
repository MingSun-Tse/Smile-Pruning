# module init
from importlib import import_module
from .finetune import finetune
from .prune import prune

module_dict = {
	'ft': finetune,
	'prune': prune,
}