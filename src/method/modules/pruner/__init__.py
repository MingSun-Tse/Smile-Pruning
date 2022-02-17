# pruner init
from importlib import import_module
from .l1_pruner import l1_pruner

pruner_dict = {
	'l1': l1_pruner,
}