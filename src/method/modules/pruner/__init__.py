# pruner init
from importlib import import_module
from .mag_pruner import mag_pruner

pruner_dict = {
	'mag': mag_pruner,
}