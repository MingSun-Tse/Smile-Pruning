# modules init
from .train import train
from .prune import prune

module_dict = {
	'train': train,
	'prune': prune,
}