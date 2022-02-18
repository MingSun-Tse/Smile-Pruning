
from .train import train
from .prune import prune
from .reinit import reinit


module_dict = {
    'train': train,
    'prune': prune,
    'reinit': reinit,
}