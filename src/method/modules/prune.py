from utils import strlist_to_list
from collections import OrderedDict
from math import ceil
from pdb import set_trace as st
import torch.nn as nn
import torch

def prune(cagos):
	pruner = mag_pruner(model, args)
	model = pruner.prune()

	model.mask = pruner.mask
