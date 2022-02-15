from utils import *
import torch
import torch.nn.functional as F

class MetaModel:
	def __init__(self, model, ):
		self.mask = {}