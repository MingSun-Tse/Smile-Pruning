from importlib import import_module
from .mlp import mlp
from .resnet import resnet56, resnet32, resnet20

model_dict = {
	'mlp': mlp,
	'resnet20': resnet20,
	'resnet32': resnet32,
	'resnet56': resnet56,
}