from importlib import import_module
from .vgg import vgg11, vgg13, vgg16, vgg19
from .resnet_cifar10 import resnet56
from .lenet5 import lenet5, lenet5_mini, lenet5_linear, lenet5_wider_linear, lenet5_wider_linear_nomaxpool
from .mlp import mlp_7_linear, mlp_7_relu

def set_up_model(args, logger):
    logger.log_printer("==> making model ...")
    module = import_module("model.model_%s" % args.method)
    model = module.make_model(args, logger)    
    return model

def is_single_branch(model_name):
    for k in single_branch_model:
        if model_name.startswith(k):
            return True
    return False


model_dict = {
    'mlp_7_linear': mlp_7_linear,
    'mlp_7_relu': mlp_7_relu,
    'lenet5': lenet5,
    'lenet5_mini': lenet5_mini,
    'lenet5_linear': lenet5_linear,
    'lenet5_wider_linear': lenet5_wider_linear,
    'lenet5_wider_linear_nomaxpool': lenet5_wider_linear_nomaxpool,
    'resnet56': resnet56,
    'vgg11': vgg11,
    'vgg13': vgg13,
    'vgg16': vgg16,
    'vgg19': vgg19,
}

num_layers = {
    'mlp_7_linear': 7,
    'mlp_7_relu': 7,
    'lenet5': 5,
    'lenet5_mini': 5,
    'lenet5_linear': 5,
    'lenet5_wider_linear': 5,
    'lenet5_wider_linear_nomaxpool': 5,
    'alexnet': 8,
    'vgg11': 11,
    'vgg13': 13,
    'vgg16': 16,
    'vgg19': 19,
}

single_branch_model = [
    'mlp_7',
    'lenet',
    'alexnet',
    'vgg',
]