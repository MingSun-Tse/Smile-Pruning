
from .pruner import pruner_dict

def prune(model, loader, args, logger):
    # Read config file, get pruner name
    pruner = pruner_dict[args.pruner].Pruner(model, loader, args, logger)
    pruner.prune()
    print(f'==> Prune is done.')
    # -- Debug to check if the pruned weights are really zero. Confirmed!
    # import torch.nn as nn
    # for name, m in pruner.model.named_modules():
    #     if isinstance(m, (nn.Conv2d, nn.Linear)):
    #         print(m.weight.data.abs().min())
    # --
    return pruner.model

