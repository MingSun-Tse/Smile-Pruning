
from .pruner import pruner_dict

def prune(model, loader, args, logger, passer):
    # Read config file, get pruner name
    pruner = pruner_dict[args.pruner].Pruner(model, loader, args, logger, passer)
    pruner.prune()
    print(f'==> Prune is done.')
    return pruner.model

