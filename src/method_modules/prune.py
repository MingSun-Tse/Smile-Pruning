
from importlib import import_module

def prune(model, loader, args, logger, passer):
    # Read config file, get pruner name
    module = import_module(f'method_modules.pruner.{args.pruner}_pruner')
    pruner = module.Pruner(model, loader, args, logger, passer)
    pruner.prune()
    print(f'==> Prune is done.')
    return pruner.model

