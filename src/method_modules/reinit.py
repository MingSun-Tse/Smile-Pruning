
from importlib import import_module

def reinit(model, loader, args, logger, passer):
    # Read config file, get pruner name
    module = import_module(f'method_modules.reiniter.{args.reiniter}')
    reiniter = module.Reiniter(model, loader, args, logger, passer)
    reiniter.reinit()
    print(f'==> Reinit is done.')
    return reiniter.model