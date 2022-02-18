
def prune(model, loader, args, logger, passer):
    # Read config file, get pruner name
    pruner_name = 'l1'
    from importlib import import_module
    pruner = import_module(f'method_modules.pruner.{pruner_name}').Pruner(model, loader, args, logger, passer)
    model = pruner.prune()
    return model

