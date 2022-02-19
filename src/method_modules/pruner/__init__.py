from . import reg_pruner, l1_pruner

pruner_dict = {
    'greg1': reg_pruner,
    'greg2': reg_pruner,
    'l1': l1_pruner,
}