from . import reg_pruner, l1_pruner, oracle_pruner, orth_preserving_pruner, merge_pruner
from . import l1_pruner_iterative

# when new pruner implementation is added in the 'pruner' dir, update this dict to maintain minimal code change.
# key: pruning method name, value: the corresponding pruner
pruner_dict = {
    'FixReg': reg_pruner,
    'GReg-1': reg_pruner,
    'GReg-2': reg_pruner,
    'L1': l1_pruner,
    'Oracle': oracle_pruner,
    'OPP': orth_preserving_pruner,
    'Merge': merge_pruner,
    'L1_Iter': l1_pruner_iterative,
}