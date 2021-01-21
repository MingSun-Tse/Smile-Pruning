import torch
import torch.nn as nn
import copy
import time
import numpy as np
from utils import _weights_init, _weights_init_orthogonal
from .meta_pruner import MetaPruner

class Pruner(MetaPruner):
    def __init__(self, model, args, logger, runner):
        super(Pruner, self).__init__(model, args, logger, runner)

    def prune(self):
        self._get_kept_wg_L1()
        self._prune_and_build_new_model()
                    
        if self.args.reinit:
            if self.args.reinit == 'orth':
                self.model.apply(_weights_init_orthogonal)
                self.logprint("==> Reinit model: orthogonal initialization")
            elif self.args.reinit == 'default':
                self.model.apply(_weights_init) # equivalent to training from scratch
                self.logprint("==> Reinit model: default initialization (kaiming_normal for Conv; 0 mean, 1 std for BN)")
            else:
                raise NotImplementedError
            
        return self.model