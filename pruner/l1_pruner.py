import torch
import torch.nn as nn
import copy
import time
import numpy as np
from utils import _weights_init, _weights_init_orthogonal
from .meta_pruner import MetaPruner


# refer to: A Signal Propagation Perspective for Pruning Neural Networks at Initialization (ICLR 2020).
def approximate_isometry_optimize(model, mask, lr, n_iter):
    def optimize(w): # w: N x M matrix
        identity = torch.eye(w.size(0)).cuda()
        w_ = torch.autograd.Variable(w, requires_grad=True)
        optim = torch.optim.Adam([w_], lr)
        for i in range(n_iter):
            loss = nn.MSELoss()(torch.matmul(w_, w_.t()), identity)
            optim.zero_grad()
            loss.backward()
            optim.step()
            w_ = torch.mul(w_, mask_) # not update the pruned params
            w_ = torch.autograd.Variable(w_, requires_grad=True)
            optim = torch.optim.Adam([w_], lr)
            if i % 10 == 0:
                print('[%d/%d] approximate_isometry_optimize for layer "%s", loss %.6f' % (i, n_iter, name, loss.item()))
        return w_
    
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            mask_ = mask[name]
            w = m.weight
            w = w.view(w.size(0), -1) # [n_filter, -1]
            w_ = optimize(w)
            m.weight.data.copy_(w_)
            print('Finished approximate_isometry_optimize for layer "%s"' % name)


class Pruner(MetaPruner):
    def __init__(self, model, args, logger, runner):
        super(Pruner, self).__init__(model, args, logger, runner)

    def prune(self):
        self._get_kept_wg_L1()
        self._prune_and_build_new_model()
        mask = self.mask if self.args.wg == 'weight' else None

        if self.args.reinit:
            if self.args.reinit == 'orth':
                self.model.apply(_weights_init_orthogonal)
                self.logprint("==> Reinit model: orthogonal initialization")
            elif self.args.reinit == 'default':
                self.model.apply(_weights_init) # equivalent to training from scratch
                self.logprint("==> Reinit model: default initialization (kaiming_normal for Conv; 0 mean, 1 std for BN)")
            elif self.args.reinit == 'approximate_isometry':
                approximate_isometry_optimize(self.model, mask=mask, lr=0.001, n_iter=10000) 
                # 10000 refers to the paper above; lr in the paper is 0.1, but not converged here. By trial-and-error, 0.001 works best
                self.logprint("==> Reinit model: approximate_isometry initialization")
            else:
                raise NotImplementedError
            
        return self.model