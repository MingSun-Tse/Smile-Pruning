import torch
import torch.nn as nn
import copy
import time
import numpy as np
import torch.optim as optim
from .meta_pruner import MetaPruner
from .reinit_model import orth_dist, deconv_orth_dist
from utils import Timer

class Pruner(MetaPruner):
    def __init__(self, model, args, logger, passer):
        super(Pruner, self).__init__(model, args, logger, passer)

    def prune(self):
        if self.args.orth_reg_iter > 0:
            self.netprint('\n--> Start orthogonal regularization training.')
            self.model = self._orth_reg_train(self.model) # update self.model
            self.netprint('<-- End orthogonal regularization training.\n')
        self._get_kept_wg_L1()
        self._prune_and_build_new_model()
        return self.model
    
    def _orth_reg_train(self, model):
        optimizer = optim.SGD(model.parameters(), 
                            lr=self.args.lr_prune,
                            momentum=self.args.momentum,
                            weight_decay=self.args.weight_decay)
        
        acc1 = acc5 = 0
        epoch = -1
        timer = Timer(self.args.orth_reg_iter / self.args.print_interval)
        self.total_iter = -1
        self.prune_state = 'orth_reg'
        while True:
            epoch += 1
            for _, (inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                self.total_iter += 1
                total_iter = self.total_iter
                    
                if total_iter % self.args.print_interval == 0:
                    print("")
                    print("Iter = %d [prune_state = %s, method = %s] " 
                        % (total_iter, self.prune_state, self.args.method) + "-"*40)
                    
                # forward
                model.train()
                y_ = model(inputs)
                
                # normal training forward
                loss = self.criterion(y_, targets)
                logtmp = f'loss_cls {loss:.4f}'

                # Orth reg
                loss_orth_reg = 0
                for name, module in model.named_modules():
                    if isinstance(module, self.learnable_layers):
                        if self.args.orth_reg_method in ['CVPR20']:
                            if self.layers[name].layer_index != 0: # per the CVPR20 paper, do not reg the 1st conv
                                shape = self.layers[name].size
                                if len(shape) == 2 or shape[-1] == 1: # FC and 1x1 conv 
                                    loss_orth_reg += orth_dist(module.weight)
                                else:
                                    loss_orth_reg += deconv_orth_dist(module.weight)
                        elif self.args.orth_reg_method in ['CVPR17']:
                            loss_orth_reg += orth_dist(module.weight)
                        else:
                            raise NotImplementedError
                loss += loss_orth_reg * self.args.lw_orth_reg
                
                # print loss
                if self.total_iter % self.args.print_interval == 0:
                    logtmp += f' loss_orth_reg (*{self.args.lw_orth_reg}) {loss_orth_reg:.10f} Iter {self.total_iter}'
                    print(logtmp)
                    print(f"predicted_finish_time of orth_reg: {timer()}")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # test
                if total_iter % self.args.test_interval == 0:
                    acc1, acc5, *_ = self.test(model)
                    self.accprint("Acc1 = %.4f Acc5 = %.4f Iter = %d (after update) [prune_state = %s, method = %s]" % 
                        (acc1, acc5, total_iter, self.prune_state, self.args.method))
                
                # save model (save model before a batch starts)
                if total_iter % self.args.save_interval == 0:
                    self._save_model(model, optimizer, acc1, acc5)
                    print('Periodically save model done. Iter = {}'.format(total_iter))

                # return
                if total_iter > self.args.orth_reg_iter:
                    return copy.deepcopy(model)
                
    def _save_model(self, model, optimizer, acc1=0, acc5=0, mark=''):
        state = {'iter': self.total_iter,
                'arch': self.args.arch,
                'model': model,
                'state_dict': model.state_dict(),
                'acc1': acc1,
                'acc5': acc5,
                'optimizer': optimizer.state_dict(),
                'ExpID': self.logger.ExpID,
        }
        self.save(state, is_best=False, mark=mark)