import torch
import torch.nn as nn
import torch.optim as optim
import os, copy, time, pickle, numpy as np, math
from .meta_pruner import MetaPruner
from .reinit_model import reinit_model, orth_regularization, orth_regularization_v3, orth_regularization_v4, deconv_orth_dist
from .reinit_model import orth_regularization_v5, orth_regularization_v5_2
from .reinit_model import orth_regularization_v6
from utils import plot_weights_heatmap, Timer
import matplotlib.pyplot as plt
from collections import OrderedDict
pjoin = os.path.join

class Pruner(MetaPruner):
    def __init__(self, model, args, logger, passer):
        super(Pruner, self).__init__(model, args, logger, passer)

        # Reg related variables
        self.reg = {}
        self.delta_reg = {}
        self.hist_mag_ratio = {}
        self.n_update_reg = {}
        self.iter_update_reg_finished = {}
        self.iter_finish_pick = {}
        self.iter_stabilize_reg = math.inf
        self.original_w_mag = {}
        self.original_kept_w_mag = {}
        self.ranking = {}
        self.pruned_wg_L1 = {}
        self.all_layer_finish_pick = False
        self.w_abs = {}
        self.mag_reg_log = {}
        if self.args.__dict__.get('AdaReg_only_picking'): # AdaReg is the old name for GReg-2
            self.original_model = copy.deepcopy(self.model)
        
        # prune_init, to determine the pruned weights
        # this will update the 'self.kept_wg' and 'self.pruned_wg' 
        if self.args.method in ['GReg-1', 'GReg-2', 'OPP']:
            self._get_kept_wg_L1()
        for k, v in self.pruned_wg.items():
            self.pruned_wg_L1[k] = v
        if self.args.method == 'GReg-2': # GReg-2 will determine which wgs to prune later, so clear it here
            self.kept_wg = {}
            self.pruned_wg = {}

        self.prune_state = "update_reg"
        for name, m in self.model.named_modules():
            if isinstance(m, self.learnable_layers):
                shape = m.weight.data.shape

                # initialize reg
                if self.args.wg == 'weight':
                    self.reg[name] = torch.zeros_like(m.weight.data).flatten().cuda()
                else:
                    self.reg[name] = torch.zeros(shape[0], shape[1]).cuda() 
                
                # get original weight magnitude
                w_abs = self._get_score(m)
                n_wg = len(w_abs)
                self.ranking[name] = []
                for _ in range(n_wg):
                    self.ranking[name].append([])
                self.original_w_mag[name] = m.weight.abs().mean().item()
                kept_wg_L1 = [i for i in range(n_wg) if i not in self.pruned_wg_L1[name]]
                self.original_kept_w_mag[name] = w_abs[kept_wg_L1].mean().item()

        # init original_column_gram
        self.original_column_gram = OrderedDict()
        for name, m in self.model.named_modules():
            if isinstance(m, self.learnable_layers):
                w = m.weight.data
                w = w.view(w.size(0), -1)
                self.original_column_gram[name] = w.t() @ w
        
        # get bn after each conv/fc
        self.next_bn = {}
        for n, m in self.model.named_modules():
            if isinstance(m, self.learnable_layers):
                next_bn = self._next_bn(self.model, m)
                self.next_bn[n] = next_bn

    def _pick_pruned_wg(self, w, pr):
        if pr == 0:
            return []
        elif pr > 0:
            w = w.flatten()
            n_pruned = min(math.ceil(pr * w.size(0)), w.size(0) - 1) # do not prune all
            return w.sort()[1][:n_pruned]
        elif pr == -1: # automatically decide lr by each layer itself
            tmp = w.flatten().sort()[0]
            n_not_consider = int(len(tmp) * 0.02)
            w = tmp[n_not_consider:-n_not_consider]

            sorted_w, sorted_index = w.flatten().sort()
            max_gap = 0
            max_index = 0
            for i in range(len(sorted_w) - 1):
                # gap = sorted_w[i+1:].mean() - sorted_w[:i+1].mean()
                gap = sorted_w[i+1] - sorted_w[i]
                if gap > max_gap:
                    max_gap = gap
                    max_index = i
            max_index += n_not_consider
            return sorted_index[:max_index + 1]
        else:
            self.logprint("Wrong pr. Please check.")
            exit(1)
    
    def _update_mag_ratio(self, m, name, w_abs, pruned=None):
        if type(pruned) == type(None):
            pruned = self.pruned_wg[name]
        kept = [i for i in range(len(w_abs)) if i not in pruned]
        ave_mag_pruned = w_abs[pruned].mean()
        ave_mag_kept = w_abs[kept].mean()
        if len(pruned):
            mag_ratio = ave_mag_kept / ave_mag_pruned 
            if name in self.hist_mag_ratio:
                self.hist_mag_ratio[name] = self.hist_mag_ratio[name]* 0.9 + mag_ratio * 0.1
            else:
                self.hist_mag_ratio[name] = mag_ratio
        else:
            mag_ratio = math.inf
            self.hist_mag_ratio[name] = math.inf
        
        # print
        mag_ratio_now_before = ave_mag_kept / self.original_kept_w_mag[name]
        if self.total_iter % self.args.print_interval == 0:
            self.logprint("    mag_ratio %.4f mag_ratio_momentum %.4f" % (mag_ratio, self.hist_mag_ratio[name]))
            self.logprint("    for kept weights, original_kept_w_mag %.6f, now_kept_w_mag %.6f ratio_now_over_original %.4f" % 
                (self.original_kept_w_mag[name], ave_mag_kept, mag_ratio_now_before))
        return mag_ratio_now_before

    def _get_score(self, m):
        shape = m.weight.data.shape
        if self.args.wg == "channel":
            w_abs = m.weight.abs().mean(dim=[0, 2, 3]) if len(shape) == 4 else m.weight.abs().mean(dim=0)
        elif self.args.wg == "filter":
            w_abs = m.weight.abs().mean(dim=[1, 2, 3]) if len(shape) == 4 else m.weight.abs().mean(dim=1)
        elif self.args.wg == "weight":
            w_abs = m.weight.abs().flatten()
        return w_abs

    def _fix_reg(self, m, name):
        if self.pr[name] == 0:
            return True
        if self.args.wg != 'weight':
            self._update_mag_ratio(m, name, self.w_abs[name])

        pruned = self.pruned_wg[name]
        if self.args.wg == "channel":
            self.reg[name][:, pruned] = self.args.reg_upper_limit
        elif self.args.wg == "filter":
            self.reg[name][pruned, :] = self.args.reg_upper_limit
        elif self.args.wg == 'weight':
            self.reg[name][pruned] = self.args.reg_upper_limit

        finish_update_reg = self.total_iter > self.args.fix_reg_interval
        return finish_update_reg

    def _greg_1(self, m, name):
        if self.pr[name] == 0:
            return True
        
        if self.args.wg != 'weight': # weight is too slow
            self._update_mag_ratio(m, name, self.w_abs[name])
        
        pruned = self.pruned_wg[name]
        if self.args.wg == "channel":
            self.reg[name][:, pruned] += self.args.reg_granularity_prune
        elif self.args.wg == "filter":
            self.reg[name][pruned, :] += self.args.reg_granularity_prune
        elif self.args.wg == 'weight':
            self.reg[name][pruned] += self.args.reg_granularity_prune
        else:
            raise NotImplementedError

        # when all layers are pushed hard enough, stop
        if self.args.wg == 'weight': # for weight, do not use the magnitude ratio condition, because 'hist_mag_ratio' is not updated, too costly
            finish_update_reg = False
        else:
            finish_update_reg = True
            for k in self.hist_mag_ratio:
                if self.hist_mag_ratio[k] < self.args.mag_ratio_limit:
                    finish_update_reg = False
        return finish_update_reg or self.reg[name].max() > self.args.reg_upper_limit

    def _update_reg(self):
        for name, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                cnt_m = self.layers[name].layer_index
                pr = self.pr[name]
                
                if name in self.iter_update_reg_finished.keys():
                    continue

                if self.total_iter % self.args.print_interval == 0:
                    self.logprint("[%d] Update reg for layer '%s'. Pr = %s. Iter = %d" 
                        % (cnt_m, name, pr, self.total_iter))
                
                # get the importance score (L1-norm in this case)
                self.w_abs[name] = self._get_score(m)
                
                # update reg functions, two things: 
                # (1) update reg of this layer (2) determine if it is time to stop update reg
                finish_update_reg = self._greg_1(m, name)

                # check prune state
                if finish_update_reg:
                    # after 'update_reg' stage, keep the reg to stabilize weight magnitude
                    self.iter_update_reg_finished[name] = self.total_iter
                    self.logprint("==> [%d] Just finished 'update_reg'. Iter = %d" % (cnt_m, self.total_iter))

                    # check if all layers finish 'update_reg'
                    self.prune_state = "stabilize_reg"
                    for n, mm in self.model.named_modules():
                        if isinstance(mm, self.learnable_layers):
                            if n not in self.iter_update_reg_finished:
                                self.prune_state = "update_reg"
                                break
                    if self.prune_state == "stabilize_reg":
                        self.iter_stabilize_reg = self.total_iter
                        self.logprint("==> All layers just finished 'update_reg', go to 'stabilize_reg'. Iter = %d" % self.total_iter)
                        self._save_model(mark='just_finished_update_reg')
                    
                # after reg is updated, print to check
                if self.total_iter % self.args.print_interval == 0:
                    self.logprint("    reg_status: min = %.5f ave = %.5f max = %.5f" % 
                                (self.reg[name].min(), self.reg[name].mean(), self.reg[name].max()))

    def _apply_reg(self):
        for name, m in self.model.named_modules():
            if name in self.reg and self.pr[name] > 0:
                reg = self.reg[name] # [N, C]
                if self.args.wg in ['filter', 'channel']:
                    if reg.shape != m.weight.data.shape:
                        reg = reg.unsqueeze(2).unsqueeze(3) # [N, C, 1, 1]
                elif self.args.wg == 'weight':
                    reg = reg.view_as(m.weight.data) # [N, C, H, W]
                m.weight.grad += reg * m.weight
                bias = False if isinstance(m.bias, type(None)) else True
                if bias:
                    if len(reg.shape) == 4:
                        m.bias.grad += reg[:,0,0,0] * m.bias
                    elif len(reg.shape) == 2:
                        m.bias.grad += reg[:,0] * m.bias
                    else:
                        raise NotImplementedError
                
                # apply reg to bn
                if self.args.bn_reg:
                    next_bn = self.next_bn[name]
                    if next_bn:
                        assert self.args.wg == 'filter'
                        next_bn.weight.grad += reg[:,0,0,0] * next_bn.weight
                        next_bn.bias.grad += reg[:,0,0,0] * next_bn.bias

    
    def _resume_prune_status(self, ckpt_path):
        state = torch.load(ckpt_path)
        self.model = state['model'].cuda()
        self.model.load_state_dict(state['state_dict'])
        self.optimizer = optim.SGD(self.model.parameters(), 
                                lr=self.args.lr_pick if self.args.__dict__.get('AdaReg_only_picking') else self.args.lr_prune, 
                                momentum=self.args.momentum,
                                weight_decay=self.args.weight_decay)
        self.optimizer.load_state_dict(state['optimizer'])
        self.prune_state = state['prune_state']
        self.total_iter = state['iter']
        self.iter_stabilize_reg = state.get('iter_stabilize_reg', math.inf)
        self.reg = state['reg']
        self.hist_mag_ratio = state['hist_mag_ratio']

    def _save_model(self, acc1=0, acc5=0, mark=''):
        state = {'iter': self.total_iter,
                'prune_state': self.prune_state, # we will resume prune_state
                'arch': self.args.arch,
                'model': self.model,
                'state_dict': self.model.state_dict(),
                'iter_stabilize_reg': self.iter_stabilize_reg,
                'acc1': acc1,
                'acc5': acc5,
                'optimizer': self.optimizer.state_dict(),
                'reg': self.reg,
                'hist_mag_ratio': self.hist_mag_ratio,
                'ExpID': self.logger.ExpID,
        }
        self.save(state, is_best=False, mark=mark)

    def prune(self):
        self.model = self.model.train()
        self.optimizer = optim.SGD(self.model.parameters(), 
                                lr=self.args.lr_pick if self.args.__dict__.get('AdaReg_only_picking') else self.args.lr_prune,
                                momentum=self.args.momentum,
                                weight_decay=self.args.weight_decay)
        
        # resume model, optimzer, prune_status
        self.total_iter = -1
        if self.args.resume_path:
            self._resume_prune_status(self.args.resume_path)
            self._get_kept_wg_L1() # get pruned and kept wg from the resumed model
            self.model = self.model.train()
            self.logprint("Resume model successfully: '{}'. Iter = {}. prune_state = {}".format(
                        self.args.resume_path, self.total_iter, self.prune_state))

        acc1 = acc5 = 0
        epoch = 0
        total_iter_reg = self.args.reg_upper_limit / self.args.reg_granularity_prune * self.args.update_reg_interval + self.args.stabilize_reg_interval
        timer = Timer(total_iter_reg / self.args.print_interval)
        while True:
            epoch += 1
            for _, (inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                self.total_iter += 1
                total_iter = self.total_iter
                    
                if total_iter % self.args.print_interval == 0:
                    self.logprint("")
                    self.logprint("Iter = %d [prune_state = %s, method = %s] " 
                        % (total_iter, self.prune_state, self.args.method) + "-"*40)
                    
                # forward
                self.model.train()
                y_ = self.model(inputs)
                
                if self.prune_state == "update_reg" and total_iter % self.args.update_reg_interval == 0:
                    self._update_reg()
                    
                # normal training forward
                loss = self.criterion(y_, targets)
                logtmp = f'loss_cls {loss:.4f}'

                # GReg implemented via loss
                if self.args.greg_via_loss:
                    loss_greg, lw = 0, 0
                    for name, module in self.model.named_modules():
                        if isinstance(module, self.learnable_layers) and self.pr[name] > 0:
                            w = module.weight
                            w = w.view(w.size(0), -1)
                            mask = torch.ones_like(w).cuda()
                            mask[self.kept_wg[name], :] = 0 # only apply reg to the pruned wgs
                            loss_greg += 0.5 * (w * w * mask).sum()
                            lw = self.reg[name].max()
                    loss += loss_greg * lw
                    logtmp += f' loss_greg (*{lw}) {loss_greg:.4f}'

                # OPP regularization
                if self.args.lw_opp:
                    loss_opp = 0
                    lw_opp = self.args.lw_opp
                    for name, module in self.model.named_modules():
                        if isinstance(module, self.learnable_layers):
                            if self.args.opp_scheme in ['1', 'v1']:
                                shape = self.layers[name].size
                                if len(shape) == 2 or shape[-1] == 1: # FC and 1x1 conv 
                                    loss_opp += orth_regularization(module.weight)
                                else:
                                    loss_opp += deconv_orth_dist(module.weight)
                            
                            elif self.args.opp_scheme in ['2', 'v2']:
                                loss_opp += orth_regularization(module.weight, transpose=self.args.transpose)
                            
                            elif self.args.opp_scheme in ['3', 'v3']:
                                if self.pr[name] > 0:
                                    loss_opp += orth_regularization_v3(module.weight, pruned_wg=self.pruned_wg[name])
                            
                            elif self.args.opp_scheme in ['4', 'v4']:
                                if self.pr[name] > 0:
                                    loss1, loss2 = orth_regularization_v4(module.weight, self.original_column_gram[name], pruned_wg=self.pruned_wg[name])
                                    loss_opp += loss1 + loss2
                                    if self.total_iter % self.args.print_interval == 0:
                                        self.logprint(f'{name} [pr {self.pr[name]}] -- loss row {loss1:.8f} loss column {loss2:.8f}')
                            
                            elif self.args.opp_scheme in ['5', 'v5']:
                                if self.pr[name] > 0:
                                    loss_opp += orth_regularization_v5(module.weight, pruned_wg=self.pruned_wg[name])
                                    lw_opp = self.args.lw_opp * self.reg[name].max()
                            
                            elif self.args.opp_scheme in ['5_2', 'v5_2']:
                                if self.pr[name] > 0:
                                    loss_opp += orth_regularization_v5_2(module.weight, pruned_wg=self.pruned_wg[name])
                                    lw_opp = self.args.lw_opp * self.reg[name].max()
                            
                            elif self.args.opp_scheme in ['6', 'v6']:
                                if self.pr[name] > 0:
                                    n_filters = module.weight.data.shape[0]
                                    penalty_map = torch.ones(n_filters, n_filters).cuda() * self.args.lw_orth_reg
                                    penalty_map[self.pruned_wg[name], :] = self.args.lw_opp * self.reg[name].max()
                                    penalty_map[:, self.pruned_wg[name]] = self.args.lw_opp * self.reg[name].max()
                                    loss_opp += orth_regularization_v6(module.weight, self.pruned_wg[name], penalty_map)
                                    lw_opp = 1 # to maintain interface
                            else:
                                raise NotImplementedError

                    loss += lw_opp * loss_opp
                    logtmp += f' loss_opp (*{lw_opp}) {loss_opp:.4f}'
                
                # print loss
                if self.total_iter % self.args.print_interval == 0:
                    logtmp += f' Iter {self.total_iter}'
                    self.logprint(logtmp)

                self.optimizer.zero_grad()
                loss.backward()
                
                # after backward but before update, apply reg to the grad
                if not self.args.not_apply_reg:
                    self._apply_reg()
                self.optimizer.step()

                # test
                if total_iter % self.args.test_interval == 0:
                    acc1, acc5, *_ = self.test(self.model)
                    self.accprint("Acc1 = %.4f Acc5 = %.4f Iter = %d (after update) [prune_state = %s, method = %s]" % 
                        (acc1, acc5, total_iter, self.prune_state, self.args.method))
                
                # save model (save model before a batch starts)
                if total_iter % self.args.save_interval == 0:
                    self._save_model(acc1, acc5)
                    self.logprint('Periodically save model done. Iter = {}'.format(total_iter))

                # log print
                if total_iter % self.args.print_interval == 0:
                    w_abs_sum = 0
                    w_num_sum = 0
                    cnt_m = 0
                    for _, m in self.model.named_modules():
                        if isinstance(m, self.learnable_layers):
                            cnt_m += 1
                            w_abs_sum += m.weight.abs().sum()
                            w_num_sum += m.weight.numel()

                    _, predicted = y_.max(1)
                    correct = predicted.eq(targets).sum().item()
                    train_acc = correct / targets.size(0)
                    self.logprint("After optim update, ave_abs_weight: %.10f current_train_loss: %.4f current_train_acc: %.4f" %
                        (w_abs_sum / w_num_sum, loss.item(), train_acc))
                
                # change prune state
                if self.prune_state == "stabilize_reg" and total_iter - self.iter_stabilize_reg == self.args.stabilize_reg_interval:
                    self._prune_and_build_new_model() 
                    self.logprint("'stabilize_reg' is done. Pruned, go to 'finetune'. Iter = %d" % total_iter)
                    return copy.deepcopy(self.model)
                
                if total_iter % self.args.print_interval == 0:
                    self.logprint(f"predicted_finish_time of reg: {timer()}")
            
            # after each epoch training, reinit
            if epoch % self.args.reinit_interval == 0:
                acc1_before, *_ = self.test(self.model)
                self.model = reinit_model(self.model, args=self.args, mask=None, print=self.logprint)
                acc1_after, *_ = self.test(self.model)
                self.logprint(f'Before reinit, acc1 {acc1_before:.4f} after reinit, acc1 {acc1_after:.4f}')