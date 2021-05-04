import torch
import torch.nn as nn
import torch.optim as optim
import os, copy, time, pickle, numpy as np, math
from .meta_pruner import MetaPruner
from .reinit_model import reinit_model, orth_regularization, orth_regularization_v3, orth_regularization_v4, deconv_orth_dist
from utils import cal_correlation, Timer
import matplotlib.pyplot as plt
from collections import OrderedDict
import torch.nn.functional as F
from sklearn.cluster import KMeans
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
        
        # init: get pruned weight groups
        self.reg_multiplier = 0
        self._get_kept_wg()

        self.prune_state = "update_reg"
        for name, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                shape = m.weight.data.shape

                # initialize reg
                if self.args.wg == 'weight':
                    self.reg[name] = torch.zeros_like(m.weight.data).flatten().cuda()
                else:
                    self.reg[name] = torch.zeros(shape[0], shape[1]).cuda() 
                
        # init original_column_gram
        self.original_column_gram = OrderedDict()
        for name, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                w = m.weight.data
                w = w.view(w.size(0), -1)
                self.original_column_gram[name] = w.t() @ w

    def _cal_distance(self, w1, w2):
        out = F.cosine_similarity(w1, w2, dim=0)
        return out.item()
        # cal_correlation(w.t(), coef=True) # can be replaced as cosine distance

    def _get_kept_wg(self):
        '''Get the pruned and kept weight groups (i.e., filters in this case). 
        And, for the pruned weight groups, get the replacing mapping.
        '''
        self.wg_clusters = {}
        for n, m in self.model.named_modules():
            if isinstance(m, self.learnable_layers):
                self.pruned_wg[n] = []
                n_filter = m.weight.size(0)
                w = m.weight.data.view(n_filter, -1)
                
                # weight group clustering
                if self.pr[n] > 0:
                    n_pruned = min(math.ceil(self.pr[n] * n_filter), n_filter - 1) # at least, keep one wg
                    n_kept = n_filter - n_pruned
                    kmeans = KMeans(n_clusters=n_kept, random_state=0).fit(w.cpu().data.numpy())
                    self.wg_clusters[n] = {}
                    for wg_ix, label in enumerate(kmeans.labels_):
                        if label in self.wg_clusters[n]:
                            self.wg_clusters[n][label] += [wg_ix]
                            self.pruned_wg[n] += [wg_ix] # will prune all the filters except the 1st one in each cluster
                        else:
                            self.wg_clusters[n][label] = [wg_ix]
                    self.logprint(f'{n} -- wg_clusters: {self.wg_clusters[n]} (n_kept/n_filter = {n_kept}/{n_filter})')
                else:
                    self.wg_clusters[n] = {i:[i] for i in range(n_filter)}
                
                self.kept_wg[n] = [x for x in range(n_filter) if x not in self.pruned_wg[n]]

    def _merge_channels(self):
        '''Merge channels in a filter.
        '''
        for name, m in self.model.named_modules():
            if isinstance(m, self.learnable_layers):
                # zero out filters that are going to be pruned
                pruned_filter = self.pruned_wg[name]
                m.weight.data[pruned_filter] *= 0
                next_bn = self._next_bn(self.model, m)

                # for a filter, merge all the channels of the same cluster to the 1st channel
                prev_learnable_layer = self._prev_learnable_layer(self.model, name, m) # previous learnable layer
                if prev_learnable_layer:
                    for label, wg_ixs in self.wg_clusters[prev_learnable_layer].items():
                        if len(wg_ixs) > 1:
                            m.weight[:, wg_ixs[0]] = m.weight[:, wg_ixs].sum(dim=1)
                    self.logprint(f'{name} merge channels: {self.wg_clusters[prev_learnable_layer]}')
            
            elif isinstance(m, nn.BatchNorm2d) and m == next_bn:
                m.weight.data[pruned_filter] *= 0
                m.bias.data[pruned_filter] *= 0

    def _get_loss_reg(self, print_log=False):
        loss_reg = 0
        for n, m in self.model.named_modules():
            if n in self.wg_clusters:
                w = m.weight
                w = w.view(w.size(0), -1)
                for label, wg_ixs in self.wg_clusters[n].items():
                    if len(wg_ixs) > 1:
                        center = w[wg_ixs].mean(dim=0) # wg center, wg is filter here
                        for ix in wg_ixs:
                            loss_reg += F.l1_loss(w[ix], center)
        return loss_reg

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

        t1 = time.time()
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
                
                # test
                if total_iter % self.args.test_interval == 0:
                    acc1, acc5, *_ = self.test(self.model)
                    self.accprint("Acc1 = %.4f Acc5 = %.4f Iter = %d (before update) [prune_state = %s, method = %s]" % 
                        (acc1, acc5, total_iter, self.prune_state, self.args.method))
                
                # save model (save model before a batch starts)
                if total_iter % self.args.save_interval == 0:
                    self._save_model(acc1, acc5)
                    self.logprint('Periodically save model done. Iter = {}'.format(total_iter))
                    
                if total_iter % self.args.print_interval == 0:
                    self.logprint("")
                    self.logprint("Iter = %d [prune_state = %s, method = %s] " 
                        % (total_iter, self.prune_state, self.args.method) + "-"*40)
                    
                # forward
                self.model.train()
                y_ = self.model(inputs)
                
                if self.prune_state == "update_reg" and total_iter % self.args.update_reg_interval == 0:
                    self.reg_multiplier += self.args.reg_granularity_prune
                    if self.reg_multiplier >= self.args.reg_upper_limit:
                        self.prune_state = "stabilize_reg"
                        self.iter_stabilize_reg = self.total_iter
                    
                # normal training forward
                loss_cls = self.criterion(y_, targets)
                loss_reg = self._get_loss_reg(print_log=self.args.verbose)
                loss = loss_cls + loss_reg * self.reg_multiplier
                # loss = loss_cls
                logtmp = f'loss_cls {loss_cls:.4f} loss_reg (*{self.reg_multiplier}) {loss_reg:.4f}'

                # print loss
                if self.total_iter % self.args.print_interval == 0:
                    self.logprint(logtmp)

                self.optimizer.zero_grad()
                loss.backward()
                
                # after backward but before update, apply reg to the grad
                self.optimizer.step()

                # log print
                if total_iter % self.args.print_interval == 0:
                    _, predicted = y_.max(1)
                    correct = predicted.eq(targets).sum().item()
                    train_acc = correct / targets.size(0)
                    self.logprint("After optim update, current_train_loss: %.4f current_train_acc: %.4f" % (loss.item(), train_acc))
                
                # change prune state
                if self.prune_state == "stabilize_reg" and total_iter - self.iter_stabilize_reg == self.args.stabilize_reg_interval:
                    acc1, *_ = self.test(self.model)
                    self.logprint(f"'stabilize_reg' is done. Acc1 {acc1}. Iter {total_iter}")
                    self._merge_channels()
                    acc1, *_ = self.test(self.model)
                    self.logprint(f'Merge channels and zero out pruned weight groups. Acc1 {acc1}')
                    
                    # # --- check accuracy to make sure '_prune_and_build_new_model' works normally
                    # # checked. works normally!
                    # for name, m in self.model.named_modules():
                    #     if isinstance(m, self.learnable_layers):
                    #         pruned_filter = self.pruned_wg[name]
                    #         m.weight.data[pruned_filter] *= 0
                    #         next_bn = self._next_bn(self.model, m)
                    #     elif isinstance(m, nn.BatchNorm2d) and m == next_bn:
                    #         m.weight.data[pruned_filter] *= 0
                    #         m.bias.data[pruned_filter] *= 0

                    # acc1_before, *_ = self.test(self.model)
                    # self._prune_and_build_new_model()
                    # acc1_after, *_ = self.test(self.model)
                    # print(acc1_before, acc1_after)
                    # exit()
                    # # ---
                    
                    self._prune_and_build_new_model() 
                    acc1, *_ = self.test(self.model)
                    self.logprint(f"Model is pruned and a new model is built. Acc1 {acc1:.4f}")
                    exit()
                    return copy.deepcopy(self.model)

                if total_iter % self.args.print_interval == 0:
                    t2 = time.time()
                    total_time = t2 - t1
                    self.logprint(f"predicted_finish_time of reg: {timer()}")
                    t1 = t2
            
            # after each epoch training, reinit
            if epoch % self.args.reinit_interval == 0:
                acc1_before, *_ = self.test(self.model)
                self.model = reinit_model(self.model, args=self.args, mask=None, print=self.logprint)
                acc1_after, *_ = self.test(self.model)
                self.logprint(f'Before reinit, acc1 {acc1_before:.4f} after reinit, acc1 {acc1_after:.4f}')