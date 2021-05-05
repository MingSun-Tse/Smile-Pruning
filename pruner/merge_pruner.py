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
        self.iter_stabilize_reg = math.inf
        
        # init: get pruned weight groups
        self.prune_state = "update_reg"
        self.reg_multiplier = 0
        self._get_kept_wg(clustering=args.clustering)
        self.next_bn = {}
        for n, m in self.model.named_modules():
            if isinstance(m, self.learnable_layers):
                next_bn = self._next_bn(self.model, m)
                self.next_bn[n] = next_bn

    def _get_kept_wg(self, clustering):
        '''Get the pruned and kept weight groups (i.e., filters in this case). 
        And, for the pruned weight groups, get the replacing mapping.
        '''
        self.wg_clusters = {}
        if clustering == 'l1':
            self._get_kept_wg_L1() # update self.kept_wg and self.pruned_wg
            for n, m in self.model.named_modules():
                if isinstance(m, self.learnable_layers):
                    self.wg_clusters[n] = {}
                    kept_wg, pruned_wg = self.kept_wg[n], self.pruned_wg[n]
                    n_kept, n_filter = len(kept_wg), len(kept_wg) + len(pruned_wg)
                    # for kept wgs, each wg (filter here) makes a cluster alone
                    for i in range(n_kept):
                        self.wg_clusters[n][i] = [kept_wg[i]]
                    
                    # for pruned wgs, assign each to a cluster based on affinity
                    for i in pruned_wg:
                        # find the closest wg in kept_wg
                        min_dist = 1e10
                        for j in kept_wg:
                            dist = F.l1_loss(m.weight.data[i], m.weight.data[j]).item() # this metric may need improvement
                            if dist < min_dist:
                                closest_wg = j
                                min_dist = dist
                        closest_wg_ix = kept_wg.index(closest_wg)
                        self.wg_clusters[n][closest_wg_ix] += [i]
                    if self.pr[n] > 0:
                        self.logprint(f'{n} -- wg_clusters: {self.wg_clusters[n]} (n_kept/n_filter = {n_kept}/{n_filter})')
        else:
            for n, m in self.model.named_modules():
                if isinstance(m, self.learnable_layers):
                    self.pruned_wg[n] = []
                    n_filter = m.weight.size(0)
                    w = m.weight.data.view(n_filter, -1)
                    
                    # weight group clustering
                    if self.pr[n] > 0:
                        n_pruned = min(math.ceil(self.pr[n] * n_filter), n_filter - 1) # at least, keep one wg
                        n_kept = n_filter - n_pruned
                        if clustering == 'kmeans':
                            kmeans = KMeans(n_clusters=n_kept, random_state=0).fit(w.cpu().data.numpy())
                            labels = kmeans.labels_
                        elif clustering == 'random':
                            labels = []
                            for i in range(n_filter):
                                labels += [torch.randint(n_kept, (1,)).item()]
                        self.wg_clusters[n] = {}
                        for wg_ix, label in enumerate(labels):
                            if label in self.wg_clusters[n]:
                                self.wg_clusters[n][label] += [wg_ix]
                                self.pruned_wg[n] += [wg_ix] # will prune all the filters except the 1st one in each cluster
                            else:
                                self.wg_clusters[n][label] = [wg_ix]
                        self.logprint(f'{n} -- wg_clusters: {self.wg_clusters[n]} (n_kept/n_filter = {n_kept}/{n_filter})')
                    else:
                        self.wg_clusters[n] = {i:[i] for i in range(n_filter)}
                    self.kept_wg[n] = [x for x in range(n_filter) if x not in self.pruned_wg[n]]

    def _get_loss_reg(self, print_log=False):
        metric = F.l1_loss
        loss_reg_w = loss_reg_bn = 0
        for n, m in self.model.named_modules():
            if n in self.wg_clusters:
                bias = False if isinstance(m.bias, type(None)) else True
                next_bn = self.next_bn[n]
                for _, wg_ixs in self.wg_clusters[n].items():
                    # choose the 1st wg as center
                    if len(wg_ixs) > 1:
                        loss_reg_w += metric( m.weight[wg_ixs[1:]], m.weight[wg_ixs[0]].unsqueeze(0).detach() )
                        if bias:
                            loss_reg_w += metric( m.bias[wg_ixs[1:]], m.bias[wg_ixs[0]].unsqueeze(0).detach() )
                        if self.args.consider_bn and next_bn:
                            loss_reg_bn += metric( next_bn.weight[wg_ixs[1:]], next_bn.weight[wg_ixs[0]].unsqueeze(0).detach() )
                            loss_reg_bn += metric( next_bn.bias[  wg_ixs[1:]], next_bn.bias[  wg_ixs[0]].unsqueeze(0).detach() )

                    # # choose the averaged wg as center 
                    # if len(wg_ixs) > 1:
                    #     center = m.weight[wg_ixs].mean(dim=0, keepdim=True) # [1,C,H,W], wg center, wg is filter here
                    #     loss_reg_w += F.l1_loss(m.weight[wg_ixs], center)
                    #     if bn:
                    #         center = bn.weight[wg_ixs].mean(dim=0, keepdim=True) # [1], bn center
                    #         loss_reg_bn += F.l1_loss(bn.weight[wg_ixs], center)
                    #         center = bn.bias[wg_ixs].mean(dim=0, keepdim=True)
                    #         loss_reg_bn += F.l1_loss(bn.bias[wg_ixs], center)

        if self.args.consider_bn:
            loss_reg = loss_reg_w + loss_reg_bn
            logstr = f'loss_reg_w {loss_reg_w:.6f} loss_reg_bn {loss_reg_bn:.6f}'
        else:
            loss_reg = loss_reg_w
            logstr = f'loss_reg_w {loss_reg_w:.6f}'
        
        if self.total_iter % self.args.print_interval == 0:
            self.logprint(logstr)
        return loss_reg
    
    def _merge_channels(self):
        '''Merge channels in a filter.
        '''
        for name, m in self.model.named_modules():
            if isinstance(m, self.learnable_layers):
                # zero out filters that are going to be pruned
                pruned_filter = self.pruned_wg[name]
                m.weight.data[pruned_filter] *= 0
                bias = False if isinstance(m.bias, type(None)) else True
                if bias:
                    m.bias.data[pruned_filter] *= 0
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
    
    def _share_weights_for_filters(self):
        '''Explicitly make filters in the same cluster share weights, as well as the BN statistics.
        '''
        for name, m in self.model.named_modules():
            if isinstance(m, self.learnable_layers):
                for label, wg_ixs in self.wg_clusters[name].items():
                    m.weight.data[wg_ixs] = m.weight.data[wg_ixs].mean(dim=0)
                    bias = False if isinstance(m.bias, type(None)) else True
                    if bias:
                        m.bias.data[wg_ixs] = m.bias.data[wg_ixs].mean(dim=0)
                next_bn = self._next_bn(self.model, m)
                learnable_layer_name = name
                
            elif isinstance(m, nn.BatchNorm2d) and m == next_bn:
                for label, wg_ixs in self.wg_clusters[learnable_layer_name].items():
                    m.weight.data[wg_ixs] = m.weight.data[wg_ixs].mean(dim=0)
                    m.bias.data[wg_ixs] = m.bias.data[wg_ixs].mean(dim=0)
                    m.running_mean[wg_ixs] = m.running_mean[wg_ixs].mean(dim=0)
                    m.running_var[wg_ixs] = m.running_var[wg_ixs].mean(dim=0)
                        
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
                    self.reg_multiplier += self.args.reg_granularity_prune
                    if self.reg_multiplier >= self.args.reg_upper_limit:
                        self.prune_state = "stabilize_reg"
                        self.iter_stabilize_reg = self.total_iter
                    
                # normal training forward
                loss = self.criterion(y_, targets)
                logtmp = f'loss_cls {loss:.4f}'
                if self.total_iter % self.args.interval_apply_cluster_reg == 0:
                    loss_reg = self._get_loss_reg(print_log=self.args.verbose)
                    loss += loss_reg * self.reg_multiplier
                    logtmp += f' loss_reg (*{self.reg_multiplier}) {loss_reg:.4f}'

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

                # test
                if total_iter % self.args.test_interval == 0:
                    acc1, acc5, *_ = self.test(self.model)
                    self.accprint("Acc1 = %.4f Acc5 = %.4f Iter = %d (after update) [prune_state = %s, method = %s]" % 
                        (acc1, acc5, total_iter, self.prune_state, self.args.method))
                
                # save model (save model before a batch starts)
                if total_iter % self.args.save_interval == 0:
                    self._save_model(acc1, acc5)
                    self.logprint('Periodically save model done. Iter = {}'.format(total_iter))

                # change prune state
                if self.prune_state == "stabilize_reg" and total_iter - self.iter_stabilize_reg == self.args.stabilize_reg_interval:
                    acc1, *_ = self.test(self.model)
                    self.logprint(f"'stabilize_reg' is done. Acc1 {acc1}. Iter {total_iter}")

                    # # --- check accuracy to make sure '_merge_channels' works normally
                    # # checked, works normally!
                    # self._share_weights_for_filters()
                    # acc1_before, *_ = self.test(self.model)
                    # self._merge_channels()
                    # acc1_after, *_ = self.test(self.model)
                    # print(acc1_before, acc1_after)
                    # exit()
                    # # ---

                    self._share_weights_for_filters()
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
                    return copy.deepcopy(self.model)

                if total_iter % self.args.print_interval == 0:
                    self.logprint(f"predicted_finish_time of reg: {timer()}")
            
            # after each epoch training, reinit
            if epoch % self.args.reinit_interval == 0:
                acc1_before, *_ = self.test(self.model)
                self.model = reinit_model(self.model, args=self.args, mask=None, print=self.logprint)
                acc1_after, *_ = self.test(self.model)
                self.logprint(f'Before reinit, acc1 {acc1_before:.4f} after reinit, acc1 {acc1_after:.4f}')