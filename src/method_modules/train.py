

import torch
import torch.nn as nn
from utils import PresetLRScheduler, adjust_learning_rate, AverageMeter, ProgressMeter, accuracy
from utils import get_n_params, get_n_flops, get_n_params_, get_n_flops_
from utils import add_noise_to_model, compute_jacobian, _weights_init_orthogonal, get_jacobian_singular_values
from utils import Timer
import shutil, time, os
import numpy as np
pjoin = os.path.join

def save_ckpt(save_dir, ckpt, is_best=False, mark=''):
    out = pjoin(save_dir, "ckpt_last.pth")
    torch.save(ckpt, out)
    if is_best:
        out_best = pjoin(save_dir, "ckpt_best.pth")
        torch.save(ckpt, out_best)
    if mark:
        out_mark = pjoin(save_dir, "ckpt_{}.pth".format(mark))
        torch.save(ckpt, out_mark)

def one_epoch_train(train_loader, model, criterion, optimizer, epoch, args, print_log=True):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # Switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # Measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        if hasattr(args, 'advanced_lr'):
            lr = adjust_learning_rate_v2(optimizer, epoch, i, len(train_loader))
            args.advanced_lr.lr = lr
            if i == 10: print(f'==> Set LR to {lr:.6f} Epoch {epoch} Iter {i}')

        # Compute output
        output = model(images)
        loss = criterion(output, target)

        # Measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # Orth regularization
        if args.orth_reg_iter_ft:
            loss_orth_reg, cnt = 0, -1
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    cnt += 1
                    if args.orth_reg_method in ['CVPR20']:
                        if cnt != 0: # per the CVPR20 paper, do not reg the 1st conv
                            shape = module.weight.shape
                            if len(shape) == 2 or shape[-1] == 1: # FC and 1x1 conv 
                                loss_orth_reg += orth_dist(module.weight)
                            else:
                                loss_orth_reg += deconv_orth_dist(module.weight)
                    elif args.orth_reg_method in ['CVPR17']:
                        loss_orth_reg += orth_dist(module.weight)
                    else:
                        raise NotImplementedError
            loss += loss_orth_reg * args.lw_orth_reg
            if i % args.print_interval == 0:
                print(f'loss_orth_reg (*{args.lw_orth_reg}) {loss_orth_reg:.10f} Epoch {epoch} Iter {i}')

        # Compute gradient and SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # After update, zero out pruned weights
        if args.wg == 'weight'and hasattr(model, 'mask'):
            apply_mask_forward(model, model.mask)

        # Util functionality, check the gradient norm of params
        if hasattr(args, 'utils') and args.utils.check_grad_norm:
            from utils import check_grad_norm
            if i % args.print_interval == 0:
                print(''); print(f'(** Start check_grad_norm. Epoch {epoch} Step {i} **)')
                check_grad_norm(model)
                print(f'(** End check_grad_norm **)'); print('')

        # Util functionality, check the gradient norm of params
        if hasattr(args, 'utils') and args.utils.check_weight_stats:
            from utils import check_weight_stats
            if i % args.print_interval == 0:
                print(''); print(f'(** Start check_weight_stats. Epoch {epoch} Step {i} **)')
                check_weight_stats(model)
                print(f'(** End check_weight_stats **)'); print('')

        # Check Jacobian singular value (JSV) # TODO-@mst: move to utility
        if args.jsv_interval == -1:
            args.jsv_interval = len(train_loader) # default: check jsv at the last iteration
        if args.jsv_loop and (i + 1) % args.jsv_interval == 0:
            jsv, jsv_diff, cn = get_jacobian_singular_values(model, train_loader, num_classes=args.passer['num_classes'], n_loop=args.jsv_loop, print_func=print, rand_data=args.jsv_rand_data)
            print('JSV_mean %.4f JSV_std %.4f JSV_max %.4f JSV_min %.4f Condition_Number_mean %.4f JSV_diff_mean %.4f JSV_diff_std %.4f -- Epoch %d Iter %d' % 
                (np.mean(jsv), np.std(jsv), np.max(jsv), np.min(jsv), np.mean(cn), np.mean(jsv_diff), np.std(jsv_diff), epoch, i))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if print_log and i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args, noisy_model_ensemble=False):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    train_state = model.training

    # switch to evaluate mode
    model.eval()

    # @mst: add noise to model
    model_ensemble = []
    if noisy_model_ensemble:
        for i in range(args.model_noise_num):
            noisy_model = add_noise_to_model(model, std=args.model_noise_std)
            model_ensemble.append(noisy_model)
        print('==> added Gaussian noise to model weights (std=%s, num=%d)' % (args.model_noise_std, args.model_noise_num))
    else:
        model_ensemble.append(model)

    time_compute = []
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            t1 = time.time()
            output = 0
            for model in model_ensemble: # @mst: test model ensemble
                output += model(images)
            output /= len(model_ensemble)
            time_compute.append((time.time() - t1) / images.size(0))
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        # print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
        #       .format(top1=top1, top5=top5))
        # @mst: commented because we will use another print outside 'validate'
    # print("time compute: %.4f ms" % (np.mean(time_compute)*1000))

    # change back to original model state if necessary
    if train_state:
        model.train()
    return top1.avg.item(), top5.avg.item(), losses.avg # @mst: added returning top5 acc and loss

def adjust_learning_rate_v2(optimizer, epoch, iteration, num_iter):
    '''More advanced LR scheduling. Refers to d-li14 MobileNetV2 ImageNet implementation:
    https://github.com/d-li14/mobilenetv2.pytorch/blob/1733532bd43743442077326e1efc556d7cfd025d/imagenet.py#L374
    '''
    assert hasattr(args, 'advanced_lr')
    
    warmup_iter = args.advanced_lr.warmup_epoch * num_iter # num_iter: num_iter_per_epoch
    current_iter = iteration + epoch * num_iter
    max_iter = args.epochs * num_iter

    if epoch < args.advanced_lr.warmup_epoch:
        lr = args.lr * current_iter / warmup_iter
    else:
        if args.advanced_lr.lr_decay == 'step':
            lr = args.lr * (args.advanced_lr.gamma ** ((current_iter - warmup_iter) / (max_iter - warmup_iter)))
        elif args.advanced_lr.lr_decay == 'cos':
            lr = args.lr * (1 + math.cos(math.pi * (current_iter - warmup_iter) / (max_iter - warmup_iter))) / 2
        elif args.advanced_lr.lr_decay == 'linear':
            lr = args.lr * (1 - (current_iter - warmup_iter) / (max_iter - warmup_iter))
        elif args.advanced_lr.lr_decay == 'schedule':
            count = sum([1 for s in args.advanced_lr.schedule if s <= epoch])
            lr = args.lr * pow(args.advanced_lr.gamma, count)
        else:
            raise ValueError('Unknown lr mode {}'.format(args.advanced_lr.lr_decay))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def apply_mask_forward(model, mask):
    for name, m in model.named_modules():
        if name in mask:
            m.weight.data.mul_(mask[name])

def train(model, loader, args, logger, passer):
    train_loader = loader.train_loader
    val_loader = loader.test_loader
    best_acc1, best_acc1_epoch = 0, 0
    print_log = True
    accprint = logger.accprint
    criterion = passer['criterion']

    # since model is new, we need a new optimizer
    if args.solver == 'Adam':
        print('==> Start to finetune: using Adam optimizer')
        optimizer = torch.optim.Adam(model.parameters(), args.lr)
    else:
        print('==> Start to finetune: using SGD optimizer')
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    
    # set lr finetune schduler for finetune
    if args.pipeline:
        assert args.lr_ft is not None
        lr_scheduler = PresetLRScheduler(args.lr_ft)
    
    acc1_list, loss_train_list, loss_test_list, last_lr  = [], [], [], 0
    timer = Timer(args.epochs - args.start_epoch)
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        # @mst: use our own lr scheduler
        if not hasattr(args, 'advanced_lr'): # 'advanced_lr' can override 'lr_scheduler' and 'adjust_learning_rate'
            lr = lr_scheduler(optimizer, epoch) if args.pipeline else adjust_learning_rate(optimizer, epoch, args)
            if print_log:
                print("==> Set lr = %s @ Epoch %d begins" % (lr, epoch))

            # save model if LR just changed
            if last_lr !=0 and lr != last_lr:
                state = {'epoch': epoch, # this is to save the model of last epoch
                        'arch': args.arch,
                        'model': model,
                        'state_dict': model.state_dict(),
                        'acc1': acc1,
                        'acc5': acc5,
                        'optimizer': optimizer.state_dict(),
                        'ExpID': logger.ExpID,
                        'prune_state': 'finetune',
                }
                if args.wg == 'weight':
                    state['mask'] = mask 
                save_ckpt(save_dir, state, mark=f'lr{last_lr}_epoch{epoch}')
                print(f'==> Save ckpt at the last epoch ({epoch}) of LR {last_lr}')

        # train for one epoch
        one_epoch_train(train_loader, model, criterion, optimizer, epoch, args, print_log=print_log)
        if hasattr(args, 'advanced_lr'): # advanced_lr will adjust lr inside the train fn
            lr = args.advanced_lr.lr
        last_lr = lr

        # @mst: check weights magnitude during finetune
        if args.pipeline in ['GReg-1', 'GReg-2'] and not isinstance(pruner, type(None)):
            for name, m in model.named_modules():
                if name in pruner.reg:
                    ix = pruner.layers[name].layer_index
                    mag_now = m.weight.data.abs().mean()
                    mag_old = pruner.original_w_mag[name]
                    ratio = mag_now / mag_old
                    tmp = '[%2d] %25s -- mag_old = %.4f, mag_now = %.4f (%.2f)' % (ix, name, mag_old, mag_now, ratio)
                    original_print(tmp, file=logger.logtxt, flush=True)
                    if args.screen_print:
                        print(tmp)

        # evaluate on validation set
        acc1, acc5, loss_test = validate(val_loader, model, criterion, args) # @mst: added acc5
        if args.dataset not in ['imagenet'] and args.test_trainset: # too costly, not test for now
            acc1_train, acc5_train, loss_train = validate(train_loader, model, criterion, args)
        else:
            acc1_train, acc5_train, loss_train = -1, -1, -1
        acc1_list.append(acc1)
        loss_train_list.append(loss_train)
        loss_test_list.append(loss_test)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        if is_best:
            best_acc1_epoch = epoch
            best_loss_train = loss_train
            best_loss_test = loss_test
        if print_log:
            accprint("Acc1 %.4f Acc5 %.4f Loss_test %.4f | Acc1_train %.4f Acc5_train %.4f Loss_train %.4f | Epoch %d (Best_Acc1 %.4f @ Best_Acc1_Epoch %d) lr %s" % 
                (acc1, acc5, loss_test, acc1_train, acc5_train, loss_train, epoch, best_acc1, best_acc1_epoch, lr))
            print('predicted finish time: %s' % timer())

        ngpus_per_node = torch.cuda.device_count()
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
                ckpt = {'epoch': epoch + 1,
                        'arch': args.arch,
                        'model': model,
                        'state_dict': model.state_dict(),
                        'acc1': acc1,
                        'acc5': acc5,
                        'optimizer': optimizer.state_dict(),
                        'ExpID': logger.ExpID,
                        'prune_state': 'finetune',
                }
                if args.wg == 'weight' and hasattr(model, 'mask'):
                    ckpt['mask'] = model.mask
                save_ckpt(logger.weights_path, ckpt, is_best)
    
    print(f'==> Train is done.')
    return model
