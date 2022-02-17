from utils import optimizer_dict, AverageMeter, ProgressMeter, accuracy, validate
import torch
import time
from pdb import set_trace as st

def train(model, loader, criterion, args, logger):
	# optimizer
	optimizer = optimizer_dict(model, args)
	scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

	for epoch in range(args.ft_epoch):
		# train
		one_epoch_train(loader, model, criterion, optimizer, epoch, args)
		scheduler.step()

		# validate on test
		test_acc1, test_acc5, test_loss = validate(loader.test_loader, model, criterion, args)
		logger.misc_results['test_acc1'] += [epoch, test_acc1]
		logger.misc_results['test_acc5'] += [epoch, test_acc5]
		logger.misc_results['test_loss'] += [epoch, test_loss]

		# validate on train
		train_acc1, train_acc5, train_loss = validate(loader.train_loader, model, criterion, args)
		logger.misc_results['train_acc1'] += [epoch, train_acc1]
		logger.misc_results['train_acc5'] += [epoch, train_acc5]
		logger.misc_results['train_loss'] += [epoch, train_loss]

def one_epoch_train(loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(loader.train_loader), [batch_time, data_time, losses, top1, top5], prefix="Epoch: [{}]".format(epoch))

    model.train()

    end = time.time()
    for i, (x, y) in enumerate(loader.train_loader):
        data_time.update(time.time() - end)

        x = x.cuda()
        y = y.cuda()

        y_ = model(x)
        loss = criterion(y_, y)

        if args.dataset == 'tsb':
            acc1, acc5 = accuracy(y_, y, topk=(1,2))
        else:
            acc1, acc5 = accuracy(y_, y, topk=(1,5))

        losses.update(loss.item(), x.size(0))
        top1.update(acc1[0], x.size(0))
        top5.update(acc5[0], x.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.ft_print_interval == 0:
            progress.display(i)

