from utils import optimizer_dict, AverageMeter, ProgressMeter
import torch
import time

def train(model, loader, criterion, args, logger):
	
	# optimizer
	optimizer = optimizer_dict(model, args)
	scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

	for epoch in range(args.ft_epoch):
		# train
		one_epoch_train(loader, model, criterion, optimizer, epoch, args)
		scheduler.step()

		# validate on test
		cagos['model'].reset()
		test_acc1, test_acc5, test_loss, test_epoch_index = validate(cagos['loader'].test_loader, 
																	 cagos['model'], 
																	 cagos['criterion'], 
																	 cagos['args']
																	 )
		cagos['test_index'].append(test_epoch_index)
		cagos['model'].next_epoch('test')
		cagos['test_acc1_list'].append(test_acc1)
		cagos['test_loss_list'].append(test_loss)

		# validate on train
		cagos['model'].reset()
		train_acc1, train_acc5, train_loss, train_epoch_index = validate(cagos['loader'].train_loader, 
																		 cagos['model'], 
																		 cagos['criterion'], 
																		 cagos['args']
																		 )
		cagos['train_index'].append(train_epoch_index)
		cagos['model'].next_epoch('train')
		cagos['train_acc1_list'].append(train_acc1)
		cagos['train_loss_list'].append(train_loss)

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

