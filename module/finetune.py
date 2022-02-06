from utils import *

def finetune(loader, train_loader, test_loader, criterion, model, args,
			 train_index, train_loss_list, train_acc1_list,
			 test_index, test_loss_list, test_acc1_list):
	
	# optimizer
	optimizer = optimizer_dict(model, args)
	scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

	for epoch in range(args.ft_epoch):
		# train
		_ = train(train_loader, model, criterion, optimizer, epoch, args)
		scheduler.step()

		# validate on test
		model.reset()
		test_acc1, test_acc5, test_loss, test_epoch_index = validate(test_loader, model, criterion, args)
		test_index.append(test_epoch_index)
		model.next_epoch('test')
		test_acc1_list.append(test_acc1)
		test_loss_list.append(test_loss)

		# validate on train
		model.reset()
		train_acc1, train_acc5, train_loss, train_epoch_index = validate(train_loader, model, criterion, args)
		train_index.append(train_epoch_index)
		model.next_epoch('train')
		train_acc1_list.append(train_acc1)
		train_loss_list.append(train_loss)

	return [loader, train_loader, test_loader, criterion, model, args,
	train_index, train_loss_list, train_acc1_list,
	test_index, test_loss_list, test_acc1_list]




