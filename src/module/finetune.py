from utils import optimizer_dict

def finetune(cagos):
	
	# optimizer
	optimizer = optimizer_dict(cagos['model'], cagos['args'])
	scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

	for epoch in range(cagos['args'].ft_epoch):
		# train
		_, epoch_mean, epoch_std, epoch_abs_mean, epoch_abs_std = train(cagos['loader'], 
																		cagos['loader'].train_loader, 
																		cagos['model'], 
																		cagos['criterion'], 
																		optimizer, 
																		epoch, 
																		cagos['args']
																		)
		cagos['grad_mean'].append(epoch_mean)
		cagos['grad_std'].append(epoch_std)
		cagos['grad_abs_mean'].append(epoch_abs_mean)
		cagos['grad_abs_std'].append(epoch_abs_std)
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



