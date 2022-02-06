from smilelogging import Logger
from data import Data
import torch.nn as nn
from model import model_dict
from module import module_dict
from utils import *
from option import args
import matplotlib.pyplot as plt
from pdb import set_trace as st
def main():
	# parallel/random setting may be added here

	main_worker(args)

def main_worker(args):
	# log
	logger = Logger(args)
	global print; print = logger.log_printer.logprint

	# data
	loader = Data(args)
	train_loader = loader.train_loader
	test_loader = loader.test_loader

	# loss
	criterion = loss_dict(args.loss).cuda()

	# model
	model = model_dict[args.model](args)
	# model = torch.nn.DataParallel(model).cuda()
	model = model.cuda()

	# pipeline
	# pipeline level variable
	train_index, train_loss_list, train_acc1_list = [], [], []
	test_index, test_loss_list, test_acc1_list = [], [], []
	input_bunch = [loader, train_loader, test_loader, criterion, model, args,
					train_index, train_loss_list, train_acc1_list,
					test_index, test_loss_list, test_acc1_list]

	pipeline = strlist_to_list(args.pipeline, str)
	for each_module_str in pipeline:
		each_module = module_dict[each_module_str]
		input_bunch = each_module(*input_bunch)

	# pipeline level analysis
	if args.ipPlot == 1:
		print('plot IP')
		information_plane_plot(input_bunch, args, logger, loader)

	if args.laPlot == 1:
		print('plot la')
		loss_acc_plot(input_bunch, args, logger)

	# edge_(input_bunch, args, logger, loader)
	# renyi_(input_bunch, args, logger, loader)

	
	

	

	

	




if __name__ == '__main__':
	main()