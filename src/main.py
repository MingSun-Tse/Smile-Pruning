from smilelogging import Logger
from dataset import Data
import torch.nn as nn
from models import model_dict
from method import method_dict
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

	# loss
	criterion = loss_dict(args.loss).cuda()

	# model
	model = model_dict[args.model](num_classes=10)
	# model = torch.nn.DataParallel(model).cuda()
	model = model.cuda()

	# method
	# assign method input: model, loader, criterion, args, logger
	logger.misc_results = {'train_acc1': [], 'train_acc5': [], 'train_loss': [], 'test_acc1': [], 'test_acc5': [], 'test_loss': []}

	method = method_dict[args.method](model, loader, criterion, args, logger)

	method.operate()

	# pipeline level analysis
	if args.ipPlot == 1:
		print('plot IP')
		information_plane_plot(input_bunch, args, logger, loader)

	if args.laPlot == 1:
		print('plot la')
		loss_acc_plot(input_bunch, args, logger)


if __name__ == '__main__':
	main()