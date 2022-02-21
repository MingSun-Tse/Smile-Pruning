'''
    This code is based on the official PyTorch ImageNet training example 'main.py'. Commit ID: 69d2798, 04/23/2020.
    URL: https://github.com/pytorch/examples/tree/master/imagenet
    Major modified parts will be indicated by '@mst' mark.
    Questions to @mingsun-tse (huan.wang.cool@gmail.com).
'''
import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

# --- @mst
import copy, math
import numpy as np
from importlib import import_module
from dataset import Data

# import sys; sys.path.insert(0, '../UtilsHub/smilelogging')
from smilelogging import Logger
from utils import get_n_params, get_n_flops, get_n_params_, get_n_flops_
from utils import add_noise_to_model, compute_jacobian, _weights_init_orthogonal, get_jacobian_singular_values
from utils import Dataset_lmdb_batch
from utils import AverageMeter, ProgressMeter, adjust_learning_rate, accuracy
from model import model_dict, is_single_branch
from dataset import num_classes_dict, input_size_dict
from option import args, check_args
pjoin = os.path.join

original_print = print
logger = Logger(args)
accprint = logger.log_printer.accprint
netprint = logger.netprint
logger.misc = {}
logger.passer = {}

class MyDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
# ---

def main():
    # @mst: move this to above, won't influence the original functions
    # args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    
    # Set up pipeline
    from method_modules import module_dict
    from utils import update_args_from_file
    pipeline, configs = get_pipeline(args.pipeline)

    # Data loading code
    train_sampler = None
    if args.dataset not in ['imagenet', 'imagenet_subset_200']:
        loader = Data(args)
        train_loader = loader.train_loader
        val_loader = loader.test_loader
    else:   
        traindir = os.path.join(args.data_path, args.dataset, 'train')
        val_folder = 'val'
        if args.debug:
            val_folder = 'val_tmp' # val_tmp is a tiny version of val to accelerate test in debugging
            val_folder_path = f'{args.data_path}/{args.dataset}/{val_folder}'
            if not os.path.exists(val_folder_path):
                os.makedirs(val_folder_path)
                dirs = os.listdir(f'{args.data_path}/{args.dataset}/val')[:3]
                [shutil.copytree(f'{args.data_path}/{args.dataset}/val/{d}', f'{val_folder_path}/{d}') for d in dirs]
        valdir = os.path.join(args.data_path, args.dataset, val_folder)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        transforms_train = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize])
        transforms_val = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize])

        if args.use_lmdb:
            lmdb_path_train = traindir + '/lmdb'
            lmdb_path_val = valdir + '/lmdb'
            assert os.path.exists(lmdb_path_train) and os.path.exists(lmdb_path_val)
            print(f'Loading data in LMDB format: "{lmdb_path_train}" and "{lmdb_path_val}"')
            train_dataset = Dataset_lmdb_batch(lmdb_path_train, transforms_train)
            val_dataset = Dataset_lmdb_batch(lmdb_path_val, transforms_val)
        else:
            train_dataset = datasets.ImageFolder(traindir, transforms_train)
            val_dataset = datasets.ImageFolder(valdir, transforms_val)

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    logger.passer['criterion'] = criterion

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    num_classes = num_classes_dict[args.dataset]
    *_, num_channels, input_height, input_width = input_size_dict[args.dataset]
    logger.passer['input_size'] = [1, num_channels, input_height, input_width]
    logger.passer['is_single_branch'] = is_single_branch
    if args.dataset in ["imagenet", "imagenet_subset_200", "tiny_imagenet"]:
        if args.pretrained:
            print("=> using pre-trained model '{}'".format(args.arch))
            model = models.__dict__[args.arch](num_classes=num_classes, pretrained=True)
        else:
            print("=> creating model '{}'".format(args.arch))
            model = models.__dict__[args.arch](num_classes=num_classes)
    else: # @mst: added non-imagenet models
        model = model_dict[args.arch](num_classes=num_classes, num_channels=num_channels, use_bn=args.use_bn, conv_type=args.conv_type)
        if args.init in ['orth', 'exact_isometry_from_scratch']:
            model.apply(lambda m: _weights_init_orthogonal(m, act=args.activation))
            print("==> Use weight initialization: 'orthogonal_'. Activation: %s" % args.activation)
    print(f'==> Use conv_type: {args.conv_type}')

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = MyDataParallel(model.features)
            model.cuda()
        else:
            model = MyDataParallel(model).cuda()

    # Load the unpruned model for pruning 
    # This may be useful for the non-imagenet cases where we use our pretrained models.
    if args.base_model_path:
        ckpt = torch.load(args.base_model_path)
        logstr = f'==> Load pretrained ckpt successfully: "{args.base_model_path}".'
        if 'model' in ckpt:
            model = ckpt['model']
            logstr += ' Use the model stored in ckpt.'
        model.load_state_dict(ckpt['state_dict'])
        if args.test_pretrained:
            acc1, acc5, loss_test = validate(val_loader, model, criterion, args)
            logstr += f'Its accuracy: {acc1:.4f}.'
        print(logstr)
    
    ################################## Core pipeline ##################################
    ix_module = 0
    for m_name, config in zip(pipeline, configs):
        ix_module += 1
        print(f'')
        print(f'***************** Model processor #{ix_module} ({m_name}) starts *****************')
        module = module_dict[m_name]
        args_copy = copy.deepcopy(args)
        if config:
            args_copy = update_args_from_file(args_copy, config)
            args_copy = check_args(args_copy)
            print(f'==> Args updated from file "{config}":')
            print_args(args_copy)
        logger.suffix = f'_{logger.ExpID}_methodmodule{ix_module}_{m_name}'
        model = module(model, loader, args_copy, logger)
    ###################################################################################
    

def get_pipeline(method:str):
    pipeline, configs = [], []
    for mp in method.split(','):
        if ':' in mp:
            module, config_path = mp.split(':')
            pipeline += [module]
            configs += [config_path]
        else:
            pipeline += [mp]
            configs += [None]
    return pipeline, configs

def print_args(args):
    # build a key map for later sorting
    key_map = {}
    for k in args.__dict__:
        k_lower = k.lower()
        if k_lower in key_map:
            key_map[k_lower + '_' + k_lower] = k
        else:
            key_map[k_lower] = k
    
    # print in the order of sorted lower keys 
    logtmp = ''
    for k_ in sorted(key_map.keys()):
        real_key = key_map[k_]
        logtmp += "('%s': %s) " % (real_key, args.__dict__[real_key])
    print(logtmp + '\n', unprefix=True)

if __name__ == '__main__':
    main()