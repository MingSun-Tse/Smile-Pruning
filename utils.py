import numpy as np
import torch.nn as nn
import torch
import time
from multiprocessing import cpu_count
from joblib import Parallel, delayed
import warnings
import matplotlib.pyplot as plt
from pdb import set_trace as st


NUM_CORES = cpu_count()
warnings.filterwarnings("ignore")

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def strlist_to_list(sstr, ttype):
    '''
        example:
        # self.args.stage_pr = [0, 0.3, 0.3, 0.3, 0, ]
        # self.args.skip_layers = ['1.0', '2.0', '2.3', '3.0', '3.5', ]
        turn these into a list of <ttype> (float or str or int etc.)
    '''
    if not sstr:
        return sstr
    out = []
    sstr = sstr.strip()
    if sstr.startswith('[') and sstr.endswith(']'):
        sstr = sstr[1:-1]
    for x in sstr.split(','):
        x = x.strip()
        if x:
            x = ttype(x)
            out.append(x)
    return out

def act_dict(act_name):
    dict_ = {'relu': nn.ReLU(),
                'softmax': nn.Softmax(),
                'tanh': nn.Tanh(),
                'sigmoid': nn.Sigmoid(),
                # 'iden': None,
                'identity': nn.Identity(),
                }

    return dict_[act_name]

def loss_dict(loss_name):
    dict_ = {'ce': nn.CrossEntropyLoss(),
                }

    return dict_[loss_name]

def optimizer_dict(model, args):
    dict_ = {
             'sgd': torch.optim.SGD(model.parameters(), 
                                    args.ft_sgd_lr,
                                    momentum=args.ft_sgd_momentum,
                                    weight_decay=args.ft_sgd_weightDecay
                                    ),
             'adam': torch.optim.Adam(model.parameters(),
                                      args.ft_adam_lr)
            }

    return dict_[args.ft_optimizer]

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t() # shape [maxk, batch_size]
        correct = pred.eq(target.view(1, -1).expand_as(pred)) # target shape: [batch_size] -> [1, batch_size] -> [maxk, batch_size]
        res = []
        for k in topk:
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True) # Because of pytorch new versions, this does not work anymore (pt1.3 is okay, pt1.9 not okay).
            correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    end = time.time()
    train_epoch_index = []

    # initialize the stats
    epoch_grad_mean = {}
    epoch_grad_std = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            epoch_grad_mean[name] = torch.zeros(module.weight.size())
            epoch_grad_std[name] = torch.zeros(module.weight.size())
    # st()

    for i, (x, y, index) in enumerate(train_loader):
        data_time.update(time.time() - end)
        train_epoch_index = train_epoch_index + index.tolist()

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
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):

        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.ft_print_interval == 0:
            progress.display(i)

        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                module.weight.grad.mean()
                module.weight.grad.std()


    st()

    return np.array(train_epoch_index)
    # TODO: train no more need to return index

def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    train_state = model.training

    model.eval()

    time_compute = []
    with torch.no_grad():
        end = time.time()
        val_epoch_index = []
        for i, (x, y, index) in enumerate(val_loader):
            val_epoch_index = val_epoch_index + index.tolist()

            x = x.cuda()
            y = y.cuda()

            t1 = time.time()
            y_ = model(x)
            time_compute.append((time.time() - t1) / x.size(0))
            loss = criterion(y_, y)

            if args.dataset == 'tsb':
                acc1, acc5 = accuracy(y_, y, topk=(1,2))
            else:
                acc1, acc5 = accuracy(y_, y, topk=(1,5))
            losses.update(loss.item(), x.size(0))
            top1.update(acc1[0], x.size(0))
            top5.update(acc5[0], x.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.ft_print_interval == 0:
                progress.display(i)

        if train_state:
            model.train()

    return top1.avg.item(), top5.avg.item(), losses.avg, np.array(val_epoch_index)

def get_aligned_feats(representations, order):
    for epoch in range(len(representations)):
        for layer in range(len(representations[0])):
            # representations[epoch][layer] = representations[epoch][layer][np.argsort(order[epoch]), :]
            representations[epoch][layer] = representations[epoch][layer].cpu().numpy()[np.argsort(order[epoch]), :]

    return representations

def get_info(ws, x, label, num_of_bins, every_n=1,
                    return_matrices=False):
    
    """
    Calculate the information for the network for all the epochs and all the layers

    ws.shape =  [n_epoch, n_layers, n_params]
    ws --- outputs of all layers for all epochs
    """

    # print('Start calculating the information...')

    bins = np.linspace(-1, 1, num_of_bins)
    label = np.array(label).astype(np.float)
    pys, _, unique_x, unique_inverse_x, unique_inverse_y, pxs = extract_probs(label, x)

    with Parallel(n_jobs=NUM_CORES, prefer='threads') as parallel:
        information_total = parallel(
            delayed(calc_information_for_epoch)(
                i, epoch_output, bins, unique_inverse_x, unique_inverse_y, pxs, pys
            ) for i, epoch_output in enumerate(ws) if i % every_n == 0
        )

    if not return_matrices:
        return information_total

    ixt_matrix = np.zeros((len(information_total), len(ws[0])))
    ity_matrix = np.zeros((len(information_total), len(ws[0])))

    for epoch, layer_info in enumerate(information_total):
        for layer, info in enumerate(layer_info):
            ixt_matrix[epoch][layer] = info['local_IXT']
            ity_matrix[epoch][layer] = info['local_ITY']

    return ixt_matrix, ity_matrix

def plot_ip(IXT_array, ITY_array, num_epochs, every_n, args, logger, mode):
    assert len(IXT_array) == len(ITY_array)

    max_index = len(IXT_array)

    plt.figure(figsize=(15, 9))
    plt.title(args.project_name)
    plt.xlabel('$I(X;T)$')
    plt.ylabel('$I(T;Y)$')

    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, num_epochs + 1)]

    for i in range(0, max_index):
        IXT = IXT_array[i, :]
        ITY = ITY_array[i, :]
        plt.plot(IXT, ITY, marker='o', markersize=args.ip_marksize, markeredgewidth=0.04,
                 linestyle=None, linewidth=0, color=colors[i * every_n], zorder=10)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm._A = []
    cbar = plt.colorbar(sm, ticks=[])
    cbar.set_label('Num epochs')
    cbar.ax.text(0.5, -0.01, 0, transform=cbar.ax.transAxes, va='top', ha='center')
    cbar.ax.text(0.5, 1.0, str(num_epochs), transform=cbar.ax.transAxes, va='bottom', ha='center')
    
    IP_path = logger.gen_img_path + '/' + args.project_name + mode + '_epoch_wise.jpg'
    plt.savefig(IP_path)

    plt.show()

def plot_ip2(IXT_array, ITY_array, num_epochs, every_n, args, logger, mode):
    assert len(IXT_array) == len(ITY_array)

    max_index = np.shape(IXT_array)[1]

    plt.figure(figsize=(15, 9))
    plt.title(args.project_name)
    plt.xlabel('$I(X;T)$')
    plt.ylabel('$I(T;Y)$')

    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, max_index + 1)]

    for i in range(0, max_index):
        IXT = IXT_array[:, i]
        ITY = ITY_array[:, i]
        plt.plot(IXT, ITY, marker='o', markersize=args.ip_marksize, markeredgewidth=0.04,
                 linestyle=None, linewidth=1, color=colors[i], zorder=10)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm._A = []
    cbar = plt.colorbar(sm, ticks=[])
    cbar.set_label('Num layers')
    cbar.ax.text(0.5, -0.01, 0, transform=cbar.ax.transAxes, va='top', ha='center')
    cbar.ax.text(0.5, 1.0, str(max_index), transform=cbar.ax.transAxes, va='bottom', ha='center')
    
    IP_path = logger.gen_img_path + '/' + args.project_name + mode + '_layer_wise.jpg'
    plt.savefig(IP_path)

    plt.show()

def extract_probs(label, x):
    """calculate the probabilities of the given data and labels p(x), p(y) and (y|x)"""
    pys = np.sum(label, axis=0) / float(label.shape[0])

    unique_x, unique_x_indices, unique_inverse_x, unique_x_counts = np.unique(
        x, axis=0,
        return_index=True, return_inverse=True, return_counts=True
    )

    pxs = unique_x_counts / np.sum(unique_x_counts)

    unique_array_y, unique_y_indices, unique_inverse_y, unique_y_counts = np.unique(
        label, axis=0,
        return_index=True, return_inverse=True, return_counts=True
    )

    return pys, None, unique_x, unique_inverse_x, unique_inverse_y, pxs

def calc_information_for_epoch(epoch_number, ws_epoch, bins, unique_inverse_x,
                               unique_inverse_y, pxs, pys):
    """Calculate the information for all the layers for specific epoch"""
    information_epoch = []

    for i in range(len(ws_epoch)):
        information_epoch_layer = layer_information(
            layer_output=ws_epoch[i],
            bins=bins,
            unique_inverse_x=unique_inverse_x,
            unique_inverse_y=unique_inverse_y,
            px=pxs, py=pys
        )
        information_epoch.append(information_epoch_layer)
    information_epoch = np.array(information_epoch)

    # print('Processed epoch {}'.format(epoch_number))

    return information_epoch

def layer_information(layer_output, bins, py, px, unique_inverse_x, unique_inverse_y):
    ws_epoch_layer_bins = bins[np.digitize(layer_output, bins) - 1]
    ws_epoch_layer_bins = ws_epoch_layer_bins.reshape(len(layer_output), -1)

    unique_t, unique_inverse_t, unique_counts_t = np.unique(
        ws_epoch_layer_bins, axis=0,
        return_index=False, return_inverse=True, return_counts=True
    )
    # old element location in new output

    pt = unique_counts_t / np.sum(unique_counts_t)

    # # I(X, Y) = H(Y) - H(Y|X)
    # # H(Y|X) = H(X, Y) - H(X)

    x_entropy = entropy(px)
    y_entropy = entropy(py)
    t_entropy = entropy(pt)

    x_t_joint_entropy = joint_entropy(unique_inverse_x, unique_inverse_t, px.shape[0], layer_output.shape[0])
    y_t_joint_entropy = joint_entropy(unique_inverse_y, unique_inverse_t, py.shape[0], layer_output.shape[0])

    return {
        'local_IXT': t_entropy + x_entropy - x_t_joint_entropy,
        'local_ITY': y_entropy + t_entropy - y_t_joint_entropy
    }

def joint_entropy(unique_inverse_x, unique_inverse_y, bins_x, bins_y):

    joint_distribution = np.zeros((bins_x, bins_y))
    np.add.at(joint_distribution, (unique_inverse_x, unique_inverse_y), 1)
    joint_distribution /= np.sum(joint_distribution)

    return entropy(joint_distribution)

def entropy(probs):
    return -np.sum(probs * np.ma.log2(probs))

def information_plane_plot(input_bunch, args, logger, loader):
	model = input_bunch[4]
	train_index, test_index = input_bunch[6], input_bunch[9]
	ip_feats = get_aligned_feats(model.feats_per_epo, train_index)
	ip_feats_test = get_aligned_feats(model.feats_per_epo_test, test_index)

	new_x = loader.train_set.data
	new_x_test = loader.test_set.data

	new_y = np.concatenate([loader.train_set.targets_ori, 1-loader.train_set.targets_ori], axis = 1)
	new_y_test = np.concatenate([loader.test_set.targets_ori, 1-loader.test_set.targets_ori], axis = 1)

	ixt_array, ity_array = get_info(ip_feats, new_x, new_y, args.ipPlot_bin_num, every_n=1, return_matrices=True)
	# st()
	plot_ip(ixt_array, ity_array, num_epochs=args.ft_epoch, every_n=1, args=args, logger=logger, mode='train')
	plot_ip2(ixt_array, ity_array, num_epochs=args.ft_epoch, every_n=1, args=args, logger=logger, mode='train')

	ixt_array_test, ity_array_test = get_info(ip_feats_test, new_x_test, new_y_test, args.ipPlot_bin_num, every_n=1, return_matrices=True)
	plot_ip(ixt_array_test, ity_array_test, num_epochs=args.ft_epoch, every_n=1, args=args, logger=logger, mode='test')
	plot_ip2(ixt_array_test, ity_array_test, num_epochs=args.ft_epoch, every_n=1, args=args, logger=logger, mode='test')

def loss_acc_plot(input_bunch, args, logger):
	train_loss_list, train_acc1_list = input_bunch[7], input_bunch[8]
	test_loss_list, test_acc1_list = input_bunch[10], input_bunch[11]

	fig, axs = plt.subplots(1,4)
	axs[0].plot(np.arange(len(train_loss_list)), train_loss_list), axs[0].set_title('train_loss')
	axs[1].plot(np.arange(len(train_acc1_list)), train_acc1_list), axs[1].set_title('train_acc1')
	axs[2].plot(np.arange(len(test_loss_list)), test_loss_list), axs[2].set_title('test_loss')
	axs[3].plot(np.arange(len(test_acc1_list)), test_acc1_list), axs[3].set_title('test_acc1')
	plt.suptitle(args.project_name)

	path = logger.gen_img_path + '/' + args.project_name + '_loss_acc.jpg'
	plt.savefig(path)

def edge_(input_bunch, args, logger, loader):
	model = input_bunch[4]
	train_index, test_index = input_bunch[6], input_bunch[9]
	ip_feats = get_aligned_feats(model.feats_per_epo, train_index)
	ip_feats_test = get_aligned_feats(model.feats_per_epo_test, test_index)

	new_x = loader.train_set.data
	new_x_test = loader.test_set.data

	new_y = np.concatenate([loader.train_set.targets_ori, 1-loader.train_set.targets_ori], axis = 1)
	new_y_test = np.concatenate([loader.test_set.targets_ori, 1-loader.test_set.targets_ori], axis = 1)
	# st()
	new_y = np.argmax(new_y, axis=1)
	# st()

	epochs = len(ip_feats)
	layers = len(ip_feats[0])

	ixt_array = np.zeros((epochs, layers))
	ity_array = np.zeros((epochs, layers))

	smt_vector_xt = np.array([0.8, 1.0, 1.2, 1.8,2.2,2.4,2.6])
	smt_vector_xt = np.array([0.002, 0.006, 0.01, 0.015,0.02,2.4,2.6])
	smt_vector_ty = np.array([0.4, 0.5, 0.6, 0.8,1.0,1.2,1.4])
	for epo in range(epochs):
		start=time.time()
		for ly in range(layers):
			ixt_array[epo][ly] = (EDGE(new_x, ip_feats[epo][ly], U=20, L_ensemble=10, gamma = [0.002, smt_vector_xt[ly]], epsilon=[0.2,0.2], epsilon_vector='range'))
			ity_array[epo][ly] = (EDGE(new_y, ip_feats[epo][ly], U=10, L_ensemble=10, gamma = [0.0001, smt_vector_ty[ly]], epsilon=[0.2,0.2], epsilon_vector='range'))
		end = time.time()
		print(epo, ':',end-start)

	plot_ip2(ixt_array, ity_array, num_epochs=args.ft_epoch, every_n=1, args=args, logger=logger, mode='train')




