from utils import _weights_init, _weights_init_orthogonal, orthogonalize_weights, delta_orthogonalize_weights
import torch
import torch.nn as nn
import numpy as np

def approximate_isometry_optimize(model, mask, lr, n_iter, wg='weight', print=print):
    '''Refer to: 2020-ICLR-A Signal Propagation Perspective for Pruning Neural Networks at Initialization (ICLR 2020).
        Code: https://github.com/namhoonlee/spp-public
    '''
    def optimize(w, layer_name):
        '''Approximate Isometry for sparse weights by iterative optimization
        '''
        flattened = w.view(w.size(0), -1) # [n_filter, -1]
        identity = torch.eye(w.size(0)).cuda() # identity matrix
        w_ = torch.autograd.Variable(flattened, requires_grad=True)
        optim = torch.optim.Adam([w_], lr)
        for i in range(n_iter):
            loss = nn.MSELoss()(torch.matmul(w_, w_.t()), identity)
            optim.zero_grad()
            loss.backward()
            optim.step()
            if not isinstance(mask, type(None)):
                w_ = torch.mul(w_, mask[layer_name].view_as(w_)) # not update the pruned params
            w_ = torch.autograd.Variable(w_, requires_grad=True)
            optim = torch.optim.Adam([w_], lr)
            # if i % 100 == 0:
            #     print('[%d/%d] approximate_isometry_optimize for layer "%s", loss %.6f' % (i, n_iter, name, loss.item()))
        return w_.view(m.weight.shape)
    
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            w_ = optimize(m.weight, name)
            m.weight.data.copy_(w_)
            print('Finished approximate_isometry_optimize for layer "%s"' % name)

def exact_isometry_based_on_existing_weights(model, act, print=print):
    '''Our proposed method.
    '''
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            w_ = orthogonalize_weights(m.weight, act=act)
            m.weight.data.copy_(w_)
            print('Finished exact_isometry for layer "%s"' % name)

def exact_isometry_based_on_existing_weights_delta(model, act, print=print):
    '''Refer to 2018-ICML-Dynamical Isometry and a Mean Field Theory of CNNs: How to Train 10,000-Layer Vanilla Convolutional Neural Networks
    '''
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            w_ = delta_orthogonalize_weights(m.weight, act=act)
            m.weight.data.copy_(w_)
            print('Finished isometry for conv layer "%s"' % name)
        elif isinstance(m, nn.Linear):
            w_ = orthogonalize_weights(m.weight, act=act)
            m.weight.data.copy_(w_)
            print('Finished isometry for linear layer "%s"' % name)

def reinit_model(model, args, mask, print):
    if args.reinit in ['default', 'kaiming_normal']:
        model.apply(_weights_init) # completely reinit weights via 'kaiming_normal'
        print("==> Reinit model: default ('kaiming_normal' for Conv/FC; 0 mean, 1 std for BN)")

    elif args.reinit in ['orth', 'exact_isometry_from_scratch']:
        model.apply(lambda m: _weights_init_orthogonal(m, act=args.activation)) # reinit weights via 'orthogonal_' from scratch
        print("==> Reinit model: exact_isometry ('orthogonal_' for Conv/FC; 0 mean, 1 std for BN)")

    elif args.reinit == 'exact_isometry_based_on_existing':
        exact_isometry_based_on_existing_weights(model, act=args.activation, print=print) # orthogonalize weights based on existing weights
        print("==> Reinit model: exact_isometry (orthogonalize Conv/FC weights based on existing weights)")

    elif args.reinit == 'exact_isometry_based_on_existing_delta':
        exact_isometry_based_on_existing_weights_delta(model, act=args.activation, print=print)
        
    elif args.reinit == 'approximate_isometry': # A Signal Propagation Perspective for Pruning Neural Networks at Initialization (ICLR 2020)
        approximate_isometry_optimize(model, mask=mask, lr=args.lr_AI, n_iter=10000, print=print) # 10000 refers to the paper above; lr in the paper is 0.1, but not converged here
        print("==> Reinit model: approximate_isometry")
    else:
        raise NotImplementedError
    return model

def orth_regularization(w):
    w_ = w.view(w.size(0), -1)
    if w_.size(0) < w.size(1):
        w_ = w_.t()
    identity = torch.eye(w_.size(0)).cuda()
    loss = nn.MSELoss()(torch.matmul(w_, w_.t()), identity)
    # torch.norm(w.t_() @ w - torch.eye(w.size(1)).cuda())
    return loss

def deconv_orth_dist(kernel, stride = 2, padding = 1):
    '''Refer to 2020-CVPR-Orthogonal Convolutional Neural Networks.
    '''
    [o_c, i_c, w, h] = kernel.shape
    output = torch.conv2d(kernel, kernel, stride=stride, padding=padding)
    target = torch.zeros((o_c, o_c, output.shape[-2], output.shape[-1])).cuda()
    ct = int(np.floor(output.shape[-1]/2))
    target[:,:,ct,ct] = torch.eye(o_c).cuda()
    return torch.norm( output - target )