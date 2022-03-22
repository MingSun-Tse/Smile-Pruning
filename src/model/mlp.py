import torch
import torch.nn as nn
import math

class FCNet(nn.Module):
    def __init__(self, dim_input, num_classes, num_fc, width=0, num_params=0, branch_layer_out_dim=[], act='relu', dropout=0):
        super(FCNet, self).__init__()
        # activation func
        if act == 'relu':
            activation = nn.ReLU()
        elif act == 'lrelu':
            activation = nn.LeakyReLU()
        elif act == 'linear':
            activation = nn.Identity()
        else:
            raise NotImplementedError
        
        num_middle = num_fc - 2
        if width == 0:
            # Given total num of parameters budget, calculate the width: num_middle * width^2 + width * (dim_input + num_classes) = num_params 
            assert num_params > 0
            Delta = (dim_input + num_classes) * (dim_input + num_classes) + 4 * num_middle * num_params
            width = (math.sqrt(Delta) - dim_input - num_classes) / 2 / num_middle
            width = int(width)
            print("FC net width = %s" % width)

        # build the stem net
        net = [nn.Linear(dim_input, width), activation]
        for i in range(num_middle):
            net.append(nn.Linear(width, width))
            if dropout and num_middle - i <= 2: # the last two middle fc layers will be applied with dropout
                net.append(nn.Dropout(dropout))
            net.append(activation)
        net.append(nn.Linear(width, num_classes))
        self.net = nn.Sequential(*net)
        
        # build branch layers
        branch = []
        for x in branch_layer_out_dim:
            branch.append(nn.Linear(width, x))
        self.branch = nn.Sequential(*branch) # so that the whole model can be put on cuda
        self.branch_layer_ix = []
    
    def forward(self, img, branch_out=False, mapping=False):
        '''
            <branch_out>: if output the internal features
            <mapping>: if the internal features go through a mapping layer
        '''
        if not branch_out:
            img = img.view(img.size(0), -1)
            return self.net(img)
        else:
            out = []
            start = 0
            y = img.view(img.size(0), -1)
            keys = [int(x) for x in self.branch_layer_ix]
            for i in range(len(keys)):
                end = keys[i] + 1
                y = self.net[start:end](y)
                y_branch = self.branch[i](y) if mapping else y
                out.append(y_branch)
                start = end
            y = self.net[start:](y)
            out.append(y)
            return out


# Refer to: A Signal Propagation Perspective for Pruning Neural Networks at Initialization (ICLR 2020).
# https://github.com/namhoonlee/spp-public/blob/32bde490f19b4c28843303f1dc2935efcd09ebc9/spp/network.py#L108
def mlp_7_linear(num_classes=10, num_channels=1, **kwargs):
    return FCNet(dim_input=1024, num_classes=num_classes, num_fc=7, width=100, act='linear')

def mlp_7_relu(num_classes=10, num_channels=1, **kwargs):
    return FCNet(dim_input=1024, num_classes=num_classes, num_fc=7, width=100, act='relu')