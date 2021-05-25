import numpy as np, os, sys
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman" # set fonts globally
import argparse
# plt.style.use(['science'])
from utils import set_ax, parse_value, parse_ExpID, check_path
from logger import Logger
from mpl_toolkits.axes_grid.inset_locator import inset_axes


# ------------------------------------------ routine params to set up the project dir
parser = argparse.ArgumentParser()
parser.add_argument('--project_name', type=str, default="")
parser.add_argument('--debug', action="store_true")
parser.add_argument('--screen_print', action="store_true")
parser.add_argument('--exp_ids', type=str, default="", help='234512,230121')
parser.add_argument('--log_file', type=str, default="log.txt")
parser.add_argument('--legends', type=str, default='', help="lr=0.001,lr=0.01")
args = parser.parse_args()

# logger = Logger(args)
# logprint = logger.log_printer.logprint
# ExpID = logger.ExpID # exp id for this plot project

'''Usage: 
    python <this_file> --exp_ids xxxxxx,yyyyyy --log_file <log.txt or sth.npy>

# Means, I want to plot the "log_file" in the experiments indicated by "exp_ids".
'''
# ------------------------------------------ some general plot settings
# set colors etc. according to log index
colors = ['red', 'blue']
linestyles = ['-', ':', '-.', '--']
markers = []
legends = args.legends.split('/')

# set up fig and needed axes
fig, ax = plt.subplots(figsize=(4, 2.7), nrows=1, ncols=1)

# set background, spines, etc.
set_ax(ax)
# ax.grid(linestyle='dashed')

PR = [0, 0.1, 0.3, 0.5, 0.7, 0.9]
JSV_mlp_7_linear1    = [2.4987, 1.7132, 0.5325, 0.1180, 0.0151, 0.0004]
JSV_mlp_7_relu1      = [1.8265, 1.6664, 1.2583, 0.6295, 0.1334, 0.0055]
JSV_lenet_5_linear1  = [3.7455, 3.3744, 2.5529, 1.5176, 0.4221, 0.1126]
JSV_lenet_5_relu1    = [3.0128, 2.8606, 2.5970, 1.8385, 0.6943, 0.1847]

JSV_mlp_7_linear2    = [2.4987, 1.7132, 0.5325, 0.1180, 0.0151, 0.0004]
JSV_mlp_7_relu2      = [1.8286, 1.6668, 1.2600, 0.6257, 0.1345, 0.0057]
JSV_lenet_5_linear2  = [3.7437, 3.3700, 2.5503, 1.5162, 0.4228, 0.1127]
JSV_lenet_5_relu2    = [2.9976, 2.8640, 2.5934, 1.8470, 0.6956, 0.1853]

def get_mean_std(x1, x2):
    out_mean, out_std = [], []
    for i1, i2 in zip(x1, x2):
        out_mean.append(np.mean([i1, i2]))
        out_std.append(np.std([i1, i2]))
    return np.array(out_mean), np.array(out_std)

def plot(fig, ax, x, y, std, color, linestyle, label, marker):
    ax.plot(x, y, color=color, linestyle=linestyle, label=label, marker=marker)
    ax.fill_between(x, y - std, y + std, facecolor=color, alpha=0.3)

# get mean and std
mean1, std1         = get_mean_std(JSV_mlp_7_linear1,   JSV_mlp_7_linear2)
mean2, std2         = get_mean_std(JSV_mlp_7_relu1,     JSV_mlp_7_relu2)
mean3, std3         = get_mean_std(JSV_lenet_5_linear1, JSV_lenet_5_linear2)
mean4, std4         = get_mean_std(JSV_lenet_5_relu1,   JSV_lenet_5_relu2)


# plot
plot(fig, ax, PR, mean1, std1, color='blue', linestyle='dotted', label='MLP-7-Linear', marker='*')
plot(fig, ax, PR, mean2, std2, color='blue', linestyle='solid', label='MLP-7-ReLU', marker='*')
plot(fig, ax, PR, mean3, std3, color='red', linestyle='dotted', label='LeNet-5-Linear', marker='d')
plot(fig, ax, PR, mean4, std4, color='red', linestyle='solid', label='LeNet-5-ReLU', marker='d')

# set legend
ax.legend(frameon=False) # loc='lower right'

# set x y label
ax.set_xlabel('Pruning ratio')
ax.set_ylabel('Mean JSV')

# save
out = 'jsv_vs_pr.pdf'
fig.tight_layout()
fig.savefig(out)
