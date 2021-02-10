import numpy as np, os, sys
import matplotlib.pyplot as plt
from scipy import stats
import argparse
from utils import set_ax, parse_value, parse_ExpID, check_path
from logger import Logger

# ------------------------------------------ routine params to set up the project dir
parser = argparse.ArgumentParser()
parser.add_argument('--project_name', type=str, default="")
parser.add_argument('--debug', action="store_true")
parser.add_argument('--screen_print', action="store_true")
parser.add_argument('--exp_ids', type=str, default="", help='234512,230121')
parser.add_argument('--log_file', type=str, default="log.txt")
parser.add_argument('--legends', type=str, default='', help="lr=0.001,lr=0.01")
args = parser.parse_args()

logger = Logger(args)
logprint = logger.log_printer.logprint
ExpID = logger.ExpID # exp id for this plot project

'''Usage: 
    python <this_file> --exp_ids xxxxxx,yyyyyy --log_file <log.txt or sth.npy>

# Means, I want to plot the "log_file" in the experiments indicated by "exp_ids".
'''
# ------------------------------------------ some general plot settings
# set colors etc. according to log index
colors = ['red', 'blue']
linestyles = ['-', ':', '-.', '--']
markers = []
legends = args.legends.split(';')

# set up fig and needed axes
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

# set background, spines, etc.
set_ax(ax1)
set_ax(ax2)

# set x ylim
ax1.set_xlim(0, 89)
ax1.set_ylim(0, 3.6)
ax2.set_ylim(0, 100)

# set x ylabel
ax1.set_xlabel('Epoch', fontsize=14)
ax1.set_ylabel('JSV', fontsize=14)
ax2.set_ylabel('Test accuracy (%)', fontsize=14)

# ------------------------------------------ the plot function for one experiment log file
def one_exp_plot(log_file, ix):
    '''ix is the index of log_file (the ix-th experiment log)
    '''
    jsv, jsv_epoch = [], []
    test_acc, test_acc_epoch = [], []
    for line in open(log_file):
        if 'JSV_mean' in line and 'Epoch' in line:
            jsv.append(parse_value(line, 'JSV_mean'))
            jsv_epoch.append(parse_value(line, 'Epoch', type_func=int))
        if 'Best_Acc1_Epoch' in line:
            test_acc.append(parse_value(line, 'Acc1'))
            test_acc_epoch.append(parse_value(line, 'Epoch', type_func=int))

    # plot ax1: JSV
    ax1.plot(jsv_epoch, jsv, label=legends[ix], color=colors[0], linestyle=linestyles[ix])
    ax1.yaxis.label.set_color(colors[0])
    ax1.tick_params(axis='y', colors=colors[0])

    # plot ax2: Test accuracy
    ax2.plot(test_acc_epoch, test_acc, label=legends[ix], color=colors[1], linestyle=linestyles[ix])
    ax2.yaxis.label.set_color(colors[1])
    ax2.tick_params(axis='y', colors=colors[1])

# ------------------------------------------ main function to deal with multi-experiment log files
exp_ids = args.exp_ids.split(';')
ix = -1
for exp_id in exp_ids:
    ix += 1
    log_file = 'Experiments/*%s*/log/%s' % (exp_id, args.log_file)
    log_file = check_path(log_file)
    # plot one log file
    logprint('[%d] Plot: "%s"' % (ix, log_file))
    one_exp_plot(log_file, ix)

# set legend
ax1.legend(loc='lower right')
leg = ax1.get_legend()
leg.legendHandles[0].set_color('k')
leg.legendHandles[1].set_color('k')

# save
out = '%s/%s.pdf' % (logger.log_path, ExpID)
fig.tight_layout()
fig.savefig(out)
logprint('save to "%s"' % out)