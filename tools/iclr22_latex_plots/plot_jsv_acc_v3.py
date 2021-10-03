import numpy as np, os, sys, copy
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman" # set fonts globally
import argparse
# plt.style.use(['science'])
sys.path.insert(0, './')
from utils import set_ax, parse_value, parse_ExpID, check_path
from logger import Logger
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# ------------------------------------------ routine params to set up the project dir
parser = argparse.ArgumentParser()
parser.add_argument('--project_name', type=str, default="")
parser.add_argument('--debug', action="store_true")
parser.add_argument('--screen_print', action="store_true")
parser.add_argument('--exp_ids', type=str, default="", help='234512,230121')
parser.add_argument('--log_file', type=str, default="log.txt")
parser.add_argument('--legends', type=str, default='', help="lr=0.001,lr=0.01")
parser.add_argument('--x_step', type=int, default=1, help="the interval of x")
parser.add_argument('--ylim', type=str)
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
linestyles = [':', '-', '-.', '--']
markers = []
legends = args.legends.split('/')
label_fs = 12
legend_fs = 12
ticklabel_fs = 12

# set up fig and needed axes
fig, axes = plt.subplots(figsize=(3.5, 3.5), nrows=2, ncols=1)
# fig.subplots_adjust(hspace=0.05)  # adjust space between axes
ax1, ax2 = axes

# set background, spines, etc.
for ax in axes:
    set_ax(ax)

# ------------------------------------------ the plot function for one experiment log file
def one_exp_plot(log_file, ix):
    '''ix is the index of log_file (the ix-th experiment log)
    '''
    jsv, jsv_epoch = [], []
    test_acc, test_acc_epoch = [], []
    lines = open(log_file).readlines()
    for i in range(len(lines)):
        line = lines[i].strip()
        if 'JSV_mean' in line:
            jsv.append(parse_value(line, 'JSV_mean'))
            if 'Epoch' in line:
                jsv_epoch.append(parse_value(line, 'Epoch', type_func=int))
            else:
                jsv_epoch.append(parse_value(lines[i+1], 'Epoch', type_func=int))
        if 'Best_Acc1_Epoch' in line:
            test_acc.append(parse_value(line, 'Acc1'))
            test_acc_epoch.append(parse_value(line, 'Epoch', type_func=int))
    
    interval = args.x_step

    # plot ax1: JSV
    ax1.plot(jsv_epoch[::interval], jsv[::interval], label=legends[ix], color=colors[0], linestyle=linestyles[ix])
    ax1.set_ylabel('Mean JSV', fontsize=label_fs)

    # plot ax2: Test accuracy
    ax2.plot(test_acc_epoch[::interval], test_acc[::interval], label=legends[ix], color=colors[1], linestyle=linestyles[ix])
    ax2.set_xlabel('Epoch', fontsize=label_fs); ax2.set_ylabel('Test accuracy (%)', fontsize=label_fs)


# ------------------------------------------ main function to deal with multi-experiment log files
exp_ids = args.exp_ids.split('/')
ix = -1
for exp_id in exp_ids:
    ix += 1
    log_file = 'Experiments/*%s*/log/%s' % (exp_id, args.log_file)
    try:
        log_file = check_path(log_file)
    except:
        logprint('Error when locating "%s"' % log_file)
    # plot one log file
    logprint('[%d] Plot: "%s"' % (ix, log_file))
    one_exp_plot(log_file, ix)

# set legend
ax1.legend(frameon=False, fontsize=legend_fs) # loc='lower right'
ax2.legend(frameon=False, fontsize=legend_fs)

# set tick label size
ax1.tick_params(axis='both', which='major', labelsize=ticklabel_fs)
ax2.tick_params(axis='both', which='major', labelsize=ticklabel_fs)

# set ylim
if args.ylim:
    y1, y2 = [float(i) for i in args.ylim.split(',')]
    ax2.set_ylim([y1, y2])

# save
out = '%s/%s.pdf' % (logger.log_path, ExpID)
if args.debug:
    out = f'{ExpID}.pdf' # save to current folder for easier check

fig.tight_layout()
fig.savefig(out)
logprint('save to "%s"' % out)
