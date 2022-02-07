import argparse
from smilelogging import argparser as parser
from pdb import set_trace as st

parser.add_argument('--data_path', type=str, default='./data')
parser.add_argument('--dataset', type=str, default='tsb')
parser.add_argument('--batch_size', type=int, default=12)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--model', type=str, default='mlp')
parser.add_argument('--mlp_layers', type=str, default='[12,10,10,2]')
parser.add_argument('--mlp_hiddenAct', type=str, default='tanh')
parser.add_argument('--mlp_finalAct', type=str, default='softmax')
parser.add_argument('--mlp_plotPreAct', type=bool, default=False)
parser.add_argument('--loss', type=str, default='ce')
parser.add_argument('--pipeline', type=str, default='[ft]')
parser.add_argument('--ft_epoch', type=int, default=100)
parser.add_argument('--ft_optimizer', type=str, default='sgd')
parser.add_argument('--ft_sgd_lr', type=float, default=0.01)
parser.add_argument('--ft_sgd_momentum', type=float, default=0.)
parser.add_argument('--ft_sgd_weightDecay', type=float, default=0.0000)
parser.add_argument('--ft_sgd_lrDecay', type=str, default='exp')
parser.add_argument('--ft_sgd_exp_gamma', type=float, default=0.97)
parser.add_argument('--ft_adam_lr', type=float, default=0.01)
parser.add_argument('--ft_print_interval', type=int, default=10)
parser.add_argument('--ip_marksize', type=int, default=5)

parser.add_argument('--ipPlot', type=int, default=1)
parser.add_argument('--ipPlot_est', type=str, default='bin')
parser.add_argument('--ipPlot_bin_num', type=int, default=40)
parser.add_argument('--laPlot', type=int, default=1)

parser.add_argument('--hacksmile.ON', action='store_true')
parser.add_argument('--hacksmile.config', type=str)
args = parser.parse_args()
from smilelogging.utils import update_args
args = update_args(args)