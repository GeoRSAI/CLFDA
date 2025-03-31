#!/usr/bin/env Python
# coding=utf-8
import torchvision.models as models
import argparse
import sys
import os, copy
from utils import check_path
pjoin = os.path.join

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Farmland Extraction by Pytorch')
parser.add_argument('--data', metavar='DIR', help='path to dataset')
parser.add_argument('--dataset', help='dataset name', choices=['iFLY', 'Whu', 'Sfarmland', 'Selfbuilding','whu512','whu512mul'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='unet', help='model architecture: ' + ' | '.join(model_names) + ' (default: DSSA)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=64, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', '--batch_size', default=4, type=int, metavar='N', help='mini-batch size (default: 256), this is the total ' 'batch size of all GPUs on the current node when ' 'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--wd', '--weight_decay', default=1e-3, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
parser.add_argument('--milestones',  default=[35,45,55], help='scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [35,45,55], gamma=args.gamma)')
parser.add_argument('-p', '--print-freq', '--print_freq', default=20, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')

# routine params
parser.add_argument('--project_name', type=str, default="")
parser.add_argument('--debug', action="store_true")
parser.add_argument('--screen_print', action="store_true")
parser.add_argument('--note', type=str, default='', help='experiment note')
parser.add_argument('--print_interval', type=int, default=100)
parser.add_argument('--test_interval', type=int, default=2000)
parser.add_argument('--plot_interval', type=int, default=100000000)
parser.add_argument('--save_interval', type=int, default=2000, help="the interval to save model")
parser.add_argument('--ExpID', type=str, default='', help='Experiment id. In default it will be assigned automatically')
parser.add_argument('--save_init_model', action="store_true", help='save the model after initialization')


parser.add_argument('--gamma', '--gm', default=0.1, type=float, help='learning rate decay parameter: Gamma')
parser.add_argument('--itersize', default=1, help='iter size')
parser.add_argument("-plr", "--pretrainlr", type=float, default=0.1)
parser.add_argument('--loss_lmbda', default=1.1, type=float, help='hype-param of loss 1.1 for BSDS 1.3 for NYUD')
parser.add_argument("-num_classes", "--num_classes", type=int, default=1)
parser.add_argument('--LABELS',default='',help='LABELS without background')
parser.add_argument('--Colors',default='',help='RGB without background')
args = parser.parse_args(args=[])

args.dataset = 'gid15'
args.root_path = "C:/DLdatasets/GID-15"

# args.dataset = 'FUSU'
# args.root_path = "E:/DLdatasets/FUSU_filter"

if args.dataset == 'gid15':
    #GID15
    args.LABELS = ["background","industrial land","urban residential","rural residential.","traffic land","paddy field",
                   "irrigated land","dry cropland","garden land","arbor forest","shrub land",
                   "natural meadow","artifical meadow","river","lake","pond"]
    args.Colors = {0:[255,255, 255], 9:[200, 0, 0], 10: [250, 0, 150], 11:[200, 150, 150], 12:[250, 150, 150],
                   1: [0, 200, 0], 2: [150, 250, 0], 3: [150, 200, 150], 4: [200, 0, 200], 5: [150, 0, 250],
                   6: [150, 150, 250], 7: [250, 200, 0], 8: [200, 200, 0], 13: [0, 0, 200], 14: [0, 150, 200],
                   15: [0, 200, 250]}
elif args.dataset == "FUSU":
    #SUFU
    args.LABELS = ["background", "traffic land", "inland water", "residential land", "cropland", "agriculture construction",
                    'blank','industrial land','orchard','park','public management and service',
                    'commercial land','public construction','special','forest','storage','wetland','grass']
    args.Colors = { 0:[255, 255, 255],1:[233, 133, 133],2:[8, 154, 230],3:[255, 0, 30],4:[126, 211, 33],5:[135, 126, 20],
                    6:[94, 47, 4], 7:[10, 82, 77],8:[184, 233, 134],9:[219, 170, 230],10:[255, 199, 2],
                    11:[252, 232, 5],12:[245, 107, 0], 13:[243, 229, 176], 14:[3, 100, 0],15:[127, 123, 127],16:[52, 205, 249],17:[18, 227, 180]}


args.num_classes = len(args.LABELS)
'''
    "unetformer":unetformer,
    "rs3mamba": rs3mamba,
    "rsm_ss": rsm_ss,
    "CLFDA":CLFDA(rsm_ss),
'''
args.arch = 'CLFDA'
# args.arch = ''
args.project_name = args.arch + '_' + args.dataset

args.epochs = 128
args.seed = 0
args.gpu = 0
# data
args.workers = 4
baseline_batch_size = 8
args.batch_size = 4
args.itersize = baseline_batch_size / args.batch_size
# opt
args.LR = 0.001
args.weight_decay = 0.001
args.momentum = 0.9
args.wd = 0.0005
args.milestones = [30,45,55]
args.gamma = 0.1

args.print_freq = 20 * args.itersize
args.save_init_model = True
args.screen_print = True

# args.resume = r"D:\DLexp\202411\exp002_RSMamba\Experiments\rsm_ss_gid15_SERVER-20241117-194119\weights\checkpoint_Last.pth"

args_tmp = {}
for k, v in args._get_kwargs():
    args_tmp[k] = v

# Above is the default setting. But if we explicitly assign new value for some arg in the shell script,
# the following will adjust the arg to the assigned value.
script = " ".join(sys.argv)
args.resume = check_path(args.resume)


