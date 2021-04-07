import argparse
import torchvision.models as models

class PCL_Parse(argparse.ArgumentParser):

    def __init__(self):
        super().__init__(description='PyTorch ImageNet Training')

        model_names = sorted(name for name in models.__dict__
            if name.islower() and not name.startswith("__")
            and callable(models.__dict__[name]))

        self.add_argument('data', metavar='DIR',
                    help='path to dataset')
        self.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                            choices=model_names,
                            help='model architecture: ' +
                                ' | '.join(model_names) +
                                ' (default: resnet50)')
        self.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                            help='number of data loading workers (default: 32)')
        self.add_argument('--epochs', default=200, type=int, metavar='N',
                            help='number of total epochs to run')
        self.add_argument('--start-epoch', default=0, type=int, metavar='N',
                            help='manual epoch number (useful on restarts)')
        self.add_argument('-b', '--batch-size', default=256, type=int,
                            metavar='N',
                            help='mini-batch size (default: 256), this is the total '
                                'batch size of all GPUs on the current node when '
                                'using Data Parallel or Distributed Data Parallel')
        self.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                            metavar='LR', help='initial learning rate', dest='lr')
        self.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                            help='learning rate schedule (when to drop lr by 10x)')
        self.add_argument('--momentum', default=0.9, type=float, metavar='M',
                            help='momentum of SGD solver')
        self.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                            metavar='W', help='weight decay (default: 1e-4)',
                            dest='weight_decay')
        self.add_argument('-p', '--print-freq', default=100, type=int,
                            metavar='N', help='print frequency (default: 10)')
        self.add_argument('--resume', default='', type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')
        self.add_argument('--world-size', default=-1, type=int,
                            help='number of nodes for distributed training')
        self.add_argument('--rank', default=-1, type=int,
                            help='node rank for distributed training')
        self.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                            help='url used to set up distributed training')
        self.add_argument('--dist-backend', default='nccl', type=str,
                            help='distributed backend')
        self.add_argument('--seed', default=None, type=int,
                            help='seed for initializing training. ')
        self.add_argument('--gpu', default=None, type=int,
                            help='GPU id to use.')
        self.add_argument('--multiprocessing-distributed', action='store_true',
                            help='Use multi-processing distributed training to launch '
                                'N processes per node, which has N GPUs. This is the '
                                'fastest way to use PyTorch for either single node or '
                                'multi node data parallel training')
        self.add_argument('--norm-p', default=2, type=float,
                            help='exponent used for Lp normalization in latent space')


        self.add_argument('--low-dim', default=128, type=int,
                            help='feature dimension (default: 128)')
        self.add_argument('--pcl-r', default=16384, type=int,
                            help='queue size; number of negative pairs; needs to be smaller than num_cluster (default: 16384)')
        self.add_argument('--moco-m', default=0.999, type=float,
                            help='moco momentum of updating key encoder (default: 0.999)')
        self.add_argument('--temperature', default=0.2, type=float,
                            help='softmax temperature')

        self.add_argument('--mlp', action='store_true',
                            help='use mlp head')
        self.add_argument('--aug-plus', action='store_true',
                            help='use moco-v2/SimCLR data augmentation')
        self.add_argument('--cos', action='store_true',
                            help='use cosine lr schedule')
        self.add_argument('--centroid-sampling', action='store_true',
                            help='prototypes are randomly sampled from cluster instead of centroid')

        self.add_argument('--num-cluster', default='25000,50000,100000', type=str, 
                            help='number of clusters')
        self.add_argument('--warmup-epoch', default=20, type=int,
                            help='number of warm-up epochs to only train with InfoNCE loss')
        self.add_argument('--exp-dir', default='experiment_pcl', type=str,
                            help='experiment directory')

        # extra clustering arguments
        self.add_argument('--clustering', default='kmeans', type=str, 
                            help='clustering algorithm to use for ProtoNCE loss')
        self.add_argument('--eps', default=0.5, type=float,
                            help='epsilon parameter for DBSCAN clustering algorithm')
        self.add_argument('--minPts', default=5, type=int,
                            help='minPts parameter for DBSCAN clustering algorithm')