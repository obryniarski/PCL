import os
import shutil
import math

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.nn import functional as F

import pcl.loader


def freeze_parameters(model, freeze_fc = False):
    # freezes all the parameters of the given model, except for the fully connected layers if include_fc=False
    # must get rid of any naming prefixes (i.e. so fc layer isn't named 'encoder.fc.weights)

    for name, parameter in model.name_parameters():
        parameter.requires_grad = False if (name[:2] != 'fc') or (name[:2] == 'fc' and freeze_fc) else True

            


def padded_cat(tensor_list):
    """
    Zero-pads each tensor in tensor_list so they are all the same length, then concatenates along dim=0.

    Parameters
    ----------
    tensor_list : list of torch.Tensor
        the list of possibly variable size tensors to concatenate
        shape of each tensor is (n_i, low_dim)

    Returns
    -------
    cat_tensor : torch.Tensor
        the new concatenated tensor
        shape of output is (len(tensor_list), sum(n_i), low_dim)
    lengths : list
        list of the lengths (n_i) of each tensor in tensor_list (for future operations)
    """
    d2 = tensor_list[0].shape[1]
    # max_length = max([s.shape[0] for s in tensor_list])

    max_length = sum([s.shape[0] for s in tensor_list]) 
    # every tensor has the same length as the original dataset

    pad = lambda ar: F.pad(ar, (0,0,0,max_length - ar.shape[0])).reshape(1, max_length, d2)
    padded_tensor_list = list(map(pad, tensor_list))
    cat_tensor = torch.cat(padded_tensor_list)
    lengths = torch.LongTensor([s.shape[0] for s in tensor_list])
    return cat_tensor, lengths


# def sample_from_lengths(lengths):
#     '''
#     i.e. lengths = [5, 10, 2, 3]
#     '''



def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

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

def load_data(args):
    """
    Parameters
    ----------
    args : argparse.ArgumentParser
        the command line arguments relevant to fetching data


    Returns
    -------
    train_dataset : torch.utils.data.Dataset
        torch dataset for training data
    eval_dataset : torch.utils.data.Dataset
        torch dataset for evaluation data
    """
    if args.data == 'cifar':

        # Data loading code
        # traindir = os.path.join(args.data, 'train')
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                        std=[0.247, 0.243, 0.261])
        
        if args.aug_plus:
            # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
            augmentation = [
                transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([pcl.loader.GaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]
        else:
            # MoCo v1's aug: same as InstDisc https://arxiv.org/abs/1805.01978
            augmentation = [
                transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
                transforms.RandomGrayscale(p=0.2),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]
            
        # center-crop augmentation 
        eval_augmentation = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            normalize
            ])  

        train_dataset = pcl.loader.CIFAR10Instance(
            'data', train=True,
            transform=pcl.loader.TwoCropsTransform(transforms.Compose(augmentation)), download=True)
        eval_dataset = pcl.loader.CIFAR10Instance(
            'data', train=True,
            transform=eval_augmentation, download=True)
    else:

        # Data loading code
        # train_dataset, eval_dataset = load_data(args)

        traindir = os.path.join(args.data, 'train')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        
        if args.aug_plus:
            # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
            augmentation = [
                transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([pcl.loader.GaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]
        else:
            # MoCo v1's aug: same as InstDisc https://arxiv.org/abs/1805.01978
            augmentation = [
                transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                transforms.RandomGrayscale(p=0.2),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]
            
        # center-crop augmentation 
        eval_augmentation = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
            ])    

        train_dataset = pcl.loader.ImageFolderInstance(
            traindir,
            pcl.loader.TwoCropsTransform(transforms.Compose(augmentation)))
        eval_dataset = pcl.loader.ImageFolderInstance(
            traindir,
            eval_augmentation)

    return train_dataset, eval_dataset

