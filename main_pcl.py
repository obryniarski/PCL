# import argparse
# import parse
import builtins
import math
import os
import random
import shutil
import time
import warnings
from tqdm import tqdm
import numpy as np
import faiss

import wandb

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import pcl.loader
import pcl.builder
from parse import PCL_Parse
from utils import *
from clustering import  *


parser = PCL_Parse()

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    
    args.num_cluster = args.num_cluster.split(',')
    
    if not os.path.exists(args.exp_dir):
        os.mkdir(args.exp_dir)
    
    ngpus_per_node = torch.cuda.device_count()
    # if args.multiprocessing_distributed:
    #     # Since we have ngpus_per_node processes per node, the total world_size
    #     # needs to be adjusted accordingly
    #     args.world_size = ngpus_per_node * args.world_size
    #     # Use torch.multiprocessing.spawn to launch distributed processes: the
    #     # main_worker process function
    #     mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    # else:
    #     # Simply call main_worker function
    
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    wandb.init(entity='aai', project='Representation Learning', name = "unsupervised - " + args.exp_dir, notes=args.exp_notes)
    wandb.config.update(args)

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = pcl.builder.MoCo(
        models.__dict__[args.arch],
        args.low_dim, args.pcl_r, args.moco_m, args.temperature, args.mlp, args.centroid_sampling, 
        args.norm_p, gpu=args.gpu)
    # print(model)

    
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        # raise NotImplementedError("Only DistributedDataParallel is supported.")
    # else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        # raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optimizer = torch.optim.Adam(model.parameters(), args.lr,
    #                             betas=(0.9, 0.999),
    #                             weight_decay=args.weight_decay)
    
    # try adam
    # remove momentum after every opoch maybe, maybe remove it completely - maybe reset
    # do more iterations between clustering, 
    # normalize somehow based on the number of clusters

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data Loading
    train_dataset, eval_dataset = load_data(args)

    train_sampler = None
    eval_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    
    # dataloader for center-cropped images, use larger batch size to increase speed
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.batch_size*5, shuffle=False,
        sampler=eval_sampler, num_workers=args.workers, pin_memory=True)
    

    cluster_result = None # move this outside the loop so that clusters stay the same for the next few iterations if args.k > 1
    for epoch in range(args.start_epoch, args.epochs):
        

        if epoch>=args.warmup_epoch and epoch % args.k == 0:

            optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay) # resetting the momentum of the optimizer if we are doing clustering, since it adds so much variance

            # compute momentum features for center-cropped images
            features = compute_features(eval_loader, model, args)         
            
            # placeholder for clustering result
            cluster_result = {'im2cluster':[],'centroids':[],'density':[],'sampled_protos':[]}
            for num_cluster in args.num_cluster:
                cluster_result['im2cluster'].append(torch.zeros(len(eval_dataset),dtype=torch.long).cuda(args.gpu))
                cluster_result['centroids'].append(torch.zeros(int(num_cluster),args.low_dim).cuda(args.gpu))
                cluster_result['density'].append(torch.zeros(int(num_cluster)).cuda(args.gpu)) 
                # cluster_result['sampled_protos'].append(torch.zeros(int(num_cluster), len(eval_dataset), args.low_dim).cuda(args.gpu))
                cluster_result['sampled_protos'].append(torch.zeros(int(args.pcl_r),args.low_dim).cuda(args.gpu))
            
        

            clustering_algs = {'kmeans': run_kmeans, 'dbscan': run_dbscan}

            # if args.gpu == 0: # I commented this out, it was necessary only for distributed code
            features[torch.norm(features,dim=1)>1.5] /= 2 #account for the few samples that are computed twice  
            # features = features.numpy()
            # cluster_result = run_kmeans(features, args)  #run kmeans clustering on master node
            cluster_result = clustering_algs[args.clustering](features, args) #run clustering on master node with given clustering alg
            # save the clustering result
            # torch.save(cluster_result,os.path.join(args.exp_dir, 'clusters_%d'%epoch))  
                

            # maybe sample the random negative samples here from all the data in each cluster
            # then distribute after you've already picked (this avoids the issue that we are putting our entire dataset X on the GPUs)
            # maybe sample n things from each cluster and average to get the prototype.

        adjust_learning_rate(optimizer, epoch, args)


        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, cluster_result)
        # validate(train_loader, model, criterion, optimizer, epoch, args, cluster_result)

        if (epoch+1)%5==0 and (not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0)):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename='{}/checkpoint_{:04d}.pth.tar'.format(args.exp_dir,epoch))


# def validate(train_loader, model, criterion, optimizer, epoch, args, cluster_result=None):


def train(train_loader, model, criterion, optimizer, epoch, args, cluster_result=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    acc_inst = AverageMeter('Acc@Inst', ':6.2f')   
    acc_proto = AverageMeter('Acc@Proto', ':6.2f')
    
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, acc_inst, acc_proto],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    end = time.time()
    for i, (images, index) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)
                
        # compute output
        output, target, output_proto, target_proto = model(im_q=images[0], im_k=images[1], cluster_result=cluster_result, index=index)

        # InfoNCE loss
        # loss = criterion(output, target) # this is the original

        # if output_proto is None: # seeing what happens if we don't use infoNCE loss during clustering
        loss = criterion(output, target)
        # else:
            # loss = torch.tensor([0], dtype=torch.float).cuda(args.gpu)

        wandb.log({'InfoNCE Loss': loss.cpu().item()})
        
        # ProtoNCE loss
        if output_proto is not None:
            loss_proto = 0
            n = 0
            for proto_out,proto_target in zip(output_proto, target_proto):
                loss_proto += criterion(proto_out, proto_target)  
                accp = accuracy(proto_out, proto_target)[0]

                wandb.log({'Proto Accuracy - {}'.format(n): accp.cpu().item()})

                acc_proto.update(accp[0], images[0].size(0))
                n += 1
                
            # average loss across all sets of prototypes
            loss_proto /= len(args.num_cluster)
            loss += loss_proto   

            wandb.log({'ProtoNCE Loss': loss_proto.cpu().item(), 'Total Loss': loss.cpu().item()})


        losses.update(loss.item(), images[0].size(0))
        acc = accuracy(output, target)[0] 

        wandb.log({'Instance Accuracy': acc.cpu().item()})

        acc_inst.update(acc[0], images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


if __name__ == '__main__':
    main()
