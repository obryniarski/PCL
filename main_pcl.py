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
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # suppress printing if not master    
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
        
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    if torch.distributed.get_rank() == 0:
        wandb.init(entity='aai', project='Representation Learning', notes='random sampling from clusters for prototypes instead of clusters')
        wandb.config.update(args)

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = pcl.builder.MoCo(
        models.__dict__[args.arch],
        args.low_dim, args.pcl_r, args.moco_m, args.temperature, args.mlp, args.centroid_sampling, 
        args.norm_p)
    # print(model)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

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
    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset,shuffle=False)
    else:
        train_sampler = None
        eval_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    
    # dataloader for center-cropped images, use larger batch size to increase speed
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.batch_size*5, shuffle=False,
        sampler=eval_sampler, num_workers=args.workers, pin_memory=True)
    
    for epoch in range(args.start_epoch, args.epochs):
        
        cluster_result = None
        if epoch>=args.warmup_epoch:
            # compute momentum features for center-cropped images
            features = compute_features(eval_loader, model, args)         
            
            # placeholder for clustering result
            cluster_result = {'im2cluster':[],'centroids':[],'density':[],'sampled_protos':[]}
            for num_cluster in args.num_cluster:
                cluster_result['im2cluster'].append(torch.zeros(len(eval_dataset),dtype=torch.long).cuda())
                cluster_result['centroids'].append(torch.zeros(int(num_cluster),args.low_dim).cuda())
                cluster_result['density'].append(torch.zeros(int(num_cluster)).cuda()) 
                # cluster_result['sampled_protos'].append(torch.zeros(int(num_cluster), len(eval_dataset), args.low_dim).cuda())
                cluster_result['sampled_protos'].append(torch.zeros(int(args.pcl_r),args.low_dim).cuda())


            if args.gpu == 0:
                features[torch.norm(features,dim=1)>1.5] /= 2 #account for the few samples that are computed twice  
                features = features.numpy()
                cluster_result = run_kmeans(features, args)  #run kmeans clustering on master node
                print('hello')
                # print('done')
                # save the clustering result
                # torch.save(cluster_result,os.path.join(args.exp_dir, 'clusters_%d'%epoch))  
                

            # maybe sample the random negative samples here from all the data in each cluster
            # then distribute after you've already picked (this avoids the issue that we are putting our entire dataset X on the GPUs)
            # maybe sample n things from each cluster and average to get the prototype.



            dist.barrier()  
            # broadcast clustering result
            print('bruh')

            for k, data_list in cluster_result.items():
                for data_tensor in data_list:                
                        dist.broadcast(data_tensor, 0, async_op=False)


            print('hello2')

    
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        print('ahhhhhhhh')

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, cluster_result)

        # print('done')

        if (epoch+1)%5==0 and (not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0)):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename='{}/checkpoint_{:04d}.pth.tar'.format(args.exp_dir,epoch))


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
    print('why is it here')
    end = time.time()
    for i, (images, index) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # print('??')
        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)
                
        print(images[0].shape)
        # compute output
        output, target, output_proto, target_proto = model(im_q=images[0], im_k=images[1], cluster_result=cluster_result, index=index)
        print(1)
        # InfoNCE loss
        loss = criterion(output, target)  

        if torch.distributed.get_rank() == 0:
            wandb.log({'InfoNCE Loss': loss.cpu().item()})
        
        # ProtoNCE loss
        if output_proto is not None:
            loss_proto = 0
            n = 0
            for proto_out,proto_target in zip(output_proto, target_proto):
                loss_proto += criterion(proto_out, proto_target)  
                accp = accuracy(proto_out, proto_target)[0]
                if torch.distributed.get_rank() == 0:
                    wandb.log({'Proto Accuracy - {}'.format(n): accp.cpu().item()})
                acc_proto.update(accp[0], images[0].size(0))
                n += 1
                
            # average loss across all sets of prototypes
            loss_proto /= len(args.num_cluster)
            loss += loss_proto   
            if torch.distributed.get_rank() == 0:
                wandb.log({'ProtoNCE Loss': loss_proto.cpu().item(), 'Total Loss': loss.cpu().item()})

        print(2)

        losses.update(loss.item(), images[0].size(0))
        acc = accuracy(output, target)[0] 
        if torch.distributed.get_rank() == 0:
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
