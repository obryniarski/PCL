# from sklearn.manifold import TSNE, LocallyLinearEmbedding
from tsnecuda import TSNE
import matplotlib.pyplot as plt
import torchvision.models as models
import torch
import wandb
import numpy as np
import pcl.loader
import pcl.builder
# from utils import *
from clustering import *
from sklearn.cluster import DBSCAN

def load_data(data='cifar', aug_plus=True):
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
    if data == 'cifar':

        # Data loading code
        # traindir = os.path.join(args.data, 'train')
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                        std=[0.247, 0.243, 0.261])
        
        if aug_plus:
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

        traindir = os.path.join(data, 'train')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        
        if aug_plus:
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

def run_dbscan(x, eps=0.4, minPts=10, temperature=0.2):
    # make sure the parser args has the necessary dbscan parameters (eps, minPts) - ADDED

    # x = x.numpy() # this is already done before calling the function
    # n = x.shape[0] # number of samples
    # d = x.shape[1] # dimension
    print('performing dbscan clustering')
    results = {'im2cluster':[],'centroids':[],'density':[],'sampled_protos':[]}


    db = DBSCAN(eps=eps, min_samples=minPts, n_jobs=-1).fit(x) # run DBSCAN

    im2cluster = db.labels_
    # print(im2cluster)
    if -1 in im2cluster: # so that noise data is in cluster 0 instead of -1
        print('help')
        im2cluster += 1 
    centroids = im2cluster_to_centroids(x, im2cluster)

    density = np.ones(len(set(im2cluster))) * temperature

    im2cluster = torch.LongTensor(im2cluster).cuda()           
    print(set(im2cluster.tolist()))
    density = torch.Tensor(density).cuda()
    centroids = torch.Tensor(centroids).cuda()
    centroids = nn.functional.normalize(centroids, p=2, dim=1) # hmmmm ? 

    results['centroids'].append(centroids)
    results['density'].append(density)
    results['im2cluster'].append(im2cluster)



    # run dbscan, and then select a random core point from each cluster to be the centroid
    return results

def run_kmeans(x, num_cluster=['350'], centroid_sampling=False, temperature=0.2):
    """
    Args:
        x: data to be clustered
    """
    
    print('performing kmeans clustering')
    results = {'im2cluster':[],'centroids':[],'density':[],'sampled_protos':[]}
    
    for seed, num_cluster in enumerate(num_cluster):
        # intialize faiss clustering parameters
        d = x.shape[1]
        k = int(num_cluster)
        clus = faiss.Clustering(d, k)
        clus.verbose = True
        clus.niter = 20
        clus.nredo = 5
        clus.seed = seed
        clus.max_points_per_centroid = 1000
        clus.min_points_per_centroid = 10

        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False # originally False
        cfg.device = 0  
        # cfg.device = 1 #REMEMBER TO CHANGE THIS
        index = faiss.GpuIndexFlatL2(res, d, cfg)  

        clus.train(x, index)   

        D, I = index.search(x, 1) # for each sample, find cluster distance and assignments
        im2cluster = [int(n[0]) for n in I]
        
        # get cluster centroids
        centroids = faiss.vector_to_array(clus.centroids).reshape(k,d)
        
        # sample-to-centroid distances for each cluster 
        Dcluster = [[] for c in range(k)]
        indices_per_cluster = [[] for c in range(k)] # for next step - random sampling
        for im,i in enumerate(im2cluster):
            Dcluster[i].append(D[im][0])
            indices_per_cluster[i].append(im)

        if centroid_sampling:
            # print("WTF")
            # sample a random point from each cluster to act as a prototype rather than the centroid
            # sampled_protos = [np.zeros((len(indices_per_cluster[i]), d)) for i in range(k)]
            sampled_protos = [0 for i in range(k)]
            for i in range(k):
                # if there are no points other than the centroid (empty), this won't work
                # print(len(indices_per_cluster[i]))
                selected_proto_id = random.choice(indices_per_cluster[i % num_cluster])
                sampled_protos[i] = selected_proto_id
                # sampled_protos[i] = x[indices_per_cluster[i]]


        # concentration estimation (phi)        
        density = np.zeros(k)
        for i,dist in enumerate(Dcluster):
            if len(dist)>1:
                d = (np.asarray(dist)**0.5).mean()/np.log(len(dist)+10)            
                density[i] = d     
                
        #if cluster only has one point, use the max to estimate its concentration        
        dmax = density.max()
        for i,dist in enumerate(Dcluster):
            if len(dist)<=1:
                density[i] = dmax 

        density = density.clip(np.percentile(density,10),np.percentile(density,90)) #clamp extreme values for stability
        density = temperature*density/density.mean()  #scale the mean to temperature 
        
        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).cuda()
        centroids = nn.functional.normalize(centroids, p=2, dim=1) # hmmmm ? 

        if centroid_sampling:
            for i in range(k):
                sampled_protos[i] = torch.Tensor(sampled_protos[i]).cuda()

        im2cluster = torch.LongTensor(im2cluster).cuda()               
        density = torch.Tensor(density).cuda()
        
        results['centroids'].append(centroids)
        results['density'].append(density)
        results['im2cluster'].append(im2cluster)
        if centroid_sampling:
            results['sampled_protos'].append(sampled_protos) 
        
    return results
    
    
low_dim=128
pcl_r=50
moco_m=0.999
temperature=0.2
mlp=True
centroid_sampling=False
norm_p = 2

arch = 'resnet50'

print("=> creating model '{}'".format(arch))
model = pcl.builder.MoCo(
    models.__dict__[arch],
    low_dim, pcl_r, moco_m, temperature, mlp, centroid_sampling, 
    norm_p)

model = model.cuda(0)

checkpoint_id=['smallbatch','0199']
checkpoint = torch.load('pcl_cifar10_{}/checkpoint_{}.pth.tar'.format(checkpoint_id[0], checkpoint_id[1]))
model.load_state_dict(checkpoint['state_dict'])

batch_size = 256
workers = 8


train_dataset, eval_dataset = load_data(data='cifar', aug_plus=True)
train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True, sampler=None, drop_last=True)
# dataloader for center-cropped images, use larger batch size to increase speed
eval_loader = torch.utils.data.DataLoader(
    eval_dataset, batch_size=batch_size*5, shuffle=True,
    sampler=None, num_workers=workers, pin_memory=True)


features = compute_features(eval_loader, model, low_dim=low_dim)
features[torch.norm(features,dim=1)>1.5] /= 2 #account for the few samples that are computed twice  
features = features.numpy()[0:10000]

results = run_dbscan(features)
# results = run_kmeans(features)




# visualizing the data

print('Visualizing Representations...')
# methods = [LocallyLinearEmbedding(n_components=2, method='standard'), TSNE(n_components=2, init='pca')]

# tsne = TSNE(n_components=2, perplexity=45, learning_rate=200, verbose=1)
# methods = tsne

hparams = [750]
fig = plt.figure(figsize=(25,25))

for i, hparam in enumerate(hparams):
    print('hparam: {}'.format(i))
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=hparam, verbose=1, n_jobs=-1, n_iter=2500)
    y = tsne.fit_transform(features)
    ax = fig.add_subplot(1, len(hparams), i + 1)
    ax.scatter(y[:, 0], y[:, 1], c=results['im2cluster'][0].tolist())
    ax.set_title('hparam: {}'.format(hparam))

fig.savefig('imgs/visualized_features_{}_{}'.format(checkpoint_id[0], checkpoint_id[1]))