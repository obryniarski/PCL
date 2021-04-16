# from sklearn.manifold import TSNE, LocallyLinearEmbedding
from tsnecuda import TSNE as cudaTSNE
import matplotlib.pyplot as plt
import torchvision.models as models
import torch
import wandb
import numpy as np
import pcl.loader
import pcl.builder
# from utils import *
import torchvision.transforms as transforms
from clustering import *
from sklearn.cluster import DBSCAN
import os
import math
import torchvision.datasets as datasets
from PIL import ImageFilter, Image
import umap
import umap.plot

class CIFAR10Instance_w_label(datasets.CIFAR10):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, index) 
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        # print(img.shape)
        if self.transform is not None:
            img = self.transform(img)
        # print(img.shape)
        return img, index, target

def compute_features(eval_loader, model, low_dim=128, gpu=0):
    print('Computing features...')
    model.eval()
    features = torch.zeros(len(eval_loader.dataset),low_dim).cuda(gpu)
    targets = torch.zeros(len(eval_loader.dataset), dtype=torch.long)
    for i, (images, index, target) in enumerate(tqdm(eval_loader)):
        with torch.no_grad():
            images = images.cuda(gpu, non_blocking=True)
            feat = model(images,is_eval=True) 
            features[index] = feat
            targets[index] = target
    # dist.barrier()        
    # dist.all_reduce(features, op=dist.ReduceOp.SUM)     
    return features.cpu(), targets
    
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

        train_dataset = CIFAR10Instance_w_label(
            'data', train=True,
            transform=pcl.loader.TwoCropsTransform(transforms.Compose(augmentation)), download=True)
        eval_dataset = CIFAR10Instance_w_label(
            'data', train=True,
            transform=eval_augmentation, download=True)

    return train_dataset, eval_dataset

def im2cluster_to_centroids(x, im2cluster):
    # returns an np.ndarray of c_i, where each c_i is the mean of all inputs with cluster i

    centroids = np.zeros((max(im2cluster) + 1, len(x[0]))) # (num_clusters, C)

    # unique_clusters = set(im2cluster)
    counts = np.zeros(max(im2cluster) + 1) # (num_clusters)

    for idx in range(len(x)):
        cluster = im2cluster[idx]
        centroids[cluster] += x[idx]
        counts[cluster] += 1
    
    centroids = centroids / np.expand_dims(counts, axis=1) # taking mean of vectors in each cluster

    # normalizing since means won't lie on the unit sphere after taking mean (could be like in the middle, this is especially likely for noise class)
    centroids = centroids / np.linalg.norm(centroids, axis=1).reshape(-1, 1)

    return centroids

def l2_distance(x, y):
    return np.linalg.norm(x - y)


def run_dbscan(x, minPts=200, minSamples=0, temperature=0.2):
    # make sure the parser args has the necessary dbscan parameters (eps, minPts) - ADDED

    # x = x.numpy() # this is already done before calling the function
    # n = x.shape[0] # number of samples
    # d = x.shape[1] # dimension
    print('performing dbscan clustering')
    results = {'im2cluster':[],'centroids':[],'density':[],'sampled_protos':[]}


    # x = x.numpy()
    pca = PCA(n_components = 20)
    features = pca.fit_transform(x)
    # print(pca.explained_variance_ratio_)

    # db = DBSCAN(eps=args.eps, min_samples=args.minPts, n_jobs=-1, metric='euclidean').fit(features) # run DBSCAN
    # im2cluster = db.labels_

    if minSamples:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=minPts, min_samples=minSamples)
    else:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=minPts)

    im2cluster = clusterer.fit_predict(features)
    

    if -1 in im2cluster: # so that noise data is in cluster 0 instead of -1
        im2cluster += 1 
    centroids = im2cluster_to_centroids(x, im2cluster)

    # density = np.ones(len(set(im2cluster))) * args.temperature

    Dcluster = [[] for c in range(len(centroids))]
    for im,i in enumerate(im2cluster):
        # Dcluster[i].append(D[im][0])
        Dcluster[i].append(l2_distance(centroids[i], x[im]))

    # concentration estimation (phi)
    density = np.zeros(len(centroids))
    for i,dist in enumerate(Dcluster):
        if len(dist)>1:
            density[i] = (np.asarray(dist)).mean()/np.log(len(dist)+10) # i got rid of the **0.5 since then we aren't actually doing l2 distances? idk why the authors did it (tbd)    
            
    #if cluster only has one point, use the max to estimate its concentration        
    dmax = density.max()
    for i,dist in enumerate(Dcluster):
        if len(dist)<=1:
            density[i] = dmax 

    density = density.clip(np.percentile(density,10),np.percentile(density,90)) #clamp extreme values for stability
    density = temperature*density/density.mean()  #scale the mean to temperature 


    im2cluster = torch.LongTensor(im2cluster).cuda(gpu)   

    unique_clusters = set(im2cluster.tolist())
    print(unique_clusters)

    density = torch.Tensor(density).cuda(gpu)
    centroids = torch.Tensor(centroids).cuda(gpu)
    # centroids = nn.functional.normalize(centroids, p=args.norm_p, dim=1) # hmmmm ? 

    results['centroids'].append(centroids)
    results['density'].append(density)
    results['im2cluster'].append(im2cluster)

    # run dbscan, and then select a random core point from each cluster to be the centroid
    return results

    # density = np.ones(len(set(im2cluster))) * args.temperature

def run_kmeans(x, num_cluster=['250'], centroid_sampling=False, temperature=0.2):
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
    


def calculate_kn_distance(X,k):

    kn_distance = []
    for i in range(len(X)):
        eucl_dist = []
        for j in range(len(X)):
            # eucl_dist.append(
            #     math.sqrt(
            #         ((X[i,0] - X[j,0]) ** 2) +
            #         ((X[i,1] - X[j,1]) ** 2)))
            eucl_dist.append(np.linalg.norm(X[i] - X[j]))
                    

        eucl_dist.sort()
        kn_distance.append(eucl_dist[k])

    return kn_distance


# visualizing the data

print('Visualizing Representations...')
# methods = [LocallyLinearEmbedding(n_components=2, method='standard'), TSNE(n_components=2, init='pca')]

# tsne = TSNE(n_components=2, perplexity=45, learning_rate=200, verbose=1)
# methods = tsne
def plot_tsne(num_classes=20, num_samples=10000):

    features, classes = compute_features(eval_loader, model, low_dim=low_dim, gpu=gpu)
    features[torch.norm(features,dim=1)>1.5] /= 2 #account for the few samples that are computed twice  
    features = features.numpy()
    # results = run_dbscan(features)
    # results = run_kmeans(features)
    # im2cluster = results['im2cluster'][0].tolist() # remember to turn this back to a list
    restricted_classes = [i for i in classes if i < num_classes]
    features = features[np.array(classes) < num_classes]
    print(len(restricted_classes))
    print(len(features))

    features = features[:num_samples]
    restricted_classes = restricted_classes[:num_samples]

    hparams = [2500]
    fig = plt.figure(figsize=(25,25))

    for i, hparam in enumerate(hparams):
        print('hparam: {}'.format(i))
        # tsne = TSNE(n_components=2, perplexity=30, learning_rate=hparam, verbose=1, n_jobs=-1, n_iter=2500)
        tsne = cudaTSNE(n_components=2, perplexity=50, learning_rate=600, verbose=1, n_iter=hparam)

        y = tsne.fit_transform(features)
        if len(hparams) == 1:
            ax = fig.add_subplot(1, len(hparams), i + 1)
        else:
            ax = fig.add_subplot(3, len(hparams)//3 + 1, i + 1)

        ax.scatter(y[:, 0], y[:, 1], c=restricted_classes)
        ax.set_title('hparam: {}'.format(hparam))

    if not os.path.exists('imgs/tsne_{}'.format(checkpoint_id[0])):
        os.makedirs('imgs/tsne_{}'.format(checkpoint_id[0]))
    
    save_path = 'imgs/tsne_{}/tsne_{}_{}'.format(checkpoint_id[0], checkpoint_id[0], checkpoint_id[1])
    fig.savefig(save_path)
    print('Figure saved to : {}'.format(save_path))



def plot_umap(num_classes=20, num_samples=10000):

    features, classes = compute_features(eval_loader, model, low_dim=low_dim, gpu=gpu)
    features[torch.norm(features,dim=1)>1.5] /= 2 #account for the few samples that are computed twice  
    features = features.numpy()
    # results = run_dbscan(features)
    # results = run_kmeans(features)
    # im2cluster = results['im2cluster'][0].tolist() # remember to turn this back to a list
    restricted_classes = np.array([i for i in classes if i < num_classes])
    features = features[np.array(classes) < num_classes]

    features = features[:num_samples]
    restricted_classes = restricted_classes[:num_samples]

    hparams = [2500]

    for i, hparam in enumerate(hparams):
        print('hparam: {}'.format(i))
        reducer = umap.UMAP(n_neighbors = 30, min_dist=0.1, n_components=2, metric='cosine')

        y = reducer.fit_transform(features)

        ax = umap.plot.points(reducer, labels=restricted_classes)

        ax.set_title('hparam: {}'.format(hparam))

    if not os.path.exists('imgs/umap_{}'.format(checkpoint_id[0])):
        os.makedirs('imgs/umap_{}'.format(checkpoint_id[0]))
    
    save_path = 'imgs/umap_{}/umap_{}_{}'.format(checkpoint_id[0], checkpoint_id[0], checkpoint_id[1])
    # save_path = 'imgs/umap_{}/umap_{}_{}'.format(checkpoint_id[0], 'cosine', checkpoint_id[1])


    # fig = ax.get_figure()
    ax.figure.savefig(save_path)
    print('Figure saved to : {}'.format(save_path))


def plot_progression(checkpoint_id, num_rows=3, num_progressions=9, num_classes=20, num_samples=10000, algorithm = 'umap', true_classes=True):


    checkpoints = os.listdir('pcl_cifar10_{}'.format(checkpoint_id[0]))
    checkpoints.sort()
    assert num_progressions <= len(checkpoints), 'Not enough checkpoints saved.'
    checkpoints = [checkpoints[i] for i in range(0, len(checkpoints), len(checkpoints) // (num_progressions-1))][:num_progressions-1] + [checkpoints[-1]]

    fig, axes = plt.subplots(num_rows, len(checkpoints)//num_rows + 1 if len(checkpoints) % num_rows != 0 else len(checkpoints)//num_rows, figsize=(40,25))

    print(checkpoints)

    for i, checkpoint_file in enumerate(checkpoints):
        # print(checkpoints)
        # print(checkpoint_file)
        epoch = checkpoint_file[11:15] # just the 4-digit checkpoint epoch

        print('epoch: {}'.format(epoch))
        checkpoint = torch.load('pcl_cifar10_{}/{}'.format(checkpoint_id[0], checkpoint_file))
        model.load_state_dict(checkpoint['state_dict'])
        features, classes = compute_features(eval_loader, model, low_dim=low_dim, gpu=gpu)
        features[torch.norm(features,dim=1)>1.5] /= 2 #account for the few samples that are computed twice  
        features = features.numpy()
        restricted_classes = np.array([i for i in classes if i < num_classes])
        features = features[np.array(classes) < num_classes]

        features = features[:num_samples]
        restricted_classes = restricted_classes[:num_samples]

        if algorithm == 'umap':
            reducer = umap.UMAP(n_neighbors = 60, min_dist=0.1, n_components=2, metric='cosine')
            y = reducer.fit_transform(features)
        elif algorithm == 'tsne':
            tsne = cudaTSNE(n_components=2, perplexity=50, learning_rate=600, verbose=1, n_iter=2500, metric='euclidean')
            y = tsne.fit_transform(features)


        if true_classes:
            scatter = axes.flat[i].scatter(y[:, 0], y[:, 1], c = restricted_classes, cmap='Spectral', s=3)
        else:
            with torch.no_grad():
                results = run_dbscan(features, minPts=50, minSamples=0, temperature=0.2)
                im2cluster = results['im2cluster'][0].tolist() # remember to turn this back to a list
            scatter = axes.flat[i].scatter(y[:, 0], y[:, 1], c = im2cluster, cmap='Spectral', s=3) # restricting num_classes does not work here


        # legend = axes.flat[i].legend(*scatter.legend_elements(), loc='lower left', title="Classes")
        # axes.flat[i].add_artist(legend)

        axes.flat[i].set_title('epoch: {}'.format(epoch))
    
    axes.flat[-1].legend(*scatter.legend_elements(), loc='lower left', title="Classes", bbox_to_anchor=(1.00, 0), prop={'size': 25})
    fig.suptitle('{}_{}: {}'.format(algorithm, checkpoint_id[0], 'Cifar Classes' if true_classes else 'Clustering Classes'), fontsize=20)

    if not os.path.exists('imgs/{}_{}'.format(algorithm, checkpoint_id[0])):
        os.makedirs('imgs/{}_{}'.format(algorithm, checkpoint_id[0]))
    save_path = 'imgs/{}_{}/{}_{}'.format(algorithm, checkpoint_id[0], 'progression', 'true_classes' if true_classes else 'cluster_classes')


    fig.savefig(save_path)
    print('Figure saved to : {}'.format(save_path))

def plot_comparison(checkpoint_id, num_progressions=5, num_classes=20, num_samples=10000, algorithm = 'umap'):

    checkpoints = os.listdir('pcl_cifar10_{}'.format(checkpoint_id[0]))
    checkpoints.sort()
    assert num_progressions <= len(checkpoints), 'Not enough checkpoints saved.'
    checkpoints = [checkpoints[i] for i in range(0, len(checkpoints), len(checkpoints) // (num_progressions-1))][:num_progressions-1] + [checkpoints[-1]]

    fig, axes = plt.subplots(len(checkpoints), 2, figsize=(30,50))

    print(checkpoints)

    for i, checkpoint_file in enumerate(checkpoints):
        # print(checkpoints)
        # print(checkpoint_file)
        epoch = checkpoint_file[11:15] # just the 4-digit checkpoint epoch

        print('epoch: {}'.format(epoch))
        checkpoint = torch.load('pcl_cifar10_{}/{}'.format(checkpoint_id[0], checkpoint_file))
        model.load_state_dict(checkpoint['state_dict'])
        features, classes = compute_features(eval_loader, model, low_dim=low_dim, gpu=gpu)
        features[torch.norm(features,dim=1)>1.5] /= 2 #account for the few samples that are computed twice  
        features = features.numpy()
        restricted_classes = np.array([i for i in classes if i < num_classes])
        features = features[np.array(classes) < num_classes]

        features = features[:num_samples]
        restricted_classes = restricted_classes[:num_samples]

        if algorithm == 'umap':
            reducer = umap.UMAP(n_neighbors = 60, min_dist=0.1, n_components=2, metric='cosine')
            y = reducer.fit_transform(features)
        elif algorithm == 'tsne':
            tsne = cudaTSNE(n_components=2, perplexity=50, learning_rate=600, verbose=1, n_iter=2500, metric='euclidean')
            y = tsne.fit_transform(features)


        scatter = axes.flat[2*i].scatter(y[:, 0], y[:, 1], c = restricted_classes, cmap='Spectral', s=3)
        with torch.no_grad():
            results = run_dbscan(features, minPts=200, minSamples=0, temperature=0.2)
            im2cluster = results['im2cluster'][0].tolist() # remember to turn this back to a list
        scatter = axes.flat[2*i + 1].scatter(y[:, 0], y[:, 1], c = im2cluster, cmap='Spectral', s=3) # restricting num_classes does not work here


        # legend = axes.flat[i].legend(*scatter.legend_elements(), loc='lower left', title="Classes")
        # axes.flat[i].add_artist(legend)

        axes.flat[2*i].set_title('Cifar Classes, epoch: {}'.format(epoch))
        axes.flat[2*i + 1].set_title('Clustering Classes, epoch: {}'.format(epoch))

    
    # axes.flat[-1].legend(*scatter.legend_elements(), loc='lower left', title="Classes", bbox_to_anchor=(1.00, 0), prop={'size': 25})
    fig.suptitle('{}_{} Comparison'.format(algorithm, checkpoint_id[0]), fontsize=50)

    if not os.path.exists('imgs/{}_{}'.format(algorithm, checkpoint_id[0])):
        os.makedirs('imgs/{}_{}'.format(algorithm, checkpoint_id[0]))
    save_path = 'imgs/{}_{}/comparison'.format(algorithm, checkpoint_id[0])


    fig.savefig(save_path)
    print('Figure saved to : {}'.format(save_path))


def plot_knn(x, k=127, num=1000):

    features, _ = compute_features(eval_loader, model, low_dim=low_dim, gpu=gpu)
    features[torch.norm(features,dim=1)>1.5] /= 2 #account for the few samples that are computed twice  
    features = features.numpy()[:num]


    fig = plt.figure(figsize=(25,25))

    # y = tsne.fit_transform(features)
    # k=127
    eps_dist = calculate_kn_distance(x, k)
    ax = fig.add_subplot(1, 1, 1)
    # ax.scatter(y[:, 0], y[:, 1], c=results['im2cluster'][0].tolist())
    ax.hist(eps_dist, bins=40)
    ax.set_title('k-distance plot, k={}'.format(k))

    if not os.path.exists('imgs/k_distance_{}'.format(checkpoint_id[0])):
        os.makedirs('imgs/k_distance_{}'.format(checkpoint_id[0]))
    save_path = 'imgs/k_distance_{}/kdistance_hist_{}_{}'.format(checkpoint_id[0], checkpoint_id[0], checkpoint_id[1])
    fig.savefig(save_path)
    print('Figure saved to : {}'.format(save_path))


def plot_knn2(x, k=127, num=1000):

    features, _ = compute_features(eval_loader, model, low_dim=low_dim, gpu=gpu)
    features[torch.norm(features,dim=1)>1.5] /= 2 #account for the few samples that are computed twice  
    features = features.numpy()[:num]


    fig = plt.figure(figsize=(25,25))

    # y = tsne.fit_transform(features)
    # k=127
    eps_dist = calculate_kn_distance(x, k) # sorted here
    eps_dist.sort()
    eps_dist.reverse()
    ax = fig.add_subplot(1, 1, 1)
    # ax.scatter(y[:, 0], y[:, 1], c=results['im2cluster'][0].tolist())
    ax.plot(list(range(len(eps_dist))), eps_dist)
    ax.set_title('k-distance plot, k={}'.format(k))

    if not os.path.exists('imgs/k_distance_{}'.format(checkpoint_id[0])):
        os.makedirs('imgs/k_distance_{}'.format(checkpoint_id[0]))
    
    save_path = 'imgs/k_distance_{}/kdistances_{}_{}'.format(checkpoint_id[0], checkpoint_id[0], checkpoint_id[1])
    fig.savefig(save_path)
    print('Figure saved to : {}'.format(save_path))

from time import time

    
low_dim=128
pcl_r=200
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

gpu = 0
model = model.cuda(gpu)


# checkpoint = torch.load('pcl_cifar10_{}/checkpoint_{}.pth.tar'.format(checkpoint_id[0], checkpoint_id[1]))
# model.load_state_dict(checkpoint['state_dict'])

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


# checkpoint_id=['dbscan_eps_point8_minpts_128','0199']
# checkpoint_id=['dbscan_eps_point41_minpts_128','0019']
# checkpoint_id=['kmeans_r_100','0019']
# checkpoint_id=['kmeans_r_200','0199']
# checkpoint_id=['kmeans_batchsize_50','0019']
# checkpoint_id=['hdbscan_minpts_200','0019']
# checkpoint_id=['InfoNCE','0109']
# checkpoint_id=['hdbscan_minsamples_5','0199']
# checkpoint_id=['hdbscan_density_fixed_1','0199']
# checkpoint_id=['hdbscan','0019']
# checkpoint_id=['kmeans','0019']
checkpoint_id=['InfoNCE','0019']



start_time = time()
# plot_knn2(features, 127)
# plot_tsne(num_classes=20, num_samples=50000)
# plot_umap(num_classes=20, num_samples=50000)
# plot_umap_progression(num_classes=20, num_samples=50000)
plot_progression(checkpoint_id, num_rows=3, num_progressions=15, num_classes=20, num_samples=50000, algorithm='umap', true_classes=True)
# plot_comparison(checkpoint_id, num_progressions=5, num_classes=20, num_samples=50000, algorithm = 'umap')



end_time = time()

print('time elapsed: {}'.format(end_time-start_time))