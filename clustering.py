import faiss
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
from sklearn.cluster import DBSCAN

from tqdm import tqdm

def compute_features(eval_loader, model, args):
    print('Computing features...')
    model.eval()
    features = torch.zeros(len(eval_loader.dataset),args.low_dim).cuda()
    for i, (images, index) in enumerate(tqdm(eval_loader)):
        with torch.no_grad():
            images = images.cuda(non_blocking=True)
            feat = model(images,is_eval=True) 
            features[index] = feat
    # dist.barrier()        
    # dist.all_reduce(features, op=dist.ReduceOp.SUM)     
    return features.cpu()

def compute_features(eval_loader, model, low_dim=128):
    print('Computing features...')
    model.eval()
    features = torch.zeros(len(eval_loader.dataset),low_dim).cuda()
    for i, (images, index) in enumerate(tqdm(eval_loader)):
        with torch.no_grad():
            images = images.cuda(non_blocking=True)
            feat = model(images,is_eval=True) 
            features[index] = feat
    # dist.barrier()        
    # dist.all_reduce(features, op=dist.ReduceOp.SUM)     
    return features.cpu()

def run_kmeans(x, args):
    """
    Args:
        x: data to be clustered
    """
    
    print('performing kmeans clustering')
    results = {'im2cluster':[],'centroids':[],'density':[],'sampled_protos':[]}
    
    for seed, num_cluster in enumerate(args.num_cluster):
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
        cfg.device = args.gpu    
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

        if args.centroid_sampling:
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
        density = args.temperature*density/density.mean()  #scale the mean to temperature 
        
        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).cuda()
        centroids = nn.functional.normalize(centroids, p=args.norm_p, dim=1) # hmmmm ? 

        if args.centroid_sampling:
            for i in range(k):
                sampled_protos[i] = torch.Tensor(sampled_protos[i]).cuda()

        im2cluster = torch.LongTensor(im2cluster).cuda()               
        density = torch.Tensor(density).cuda()
        
        results['centroids'].append(centroids)
        results['density'].append(density)
        results['im2cluster'].append(im2cluster)
        if args.centroid_sampling:
            results['sampled_protos'].append(sampled_protos) 
        
    return results

def im2cluster_to_centroids(x, im2cluster):
    # returns an np.ndarray of c_i, where each c_i is the mean of all inputs with cluster i

    centroids = np.zeros((max(im2cluster) + 1, len(x[0]))) # (num_clusters, C)

    # unique_clusters = set(im2cluster)
    counts = np.zeros(max(im2cluster) + 1) # (num_clusters)

    for idx in range(len(x)):
        cluster = im2cluster[idx]
        centroids[cluster] += x[idx]
        counts[cluster] += 1

    return centroids / np.expand_dims(counts, axis=1)

    # returns 





from sklearn.manifold import TSNE, LocallyLinearEmbedding
import matplotlib.pyplot as plt
import wandb

def run_dbscan(x, args):
    # make sure the parser args has the necessary dbscan parameters (eps, minPts) - ADDED

    # x = x.numpy() # this is already done before calling the function
    # n = x.shape[0] # number of samples
    # d = x.shape[1] # dimension
    print('performing dbscan clustering')
    results = {'im2cluster':[],'centroids':[],'density':[],'sampled_protos':[]}


    print(x)
    print( np.linalg.norm(x[0] - x[1], ord=2) )
    print( np.linalg.norm(x[7] - x[3], ord=2) )

    # visualizing the data

    # print('Visualizing Representations...')
    # methods = [LocallyLinearEmbedding(n_components=2, method='standard'), TSNE(n_components=2, init='pca')]
    # fig = plt.figure(figsize=(15,8))
    # for i, method in enumerate(methods):
    #     y = method.fit_transform(x)
    #     ax = fig.add_subplot(1, len(methods), i + 1)
    #     ax.scatter(y[:, 0], y[:, 1])

    # fig.savefig('visualized_features')

    # plt.show()

    db = DBSCAN(eps=args.eps, min_samples=args.minPts, n_jobs=-1).fit(x) # run DBSCAN

    im2cluster = db.labels_
    # print(im2cluster)
    if -1 in im2cluster: # so that noise data is in cluster 0 instead of -1
        print('help')
        im2cluster += 1 
    centroids = im2cluster_to_centroids(x, im2cluster)

    density = np.ones(len(set(im2cluster))) * args.temperature

    wandb.log({'Number of Clusters': len(set(im2cluster))})

    im2cluster = torch.LongTensor(im2cluster).cuda()           
    print(set(im2cluster.tolist()))
    density = torch.Tensor(density).cuda()
    centroids = torch.Tensor(centroids).cuda()
    centroids = nn.functional.normalize(centroids, p=args.norm_p, dim=1) # hmmmm ? 

    results['centroids'].append(centroids)
    results['density'].append(density)
    results['im2cluster'].append(im2cluster)



    # run dbscan, and then select a random core point from each cluster to be the centroid
    return results