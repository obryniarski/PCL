import torch
import torch.nn as nn
from random import sample, choices

class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, r=16384, m=0.999, T=0.1, mlp=False, proto_sampling=False, p=2, gpu=0):
        """
        dim: feature dimension (default: 128)
        r: queue size; number of negative samples/prototypes (default: 16384)
        m: momentum for updating key encoder (default: 0.999)
        T: softmax temperature 
        mlp: whether to use mlp projection
        proto_sampling: whether to use k-means centroids or samples from each cluster as prototype
        p: Lp normalization exponent
        """
        super(MoCo, self).__init__()

        self.r = r
        self.m = m
        self.T = T
        self.dim = dim
        self.gpu = gpu

        self.proto_sampling = proto_sampling
        self.p = p

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, r))
        self.queue = nn.functional.normalize(self.queue, dim=0, p=self.p)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.r % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.r  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        batch_size_this = x.shape[0]
        
        idx_shuffle = torch.randperm(batch_size_this).cuda(self.gpu)
        idx_unshuffle = torch.argsort(idx_shuffle)

        idx_this = idx_shuffle.view(-1)
        return x[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # batch_size_this = x.shape[0]
        idx_this = idx_unshuffle.view(-1)
        return x[idx_this]


    def select_prototypes(self, prototypes, im2cluster, index):
        """
        Selects the positive and negative prototypes for use in forward method.
        """

        # sample positive prototypes
        pos_proto_id = im2cluster[index] # 1-dim torch tensor mapping image index -> cluster


        pos_prototypes = prototypes[pos_proto_id] 
        
        # sample negative prototypes
        all_proto_id = [i for i in range(im2cluster.max())]     # list of all possible cluster id's
        # print('1', all_proto_id)
        # all_proto_id = torch.unique(im2cluster).tolist() # list of all possible cluster id's
        # print('2', all_proto_id)


        # print(set(all_proto_id))
        # print(set(pos_proto_id.tolist()))
        # print(pos_proto_id)
        # print(all_proto_id)
        # neg_proto_id = set(all_proto_id)-set(pos_proto_id.tolist())
        # # print(neg_proto_id)
        # neg_proto_id = sample(neg_proto_id,self.r) # sample r negative prototypes 
        # neg_prototypes = prototypes[neg_proto_id]

        neg_proto_id = all_proto_id[:self.r] # take the first r classes, or all of them if there are less
        neg_prototypes = prototypes[neg_proto_id]

        proto_selected = torch.cat([pos_prototypes,neg_prototypes],dim=0)
        return proto_selected, pos_proto_id, neg_proto_id

    def forward(self, im_q, im_k=None, is_eval=False, cluster_result=None, index=None):
        """
        Input:
            im_q: a batch of query images 
            im_k: a batch of key images 
            is_eval: return momentum embeddings (used for clustering)
            cluster_result: cluster assignments, centroids, and density
            index: indices for training samples [N]
        Output:
            logits, targets, proto_logits, proto_targets
        """
        # if index != None:
        #     print(index.shape)
        if is_eval:
            k = self.encoder_k(im_q)  
            k = nn.functional.normalize(k, dim=1, p=self.p)            
            return k


        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            # print(k.shape)
            k = nn.functional.normalize(k, dim=1, p=self.p) # sidenote: this projects the representations onto the hypersphere

            # undo shuffle
            k = self._batch_unshuffle(k, idx_unshuffle)


        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1, p=self.p)
        
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: Nxr
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+r)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda(self.gpu)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        # prototypical contrast
        if cluster_result is not None:  
            proto_labels = []
            proto_logits = []

            if self.proto_sampling:
                sampler = enumerate(zip(cluster_result['im2cluster'],cluster_result['sampled_protos'],cluster_result['density']))
                # assume that cluster_result['sampled_protos'] has an array of every element of a cluster i at the ith index
                # shape of cluster_result['sampled_protos'][i] = (num_elements_in_cluster_i, 128)
                # THIS DOESN'T WORK ^
            else:
                sampler = enumerate(zip(cluster_result['im2cluster'],cluster_result['centroids'],cluster_result['density']))
                # assume that cluster_result['centroids'] has just a single centroid at each index i 

            for n, (im2cluster,prototypes,density) in sampler:

                # lengths = None # THIS IS A PLACEHOLDER, PROTO_SAMPLING=TRUE WONT WORK WITH THIS HERE
                proto_selected, pos_proto_id, neg_proto_id = self.select_prototypes(prototypes, im2cluster, index) #NEED TO GET LENGTHS HERE SOMEHOW
                # proto_selected.shape = [N, C]

                # I FIXED THIS PART -------------------------
                # compute prototypical logits
                logits_proto = torch.mm(q,proto_selected.t()) # q is NxC and proto_selected.t() is Cx(N+r) (actually Cx min(N+r, N+num_clusters))
                # scaling temperatures for the selected prototypes
                temp_proto = density[torch.cat([pos_proto_id,torch.LongTensor(neg_proto_id).cuda()],dim=0)] #neg_proto_id should just be like the first min(pcl_r, num_clusters) values 
                logits_proto /= temp_proto

                # we want to turn the left CxN matrix into just a single column of its diagonal
                pos_logits = torch.diag(logits_proto, 0).view(-1, 1)
                neg_logits = logits_proto[:, pos_logits.shape[0]:] # shape is Nx min(r, num_clusters)


                # zero out any logits corresponding to the positive class in a given row
                mask = torch.ones_like(neg_logits)
                for i in range(len(mask)):
                    if pos_proto_id[i] < mask.shape[1]:
                        mask[i, pos_proto_id[i]] -= 1
                neg_logits = torch.einsum('nr, nr -> nr', [neg_logits, mask])

                # logits_proto = torch.cat([pos_logits.view(pos_logits.shape[0], 1), neg_logits, dim=1)
                logits_proto = torch.cat([pos_logits, neg_logits], dim=1)


                # targets for prototype assignment
                # labels_proto = torch.linspace(0, q.size(0)-1, steps=q.size(0)).long().cuda() # basically range(0, q.size(0)) but in pytorch
                labels_proto = torch.zeros(logits_proto.shape[0], dtype=torch.long).cuda(self.gpu)

                # print("NEW --------", labels_proto)
                # I FIXED THIS PART -------------------------


                # # compute prototypical logits
                # logits_proto = torch.mm(q,proto_selected.t())
                
                # # targets for prototype assignment
                # labels_proto = torch.linspace(0, q.size(0)-1, steps=q.size(0)).long().cuda()
                
                # # scaling temperatures for the selected prototypes
                # temp_proto = density[torch.cat([pos_proto_id,torch.LongTensor(neg_proto_id).cuda()],dim=0)]  
                # logits_proto /= temp_proto
                
                
                proto_labels.append(labels_proto)
                proto_logits.append(logits_proto)
            return logits, labels, proto_logits, proto_labels
        else:
            return logits, labels, None, None