import torch
import torch.nn as nn

from einops import rearrange


class DetCo(nn.Module):
    """
    DetCo: Unsupervised Contrastive Learning for Object Detection
    https://arxiv.org/pdf/2102.04803.pdf
    """
    def __init__(self, cfg, base_encoder, dim=128, K=65536, m=0.999, T=0.07):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super().__init__()

        self.K = K
        self.m = m
        self.T = T
        self.n = 8 #  4 global vectors and 4 local vectors

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder
        import copy
        self.encoder_k=copy.deepcopy(self.encoder_q)
        
        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES


        

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        
        self.queue_list=[]
        dim1_list=[256,256,256,256]
        dim2_list=[64,64,64,64]
        for i in range(4):
            self.register_buffer("queue", torch.randn(dim1_list[i],dim2_list[i], dim2_list[i], K))
            self.queue_list.append(nn.functional.normalize(self.queue, dim=1))
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
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, :, ptr:ptr + batch_size] = keys.permute(1,2,0)
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]
    
    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        batch_size = im_q.size(0)
        
        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        
        features = list()
        for f in self.in_features:
            feature = q[f]
            features.append(feature)
        #features= torch.stack(features)
        
        q=[]
        for f in features:
            q.append(nn.functional.normalize(f, dim=2))
            
        
        #q = nn.functional.normalize(features, dim=2)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            features2 = list()

            for f in self.in_features:
                feature = k[f]
                features2.append(feature)
            #features= torch.stack(features)
        
            k=[]
            for f in features2:
                k.append(nn.functional.normalize(f, dim=2))
            
            #k = nn.functional.normalize(k, dim=2)

            # undo shuffle
            for i in range(len(k)):
                k[i] = self._batch_unshuffle_ddp(k[i], idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # g2g and l2l logits as shown in Fig.4(b) of DetCo paper
        queue = self.queue.clone().detach()
        
        l_pos=0
        l_neg=0
        l_pos_cross=0
        l_neg_cross=0
        
        for i in range(len(q)):
            print(q[i].shape)
            print(queue.shape)
            l_pos += torch.einsum('bnwc,bnwc->bn', [q[i], k[i]]).unsqueeze(2)
            l_neg += torch.einsum('bnw,nwk->bnk', [q[i], queue])

            # l2g logits
            #n=n, w=dim, k=k
            l_pos_cross += torch.einsum('bnw,bnw->bn', [q[i][:,4:,:], k[i][:,:4,:]]).unsqueeze(2)
            l_neg_cross += torch.einsum('bnw,nwk->bnk', [q[i][:,4:,:], queue[:4,:,:]])

        l_pos = torch.cat([l_pos, l_pos_cross], dim=1)
        l_neg = torch.cat([l_neg, l_neg_cross], dim=1)

        # logits: BxNx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=2)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[:2], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
