""""
Data: 2021/09/15
Target: Store and Update tier Embedding
Method: element-wise add all the embeddings belong to this tier and devided by the number of embeddings for scaling
Variable: self.counts_dict
            {
            'flops':[f1,f2,f3,f4,f5]
            'params':[p1,p2,p3,p4,p5]
            'edges':[e1,e2,e,3,e4,e5]
            } 
"""

import torch
import copy

class Bucket(object):

    n_tier = 0

    def __init__(self, flag_tier, name_tier, n_arch_patch=19, d_patch_vec=512, space='nasbench'):
        
        assert flag_tier == Bucket.n_tier, "tier flag should be the same with the number of the Bucket instances"
        self.space = space
        self.flag_tier = flag_tier
        self.name_tier = name_tier
        self._n_arch_patch = n_arch_patch
        self._d_patch_vec = d_patch_vec

        self._total_bucket_emb = torch.zeros(
            n_arch_patch, d_patch_vec).unsqueeze(dim=0)
        self.current_bucket_emb = torch.zeros(
            n_arch_patch, d_patch_vec).unsqueeze(dim=0)

        self.counts_dict = {}

        self._emb_count = 0

        Bucket.n_tier += 1

    def get_bucket_emb(self):
        return copy.deepcopy(self.current_bucket_emb)

    def get_bucket_counts(self):
        return copy.deepcopy(self.counts_dict)

    def updata_bucket_emb(self, input_emb):
        assert self._n_arch_patch == input_emb.size(1), "Wrong patch length"
        assert self._d_patch_vec == input_emb.size(2), "Wrong patch embedding dimension"
        n_input_emb = input_emb.size(0)

        if input_emb.is_cuda:
            self._total_bucket_emb = self._total_bucket_emb.cuda(input_emb.device)

        added_input_emb = input_emb.sum(dim=0, keepdim=True)
        self._total_bucket_emb += added_input_emb
        self._emb_count += n_input_emb

        self.current_bucket_emb = self._total_bucket_emb / self._emb_count

        assert self.current_bucket_emb.size(0) == 1, "The length of bucket emb dimension should be 1"

    def update_counts_dict(self, params, flops, y):
        self.counts_dict['params'] = params
        self.counts_dict['flops'] = flops
        
        if self.space == 'nasbench':
            self.counts_dict['n_nodes'] = y
        if self.space == 'nasbench201':
            self.counts_dict['edges'] = y

    @property
    def emb_count(self):
        return self._emb_count        

    @classmethod
    def get_n_tier(cls):
        return cls.n_tier

    @classmethod
    def reset_n_tier(cls):
        cls.n_tier = 0

    def __del__(self):
        Bucket.n_tier -= 1
