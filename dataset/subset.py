import torch
from torch.utils.data import Dataset

from architecture import ModelSpec


class SplitSubet(Dataset):
    def __init__(self, full_dataset, indices: list, n_tier=5):
        self.full_dataset = full_dataset
        self.indices = indices

        self.subset = [self.full_dataset[i] for i in self.indices]
        self._sort(n_tier)

    def __getitem__(self, index):
        return self.subset[index]

    def __len__(self):
        return len(self.indices)

    def _sort(self, n_tier):
        assert len(self.indices) >= n_tier, 'batch_sz should be larger than n_tier'
        step = int(len(self.indices) / n_tier)
        
        sorted_list = sorted(enumerate(self.subset),key=lambda x:x[1][1],reverse=True)
        
        for i in range(n_tier):
            if i == n_tier - 1:
                t_list = sorted_list[i * step:]
            else:
                t_list = sorted_list[i * step:(i + 1) * step]
            for idx, _ in t_list:
                target = torch.zeros(1, n_tier)
                target[0, i].add_(1)
                target = tuple(target)
                self.subset[idx] = self.subset[idx] + target

    # Query subnet inside the sub dataset
    def query_stats_by_spec(self, model_spec: ModelSpec):

        arch_dict = self.full_dataset.database.check_arch_inside_dataset(model_spec)
        if arch_dict is None:
            return None, None, None
        model_hash = arch_dict['unique_hash']

        hash_list_idx = self.full_dataset.keys_list.index(model_hash)

        index_list_idx = self.full_dataset.index_list.index(hash_list_idx)

        if index_list_idx not in self.indices:
            return None, None, None

        return arch_dict['flops'], arch_dict['trainable_parameters'], index_list_idx


class SplitSubet201(Dataset):
    def __init__(self, full_dataset, indices: list, n_tier=5):
        self.full_dataset = full_dataset
        self.indices = indices

        self.subset = [self.full_dataset[i] for i in self.indices]

        self._sort(n_tier)

    def __getitem__(self, index):
        return self.subset[index]

    def __len__(self):
        return len(self.indices)

    def _sort(self, n_tier):
        assert len(self.indices) >= n_tier, 'batch_sz should be larger than n_tier'
        step = int(len(self.indices) / n_tier)
        
        for datatype in ['cifar10', 'imagenet16', 'cifar100']:

            sorted_list = sorted(enumerate(self.subset),key=lambda x:x[1][datatype][1],reverse=True)
            
            for i in range(n_tier):
                
                if i == n_tier - 1:
                    t_list = sorted_list[i * step:]
                else:
                    t_list = sorted_list[i * step:(i + 1) * step]
                for idx, _ in t_list:
                    target = torch.zeros(1, n_tier)
                    target[0, i].add_(1)
                    target = tuple(target)
                    self.subset[idx][datatype] = self.subset[idx][datatype] + target
