import torch
import copy
from torch.utils.data import Dataset

from architecture import ModelSpec, feature_tensor_encoding

from .nasbench_database import NASBenchDataBase


class NASBenchDataset(Dataset):
    def __init__(self, database: NASBenchDataBase, seed):
        self.database = database
        g_cpu = torch.Generator()
        g_cpu.manual_seed(seed)
        self.index_list = torch.randperm(self.database.size, generator=g_cpu).tolist()
        self.keys_list = list(self.database.hash_iterator())


    def __getitem__(self, index):
        model_hash = self.keys_list[self.index_list[index]]
        arch = self.database.query_by_hash(model_hash)
        
        arch_feature = feature_tensor_encoding(copy.deepcopy(arch))
        validation_accuracy = arch['avg_validation_accuracy']
        test_accuracy = arch['avg_test_accuracy']
        params = arch['trainable_parameters']
        flops = arch['flops']
        n_nodes = len(arch['module_adjacency'])
        rank = arch['rank']
        
        return arch_feature, validation_accuracy, test_accuracy, params, flops, n_nodes, rank

    # Query subnet in the entire set
    def query_stats_by_spec(self, model_spec: ModelSpec):
        arch_dict = self.database.check_arch_inside_dataset(model_spec)
        if arch_dict is None:
            return None, None, None

        model_hash = arch_dict['unique_hash']
        hash_list_idx = self.keys_list.index(model_hash)
        index_list_idx = self.index_list.index(hash_list_idx)

        return arch_dict['flops'], arch_dict['trainable_parameters'], index_list_idx

    def __len__(self):
        return self.database.size