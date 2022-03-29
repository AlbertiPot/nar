import torch
import copy

from torch.utils.data import Dataset

from architecture import feature_tensor_encoding_201

from .nasbench201_database import NASBench201DataBase


class NASBench201Dataset(Dataset):
    def __init__(self, database: NASBench201DataBase, seed):
        self.database = database
        g_cpu = torch.Generator()
        g_cpu.manual_seed(seed)

        self.index_list = torch.randperm(self.database.size,
                                         generator=g_cpu).tolist()
        self.keys_list = list(self.database.index_iterator())

    def __getitem__(self, index):
        arch_id = self.keys_list[self.index_list[index]]
        arch = self.database.query_by_id(arch_id)
        arch_feature, edges_type_counts = feature_tensor_encoding_201(copy.deepcopy(arch))

        network_data = {}
        for net_type in ['cifar10-valid', 'cifar100', 'ImageNet16-120']:
            params = arch['{}_total_params'.format(net_type)]
            flops = arch['{}_total_flops'.format(net_type)]
            # n_edges = arch['{}']
            
            if net_type == 'cifar10-valid':
                val_acc = arch['cifar10_val_acc']
                test_acc = arch['cifar10_test_acc']
                rank = arch['cifar10_rank']

                network_data['cifar10'] = (arch_feature[net_type], val_acc, test_acc, params, flops, edges_type_counts, rank)
            elif net_type == 'ImageNet16-120':
                val_acc = arch['imagenet16_val_acc']
                test_acc = arch['imagenet16_test_acc']
                rank = arch['imagenet16_rank']
                
                network_data['imagenet16'] = (arch_feature[net_type], val_acc, test_acc, params, flops, edges_type_counts, rank)
            else:
                val_acc = arch['cifar100_val_acc']
                test_acc = arch['cifar100_test_acc']
                rank = arch['cifar100_rank']

                network_data['cifar100'] = (arch_feature[net_type], val_acc, test_acc, params, flops, edges_type_counts, rank)

        return network_data

    def query_arch_by_str(self, arch_str:str, network_type):
        assert network_type in [
            'cifar10-valid','cifar100','ImageNet16-120'
            ], 'The network_type arg should choose from cifar10-valid,cifar100,ImageNet16-120'
        
        arch_key = self.database.check_arch_inside_dataset(arch_str)
        if arch_key is None:
            return None, None, None

        keys_list_idx = self.keys_list.index(arch_key)
        index_list_idx = self.index_list.index(keys_list_idx)

        flops = self.database.archs[arch_key]['{}_total_flops'.format(network_type)]
        params = self.database.archs[arch_key]['{}_total_params'.format(network_type)]

        return flops, params, index_list_idx

    def __len__(self):
        return self.database.size
                

            

            
