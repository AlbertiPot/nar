"""
Date: 2021/10/09
Target: encode NasBench201 arch into 31*4*4 feature tensor for each set in {'cifar10-valid', 'cifar100', 'ImageNet16-120'}
Input:  arch:dict from cell information dataset of NAS-Bench-201
Output: all_type_tensors_list: {'cifar10-valid': torch.Tensor(31,4,4), 'cifar100': torch.Tensor(31,4,4), 'ImageNet16-120': torch.Tensor(31,4,4)}
        edges_type_counts: torch.Tensor(["none" counts, "skip_connect" counts, "nor_conv_1x1" counts, "nor_conv_3x3" counts, "avg_pool_3x3 counts"])
"""

import torch
import copy

from .nasbench201 import str2lists

NODE_TYPE_DICT = {
    "none": 0,
    "skip_connect": 1,
    "nor_conv_1x1": 2,
    "nor_conv_3x3": 3,
    "avg_pool_3x3": 4
}


def feature_tensor_encoding_201(arch: dict,
                                arch_feature_dim=4,
                                arch_feature_channels=31):

    matrix = arch['cell_adjacency']
    assert len(matrix) == arch_feature_dim, 'Wrong length of adjacency matrix'
    matrix = torch.tensor(matrix)

    arch_str = arch['arch_str']
    arch_opt_list = str2lists(arch_str)
    coordi_list = []
    edges_type_counts = [0] * len(NODE_TYPE_DICT.values())
    for col_id, node_ops in enumerate(arch_opt_list, start=1):
        for op in node_ops:
            coordi_list.append([op[1], col_id])
            edges_type_counts[NODE_TYPE_DICT[op[0]]] += 1
    # [start_node, end_node] [[0, 1], [0, 2], [1, 2], [0, 3], [1, 3], [2, 3]]

    all_type_tensors_list = {}

    for net_type in ['cifar10-valid', 'cifar100', 'ImageNet16-120']:
        opt_flops = arch['{}_opt_flops'.format(net_type)]
        opt_params = arch['{}_opt_params'.format(net_type)]
        feature_tensor_list = []
        feature_tensor_list.append(copy.deepcopy(matrix).unsqueeze(dim=0))
        for cell_id, (flops, params) in enumerate(
                zip(opt_flops.values(), opt_params.values())):
            f_patch = torch.zeros(arch_feature_dim, arch_feature_dim)
            p_patch = torch.zeros(arch_feature_dim, arch_feature_dim)
            for edge_id, (coord, edge_flops, edge_params) in enumerate(
                    zip(coordi_list, flops, params)):
                f_patch[coord[0]][coord[1]] = edge_flops
                p_patch[coord[0]][coord[1]] = edge_params

            feature_tensor_list.append(f_patch.unsqueeze(dim=0))
            feature_tensor_list.append(p_patch.unsqueeze(dim=0))

        arch_feature_tensor = torch.cat(feature_tensor_list, dim=0)
        assert arch_feature_tensor.size(
            0) == arch_feature_channels, 'Wrong arch feature_channels'

        all_type_tensors_list[net_type] = arch_feature_tensor

    return all_type_tensors_list, torch.tensor(edges_type_counts)


if __name__ == '__main__':

    # for test
    import json
    data_path = '../data/nasbench201_vertice_example.json'
    with open(data_path, 'r') as f:
        dataset = json.load(f)
    f.close()
    assert isinstance(dataset, dict)
    arch = dataset['0']

    encoded_archs, edges_type_counts = feature_tensor_encoding_201(arch)
    print(encoded_archs['cifar100'])
    print(edges_type_counts)