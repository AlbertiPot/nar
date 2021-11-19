"""
Date: 2021/09/07
Target: The implementation of the feature tensor encoding method proposed in ReNAS
Input:  arch:dict from cell information dataset of NAS-Bench-101
Output: torch.Tensor(C,H,W) default= 19*7*7 for nasbench101
"""

import torch
from torch.nn import functional as F

type_dict = {
    'input': 1,
    'conv1x1-bn-relu': 2,
    'conv3x3-bn-relu': 3,
    'maxpool3x3': 4,
    'output': 5
}


def adjacency_matrix_padding(matrix, arch_feature_dim, num_vertices):

    assert isinstance(
        matrix, torch.Tensor), 'adjacency matrix type should be torch.Tensor'

    pad_matrix = matrix

    for _ in range(arch_feature_dim - num_vertices):

        pd = (0, 1, 0, 1)
        pad_matrix = F.pad(pad_matrix, pd, 'constant', 0)

        index = [i for i in range(len(pad_matrix))]
        index[-2], index[-1] = index[-1], index[-2]

        pad_matrix = pad_matrix[index]
        pad_matrix = pad_matrix.transpose(0, 1)
        pad_matrix = pad_matrix[index]
        pad_matrix = pad_matrix.transpose(0, 1)

    return pad_matrix


def vector_padding(vector, arch_feature_dim, num_vertices):

    assert isinstance(vector,
                      torch.Tensor), 'vector type should be torch.Tensor'

    pad_vector = vector

    for _ in range(arch_feature_dim - num_vertices):

        pd = (0, 1)
        pad_vector = F.pad(pad_vector, pd, 'constant', 0)

        index = [i for i in range(len(pad_vector))]
        index[-2], index[-1] = index[-1], index[-2]
        pad_vector = pad_vector[index]

    return pad_vector


def feature_tensor_encoding(arch: dict,
                            arch_feature_dim=7,
                            arch_feature_channels=19):

    matrix = arch['module_adjacency']
    ops = arch['module_operations']
    vertex_flops_dict_ = arch['vertex_flops']
    vertex_params_dict_ = arch['vertex_params']

    num_vertices = len(matrix)

    ops_vector = torch.tensor([type_dict[v] for v in ops])
    matrix = torch.tensor(matrix)

    if num_vertices < arch_feature_dim:
        matrix = adjacency_matrix_padding(matrix, arch_feature_dim,
                                          num_vertices)
        ops_vector = vector_padding(ops_vector, arch_feature_dim, num_vertices)

    ops_vector_matrix = torch.mul(matrix, ops_vector)

    arch_feature = ops_vector_matrix.unsqueeze(dim=0)

    for _, (fk, pk) in enumerate(zip(vertex_flops_dict_, vertex_params_dict_)):

        cell_flops = torch.tensor(vertex_flops_dict_[fk])
        cell_params = torch.tensor(vertex_params_dict_[pk])

        cell_flops = torch.true_divide(cell_flops, 1e7)
        cell_params = torch.true_divide(cell_params, 1e5)

        if num_vertices < arch_feature_dim:
            cell_flops = vector_padding(cell_flops, arch_feature_dim,
                                        num_vertices)
            cell_params = vector_padding(cell_params, arch_feature_dim,
                                         num_vertices)

        cell_flops_matrix = torch.mul(matrix, cell_flops).unsqueeze(dim=0)
        cell_params_matrix = torch.mul(matrix, cell_params).unsqueeze(dim=0)

        arch_feature = torch.cat(
            [arch_feature, cell_flops_matrix, cell_params_matrix], dim=0)

    assert len(
        arch_feature
    ) == arch_feature_channels, 'Wrong channels of arch feature tensor'
    return arch_feature


if __name__ == "__main__":

    # for test
    import json
    data_path = '../data/nasbench101_vertice_example.json'
    with open(data_path, 'r') as f:
        dataset = json.load(f)
    f.close()
    assert isinstance(dataset, list)
    arch = dataset[0]

    encoded_arch = feature_tensor_encoding(arch)
    print(encoded_arch)
    print(encoded_arch.shape)