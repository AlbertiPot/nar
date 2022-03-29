import numpy as np

DUMMY = 'dummy'
INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'

NODE_TYPE = [DUMMY, INPUT, CONV1X1, CONV3X3, MAXPOOL3X3, OUTPUT]


def seq_decode_to_arch(seq):
    n_nodes = seq[0]

    opt = [INPUT]
    matrix = np.zeros(shape=(n_nodes, n_nodes), dtype=np.int32)

    output_node = seq[2 * (n_nodes - 2) + 1]
    inter_nodes = seq[1:2 * (n_nodes - 2) + 1]
    inter_nodes_edges = inter_nodes[::2]
    inter_nodes_type = inter_nodes[1::2]
    remain_edges = seq[2 * (n_nodes - 2) + 2:]

    for end_node, (begin_node,
                   opt_type) in enumerate(zip(inter_nodes_edges,
                                              inter_nodes_type),
                                          start=1):
        opt.append(NODE_TYPE[opt_type])
        matrix[begin_node][end_node] = 1

    opt.append(OUTPUT)
    matrix[output_node][n_nodes - 1] = 1

    # remain edges
    re_start_ndoes = remain_edges[::2]
    re_end_nodes = remain_edges[1::2]
    for (begin_node, end_node) in zip(re_start_ndoes, re_end_nodes):
        if begin_node == None or end_node == None:
            continue
        matrix[begin_node][end_node] = 1

    return matrix, opt


def edges_to_str(total_edges_list: list):

    str_list = []
    for node_edges in total_edges_list:
        node_str_list = list(
            map(lambda item: item[0] + '~' + str(item[1]),
                [item for item in node_edges]))
        str_list.append('|' + '|'.join(node_str_list) + '|')

    arch_str = '+'.join(str_list)

    return arch_str