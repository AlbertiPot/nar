"""
Data: 2021/10/11
Target: Sample subnets under FLOPS and Parameters contraints for 【Nasbench201】

input: batch_statics_dict = 
        {
            'flops':[{tier1},{tier2},{tier3},{tier4},{tier5}]
            'params':[{tier1},{tier2},{tier3},{tier4},{tier5}]
            'edges':[{tier1},{tier2},{tier3},{tier4},{tier5}]
            } tier是counts_dict
"""
import random

from architecture import edges_to_str
from dataset import NASBench201Dataset

from .prob_calculate import extract_dict_value_to_list, select_distri

N_NODES = 4
NODE_TYPE_DICT = {
    0: "none",
    1: "skip_connect",
    2: "nor_conv_1x1",
    3: "nor_conv_3x3",
    4: "avg_pool_3x3"
}


def sample_helper(disti_dict):
    target_list = extract_dict_value_to_list(disti_dict, is_key=True)
    prob_list = extract_dict_value_to_list(disti_dict)
    return target_list, prob_list


class ArchSampler201(object):
    def __init__(self, top_tier, last_tier, batch_factor, reuse_step=None):

        self.top_tier = top_tier
        self.last_tier = last_tier
        self.batch_factor = batch_factor
        self.reuse_step = reuse_step

    def reset_parameters(self, batch_statics_dict):
        self.batch_flops_list = batch_statics_dict['flops']
        self.batch_params_list = batch_statics_dict['params']
        self.edges_list = batch_statics_dict['edges']

    def _sample_target_value(self,
                             candi_list,
                             threshold_kl_div=2,
                             force_uniform=False):

        target_distri = None
        if not force_uniform:
            group_size = 0
            for dic in candi_list:
                group_size += sum(extract_dict_value_to_list(dic))
            target_distri = select_distri(candi_list, self.top_tier,
                                          self.last_tier, threshold_kl_div,
                                          group_size, self.batch_factor)

        if target_distri is not None and not force_uniform:
            target_list, prob_list = sample_helper(target_distri)
            target_value = random.choices(target_list, weights=prob_list)[0]
            return target_value
        else:
            target_list = extract_dict_value_to_list(candi_list[0],
                                                     is_key=True)
            target_value = random.choice(target_list)
            return target_value

    def _sample_edges_opts(self, edges_kl_thred, force_uniform):
        edges_for_all_nodes = []
        for end_node in range(1, N_NODES):
            edges_for_one_node = []
            for start_node in range(end_node):
                opt_id = self._sample_target_value(self.edges_list,
                                                   edges_kl_thred,
                                                   force_uniform)
                edges_for_one_node.append((NODE_TYPE_DICT[opt_id], start_node))

            edges_for_all_nodes.append(
                tuple((op, int(start_node))
                      for op, start_node in edges_for_one_node))

        return edges_for_all_nodes

    def sample_arch(self,
                    batch_statics_dict,
                    n_subnets,
                    dataset: NASBench201Dataset,
                    network_type,
                    kl_thred=[2, 2, 2],
                    max_trails=100,
                    force_uniform=False):

        self.reset_parameters(batch_statics_dict)
        flops_kl_thred = kl_thred[0]
        params_kl_thred = kl_thred[1]
        edges_kl_thred = kl_thred[2]

        sampled_arch_datast_idx = []
        flops, params = 0, 0

        self.reuse_step = 1 if self.reuse_step is None else self.reuse_step
        assert self.reuse_step > 0, 'the reuse step must greater than 0'

        while len(sampled_arch_datast_idx) < n_subnets:
            # step1: sample flops and params constraints
            if len(sampled_arch_datast_idx) % self.reuse_step == 0:
                flops = self._sample_target_value(self.batch_flops_list,
                                                  flops_kl_thred,
                                                  force_uniform=force_uniform)

                params = self._sample_target_value(self.batch_params_list,
                                                   params_kl_thred,
                                                   force_uniform=force_uniform)

            for trail in range(max_trails + 1):
                # step2: sample each edges operation
                edges_for_all_nodes = self._sample_edges_opts(
                    edges_kl_thred, force_uniform=force_uniform)

                arch_str = edges_to_str(edges_for_all_nodes)

                # step3: query from the dataset and check whether satisfy the constraints
                f, p, index_list_idx = dataset.query_arch_by_str(
                    arch_str, network_type)

                if index_list_idx is None:
                    continue
                if f <= flops and p <= params:
                    break

            sampled_arch_datast_idx.append(index_list_idx)

        return sampled_arch_datast_idx
