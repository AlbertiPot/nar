"""
Data: 2021/09/18
Target: Sample subnets under FLOPS and Parameters contraints
Method: 1 sample target FLOPS and Parameters constraints according to top tier distribution
        2 sample n_nodes according to top tier distribution
        3 Given the sampled n_nodes, sample edges and nodes type according to uniform distribution
        4 Check whether the sampled subnets satisfy the FLOPS and Parameters constraints

input: 1 batch_statics_dict = 
        {
            'flops':[{tier1},{tier2},{tier3},{tier4},{tier5}]
            'params':[{tier1},{tier2},{tier3},{tier4},{tier5}]
            'n_nodes':[{tier1},{tier2},{tier3},{tier4},{tier5}]
            }
"""
import random

from architecture import ModelSpec, seq_decode_to_arch
from dataset import NASBenchDataset

from .prob_calculate import select_distri, extract_dict_value_to_list

def sample_helper(disti_dict):
    target_list = extract_dict_value_to_list(disti_dict, is_key= True)
    prob_list = extract_dict_value_to_list(disti_dict)
    return target_list, prob_list

class ArchSampler(object):
    def __init__(self, top_tier, last_tier, batch_factor, node_type_dict, max_edges = 9, reuse_step=None):
        self.top_tier = top_tier
        self.last_tier = last_tier
        self.batch_factor = batch_factor
        self.reuse_step = reuse_step
        self.node_type_dict = node_type_dict
        self.max_edges = max_edges

    def reset_parameters(self, batch_statics_dict):
        self.batch_flops_list = batch_statics_dict['flops']
        self.batch_params_list = batch_statics_dict['params']
        self.batch_n_nodes_list = batch_statics_dict['n_nodes']

    def _sample_target_value(self, candi_list, threshold_kl_div=2, force_uniform=False):
        target_distri = None
        if not force_uniform:
            batch_size = 0
            for dic in candi_list:
                batch_size += sum(extract_dict_value_to_list(dic))
            target_distri = select_distri(candi_list, self.top_tier, self.last_tier, threshold_kl_div, batch_size, self.batch_factor)

        if target_distri is not None and not force_uniform:
            target_list, prob_list = sample_helper(target_distri)
            target_value = random.choices(target_list, weights=prob_list)[0]
            return target_value
        else:
            target_list = extract_dict_value_to_list(candi_list[0], is_key=True)
            target_value = random.choice(target_list)
            return target_value

    def _sample_edges_and_types(self, n_nodes, arch_struct_list:list):
        type_candi_list = extract_dict_value_to_list(self.node_type_dict)[1:-1]

        for node_idx in range(1, n_nodes):
            # 1 sample previous node
            pre_node_id = random.choice([i for i in range(node_idx)])
            arch_struct_list.append(pre_node_id)

            # 2 sample node type
            if node_idx == n_nodes-1:
                break
            node_opt_type = random.choice(type_candi_list)
            arch_struct_list.append(node_opt_type)
            
        # 3 sampel rest edges
        # max edges = 9, the rest edges are (max_edges -(n_nodes-1))
        for i in range(self.max_edges-n_nodes+1):
            # sample begin node for the edge
            begin_node_idx = random.choice([b_idx for b_idx in range(n_nodes)])
            if begin_node_idx == n_nodes-1:   # if begin node is the output node, this edge does not exist
                arch_struct_list += [None, None]
                continue
            
            # sample end node for the edge
            end_node_idx = random.choice([l_idx for l_idx in range(begin_node_idx, n_nodes)])
            if end_node_idx == begin_node_idx:
                arch_struct_list += [None, None]
                continue

            arch_struct_list += [begin_node_idx, end_node_idx]

        # 1st for loop: (n_nodes-1)times each add 2(edge and opt type) excep the last output node, yield (n_nodes-1)*2-1
        # 2nd for loop: (max_edges-n_nodes+1)times each add 2(start and end nodes), yield (max_edges-n_nodes+1)*2
        # n_nodes stored in the first place of the list, yield 1
        # total (n_nodes-1)*2-1+(max_edges-n_nodes+1)*2+1
        assert len(arch_struct_list) == self.max_edges * 2, 'Wrong length of sampled arch_struct_list'
        return arch_struct_list

    def sample_arch(self, batch_statics_dict, n_subnets, dataset: NASBenchDataset, kl_thred=[2,2], max_trails=100, force_uniform=False):
        self.reset_parameters(batch_statics_dict)
        flops_kl_thred = kl_thred[0]
        params_kl_thred = kl_thred[1]

        sampled_arch = []
        sampled_arch_datast_idx = []
        flops, params, n_nodes = 0,0,0

        self.reuse_step = 1 if self.reuse_step is None else self.reuse_step
        assert self.reuse_step > 0, 'the reuse step must greater than 0'
        
        reuse_count = 0
        while len(sampled_arch) < n_subnets:
            # step1 sample flops and params constraints
            if reuse_count % self.reuse_step == 0:
                # sample target flops
                flops = self._sample_target_value(self.batch_flops_list, flops_kl_thred, force_uniform=force_uniform)

                # sample target params
                params = self._sample_target_value(self.batch_params_list, params_kl_thred, force_uniform=force_uniform)

            for trail in range(max_trails+1):
                arch_struct_list = []   # store arch
                
                # step2 sample target n_nodes
                n_nodes = self._sample_target_value(self.batch_n_nodes_list, force_uniform=True)
                arch_struct_list.append(n_nodes)
                
                # step3 sample nodes type and connectoin
                arch_struct_list = self._sample_edges_and_types(n_nodes, arch_struct_list)

                # step4 check wheth satisfy the flops and params constraints
                matrix, opt = seq_decode_to_arch(arch_struct_list)
                arch_spec = ModelSpec(matrix=matrix, ops=opt)

                f, p, dataset_idx = dataset.query_stats_by_spec(arch_spec)
                if dataset_idx is None:
                    continue
                if f <= flops and p <= params:
                    break
            
            sampled_arch_datast_idx.append(dataset_idx)
            if dataset_idx is None:
                sampled_arch.append(None)
            else:
                sampled_arch.append(arch_spec)
            
            reuse_count +=1

        return sampled_arch, sampled_arch_datast_idx

    
    