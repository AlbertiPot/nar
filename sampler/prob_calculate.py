""""
Data: 2021/09/16
Target: Calculate the probability and distribution of the FLOPS, Parametes, opts
Method: Distribution and probability are modified from AttentiveNAS
"""
import torch
import math
import copy
import torch.nn.functional as F

def compute_kl_div(input_seq, target_seq): 
    assert len(input_seq)==len(target_seq)
    
    input_seq = torch.tensor(input_seq, dtype = torch.float) if not isinstance(input_seq, torch.Tensor) else input_seq
    target_seq = torch.tensor(target_seq, dtype = torch.float) if not isinstance(target_seq, torch.Tensor) else target_seq
    input_seq = F.log_softmax(input_seq, dim=-1)
    target_seq = F.log_softmax(target_seq, dim=-1)
    
    kl_value = F.kl_div(input_seq, target_seq, reduction='batchmean', log_target=True)
    return kl_value


def compute_constraint_value(value, step, batch_min):
    return math.ceil((value-batch_min)/step)*step+batch_min

def convert_count_to_prob(counts_dict):

    total = sum(counts_dict.values())
    for idx in counts_dict:
        counts_dict[idx] = 1.0 * counts_dict[idx] / total
    
    return counts_dict

def build_counts_dict(raw_list, batch_min, batch_max, bins=8, scail = 1):
    batch_min = int(batch_min)
    batch_max = int(batch_max)
    raw_list.sort()
    step = math.ceil((batch_max-batch_min)/bins)
    
    counts_dict = {}
    for i in range(1, bins+1):
        counts_dict[int(scail*(i * step + batch_min))] = 0
    
    for value in raw_list:
        if value == batch_min:
            value+=step
        value = compute_constraint_value(value, step, batch_min)
        counts_dict[int(scail*value)] += 1

    return counts_dict

def build_n_nodes_counts_dict(raw_list, batch_min, batch_max):
    raw_list.sort()
    counts_dict = {}
    for idx in range(batch_min, batch_max+1):
        counts_dict[idx] = 0
    for item in raw_list:
        counts_dict[item] += 1

    return counts_dict

def build_edges_counts_dict(raw_list):
    l = raw_list.size(1)
    counts = raw_list.sum(dim=0)
    
    counts_dict = {}
    for i in range(l):
        counts_dict[i] = counts[i].item()
    
    return counts_dict

def extract_dict_value_to_list(target_dict, is_key=False):
    assert type(target_dict) == dict, 'Not a dictionary'
    if is_key:
        return list(target_dict.keys())
    return list(target_dict.values())

def select_distri(candi_list, top_tier, last_tier, threshold_kl_div, batch_size, batch_factor):
    assert top_tier < len(candi_list), 'The candidates tier indexs should be smaller than the length of candidates list'
    for i in range(top_tier):
        candi = candi_list[i]
        if sum(candi.values()) < batch_factor * batch_size:
            continue
        
        candi_counts = extract_dict_value_to_list(candi)
        assert last_tier > top_tier, 'last tier index should be larger than top tier'
        for j in range(last_tier-1, len(candi_list)):
            low_tier_counts = extract_dict_value_to_list(candi_list[j])
            if compute_kl_div(low_tier_counts, candi_counts) < threshold_kl_div:
                return None

        return convert_count_to_prob(copy.deepcopy(candi))
