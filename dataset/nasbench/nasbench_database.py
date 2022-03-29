"""
Date: 2021/09/21
Target: Load entire dataset from json file 
        query by spec and hash 
        check whether the passed spec is inside the dataset

Note: Do Not use query by hash when benchmarking

Modified from https://github.com/google-research/nasbench
Apache License 2.0: https://github.com/google-research/nasbench/blob/master/LICENSE
"""

import json
import time
import copy
import numpy as np

from architecture import ModelSpec

MODULE_VERTICES = 7
MAX_EDGES = 9
INPUT = 'input'
OUTPUT = 'output'
VALID_OPERATIONS = ['conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3']


class NASBenchDataBase(object):
    def __init__(self, data_file):
        self.archs = {}

        self._load_json_file(data_file)

    def _load_json_file(self, data_file):
        print('Loading nasbench101 dataset from file')
        start = time.time()

        with open(data_file, 'r') as f:
            dataset = json.load(f)
        f.close()

        for arch in dataset:
            self.archs[arch['unique_hash']] = arch
        
        self._sort()

        elapsed = time.time() - start
        print('Loaded dataset in {:.4f} seconds'.format(elapsed))

    def _sort(self):
        sorted_list = []
        for hash_key, arch in self.archs.items():
            sorted_list.append((hash_key, arch['avg_test_accuracy']))
        
        sorted_list = sorted(sorted_list, key=lambda item:item[1], reverse=True)
        
        for rank, (hash, _) in enumerate(sorted_list, start=1):
            self.archs[hash]['rank'] = rank

    def _check_spec(self, model_spec: ModelSpec):
        """Checks that the model spec is within the dataset."""
        if not model_spec.valid_spec:
            raise ValueError('invalid spec, provided graph is disconnected.')

        num_vertices = len(model_spec.ops)
        num_edges = np.sum(model_spec.matrix)

        if num_vertices > MODULE_VERTICES:
            raise ValueError('too many vertices.')
        if num_edges > MAX_EDGES:
            raise ValueError('too many edges.')
        if model_spec.ops[0] != INPUT:
            raise ValueError('first operation should be \'input\'')
        if model_spec.ops[-1] != OUTPUT:
            raise ValueError('last operation should be \'output\'')
        for op in model_spec.ops[1:-1]:
            if op not in VALID_OPERATIONS:
                raise ValueError('unsupported op.')

    def _hash_spec(self, model_spec: ModelSpec):
        """Returns the MD5 hash for a provided model_spec."""
        return model_spec.hash_spec(VALID_OPERATIONS)

    def query_by_hash(self, arch_hash):
        arch_data_dict = copy.deepcopy(self.archs[arch_hash])
        return arch_data_dict

    def query_by_spec(self, model_spec: ModelSpec):
        self._check_spec(model_spec)
        arch_hash = self._hash_spec(model_spec)

        return self.query_by_hash(arch_hash)

    def check_arch_inside_dataset(self, model_spec: ModelSpec):
        try:
            self._check_spec(model_spec)
        except ValueError:
            return None
        
        arch_hash = self._hash_spec(model_spec)
        if arch_hash not in list(self.archs.keys()):
            return None
        return self.query_by_hash(arch_hash)

    def hash_iterator(self):
        return self.archs.keys()

    @property
    def size(self):
        return len(self.archs)
