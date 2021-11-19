"""
Data: 2021/10/09
Modified from https://github.com/AlbertiPot/NAS-Bench-201.git
MIT License: https://github.com/D-X-Y/NAS-Bench-201/blob/master/LICENSE.md
"""

from typing import List, Text

def str2lists(arch_str: Text) -> List[tuple]:
    """
    This function shows how to read the string-based architecture encoding.
      It is the same as the `str2structure` func in `AutoDL-Projects/lib/models/cell_searchs/genotypes.py`

    :param
      arch_str: the input is a string indicates the architecture topology, such as
                    |nor_conv_1x1~0|+|none~0|none~1|+|none~0|none~1|skip_connect~2|
    :return: a list of tuple, contains multiple (op, input_node_index) pairs.

    :usage
      arch = api.str2lists( '|nor_conv_1x1~0|+|none~0|none~1|+|none~0|none~1|skip_connect~2|' )
      print ('there are {:} nodes in this arch'.format(len(arch)+1)) # arch is a list
      for i, node in enumerate(arch):
        print('the {:}-th node is the sum of these {:} nodes with op: {:}'.format(i+1, len(node), node))
    """
    node_strs = arch_str.split('+')
    genotypes = []
    for i, node_str in enumerate(node_strs):
      inputs = list(filter(lambda x: x != '', node_str.split('|')))
      for xinput in inputs: assert len(xinput.split('~')) == 2, 'invalid input length : {:}'.format(xinput)
      inputs = ( xi.split('~') for xi in inputs )
      input_infos = tuple( (op, int(IDX)) for (op, IDX) in inputs)
      genotypes.append( input_infos )
    return genotypes