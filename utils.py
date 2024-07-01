import torch
import torch.nn.functional as F
# from model import GraphSage
import numpy as np
def sampling(src_nodes, sample_num, adj_list):
    
    result = []
    for s in src_nodes:
        if len(adj_list[s]) >= sample_num:
            res = np.random.choice(list(adj_list[s]), size = (sample_num,), replace=False)
        else:
            res = np.random.choice(list(adj_list[s]), size = (sample_num,), replace=True)
        result.append(res)
    return np.asarray(result).flatten()

def mul_sampling(src_nodes, sample_nums, adj_list):
    result = [src_nodes]
    for k, hop_num in enumerate(sample_nums):
        res = sampling(result[k], hop_num, adj_list)
        result.append(res)
    return result

# sampling test
# from datasets import *
# ds = DataCenter({})
# ds.load_dataSet()
# adj_list = ds.adj_lists
# print(adj_list[2702])
# a = sampling([2702], 1, adj_list)
# print(a)
# b = mul_sampling([2702], [2, 3], adj_list)
# print(b)



