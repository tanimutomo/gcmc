import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter_

def split_stack(features, index, relations, dim_size):
    """
    Official Stack accumulation function

    Parameters
    ----------
    features : tensor (relation * num_nodes) x features
        output of messge method in RGCLayer class
    index : tensor (edges)
        edge_index[0]
    relations : teonsor(edges)
        edge_type
    dim_size : tensor(num_nodes)
        input size (the number of nodes)

    Return
    ------
    stacked_out : tensor(relation * nodes x out_dim)
    """
    out_dim = features.shape[0]
    np_index = index.numpy()
    np_relations = relations.numpy()
    splited_features = torch.split(features, int(out_dim / 5), dim=1)

    stacked_out = []
    for r, feature in enumerate(splited_features):
        relation_only_r = torch.from_numpy(np.where(np_relations==r)[0])
        r_index = index[relation_only_r]
        r_feature = feature[relation_only_r]
        stacked_out.append(scatter_('add', r_feature, r_index, dim_size=dim_size))

    stacked_out = torch.cat(stacked_out, 1)

    return stacked_out


def stack(features, index, relations, dim_size):
    """
    Stack accumulation function in RGCLayer.

    Parameters
    ----------
    features : tensor (relation * num_nodes)
        output of messge method in RGCLayer class
    index : tensor (edges)
        edge_index[0]
    relations : teonsor(edges)
        edge_type
    dim_size : tensor(num_nodes)
        input size (the number of nodes)

    Return
    ------
    out : tensor(relation * nodes x out_dim)
    """
    out = torch.zeros(dim_size * (torch.max(relations) + 1), features.shape[1])
    tar_idx = relations * dim_size + index
    out[tar_idx] = features
    # for feature, idx, relation in zip(features, index, relations):
    #     tar_idx = relation * dim_size + index
    #     out[tar_idx] = feature
    return out
    

def uniform(size, tensor):
    stdv = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)


def random_init(tensor, in_dim, out_dim):
    thresh = math.sqrt(6.0 / (in_dim + out_dim))
    if tensor is not None:
        try:
            tensor.data.uniform_(-thresh, thresh)
        except:
            nn.init.uniform_(tensor, a=-thresh, b=thresh)


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        # m.bias.data.fill_(0.01)
        try:
            truncated_normal(m.bias)
        except:
            pass

def truncated_normal(tensor, mean=0, std=1):
    tensor.data.fill_(std * 2)
    with torch.no_grad():
        while(True):
            if tensor.max() >= std * 2:
                tensor[tensor>=std * 2] = tensor[tensor>=std * 2].normal_(mean, std)
                tensor.abs_()
            else:
                break

def calc_rmse(pred, gt):
    pred = F.softmax(pred, dim=1)
    expected_pred = torch.zeros(gt.shape)
    for relation in range(pred.shape[1]):
        expected_pred += pred[:, relation] * (relation + 1)

    rmse = (gt.to(torch.float) + 1) - expected_pred
    rmse = torch.pow(rmse, 2)
    rmse = torch.pow(torch.sum(rmse) / gt.shape[0], 0.5)

    return rmse


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
