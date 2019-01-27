import time
import math
import torch
import torch.nn.functional as F

def stack(features, index, relations, dim_size):
    out = torch.zeros(dim_size * (torch.max(relations) + 1), features.shape[1])
    start = time.time()
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


def random_init(ster, tensor):
    if tensor is not None:
        tensor.data.uniform_(-ster, ster)


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
        
