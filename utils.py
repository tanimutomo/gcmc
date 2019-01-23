import math
import torch

def uniform(size, tensor):
    stdv = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)

def random_init(ster, tensor):
    if tensor is not None:
        tensor.data.uniform_(-ster, ster)

def calc_rmse(pred, gt):
    expected_pred = torch.zeros(gt.shape)
    for relation in range(pred.shape[1]):
        expected_pred += pred[:, relation] * (relation + 1)

    rmse = torch.sum(((gt.to(torch.float) + 1) - expected_pred) ** 2)
    rmse = torch.pow(rmse, 0.5) / gt.shape[0]

    return rmse


        
