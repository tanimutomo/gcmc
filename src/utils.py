import math
import torch
import torch.nn.functional as F

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
    # print(rmse)
    rmse = torch.pow(rmse, 2)
    # print(rmse)
    rmse = torch.pow(torch.sum(rmse) / gt.shape[0], 0.5)
    # print(rmse)

    return rmse


        
