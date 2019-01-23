import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing

from utils import uniform, random_init

class GAE(nn.Module):
    def __init__(self, in_c, hid_c, out_c, num_basis, num_relations, num_user, drop_prob):
        super(GAE, self).__init__()
        self.gcenc = GCEncoder(in_c, hid_c, out_c, num_relations, num_user, drop_prob)
        self.bidec = BiDecoder(out_c, num_basis, num_relations)

    def forward(self, x, edge_index, edge_type, edge_norm):
        u_features, i_features = self.gcenc(x, edge_index, edge_type, edge_norm)
        adj_matrices = self.bidec(u_features, i_features)

        return adj_matrices


class GCEncoder(nn.Module):
    def __init__(self, in_c, hid_c, out_c, num_relations, num_user, drop_prob):
        super(GCEncoder, self).__init__()
        self.num_user = num_user
        self.rgc_layer = RGCLayer(in_c, hid_c, num_relations, drop_prob)
        self.dense_layer = DenseLayer(hid_c, out_c, drop_prob)

    def forward(self, x, edge_index, edge_type, edge_norm):
        features = self.rgc_layer(x, edge_index, edge_type, edge_norm)
        u_features, i_features = self.separate_features(features)
        u_features, i_features = self.dense_layer(u_features, i_features)

        return u_features, i_features

    def separate_features(self, features):
        u_features = features[:self.num_user]
        i_features = features[self.num_user:]

        return u_features, i_features



class RGCLayer(MessagePassing):
    def __init__(self, in_c, out_c, num_relations, drop_prob):
        super(RGCLayer, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_relations = num_relations
        self.drop_prob = drop_prob
        
        ord_basis = [nn.Parameter(torch.Tensor(1, in_c * out_c)) for r in range(num_relations)]
        self.ord_basis = nn.ParameterList(ord_basis)
        self.relu = nn.ReLU()

        self.reset_parameters()
        # self.ord_basis = nn.Parameter(torch.Tensor(num_relations, 1, in_c * out_c))


    def reset_parameters(self):
        # self.ord_basis = nn.ParameterList()
        # size = self.in_c * self.out_c
        # for basis in self.ord_basis:
        #     basis = uniform(size, basis)
        #     self.ord_basis.append(basis)
        for basis in self.ord_basis:
            basis = random_init(1e-3, basis)

    def forward(self, x, edge_index, edge_type, edge_norm=None):
        return self.propagate('add', edge_index, x=x, edge_type=edge_type, edge_norm=edge_norm)

    def message(self, x_j, edge_type, edge_norm):
        # create weight using ordinal weight sharing
        for relation in range(self.num_relations):
            if relation == 0:
                weight = self.ord_basis[relation]
            else:
                weight = torch.cat((weight, weight[-1] 
                    + self.ord_basis[relation]), 0)

        weight = weight.reshape(-1, self.out_c)
        weight = self.node_dropout(weight)
        index = edge_type * self.in_c + x_j
        out = weight[index]
        out = self.relu(out)

        return out if edge_norm is None else out * edge_norm.reshape(-1, 1)

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channles]
        return aggr_out

    def node_dropout(self, weight):
        drop_mask = torch.rand(self.in_c) + (1 - self.drop_prob)
        # drop_mask = torch.floor(drop_mask).type(torch.uint8)
        drop_mask = torch.floor(drop_mask).type(torch.float)
        drop_mask = torch.cat([drop_mask 
            for r in range(self.num_relations)], 0).unsqueeze(1)
        drop_mask = drop_mask.expand(drop_mask.size(0), self.out_c)

        # weight = torch.where(drop_mask, weight, 
        #         torch.tensor(0, dtype=weight.dtype))
        assert weight.shape == drop_mask.shape
        weight = weight * drop_mask

        return weight



class DenseLayer(nn.Module):
    def __init__(self, in_c, out_c, drop_prob, bias=False):
        super(DenseLayer, self).__init__()

        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(in_c, out_c, bias=bias)

    def forward(self, u_features, i_features):
        u_features = self.dropout(u_features)
        u_features = self.fc(u_features)

        i_features = self.dropout(i_features)
        i_features = self.fc(i_features)

        return u_features, i_features


class BiDecoder(nn.Module):
    def __init__(self, feature_dim, num_basis, num_relations):
        super(BiDecoder, self).__init__()
        self.num_basis = num_basis
        self.num_relations = num_relations
        self.feature_dim = feature_dim
        self.basis_matrix = nn.Parameter(
                torch.Tensor(num_basis, feature_dim * feature_dim))
        # self.coefs = nn.Parameter(torch.Tensor(num_relations, num_basis))
        # basis_matrix = [nn.Parameter(torch.Tensor(
        #     feature_dim * feature_dim)) for b in range(num_basis)]
        coefs = [nn.Parameter(torch.Tensor(num_basis)
            ) for b in range(num_relations)]
        self.coefs = nn.ParameterList(coefs)
        # self.basis_matrix = nn.ParameterList(basis_matrix)
        self.log_softmax = nn.LogSoftmax(dim=1)

        self.reset_parameters()

    def reset_parameters(self):
        # size_basis = self.num_basis * self.feature_dim
        # size_coef = self.num_basis * self.num_relations
        random_init(1e-3, self.basis_matrix)
        for coef in self.coefs:
            random_init(1e-3, coef)

    def forward(self, u_features, i_features):
        for relation in range(self.num_relations):
            q_matrix = torch.sum(self.coefs[relation].unsqueeze(1) * self.basis_matrix, 0)
            q_matrix = q_matrix.reshape(self.feature_dim, self.feature_dim)
            if relation == 0:
                out = torch.chain_matmul(
                        u_features, q_matrix, i_features.t()).unsqueeze(-1)
            else:
                out = torch.cat((out, torch.chain_matmul(
                    u_features, q_matrix, i_features.t()).unsqueeze(-1)),dim=2)

        out = out.view(u_features.shape[0] * i_features.shape[0], -1)
        out = self.log_softmax(out)
        # out = F.log_softmax(out, dim=1)

        return out

if __name__ == '__main__':
    in_c = 2625
    hid_c = 500
    out_c = 75
    num_basis = 2
    num_relations = 5
    drop_prob = 0.7
    gcenc = GCEncoder(in_c, hid_c, out_c, num_relations, drop_prob)
    bidec = BiDecoder(out_c, num_basis, num_relations)
    # print(gcenc.rgc_layer)
    rgc = RGCLayer(in_c, hid_c, num_relations, drop_prob)
    dense = DenseLayer(in_c, out_c, drop_prob)
    # print(rgc)
    # print(dense)
    # print(bidec)
    # print(rgc.named_parameters())
    # print(dense.named_parameters())
    # print(bidec.named_parameters())

    gae = GAE(in_c, hid_c, out_c, num_basis, num_relations, drop_prob)
    print(gae.parameters())

    params = rgc.parameters()
    print(bidec.named_parameters())
    for param in params:
        print(param)

    print(bidec.basis_matrix)
    print(bidec.coefs)

    print(bidec.basis_matrix.requires_grad)
    print(bidec.coefs.requires_grad)
