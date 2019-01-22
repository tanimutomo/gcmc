import torch
import torch.nn.functional as F
from torch.nn import Parameter as Param
from torch_geometric.nn.conv import MessagePassing

from utils import uniform

class GAE(nn.Module):
    def __init__(self, in_c, hid_c, out_c, num_relations):
        super(GAE, self).__init__()
        self.gcenc = GCEncoder(in_c, hid_c, out_c, num_relations)
        self.bidec = BiDecoder()

    def forward():
        u_features, i_features = self.gcenc(x, edge_index, edge_type, edge_norm)
        adj_matrices = self.bidec()

        return adj_matrices



class GCEncoder(nn.Module):
    def __init__(self, in_c, hid_c, out_c, num_relations):
        super(GCEncoder, self).__init__()
        self.rgc_layer = RGCLayer(in_c, hid_c, num_relations)
        self.dence_layer = DenceLayer(hid_c, out_c)

    def forward(self, x, edge_index, edge_type, edge_norm):
        features = rgc_layer(x, edge_index, edge_type, edge_norm)
        u_features, i_features = self.separate_features(features)
        u_features, i_features = dence_layer(u_features, i_features)

        return u_features, i_features

    def separate_features(self, features, num_user):
        u_features = features[:num_user]
        i_features = features[num_user:]

        return u_features, i_features



class RGCLayer(MessagePassing):
    def __init__(self, in_c, out_c, num_relations, drop_prob):
        self.in_c = in_c
        self.out_c = out_c
        self.num_relations = num_relations
        self.drop_prob = drop_prob
        
        self.ord_basis = [Param(torch.Tensor(1, in_c * out_c)) for r in range(num_relations)]

        # weightの初期値についてはわからないので，とりあえずなしに．
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, x, edge_index, edge_type, edge_norm=None):
        return self.propagate('add', edge_index, x=x, edge_typ=edge_type, edge_norm)

    def message(self, x_j, edge_type, edge_norm):
        # create weight using ordinal weight sharing
        for relation in range(self.num_relations):
            if relation == 0:
                weight = ord_basis[relation]
            else:
                weight = torch.cat((weight, weight[-1] 
                    + self.ord_basis[relation]), 0)

        weight = weight.reshape(-1, self.out_c)
        weight = self.node_dropout(weight)
        index = edge_type * self.in_c + x_j
        out = weight[index]

        return out if edge_norm is None else out * edge_norm.reshape(-1, 1)

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channles]
        return aggr_out

    def node_dropout(self, weight):
        drop_mask = torch.rand(self.in_c) + (1 - self.drop_prob)
        drop_mask = torch.floor(drop_mask).type(torch.ByteTensor)
        drop_mask = torch.cat([drop_mask 
            for r in range(self.num_relation)], 0).unsqueeze(1)
        drop_mask = drop_out.expand(drop_mask.size(0), self.out_c)

        weight = torch.where(drop_mask, weight, 0)

        return weight



class DenceLayer(nn.Module):
    def __init__(self, in_c, out_c, drop_prob, bias=False):
        super(DenceLayer, self).__init__()

        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(in_c, out_c, bias=bias)

    def forward(self, u_features, i_features):
        u_features = self.dropout(u_features)
        u_features = self.fc(u_features)

        i_features = self.dropout(i_features)
        i_features = self.fc(i_features)

        return u_features, i_features



class BiDecoder(nn.Module):
    def __init__(self, out_c, num_basis, num_relations):
        super(BiDecoder, self).__init__()
        self.out_c = out_c
        self.basis_matrix = Param(torch.Tensor(num_basis, out_c * out_c))
        self.coefs = [Param(torch.Tensor(num_basis)) for r in range(num_relations)]

    def forward(self, u_features, i_features):
        for relation in range(num_relations):
            q_matrix = torch.sum(self.coefs[relation] * self.basis_matrix, 0)
            q_matrix = q_matrix.reshape(self.out_c, self.out_c)
            try:
                out = torch.cat((
                        out_adj_matrices,
                        torch.chain_matmul(u_features, q_matrix, i_features.t())),
                        dim=0)
            except:
                out = torch.chain_matmul(
                        u_features, q_matrix, i_features.t()).unsqueeze(0)

        return out
