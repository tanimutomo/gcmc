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
        hid_features = rgc_layer(x, edge_index, edge_type, edge_norm)
        out_features = dence_layer(hid_features)
        u_features, i_features = self.separate_features(out_features)

        return u_features, i_features

    def separate_features(self, features, num_user):
        u_features = features[:num_user]
        i_features = features[num_user:]

        return u_features, i_features




class RGCLayer(MessagePassing):
    def __init__(self, in_c, out_c, num_relations):
        self.in_c = in_c
        self.out_c = out_c
        self.num_relations = num_relations
        
        self.ord_basis = [Param(torch.Tensor(1, in_c * out_c)) for r in range(num_relations)]

        # weightの初期値についてはわからないので，とりあえずなしに．
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, x, edge_index, edge_type, edge_norm=None):
        return self.propagate('add', edge_index, x=x, edge_typ=edge_type, edge_norm)

    def message(self, x_j, edge_type, edge_norm):
        # create weight using ordinal weight sharing
        tmp, self.w = 0, 0
        for relation in range(self.num_relations):
            tmp = ord_basis[relation]
            if relation == 0:
                self.w = tmp
            else:
                self.w = torch.cat((self.w, tmp), 0)

        index = edge_type * self.in_c + x_j
        out = w[index]

        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channles]
        return aggr_out


class DenceLayer(nn.Module):
    def __init__(self, in_c, out_c, bias=False):
        super(DenceLayer, self).__init__()

        self.fc = nn.Linear(in_c, out_c, bias=bias)

    def forward(self, x):
        out = self.fc(x)

        return out



class BiDecoder(nn.Module):
    def __init__(self, out_c, num_basis, num_relations):
        super(BiDecoder, self).__init__()
        self.out_c = out_c
        self.basis_matrix = Param(torch.Tensor(num_basis, out_c * out_c))
        self.coefs = [Param(torch.Tensor(num_basis)) for r in range(num_relations)]

    def forward(self, u_features, i_features):
        for relation in range(num_relations):
            q_matrix = torch.sum(self.coefs[relation] * self.basis_matrix, 0)
            q_matrix = q_matrix.view(self.out_c, self.out_c)
            try:
                out = torch.cat((
                        out_adj_matrices,
                        torch.chain_matmul(u_features, q_matrix, i_features.t())),
                        dim=0)
            except:
                out = torch.chain_matmul(
                        u_features, q_matrix, i_features.t()).unsqueeze(0)

        return out
