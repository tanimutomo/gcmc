import torch
import torch.nn as nn
from src.layers import RGCLayer, DenseLayer


class GAE(nn.Module):
    def __init__(self, in_c, hid_c, out_c, num_basis, num_relations, 
            num_user, drop_prob, weight_init, accum,
            rgc_bn, rgc_relu, dense_bn, dense_relu, bidec_drop):
        super(GAE, self).__init__()
        self.gcenc = GCEncoder(in_c, hid_c, out_c, num_relations, num_user, 
                drop_prob, weight_init, accum, rgc_bn, rgc_relu, dense_bn, dense_relu)
        self.bidec = BiDecoder(out_c, num_basis, num_relations, drop_prob, weight_init, accum, bidec_drop)

    def forward(self, x, edge_index, edge_type, edge_norm):
        u_features, i_features = self.gcenc(x, edge_index, edge_type, edge_norm)
        adj_matrices = self.bidec(u_features, i_features)

        return adj_matrices


class GCEncoder(nn.Module):
    def __init__(self, in_c, hid_c, out_c, num_relations, num_user, drop_prob,  
            weight_init, accum, rgc_bn, rgc_relu, dense_bn, dense_relu):
        super(GCEncoder, self).__init__()
        self.num_relations = num_relations
        self.num_user = num_user
        self.accum = accum

        self.rgc_layer = RGCLayer(in_c, hid_c, num_relations, drop_prob,  
                weight_init, accum, rgc_bn, rgc_relu)
        self.dense_layer = DenseLayer(hid_c, out_c, num_relations, drop_prob, in_c,
                num_user, weight_init, accum, dense_bn, dense_relu)

    def forward(self, x, edge_index, edge_type, edge_norm):
        features = self.rgc_layer(x, edge_index, edge_type, edge_norm)
        u_features, i_features = self.separate_features(features)
        u_features, i_features = self.dense_layer(u_features, i_features)

        return u_features, i_features

    def separate_features(self, features):
        if self.accum == 'stack':
            num_nodes = int(features.shape[0] / self.num_relations)
            for r in range(self.num_relations):
                if r == 0:
                    u_features = features[:self.num_user]
                    i_features = features[self.num_user: (r+1) * num_nodes]
                else:
                    u_features = torch.cat((u_features,
                        features[r * num_nodes: r * num_nodes + self.num_user]), dim=0)
                    i_features = torch.cat((i_features,
                        features[r * num_nodes + self.num_user: (r+1) * num_nodes]), dim=0)

        else:
            u_features = features[:self.num_user]
            i_features = features[self.num_user:]

        return u_features, i_features


class BiDecoder(nn.Module):
    def __init__(self, feature_dim, num_basis, num_relations, drop_prob, weight_init, accum, apply_drop):
        super(BiDecoder, self).__init__()
        self.num_basis = num_basis
        self.num_relations = num_relations
        self.feature_dim = feature_dim
        self.accum = accum
        self.apply_drop = apply_drop

        self.dropout = nn.Dropout(drop_prob)
        self.basis_matrix = nn.Parameter(
                torch.Tensor(num_basis, feature_dim * feature_dim))
        coefs = [nn.Parameter(torch.Tensor(num_basis)
            ) for b in range(num_relations)]
        self.coefs = nn.ParameterList(coefs)

        self.reset_parameters(weight_init)

    def reset_parameters(self, weight_init):
        # weight_init(self.basis_matrix, self.feature_dim, self.feature_dim)
        nn.init.orthogonal_(self.basis_matrix)
        for coef in self.coefs:
            weight_init(coef, self.num_basis, self.num_relations)

    def forward(self, u_features, i_features):
        if self.apply_drop:
            u_features = self.dropout(u_features)
            i_features = self.dropout(i_features)
        if self.accum == 'stack':
            u_features = u_features.reshape(self.num_relations, -1, self.feature_dim)
            i_features = i_features.reshape(self.num_relations, -1, self.feature_dim)
            num_users = u_features.shape[1]
            num_items = i_features.shape[1]
        else:
            num_users = u_features.shape[0]
            num_items = i_features.shape[0]
            
        for relation in range(self.num_relations):
            q_matrix = torch.sum(self.coefs[relation].unsqueeze(1) * self.basis_matrix, 0)
            q_matrix = q_matrix.reshape(self.feature_dim, self.feature_dim)
            if self.accum == 'stack':
                if relation == 0:
                    out = torch.chain_matmul(
                            u_features[relation], q_matrix, 
                            i_features[relation].t()).unsqueeze(-1)
                else:
                    out = torch.cat((out, torch.chain_matmul(
                        u_features[relation], q_matrix, 
                        i_features[relation].t()).unsqueeze(-1)),dim=2)
            else:
                if relation == 0:
                    out = torch.chain_matmul(
                            u_features, q_matrix, i_features.t()).unsqueeze(-1)
                else:
                    out = torch.cat((out, torch.chain_matmul(
                        u_features, q_matrix, i_features.t()).unsqueeze(-1)),dim=2)

        out = out.reshape(num_users * num_items, -1)

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
    rgc = RGCLayer(in_c, hid_c, num_relations, drop_prob)
    dense = DenseLayer(in_c, out_c, drop_prob)

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
