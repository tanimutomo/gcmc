import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import scatter_
from src.utils import stack, split_stack


class RGCLayer(MessagePassing):
    def __init__(self, in_c, out_c, num_relations, drop_prob, 
            weight_init, accum, bn, relu):
        super(RGCLayer, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_relations = num_relations
        self.drop_prob = drop_prob
        self.weight_init = weight_init
        self.accum = accum
        self.bn = bn
        self.relu = relu
        
        if accum == 'split_stack':
            # each 100 dimention has each realtion node features
            self.ord_basis = nn.Parameter(torch.Tensor(5, in_c * out_c))
        else:
            # ordinal basis matrices in_c * out_c = 2625 * 500
            ord_basis = [nn.Parameter(torch.Tensor(1, in_c * out_c)) for r in range(num_relations)]
            self.ord_basis = nn.ParameterList(ord_basis)
        self.relu = nn.ReLU()

        if accum == 'stack':
            self.bn = nn.BatchNorm1d(self.in_c * num_relations)
        else:
            self.bn = nn.BatchNorm1d(self.in_c)

        self.reset_parameters(weight_init)


    def reset_parameters(self, weight_init):
        if self.accum == 'split_stack':
            weight_init(self.ord_basis, self.in_c * self.num_relations, self.out_c)
        else:
            for basis in self.ord_basis:
                weight_init(basis, self.in_c, self.out_c)


    def forward(self, x, edge_index, edge_type, edge_norm=None):
        return self.propagate(self.accum, edge_index, x=x, edge_type=edge_type, edge_norm=edge_norm)

    def propagate(self, aggr, edge_index, **kwargs):
        r"""The initial call to start propagating messages.
        Takes in an aggregation scheme (:obj:`"add"`, :obj:`"mean"` or
        :obj:`"max"`), the edge indices, and all additional data which is
        needed to construct messages and to update node embeddings."""

        assert aggr in ['split_stack', 'stack', 'add', 'mean', 'max']
        kwargs['edge_index'] = edge_index

        size = None
        message_args = []
        for arg in self.message_args:
            if arg[-2:] == '_i':
                # tmp is x
                tmp = kwargs[arg[:-2]]
                size = tmp.size(0)
                message_args.append(tmp[edge_index[0]])
            elif arg[-2:] == '_j':
                tmp = kwargs[arg[:-2]]
                size = tmp.size(0)
                message_args.append(tmp[edge_index[1]])
            else:
                message_args.append(kwargs[arg])

        update_args = [kwargs[arg] for arg in self.update_args]

        out = self.message(*message_args)
        if aggr == 'split_stack':
            out = split_stack(out, edge_index[0], kwargs['edge_type'], dim_size=size)
        elif aggr == 'stack':
            out = stack(out, edge_index[0], kwargs['edge_type'], dim_size=size)
        else:
            out = scatter_(aggr, out, edge_index[0], dim_size=size)
        out = self.update(out, *update_args)

        return out


    def message(self, x_j, edge_type, edge_norm):
        # create weight using ordinal weight sharing
        if self.accum == 'split_stack':
            weight = self.ord_basis
        else:
            for relation in range(self.num_relations):
                if relation == 0:
                    weight = self.ord_basis[relation]
                else:
                    weight = torch.cat((weight, weight[-1] 
                        + self.ord_basis[relation]), 0)

        # weight (R x (in_dim * out_dim)) reshape to (R * in_dim) x out_dim
        # weight has all nodes features
        weight = weight.reshape(-1, self.out_c)
        weight = self.node_dropout(weight)
        # index has target features index in weitht matrix
        index = edge_type * self.in_c + x_j
        # this opration is that index(160000) specify the nodes idx in weight matrix
        # for getting the features corresponding edge_index
        out = weight[index]

        # out is edges(160000) x hidden(500)
        return out if edge_norm is None else out * edge_norm.reshape(-1, 1)

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channles]
        if self.bn:
            aggr_out = self.bn(aggr_out.unsqueeze(0)).squeeze(0)
        if self.relu:
            aggr_out = self.relu(aggr_out)
        return aggr_out

    def node_dropout(self, weight):
        drop_mask = torch.rand(self.in_c) + (1 - self.drop_prob)
        drop_mask = torch.floor(drop_mask).type(torch.float)
        drop_mask = torch.cat([drop_mask 
            for r in range(self.num_relations)], 0).unsqueeze(1)

        drop_mask = drop_mask.expand(drop_mask.size(0), self.out_c)

        assert weight.shape == drop_mask.shape
        weight = weight * drop_mask

        return weight



class DenseLayer(nn.Module):
    def __init__(self, in_c, out_c, num_relations, drop_prob, num_nodes, num_user, 
            weight_init, accum, bn, relu, bias=False):
        super(DenseLayer, self).__init__()
        # self.in_c = in_c
        # self.out_c = out_c
        self.num_nodes = num_nodes
        self.num_user = num_user
        self.bn = bn
        self.relu = relu
        self.weight_init = weight_init

        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(in_c, out_c, bias=bias)
        if accum == 'stack':
            self.bn_u = nn.BatchNorm1d(num_user * num_relations)
            self.bn_i = nn.BatchNorm1d((num_nodes - num_user) * num_relations)
        else:
            self.bn_u = nn.BatchNorm1d(num_user)
            self.bn_i = nn.BatchNorm1d(num_nodes - num_user)
        self.relu = nn.ReLU()

        # self.reset_parameters(weight_init)

    # def reset_parameters(self, weight_init):
    #     weight_init(self.fc, self.in_c, self.out_c)

    def forward(self, u_features, i_features):
        u_features = self.dropout(u_features)
        u_features = self.fc(u_features)
        if self.bn:
            u_features = self.bn_u(
                    u_features.unsqueeze(0)).squeeze()
        if self.relu:
            u_features = self.relu(u_features)

        i_features = self.dropout(i_features)
        i_features = self.fc(i_features)
        if self.bn:
            i_features = self.bn_i(
                    i_features.unsqueeze(0)).squeeze()
        if self.relu:
            i_features = self.relu(i_features)

        return u_features, i_features

