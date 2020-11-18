from collections import *

import torch
from torch import nn
from torch.nn import functional as F
from BioChemGNN import data


class GraphIsomorphismNetworkLayer(nn.Module):
    def __init__(self, input_dim, output_dim, edge_feature_dim=None,
                 epsilon=0, activation='relu', batch_norm=False):
        super(GraphIsomorphismNetworkLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.edge_feature_dim = edge_feature_dim
        if self.edge_feature_dim:
            self.edge_linear = nn.Linear(edge_feature_dim, output_dim)
        else:
            self.edge_linear = None
        self.epsilon = epsilon

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = False

        if activation is not None:
            self.activation = getattr(F, activation)
        else:
            self.activation = None
        return

    def forward(self, x, graph):
        b_from_a = graph.b_from_a
        a_from_b = graph.a_from_b
        a2a = b_from_a[a_from_b]
        x = self.linear(x)
        x[0].fill_(0)
        message = x[a2a]

        if self.edge_feature_dim:
            edge_message = graph.bond_feat_list
            edge_message = self.edge_linear(edge_message)
            edge_message[0].fill_(0)
            neighbor_edge_message = data.index_select_ND(edge_message, a_from_b)
            message += neighbor_edge_message

        x = (1 + self.epsilon) * x + torch.sum(message, dim=1)

        if self.batch_norm:
            x = self.batch_norm(x)
            x[0].fill_(0)
        if self.activation:
            x = self.activation(x)
        return x


class GraphIsomorphismNetwork(torch.nn.Module):
    def __init__(self, node_feature_dim=None, edge_feature_dim=None, hidden_dim=None, layer_size=None, output_dim=None,
                 dropout=0., epsilon=0, activation='relu', batch_norm=True):
        super(GraphIsomorphismNetwork, self).__init__()
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.layer_size = layer_size
        self.output_dim = output_dim
        self.dropout = dropout
        self.epsilon = epsilon

        self.hidden_dim_list = [node_feature_dim] + [hidden_dim] * self.layer_size

        self.gin_layers = nn.ModuleList()
        for layer_idx, (in_dim, out_dim) in enumerate(zip(self.hidden_dim_list[:-1], self.hidden_dim_list[1:])):
            self.gin_layers.append(GraphIsomorphismNetworkLayer(
                in_dim, out_dim, edge_feature_dim=edge_feature_dim,
                epsilon=epsilon, activation=activation, batch_norm=batch_norm)
            )

        self.out_layer = nn.Sequential(
            nn.Linear(sum(self.hidden_dim_list), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim)
        )

    def forward(self, graph):
        x = graph.atom_feat_list
        h = []
        h.append(x)
        for idx in range(self.layer_size):
            x = self.gin_layers[idx](x, graph)
            h.append(x)
        h = torch.cat(h, dim=1)

        x = F.dropout(h, self.dropout, training=self.training)

        node_repr = x
        graph_repr = []
        size_list = []
        atom_scope = graph.atom_scope
        for i, (a_start, a_size) in enumerate(atom_scope):
            size_list.append(a_size)
            node_index = slice(a_start, a_start+a_size)
            temp = torch.mean(node_repr[node_index], dim=0)
            graph_repr.append(temp)

        graph_repr = torch.stack(graph_repr, dim=0)
        y = self.out_layer(graph_repr)

        return {'node_repr': node_repr, 'graph_repr': graph_repr, 'y': y}
