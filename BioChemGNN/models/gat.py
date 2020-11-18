from collections import *

import torch
from torch import nn
from torch.nn import functional as F
from BioChemGNN import data


class GraphAttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()

        self.dropout = dropout
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha
        self.concat = concat

        self.linear = nn.Linear(input_dim, output_dim)
        self.act = nn.LeakyReLU(self.alpha)
        # self.act = nn.ReLU()

        return

    def forward(self, x, a2a):
        h = self.linear(x)
        h[0].fill_(float('-9e15'))

        neighbors = torch.cat([h.unsqueeze(1), h[a2a]], dim=1)
        attention = torch.bmm(neighbors, h.unsqueeze(2))

        attention = nn.Softmax(dim=1)(attention)
        h = h * attention[:, 0]

        h[0].fill_(0)

        return h


class MultiHeadGraphAttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, head_num, dropout, alpha, concat=True):
        super(MultiHeadGraphAttentionLayer, self).__init__()

        self.dropout = dropout
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha
        self.concat = concat

        self.attention_layers = nn.ModuleList()

        for i in range(head_num):
            self.attention_layers.append(
                GraphAttentionLayer(input_dim, output_dim, dropout=dropout, alpha=alpha, concat=concat))

    def forward(self, x, a2a):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, a2a) for att in self.attention_layers], dim=1)
        return x


class GraphAttentionNetwork(torch.nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, head_num, layer_size, output_dim, dropout, alpha):
        super(GraphAttentionNetwork, self).__init__()
        self.layer_size = layer_size
        self.output_dim = output_dim
        self.dropout = dropout

        self.hidden_dim_list = [node_feature_dim] + [hidden_dim * head_num] * self.layer_size
        self.multihead_attention_layers = nn.ModuleList()
        for idx in range(self.layer_size):
            input_dim = self.hidden_dim_list[idx]
            self.multihead_attention_layers.append(
                MultiHeadGraphAttentionLayer(input_dim=input_dim, output_dim=hidden_dim, head_num=head_num, dropout=dropout, alpha=alpha, concat=True)
            )

        # self.out_layer = GraphAttentionLayer(input_dim=self.hidden_dim_list[-1], output_dim=output_dim, dropout=dropout, alpha=alpha, concat=False)
        self.out_layer = nn.Sequential(
            nn.Linear(self.hidden_dim_list[-1], hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim)
        )

    def forward(self, graph):
        x = graph.atom_feat_list
        b_from_a = graph.b_from_a
        a_from_b = graph.a_from_b
        a2a = b_from_a[a_from_b]

        for idx in range(self.layer_size):
            x = self.multihead_attention_layers[idx](x, a2a)

        x = F.dropout(x, self.dropout, training=self.training)

        node_repr = x
        graph_repr = []
        size_list = []
        atom_scope = graph.atom_scope
        for i, (a_start, a_size) in enumerate(atom_scope):
            size_list.append(a_size)
            node_index = slice(a_start, a_start+a_size)
            temp = torch.sum(node_repr[node_index], dim=0)
            graph_repr.append(temp)

        graph_repr = torch.stack(graph_repr, dim=0)
        y = self.out_layer(graph_repr)

        return {'node_repr': node_repr, 'graph_repr': graph_repr, 'y': y}
