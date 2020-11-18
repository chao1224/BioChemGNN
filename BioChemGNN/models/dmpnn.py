from collections import *

import torch
from torch import nn
from torch.nn import functional as F
from BioChemGNN import data


class DirectedMessagePassingNeuralNetwork(torch.nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, hidden_dim, layer_size, output_dim):
        super(DirectedMessagePassingNeuralNetwork, self).__init__()

        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layer_size = layer_size

        self.activation = nn.ReLU()

        self.W_input = nn.Linear(node_feature_dim+edge_feature_dim, hidden_dim, bias=False)
        self.W_hidden = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_output = nn.Linear(node_feature_dim+hidden_dim, hidden_dim)
        
        self.hidden_dim_list = [hidden_dim]
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim)
        )

        return

    def represent(self, graph):
        node_feat, edge_feat = graph.atom_feat_list, graph.bond_feat_list
        a_to_b = graph.a_to_b
        a_from_b = graph.a_from_b
        b_to_a = graph.b_to_a
        b_from_a = graph.b_from_a
        b2revb = graph.b2revb

        message = self.W_input(torch.cat([node_feat[b_to_a], edge_feat], dim=-1))
        edge_input = message
        message = self.activation(message)
        message_layers = [message]

        for i in range(self.layer_size - 1):
            message = message_layers[-1]
            neighbor_edge_message = data.index_select_ND(message, a_from_b)
            node_message = neighbor_edge_message.sum(dim=1)
            rev_edge_message = message[b2revb]
            message = node_message[b_from_a] - rev_edge_message

            message = self.W_hidden(message)
            message = self.activation(edge_input + message)
            message_layers.append(message)

        message = message_layers[-1]
        neighbor_edge_message = data.index_select_ND(message, a_from_b)
        node_message = neighbor_edge_message.sum(dim=1)
        node_repr = torch.cat([node_feat, node_message], dim=1)
        node_repr = self.activation(self.W_output(node_repr))
        return node_repr

    def forward(self, graph):
        node_repr = self.represent(graph)

        graph_repr = []
        size_list = []
        atom_scope = graph.atom_scope
        for i, (a_start, a_size) in enumerate(atom_scope):
            size_list.append(a_size)
            node_index = slice(a_start, a_start+a_size)
            temp = torch.sum(node_repr[node_index], dim=0)
            graph_repr.append(temp)

        graph_repr = torch.stack(graph_repr, dim=0)
        y = self.fc_layers(graph_repr)
        return {'node_repr': node_repr, 'graph_repr': graph_repr, 'y': y}
