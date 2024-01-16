import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, Linear

import dgl
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
"""
    GIN: Graph Isomorphism Networks
    HOW POWERFUL ARE GRAPH NEURAL NETWORKS? (Keyulu Xu, Weihua Hu, Jure Leskovec and Stefanie Jegelka, ICLR 2019)
    https://arxiv.org/pdf/1810.00826.pdf
"""

from gnns.gin_layer import GINLayer, ApplyNodeFunc, MLP
import pdb
import time


class GINNet(nn.Module):

    def __init__(self, net_params, graph):
        super().__init__()
        in_dim = net_params[0]
        hidden_dim = net_params[1]
        n_classes = net_params[2]
        dropout = 0.5
        self.n_layers = 2
        self.edge_num = graph.all_edges()[0].numel()
        n_mlp_layers = 1  # GIN
        learn_eps = True  # GIN
        neighbor_aggr_type = 'mean'  # GIN
        graph_norm = False
        batch_norm = False
        residual = False
        self.n_classes = n_classes

        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()

        for layer in range(self.n_layers):
            if layer == 0:
                mlp = MLP(n_mlp_layers, in_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(n_mlp_layers, hidden_dim, hidden_dim, n_classes)

            self.ginlayers.append(
                GINLayer(ApplyNodeFunc(mlp), neighbor_aggr_type, dropout, graph_norm, batch_norm,
                         residual, 0, learn_eps))

        # Linear function for output of each layer
        # which maps the output of different layers into a prediction score

        self.linears_prediction = nn.Linear(hidden_dim, n_classes, bias=False)
        # self.adj_mask1_train = nn.Parameter(torch.ones(self.edge_num, 1), requires_grad=True)
        # self.adj_mask2_fixed = nn.Parameter(torch.ones(self.edge_num, 1), requires_grad=False)

        ################
        # Debug_yin_feat Replace adj_mask to feature mask
        vertex_num = graph.num_of_nodes()
        embedding_dim = hidden_dim
        # Only set mask for the 2nd-layer's features
        self.adj_mask1_train = nn.Parameter(self.generate_feat_mask(vertex_num, embedding_dim[1]))
        self.adj_mask2_fixed = nn.Parameter(self.generate_feat_mask(vertex_num, embedding_dim[1]),
                                            requires_grad=False)
        ################

        self.feats = []  # Record features of each layer
        self.mm_time = 0

    def forward(self, g, h, snorm_n, snorm_e):

        # g.edata['mask'] = self.adj_mask1_train * self.adj_mask2_fixed
        hidden_rep = []

        self.feats = []  # Reset before each forward
        self.feats.append(h)

        for i in range(self.n_layers):
            ##########
            # Only add mask on features of the 2nd layer
            if i == 1:
                h = torch.mul(h, self.adj_mask1_train)
                h = torch.mul(h, self.adj_mask2_fixed)
            ##########

            t0 = time.time()
            h = self.ginlayers[i](g, h, snorm_n)
            self.mm_time += time.time() - t0
            hidden_rep.append(h)

            self.feats.append(h)  # Record feats
        score_over_layer = (self.linears_prediction(hidden_rep[0]) + hidden_rep[1]) / 2

        return score_over_layer

    def generate_feat_mask(self, vertex_num, embedding_dim):
        mask = torch.ones([vertex_num, embedding_dim])
        return mask
    

class GINNet(nn.Module):

    def __init__(self, net_params, graph):
        super().__init__()

        in_feats = net_params[0]
        n_hidden = net_params[1]
        n_classes = net_params[2]
        n_layers = 2

        self.convs = nn.ModuleList()
        if n_layers > 1:
            self.convs.append(GINConv(Linear(in_feats, n_hidden)))
            for i in range(1, n_layers - 1):
                self.convs.append(GINConv(Linear(n_hidden, n_hidden)))
            self.convs.append(GINConv(Linear(n_hidden, n_classes)))
        else:
            self.convs.append(GINConv(Linear(in_feats, n_classes)))

        self.dropout = 0.5

        ################
        # Debug_yin_feat Replace adj_mask to feature mask
        vertex_num = graph.num_of_nodes()
        embedding_dim = n_hidden
        # Only set mask for the 2nd-layer's features
        self.adj_mask1_train = nn.Parameter(self.generate_feat_mask(vertex_num, embedding_dim[1]))
        self.adj_mask2_fixed = nn.Parameter(self.generate_feat_mask(vertex_num, embedding_dim[1]),
                                            requires_grad=False)
        ################

        self.feats = []  # Record features of each layer
        self.mm_time = 0

    def reset_parameters(self):
        for layer in self.convs:
            layer.reset_parameters()

    def forward(self, features, edge_index):
        h = features
        for i, layer in enumerate(self.convs[:-1]):
            h = layer(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.convs[-1](h, edge_index)
        # return h.log_softmax(dim=-1)
        return h
    
    def generate_feat_mask(self, vertex_num, embedding_dim):
        mask = torch.ones([vertex_num, embedding_dim])
        return mask


class GINNet_ss(nn.Module):

    def __init__(self, net_params, num_par):
        super().__init__()
        in_dim = net_params[0]
        hidden_dim = net_params[1]
        n_classes = net_params[2]
        dropout = 0.5
        self.n_layers = 2
        n_mlp_layers = 1  # GIN
        learn_eps = True  # GIN
        neighbor_aggr_type = 'mean'  # GIN
        graph_norm = False
        batch_norm = False
        residual = False
        self.n_classes = n_classes

        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()

        for layer in range(self.n_layers):
            if layer == 0:
                mlp = MLP(n_mlp_layers, in_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(n_mlp_layers, hidden_dim, hidden_dim, n_classes)

            self.ginlayers.append(
                GINLayer(ApplyNodeFunc(mlp), neighbor_aggr_type, dropout, graph_norm, batch_norm,
                         residual, 0, learn_eps))

        # Linear function for output of each layer
        # which maps the output of different layers into a prediction score

        self.linears_prediction = nn.Linear(hidden_dim, n_classes, bias=False)
        self.classifier_ss = nn.Linear(hidden_dim, num_par, bias=False)

    def forward(self, g, h, snorm_n, snorm_e):

        # list of hidden representation at each layer (including input)
        hidden_rep = []

        for i in range(self.n_layers):
            h = self.ginlayers[i](g, h, snorm_n)
            hidden_rep.append(h)

        score_over_layer = (self.linears_prediction(hidden_rep[0]) + hidden_rep[1]) / 2
        h_ss = self.classifier_ss(hidden_rep[0])

        return score_over_layer, h_ss
