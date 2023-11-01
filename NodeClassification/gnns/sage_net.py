import torch
import torch as th
import torch.nn as nn
from .sage_layer import SAGEConv
import torch.nn.functional as F
# from tqdm import tqdm
# from NodeClassification import utils
import utils
import time

class GraphSAGE(nn.Module):

    def __init__(self, embedding_dim, adj, aggr='mean'):
        super().__init__()
        n_layers = len(embedding_dim) - 1
        in_feats = embedding_dim[0]
        n_classes = embedding_dim[-1]
        n_hidden = embedding_dim[1]

        self.net_layer = nn.ModuleList()
        if n_layers > 1:
            self.net_layer.append(SAGEConv(in_feats, n_hidden, aggr=aggr, bias=False))
            for i in range(1, n_layers - 1):
                self.net_layer.append(SAGEConv(n_hidden, n_hidden, aggr=aggr, bias=False))
            # self.net_layer.append(SAGEConv(n_hidden, n_classes, aggr=aggr))
            self.net_layer.append(nn.Linear(n_hidden, n_classes, bias=False))
        else:
            # self.net_layer.append(SAGEConv(in_feats, n_classes, aggr=aggr))
            self.net_layer.append(nn.Linear(in_feats, n_classes, bias=False))

        self.dropout = 0.5

        # self.adj_nonzero = torch.nonzero(adj, as_tuple=False).shape[0]
        self.adj_nonzero = adj._nnz()
        # self.adj_mask1_train = nn.Parameter(self.generate_adj_mask(adj))
        # self.adj_mask2_fixed = nn.Parameter(self.generate_adj_mask(adj), requires_grad=False)

        ################
        # Debug_yin_feat Replace adj_mask to feature mask
        vertex_num = adj.shape[0]
        # Only set mask for the 2nd-layer's features
        self.adj_mask1_train = nn.Parameter(self.generate_feat_mask(vertex_num, embedding_dim[1]))
        self.adj_mask2_fixed = nn.Parameter(self.generate_feat_mask(vertex_num, embedding_dim[1]),
                                            requires_grad=False)
        ################

        self.normalize = utils.torch_normalize_adj

        self.feats = []  # Record features of each layer
        self.mm_time = 0

    # def reset_parameters(self):
    #     for layer in self.net_layer:
    #         layer.reset_parameters()

    def forward(self, features, edge_index):
        h = features

        self.feats = []  # Reset before each forward
        self.feats.append(h)

        for i, layer in enumerate(self.net_layer[:-1]):

            t0 = time.time()
            # h = torch.mul(h, self.adj_mask1_train)
            # h = torch.mul(h, self.adj_mask2_fixed)
            h = layer(h, edge_index)
            self.mm_time += time.time() - t0

            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            self.feats.append(h)
        
        # h = self.net_layer[-1](h, edge_index)
        # self.feats.append(h)
        # return h

        ##########
        # Only add mask on features of the 2nd layer
        h = torch.mul(h, self.adj_mask1_train)
        h = torch.mul(h, self.adj_mask2_fixed)
        ##########

        h = self.net_layer[-1](h)
        values = torch.ones(edge_index[0].shape[0]).to(edge_index.device)
        adj = torch.sparse_coo_tensor(indices=edge_index, values=values)
        h = torch.mm(adj, h)
        self.feats.append(h)

        return h
    
        
    def generate_adj_mask(self, input_adj):

        sparse_adj = input_adj
        zeros = torch.zeros_like(sparse_adj)
        ones = torch.ones_like(sparse_adj)
        mask = torch.where(sparse_adj != 0, ones, zeros)
        return mask
    
    def generate_feat_mask(self, vertex_num, embedding_dim):
        mask = torch.ones([vertex_num, embedding_dim])
        return mask