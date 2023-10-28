import torch
import torch.nn as nn
import pdb
import copy
import utils
import time


class net_gcn(nn.Module):

    def __init__(self, embedding_dim, adj):
        super().__init__()

        self.layer_num = len(embedding_dim) - 1
        self.net_layer = nn.ModuleList([
            nn.Linear(embedding_dim[ln], embedding_dim[ln + 1], bias=False)
            for ln in range(self.layer_num)
        ])
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
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

    def forward(self, x, adj, val_test=False):

        ################
        # adj = torch.mul(adj, self.adj_mask1_train)
        # adj = torch.mul(adj, self.adj_mask2_fixed)
        ################

        adj = self.normalize(adj)
        #adj = torch.mul(adj, self.adj_mask2_fixed)

        # feats = []  # Record features of each layer
        self.feats = []  # Reset before each forward
        self.feats.append(x)

        # w0 = getattr(self.net_layer[0], 'weight')
        # w1 = getattr(self.net_layer[1], 'weight')
        # spar_w0 = utils.count_sparsity(w0)
        # spar_w1 = utils.count_sparsity(w1)
        # print()
        # print('spar_weight:', spar_w0, spar_w1)

        for ln in range(self.layer_num):
            # x = torch.mm(adj, x)
            # w_spar = utils.count_sparsity(getattr(self.net_layer[ln], 'weight'))
            # spar_x_pre = utils.count_sparsity(x)

            ##########
            # Only add mask on features of the 2nd layer
            if ln == 1:
                x = torch.mul(x, self.adj_mask1_train)
                x = torch.mul(x, self.adj_mask2_fixed)
            ##########

            x = self.net_layer[ln](x)
            # spar_x_after = utils.count_sparsity(x)
            # print('spar: ', spar_x_pre, w_spar, spar_x_after)

            t0 = time.time()
            x = torch.mm(adj, x)  # (AX)W -> A(XW)
            self.mm_time += time.time() - t0

            # # adj_sp = adj.to_sparse()   
            # x_sp = x.to_sparse()
            # t0 = time.time()
            # x = torch.sparse.mm(adj, x_sp)  # (AX)W -> A(XW)
            # self.mm_time += time.time() - t0
            # x = x.to_dense()

            if ln < self.layer_num - 1:
                x = self.relu(x)

                if not val_test:
                    x = self.dropout(x)

            self.feats.append(x)

            # spar_x_pre = utils.count_sparsity(x)
            # utils.plot_val_distribution(x, 'x_pre_relu')
            # x = self.relu(x)
            # utils.plot_val_distribution(x, 'x_after_relu')

            # spar_x_after = utils.count_sparsity(x)
            # print('spar_x', spar_x_pre, spar_x_after)

            # if val_test:
            #     x_total = x.numel()
            #     zeros = torch.zeros_like(x)
            #     ones = torch.ones_like(x)
            #     x_nonzero_norm = torch.where(x != 0, ones, zeros)
            #     x_nonzero = x_nonzero_norm.sum().item()
            #     x_dense = x_nonzero * 100 / x_total
            #     print('layer: {:d}, x_dense: {:.1f}'.format(ln, x_dense))

            # if val_test:
            #     self.feats.append(x)
            #     continue
            # x = self.dropout(x)

            # self.feats.append(x)

            # for i in self.feats:
            #     print(utils.count_sparsity(i))

            # x_total = x.numel()
            # zeros = torch.zeros_like(x)
            # ones = torch.ones_like(x)
            # x_nonzero_norm = torch.where(x != 0, ones, zeros)
            # x_nonzero = x_nonzero_norm.sum().item()
            # x_dense = x_nonzero * 100 / x_total
            # print('layer: {:d}, x_dense: {:.1f}'.format(ln, x_dense))

        # feat0 = self.feats[0]
        # feat1 = self.feats[1]
        # spar_feat0 = utils.count_sparsity(feat0)
        # spar_feat1 = utils.count_sparsity(feat1)
        # print('spar_feat: ', spar_feat0, spar_feat1)

        return x

    def generate_adj_mask(self, input_adj):

        sparse_adj = input_adj
        zeros = torch.zeros_like(sparse_adj)
        ones = torch.ones_like(sparse_adj)
        mask = torch.where(sparse_adj != 0, ones, zeros)
        return mask

    def generate_feat_mask(self, vertex_num, embedding_dim):
        mask = torch.ones([vertex_num, embedding_dim])
        return mask


class net_gcn_admm(nn.Module):

    def __init__(self, embedding_dim, adj):
        super().__init__()

        self.layer_num = len(embedding_dim) - 1
        self.net_layer = nn.ModuleList([
            nn.Linear(embedding_dim[ln], embedding_dim[ln + 1], bias=False)
            for ln in range(self.layer_num)
        ])
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.adj_nonzero = torch.nonzero(adj, as_tuple=False).shape[0]
        self.adj_layer1 = nn.Parameter(copy.deepcopy(adj), requires_grad=True)
        self.adj_layer2 = nn.Parameter(copy.deepcopy(adj), requires_grad=True)

    def forward(self, x, adj, val_test=False):

        for ln in range(self.layer_num):
            if ln == 0:
                x = torch.mm(self.adj_layer1, x)
            elif ln == 1:
                x = torch.mm(self.adj_layer2, x)
            else:
                assert False
            x = self.net_layer[ln](x)
            if ln == self.layer_num - 1:
                break
            x = self.relu(x)
            if val_test:
                continue
            x = self.dropout(x)
        return x

    # def forward(self, x, adj, val_test=False):

    #     for ln in range(self.layer_num):
    #         x = torch.mm(self.adj_list[ln], x)
    #         x = self.net_layer[ln](x)
    #         if ln == self.layer_num - 1:
    #             break
    #         x = self.relu(x)
    #         if val_test:
    #             continue
    #         x = self.dropout(x)
    #     return x

    def generate_adj_mask(self, input_adj):

        sparse_adj = input_adj
        zeros = torch.zeros_like(sparse_adj)
        ones = torch.ones_like(sparse_adj)
        mask = torch.where(sparse_adj != 0, ones, zeros)
        return mask


class net_gcn_baseline(nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()

        self.layer_num = len(embedding_dim) - 1
        self.net_layer = nn.ModuleList([
            nn.Linear(embedding_dim[ln], embedding_dim[ln + 1], bias=False)
            for ln in range(self.layer_num)
        ])
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, adj, val_test=False):

        for ln in range(self.layer_num):
            x = torch.mm(adj, x)
            # x = torch.spmm(adj, x)
            x = self.net_layer[ln](x)
            if ln == self.layer_num - 1:
                break
            x = self.relu(x)
            if val_test:
                continue
            x = self.dropout(x)
        return x


class net_gcn_multitask(nn.Module):

    def __init__(self, embedding_dim, ss_dim):
        super().__init__()

        self.layer_num = len(embedding_dim) - 1
        self.net_layer = nn.ModuleList([
            nn.Linear(embedding_dim[ln], embedding_dim[ln + 1], bias=False)
            for ln in range(self.layer_num)
        ])
        self.ss_classifier = nn.Linear(embedding_dim[-2], ss_dim, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, adj, val_test=False):

        x_ss = x

        for ln in range(self.layer_num):
            x = torch.spmm(adj, x)
            x = self.net_layer[ln](x)
            if ln == self.layer_num - 1:
                break
            x = self.relu(x)
            if val_test:
                continue
            x = self.dropout(x)

        if not val_test:
            for ln in range(self.layer_num):
                x_ss = torch.spmm(adj, x_ss)
                if ln == self.layer_num - 1:
                    break
                x_ss = self.net_layer[ln](x_ss)
                x_ss = self.relu(x_ss)
                x_ss = self.dropout(x_ss)
            x_ss = self.ss_classifier(x_ss)

        return x, x_ss
