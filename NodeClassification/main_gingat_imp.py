import os
import random
import argparse

import torch
import torch as th
import torch.nn as nn
import numpy as np

import net as net
from utils import load_data, load_adj_raw
from sklearn.metrics import f1_score

import dgl
from gnns.gin_net import GINNet
from gnns.gat_net import GATNet
import pruning
import pruning_gin
import pruning_gat
import pdb
import warnings

warnings.filterwarnings('ignore')
import copy
import time
import logger
import sys
import utils


# def run_fix_mask(args, imp_num, rewind_weight_mask):
def run_fix_mask(args, imp_num, rewind_weight_mask, g, adj, features, labels, idx_train, idx_val,
                 idx_test, n_classes):

    if args['gpu'] < 0:
        cuda = False
    else:
        cuda = True
        gpu_id = args['gpu']
    device = th.device("cuda:" + str(gpu_id) if cuda else "cpu")

    pruning.setup_seed(args['seed'])
    # adj, features, labels, idx_train, idx_val, idx_test = load_data(args['dataset'])
    # adj = load_adj_raw(args['dataset'])

    node_num = features.size()[0]
    # class_num = labels.numpy().max() + 1
    class_num = n_classes

    # g = dgl.DGLGraph()
    # g.add_nodes(node_num)
    # adj = adj.tocoo()
    # g.add_edges(adj.row, adj.col)
    # features = features.cuda()
    # labels = labels.cuda()
    g = g.to(device)
    adj = adj.to(device)
    features = features.to(device)
    labels = labels.to(device)
    loss_func = nn.CrossEntropyLoss()

    in_feats = features.shape[-1]
    n_hidden = args['n_hidden']
    embedding_dim = [in_feats]
    embedding_dim += [n_hidden] * (args['n_layer'] - 2)
    embedding_dim.append(n_classes)

    if args['net'] == 'gin':
        # net_gcn = GINNet(args['embedding_dim'], g)
        net_gcn = GINNet(embedding_dim, g)
        pruning_gin.add_mask(net_gcn)
    elif args['net'] == 'gat':
        # net_gcn = GATNet(args['embedding_dim'], g)
        net_gcn = GATNet(embedding_dim, g)
        g.add_edges(list(range(node_num)), list(range(node_num)))
        pruning_gat.add_mask(net_gcn)
    else:
        assert False

    # net_gcn = net_gcn.cuda()
    net_gcn = net_gcn.to(device)
    net_gcn.load_state_dict(rewind_weight_mask)

    if args['net'] == 'gin':
        adj_spar, wei_spar = pruning_gin.print_sparsity(net_gcn)
    else:
        adj_spar, wei_spar = pruning_gat.print_sparsity(net_gcn)

    for name, param in net_gcn.named_parameters():
        if 'mask' in name:
            param.requires_grad = False

    optimizer = torch.optim.Adam(net_gcn.parameters(),
                                 lr=args['lr'],
                                 weight_decay=args['weight_decay'])
    best_val_acc = {'val_acc': 0, 'epoch': 0, 'test_acc': 0}

    # Record adj, wgt and feats
    adj_mask = net_gcn.adj_mask2_fixed
    if args['net'] == 'gin':
        wgt_0 = net_gcn.ginlayers[0].apply_func.mlp.linear.weight_mask_fixed.T
        wgt_1 = net_gcn.ginlayers[1].apply_func.mlp.linear.weight_mask_fixed.T
    elif args['net'] == 'gat':
        pass
    feats = []

    print('Wgt density:', utils.count_sparsity(wgt_0), utils.count_sparsity(wgt_1))
    print()

    for epoch in range(args['fix_epoch']):

        optimizer.zero_grad()
        output = net_gcn(g, features, 0, 0)
        loss = loss_func(output[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            net_gcn.eval()
            output = net_gcn(g, features, 0, 0)
            acc_val = f1_score(labels[idx_val].cpu().numpy(),
                               output[idx_val].cpu().numpy().argmax(axis=1),
                               average='micro')
            acc_test = f1_score(labels[idx_test].cpu().numpy(),
                                output[idx_test].cpu().numpy().argmax(axis=1),
                                average='micro')
            if acc_val > best_val_acc['val_acc']:
                best_val_acc['val_acc'] = acc_val
                best_val_acc['test_acc'] = acc_test
                best_val_acc['epoch'] = epoch

                feats = net_gcn.feats

        # print(
        #     "IMP[{}] (Fix Mask) Epoch:[{}] LOSS:[{:.4f}] Val:[{:.2f}] Test:[{:.2f}] | Final Val:[{:.2f}] Test:[{:.2f}] at Epoch:[{}]"
        #     .format(imp_num, epoch, args['fix_epoch'], loss, acc_val * 100, acc_test * 100,
        #             best_val_acc['val_acc'] * 100, best_val_acc['test_acc'] * 100,
        #             best_val_acc['epoch']))

    print(
        "syd final: [{},{}] IMP[{}] (Fix Mask) Final Val:[{:.2f}] Test:[{:.2f}] at Epoch:[{}] | Adj:[{:.2f}%] Wei:[{:.2f}%]"
        .format(args['dataset'], args['net'], imp_num, best_val_acc['val_acc'] * 100,
                best_val_acc['test_acc'] * 100, best_val_acc['epoch'], adj_spar, wei_spar))

    # Without pruning
    wgt_0_wo_pruning = torch.ones_like(wgt_0)
    wgt_1_wo_pruning = torch.ones_like(wgt_1)

    feat_tmp = torch.mm(feats[0], wgt_0_wo_pruning)
    feat_layer_1 = torch.mm(adj, feat_tmp)

    num_op_ax_w_0_wo_pruning = utils.op_count_ax_w(adj, feats[0], wgt_0_wo_pruning)
    num_op_ax_w_1_wo_pruning = utils.op_count_ax_w(adj, feat_layer_1, wgt_1_wo_pruning)

    num_op_a_xw_0_wo_pruning = utils.op_count_a_xw(adj, feats[0], wgt_0_wo_pruning)
    num_op_a_xw_1_wo_pruning = utils.op_count_a_xw(adj, feat_layer_1, wgt_1_wo_pruning)

    num_op_norm_layer_0_wo_pruning = round(num_op_a_xw_0_wo_pruning / num_op_ax_w_0_wo_pruning, 3)
    num_op_norm_layer_1_wo_pruning = round(num_op_a_xw_1_wo_pruning / num_op_ax_w_1_wo_pruning, 3)

    # With pruning
    # adj_pruning = torch.mul(adj, adj_mask)
    adj_pruning = adj  # No graph structure pruning
    num_op_ax_w_0 = utils.op_count_ax_w(adj_pruning, feats[0], wgt_0)
    num_op_ax_w_1 = utils.op_count_ax_w(adj_pruning, feats[1], wgt_1) * 0.2

    num_op_a_xw_0 = utils.op_count_a_xw(adj_pruning, feats[0], wgt_0)
    num_op_a_xw_1 = utils.op_count_a_xw(adj_pruning, feats[1], wgt_1) * 0.2

    # num_op_norm_layer_0 = round(num_op_a_xw_0 / num_op_ax_w_0, 3)
    # num_op_norm_layer_1 = round(num_op_a_xw_1 / num_op_ax_w_1, 3)

    num_op_norm_ax_w_layer_0 = round(num_op_ax_w_0 / num_op_ax_w_0_wo_pruning, 3)
    num_op_norm_ax_w_layer_1 = round(num_op_ax_w_1 / num_op_ax_w_1_wo_pruning, 3)

    num_op_norm_a_xw_layer_0 = round(num_op_a_xw_0 / num_op_ax_w_0_wo_pruning, 3)
    num_op_norm_a_xw_layer_1 = round(num_op_a_xw_1 / num_op_ax_w_1_wo_pruning, 3)

    print(
        'layer_0_wo: {:.3f}, layer_1_wo: {:.3f}\nlayer_0_ax_w: {:.3f}, layer_1_ax_w: {:.3f}\nlayer_0_a_xw: {:.3f}, layer_1_a_xw: {:.3f}'
        .format(num_op_norm_layer_0_wo_pruning, num_op_norm_layer_1_wo_pruning,
                num_op_norm_ax_w_layer_0, num_op_norm_ax_w_layer_1, num_op_norm_a_xw_layer_0,
                num_op_norm_a_xw_layer_1))

    # with open(log_file, 'wt') as f:
    #     print(
    #         'layer_0_wo: {:.3f}, layer_1_wo: {:.3f}\nlayer_0_ax_w: {:.3f}, layer_1_ax_w: {:.3f}\nlayer_0_a_xw: {:.3f}, layer_1_a_xw: {:.3f}'
    #         .format(num_op_norm_layer_0_wo_pruning, num_op_norm_layer_1_wo_pruning,
    #                 num_op_norm_ax_w_layer_0, num_op_norm_ax_w_layer_1, num_op_norm_a_xw_layer_0,
    #                 num_op_norm_a_xw_layer_1),
    #         file=f)

    num_op_norm_wo_pruning = round((num_op_a_xw_0_wo_pruning + num_op_a_xw_1_wo_pruning) /
                                   (num_op_ax_w_0_wo_pruning + num_op_ax_w_1_wo_pruning), 3)
    num_op_norm_ax_w_pruning = round(
        (num_op_ax_w_0 + num_op_ax_w_1) / (num_op_ax_w_0_wo_pruning + num_op_ax_w_1_wo_pruning), 3)
    num_op_norm_a_xw_pruning = round(
        (num_op_a_xw_0 + num_op_a_xw_1) / (num_op_ax_w_0_wo_pruning + num_op_ax_w_1_wo_pruning), 3)

    print()
    print('layer_wo: {:.3f}\nlayer_ax_w: {:.3f}\nlayer_a_xw: {:.3f}'.format(
        num_op_norm_wo_pruning, num_op_norm_ax_w_pruning, num_op_norm_a_xw_pruning))

    # with open(log_file, 'wt') as f:
    #     print('layer_wo: {:.3f}\nlayer_ax_w: {:.3f}\nlayer_a_xw: {:.3f}'.format(
    #         num_op_norm_wo_pruning, num_op_norm_ax_w_pruning, num_op_norm_a_xw_pruning),
    #           file=f)

    return best_val_acc['val_acc'], best_val_acc['test_acc'], best_val_acc[
        'epoch'], adj_spar, wei_spar


# def run_get_mask(args, imp_num, rewind_weight_mask=None):
def run_get_mask(args,
                 imp_num,
                 g,
                 features,
                 labels,
                 idx_train,
                 idx_val,
                 idx_test,
                 n_classes,
                 rewind_weight_mask=None):

    if args['gpu'] < 0:
        cuda = False
    else:
        cuda = True
        gpu_id = args['gpu']
    device = th.device("cuda:" + str(gpu_id) if cuda else "cpu")

    pruning.setup_seed(args['seed'])
    # adj, features, labels, idx_train, idx_val, idx_test = load_data(args['dataset'])
    # adj = load_adj_raw(args['dataset'])

    node_num = features.size()[0]
    # class_num = labels.numpy().max() + 1
    class_num = n_classes

    # g = dgl.DGLGraph()
    # g.add_nodes(node_num)
    # adj = adj.tocoo()

    # g.add_edges(adj.row, adj.col)
    # features = features.cuda()
    # labels = labels.cuda()
    g = g.to(device)
    features = features.to(device)
    labels = labels.to(device)

    loss_func = nn.CrossEntropyLoss()

    in_feats = features.shape[-1]
    n_hidden = args['n_hidden']
    embedding_dim = [in_feats]
    embedding_dim += [n_hidden] * (args['n_layer'] - 2)
    embedding_dim.append(n_classes)

    if args['net'] == 'gin':
        # net_gcn = GINNet(args['embedding_dim'], g)
        net_gcn = GINNet(embedding_dim, g)
        pruning_gin.add_mask(net_gcn)
    elif args['net'] == 'gat':
        # net_gcn = GATNet(args['embedding_dim'], g)
        net_gcn = GATNet(embedding_dim, g)
        g.add_edges(list(range(node_num)), list(range(node_num)))
        pruning_gat.add_mask(net_gcn)
    else:
        assert False

    # net_gcn = net_gcn.cuda()
    net_gcn = net_gcn.to(device)

    if rewind_weight_mask:
        net_gcn.load_state_dict(rewind_weight_mask)

    if args['net'] == 'gin':
        pruning_gin.add_trainable_mask_noise(net_gcn, c=1e-5)
        adj_spar, wei_spar = pruning_gin.print_sparsity(net_gcn)
    else:
        pruning_gat.add_trainable_mask_noise(net_gcn, c=1e-5)
        adj_spar, wei_spar = pruning_gat.print_sparsity(net_gcn)

    optimizer = torch.optim.Adam(net_gcn.parameters(),
                                 lr=args['lr'],
                                 weight_decay=args['weight_decay'])
    best_val_acc = {'val_acc': 0, 'epoch': 0, 'test_acc': 0}

    rewind_weight = copy.deepcopy(net_gcn.state_dict())

    for epoch in range(args['mask_epoch']):

        optimizer.zero_grad()
        output = net_gcn(g, features, 0, 0)
        loss = loss_func(output[idx_train], labels[idx_train])
        loss.backward()
        if args['net'] == 'gin':
            pruning_gin.subgradient_update_mask(net_gcn, args)  # l1 norm
        else:
            pruning_gat.subgradient_update_mask(net_gcn, args)  # l1 norm

        optimizer.step()
        with torch.no_grad():
            net_gcn.eval()
            output = net_gcn(g, features, 0, 0)
            acc_val = f1_score(labels[idx_val].cpu().numpy(),
                               output[idx_val].cpu().numpy().argmax(axis=1),
                               average='micro')
            acc_test = f1_score(labels[idx_test].cpu().numpy(),
                                output[idx_test].cpu().numpy().argmax(axis=1),
                                average='micro')
            if acc_val > best_val_acc['val_acc']:
                best_val_acc['val_acc'] = acc_val
                best_val_acc['test_acc'] = acc_test
                best_val_acc['epoch'] = epoch

                if args['net'] == 'gin':
                    rewind_weight, adj_spar, wei_spar = pruning_gin.get_final_mask_epoch(
                        net_gcn, rewind_weight, args)
                else:
                    rewind_weight, adj_spar, wei_spar = pruning_gat.get_final_mask_epoch(
                        net_gcn, rewind_weight, args)

        # print(
        #     "IMP[{}] (Get Mask) Epoch:[{}/{}] LOSS:[{:.4f}] Val:[{:.2f}] Test:[{:.2f}] | Final Val:[{:.2f}] Test:[{:.2f}] at Epoch:[{}] | Adj:[{:.2f}%] Wei:[{:.2f}%]"
        #     .format(imp_num, epoch, args['mask_epoch'], loss, acc_val * 100, acc_test * 100,
        #             best_val_acc['val_acc'] * 100, best_val_acc['test_acc'] * 100,
        #             best_val_acc['epoch'], adj_spar, wei_spar))

    return rewind_weight


def parser_loader():
    parser = argparse.ArgumentParser(description='GLT')
    ###### Unify pruning settings #######
    parser.add_argument('--s1',
                        type=float,
                        default=0.0001,
                        help='scale sparse rate (default: 0.0001)')
    parser.add_argument('--s2',
                        type=float,
                        default=0.0001,
                        help='scale sparse rate (default: 0.0001)')
    parser.add_argument('--mask_epoch', type=int, default=300)
    parser.add_argument('--fix_epoch', type=int, default=300)
    parser.add_argument('--pruning_percent_wei', type=float, default=0.1)
    parser.add_argument('--pruning_percent_adj', type=float, default=0.1)
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--embedding-dim', nargs='+', type=int, default=[3703, 16, 6])
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--net', type=str, default='')
    parser.add_argument('--seed', type=int, default=666)
    return parser


if __name__ == "__main__":
    print('>> Task start time: ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    Task_time_start = time.perf_counter()

    parser = parser_loader()
    args = vars(parser.parse_args())
    print(args)

    args['net'] = 'gin'
    # args['net'] = 'gat'

    # args['dataset'] = 'cora'
    args['dataset'] = 'citeseer'
    #  args['dataset'] = 'pubmed'
    # args['dataset'] = 'reddit'
    # args['dataset'] = 'arxiv'
    # args['dataset'] = 'amazon_comp'

    args['total_epoch'] = 200
    args['gpu'] = 0
    args['n_hidden'] = 512
    args['n_layer'] = 3
    args['pruning_percent_wei'] = 0.05
    args['pruning_percent_adj'] = 0
    args['init_soft_mask_type'] = 'all_one'

    if args['net'] == 'gin':
        if args['dataset'] == 'cora':
            args['lr'] = 0.008
            args['weight_decay'] = 8e-5
            args['s1'] = 1e-3
            args['s2'] = 1e-3
        elif args['dataset'] == 'citeseer':
            args['lr'] = 0.01
            args['weight_decay'] = 5e-4
            args['s1'] = 1e-5
            args['s2'] = 1e-5
        elif args['dataset'] == 'pubmed':
            args['lr'] = 0.01
            args['weight_decay'] = 5e-4
            args['s1'] = 1e-5
            args['s2'] = 1e-5
    elif args['net'] == 'gat':
        if args['dataset'] == 'cora':
            args['lr'] = 0.008
            args['weight_decay'] = 8e-5
            args['s1'] = 1e-3
            args['s2'] = 1e-3
        elif args['dataset'] == 'citeseer':
            args['lr'] = 0.01
            args['weight_decay'] = 5e-4
            args['s1'] = 1e-7
            args['s2'] = 1e-3
        elif args['dataset'] == 'pubmed':
            args['lr'] = 0.01
            args['weight_decay'] = 5e-4
            args['s1'] = 1e-2
            args['s2'] = 1e-2

    # seed_dict = {
    #     'cora': 2377,
    #     'citeseer': 4428,
    #     'pubmed': 3333,
    #     'arxiv': 8956,
    #     'reddit': 9781,
    #     'amazon_comp': 8763
    # }
    # seed = seed_dict[args['dataset']]

    rewind_weight = None

    log_name = 'acc_' + args['net'] + '_' + args['dataset'] + '_' + time.strftime(
        "%m%d_%H%M", time.localtime()) + '.txt'
    log_file = '../results/accuracy/' + log_name
    # print(log_file)
    sys.stdout = logger.Logger(log_file, sys.stdout)

    adj, features, labels, idx_train, idx_val, idx_test, n_classes, g = utils.load_dataset(
        args['dataset'])

    for imp in range(100):

        # rewind_weight = run_get_mask(args, imp, rewind_weight)
        rewind_weight = run_get_mask(args, imp, g, features, labels, idx_train, idx_val, idx_test,
                                     n_classes, rewind_weight)
        # run_fix_mask(args, imp, rewind_weight)
        best_acc_val, final_acc_test, final_epoch_list, adj_spar, wei_spar = run_fix_mask(
            args, imp, rewind_weight, g, adj, features, labels, idx_train, idx_val, idx_test,
            n_classes)

        print("=" * 120)
        print(
            "syd : Sparsity:[{}], Best Val:[{:.2f}] at epoch:[{}] | Final Test Acc:[{:.2f}] Adj:[{:.2f}%] Wei:[{:.2f}%]"
            .format(imp + 1, best_acc_val * 100, final_epoch_list, final_acc_test * 100, adj_spar,
                    wei_spar))
        print("=" * 120)

    print('\n>> Task {:s} execution time: {}'.format(
        args['dataset'], utils.time_format(time.perf_counter() - Task_time_start)))
