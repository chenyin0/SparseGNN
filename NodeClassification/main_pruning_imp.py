# import os
# import random
import argparse

import torch
import torch as th
import torch.nn as nn
import numpy as np

import net as net
from gnns.sage_net import GraphSAGE
# from utils import load_data
from sklearn.metrics import f1_score
# import pdb
import pruning
import pruning_sage
import copy
# from scipy.sparse import coo_matrix
import warnings
import utils
import time
import logger
import sys
from graph_pruning import GraphPruning
import random
from gpu_mem_track import MemTracker

# warnings.filterwarnings('ignore')


def run_base(args, adj, features, labels, idx_train, idx_val, idx_test, n_classes, edge_index):
    # gpu_tracker = MemTracker()
    # gpu_tracker.track()

    if args['gpu'] < 0:
        cuda = False
    else:
        cuda = True
        gpu_id = args['gpu']
    device = th.device("cuda:" + str(gpu_id) if cuda else "cpu")

    # adj = adj.cuda()
    # features = features.cuda()
    # labels = labels.cuda()
    # adj = adj.to(device)
    features = features.to(device)
    labels = labels.to(device)
    # edge_index = edge_index.to(device)

    if args['net'] == 'gcn':
        adj = adj.to(device)
    elif args['net'] == 'graphsage':
        edge_index = edge_index.to(device)

    loss_func = nn.CrossEntropyLoss()

    in_feats = features.shape[-1]
    n_hidden = args['n_hidden']
    embedding_dim = [in_feats]
    embedding_dim += [n_hidden] * (args['n_layer'] - 2)
    embedding_dim.append(n_classes)

    # net_gcn = net.net_gcn(embedding_dim=args['embedding_dim'], adj=adj)
    # net_gcn = net.net_gcn(embedding_dim=embedding_dim, adj=adj)
    if args['net'] == 'gcn':
        net_gcn = net.net_gcn(embedding_dim=embedding_dim, adj=adj)
        pruning.add_mask(net_gcn)
    elif args['net'] == 'graphsage':
        net_gcn = GraphSAGE(embedding_dim=embedding_dim, adj=adj)
        pruning_sage.add_mask(net_gcn)
    # pruning.add_mask(net_gcn)
    # net_gcn = net_gcn.cuda()
    # gpu_tracker.track()
    net_gcn = net_gcn.to(device)
    # net_gcn.load_state_dict(rewind_weight_mask)

    # gpu_tracker.track()

    # adj_spar, wei_spar = pruning.print_sparsity(net_gcn)
    if args['net'] == 'gcn':
        adj_spar, wei_spar, feat_spar = pruning.print_sparsity(net_gcn)
        # gpu_tracker.track()
    elif args['net'] == 'graphsage':
        adj_spar, wei_spar, feat_spar = pruning_sage.print_sparsity(net_gcn)

    # weight = net_gcn.net_layer[0].weight_mask_fixed
    # print(weight)

    for name, param in net_gcn.named_parameters():
        if 'mask' in name:
            param.requires_grad = False

    optimizer = torch.optim.Adam(net_gcn.parameters(),
                                 lr=args['lr'],
                                 weight_decay=args['weight_decay'])
    acc_test = 0.0
    best_val_acc = {'val_acc': 0, 'epoch': 0, 'test_acc': 0}

    # Record adj, wgt and feats
    adj_mask = net_gcn.adj_mask2_fixed
    if args['net'] == 'gcn':
        wgt_0 = net_gcn.net_layer[0].weight_mask_fixed.T
        wgt_1 = net_gcn.net_layer[1].weight_mask_fixed.T
    elif args['net'] == 'graphsage':
        wgt_0 = net_gcn.net_layer[0].lin_l.weight_mask_fixed.T
        wgt_1 = net_gcn.net_layer[1].weight_mask_fixed.T
    feats = []

    print('Wgt density:', utils.count_sparsity(wgt_0), utils.count_sparsity(wgt_1))
    print()

    # with open(log_file, 'wt') as f:
    #     print('Wgt density:', utils.count_sparsity(wgt_0), utils.count_sparsity(wgt_1), file=f)

    for epoch in range(args['total_epoch']):
        optimizer.zero_grad()
        # output = net_gcn(features, adj)
        if args['net'] == 'gcn':
            output = net_gcn(features, adj)
        elif args['net'] == 'graphsage':
            output = net_gcn(features, edge_index)
        loss = loss_func(output[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            if args['net'] == 'gcn':
                output = net_gcn(features, adj, val_test=True)
                # output = net_gcn(features, adj)
            elif args['net'] == 'graphsage':
                output = net_gcn(features, edge_index)
            acc_val = f1_score(labels[idx_val].cpu().numpy(),
                               output[idx_val].cpu().numpy().argmax(axis=1),
                               average='micro')
            acc_test = f1_score(labels[idx_test].cpu().numpy(),
                                output[idx_test].cpu().numpy().argmax(axis=1),
                                average='micro')
            # print(acc_val)
            if acc_val > best_val_acc['val_acc']:
                # print()
                # print(acc_val, best_val_acc['val_acc'])

                best_val_acc['val_acc'] = acc_val
                best_val_acc['test_acc'] = acc_test
                best_val_acc['epoch'] = epoch

                feats = net_gcn.feats

                # print()
                # for i in range(len(feats)):
                #     print(i, utils.count_sparsity(feats[i]))

                # feat_total = output.numel()
                # zeros = torch.zeros_like(output)
                # ones = torch.ones_like(output)
                # feat_nonzero_norm = torch.where(output != 0, ones, zeros)
                # feat_nonzero = feat_nonzero_norm.sum().item()
                # feat_dense = feat_nonzero * 100 / feat_total
                # print('feat_dense: ', feat_dense)

        # print(
        #     "(Fix Mask) Epoch:[{}] Val:[{:.2f}] Test:[{:.2f}] | Final Val:[{:.2f}] Test:[{:.2f}] at Epoch:[{}]"
        #     .format(epoch, acc_val * 100, acc_test * 100, best_val_acc['val_acc'] * 100,
        #             best_val_acc['test_acc'] * 100, best_val_acc['epoch']))

    print('Feat density:', utils.count_sparsity(feats[0]), utils.count_sparsity(feats[1]))
    print()

    time_cost = net_gcn.mm_time

    return best_val_acc['val_acc'], best_val_acc['test_acc'], best_val_acc[
        'epoch'], adj_spar, wei_spar, feat_spar, time_cost


# def run_fix_mask(args, seed, rewind_weight_mask):
def run_fix_mask(args, seed, rewind_weight_mask, adj, features, labels, idx_train, idx_val,
                 idx_test, n_classes, edge_index):

    if args['gpu'] < 0:
        cuda = False
    else:
        cuda = True
        gpu_id = args['gpu']
    device = th.device("cuda:" + str(gpu_id) if cuda else "cpu")

    pruning.setup_seed(seed)

    # # adj, features, labels, idx_train, idx_val, idx_test = load_data(args['dataset'])
    # adj, features, labels, idx_train, idx_val, idx_test, n_classes = utils.load_dataset(
    #     args['dataset'])

    # node_num = features.size()[0]
    # class_num = labels.numpy().max() + 1

    # adj = adj.cuda()
    # features = features.cuda()
    # labels = labels.cuda()
    adj = adj.to(device)
    features = features.to(device)
    labels = labels.to(device)
    edge_index = edge_index.to(device)
    loss_func = nn.CrossEntropyLoss()

    in_feats = features.shape[-1]
    n_hidden = args['n_hidden']
    embedding_dim = [in_feats]
    embedding_dim += [n_hidden] * (args['n_layer'] - 2)
    embedding_dim.append(n_classes)

    # net_gcn = net.net_gcn(embedding_dim=args['embedding_dim'], adj=adj)
    # net_gcn = net.net_gcn(embedding_dim=embedding_dim, adj=adj)
    if args['net'] == 'gcn':
        net_gcn = net.net_gcn(embedding_dim=embedding_dim, adj=adj)
        pruning.add_mask(net_gcn)
    elif args['net'] == 'graphsage':
        net_gcn = GraphSAGE(embedding_dim=embedding_dim, adj=adj)
        pruning_sage.add_mask(net_gcn)
    # pruning.add_mask(net_gcn)
    # net_gcn = net_gcn.cuda()
    net_gcn = net_gcn.to(device)
    net_gcn.load_state_dict(rewind_weight_mask)

    # module = net_gcn.net_layer[0]
    # print(list(module.named_parameters()))
    # print(list(module.named_buffers()))
    # print(module._forward_pre_hooks)

    # adj_spar, wei_spar = pruning.print_sparsity(net_gcn)
    if args['net'] == 'gcn':
        adj_spar, wei_spar, feat_spar = pruning.print_sparsity(net_gcn)
    elif args['net'] == 'graphsage':
        adj_spar, wei_spar, feat_spar = pruning_sage.print_sparsity(net_gcn)

    # weight = net_gcn.net_layer[0].weight_mask_fixed
    # print(weight)

    for name, param in net_gcn.named_parameters():
        if 'mask' in name:
            param.requires_grad = False

    optimizer = torch.optim.Adam(net_gcn.parameters(),
                                 lr=args['lr'],
                                 weight_decay=args['weight_decay'])
    acc_test = 0.0
    best_val_acc = {'val_acc': 0, 'epoch': 0, 'test_acc': 0}

    # Record adj, wgt and feats
    adj_mask = net_gcn.adj_mask2_fixed
    if args['net'] == 'gcn':
        wgt_0 = net_gcn.net_layer[0].weight_mask_fixed.T
        wgt_1 = net_gcn.net_layer[1].weight_mask_fixed.T
    elif args['net'] == 'graphsage':
        wgt_0 = net_gcn.net_layer[0].lin_l.weight_mask_fixed.T
        wgt_1 = net_gcn.net_layer[1].weight_mask_fixed.T
    feats = []

    print('Wgt density:', utils.count_sparsity(wgt_0), utils.count_sparsity(wgt_1))
    print()

    # with open(log_file, 'wt') as f:
    #     print('Wgt density:', utils.count_sparsity(wgt_0), utils.count_sparsity(wgt_1), file=f)

    for epoch in range(args['total_epoch']):

        optimizer.zero_grad()
        # output = net_gcn(features, adj)
        if args['net'] == 'gcn':
            output = net_gcn(features, adj)
        elif args['net'] == 'graphsage':
            output = net_gcn(features, edge_index)
        loss = loss_func(output[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            if args['net'] == 'gcn':
                output = net_gcn(features, adj, val_test=True)
                # output = net_gcn(features, adj)
            elif args['net'] == 'graphsage':
                output = net_gcn(features, edge_index)
            acc_val = f1_score(labels[idx_val].cpu().numpy(),
                               output[idx_val].cpu().numpy().argmax(axis=1),
                               average='micro')
            acc_test = f1_score(labels[idx_test].cpu().numpy(),
                                output[idx_test].cpu().numpy().argmax(axis=1),
                                average='micro')
            # print(acc_val)
            if acc_val > best_val_acc['val_acc']:
                # print()
                # print(acc_val, best_val_acc['val_acc'])

                best_val_acc['val_acc'] = acc_val
                best_val_acc['test_acc'] = acc_test
                best_val_acc['epoch'] = epoch

                feats = net_gcn.feats

                # print()
                # for i in range(len(feats)):
                #     print(i, utils.count_sparsity(feats[i]))

                # feat_total = output.numel()
                # zeros = torch.zeros_like(output)
                # ones = torch.ones_like(output)
                # feat_nonzero_norm = torch.where(output != 0, ones, zeros)
                # feat_nonzero = feat_nonzero_norm.sum().item()
                # feat_dense = feat_nonzero * 100 / feat_total
                # print('feat_dense: ', feat_dense)

        # print(
        #     "(Fix Mask) Epoch:[{}] Val:[{:.2f}] Test:[{:.2f}] | Final Val:[{:.2f}] Test:[{:.2f}] at Epoch:[{}]"
        #     .format(epoch, acc_val * 100, acc_test * 100, best_val_acc['val_acc'] * 100,
        #             best_val_acc['test_acc'] * 100, best_val_acc['epoch']))

    print('Feat density:', utils.count_sparsity(feats[0]), utils.count_sparsity(feats[1]))
    print()

    # with open(log_file, 'wt') as f:
    #     print('Feat density:',
    #           utils.count_sparsity(feats[0]),
    #           utils.count_sparsity(feats[1]),
    #           file=f)

    # Without pruning
    wgt_0_wo_pruning = torch.ones_like(wgt_0)
    wgt_1_wo_pruning = torch.ones_like(wgt_1)

    # wgt_0_sp = wgt_0.to_sparse()
    # wgt_1_sp = wgt_1.to_sparse()

    feat_tmp = torch.mm(feats[0], wgt_0_wo_pruning)
    feat_layer_1 = torch.mm(adj, feat_tmp)

    # feat_0_sp = feats[0].to_sparse()
    # feat_layer_1_sp = feat_layer_1.to_sparse()
    # feat_1_sp = feats[1].to_sparse()

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
    # num_op_ax_w_1 = utils.op_count_ax_w(adj_pruning, feats[1], wgt_1) * 0.2
    num_op_ax_w_1 = utils.op_count_ax_w(adj_pruning, feats[1], wgt_1)

    num_op_a_xw_0 = utils.op_count_a_xw(adj_pruning, feats[0], wgt_0)
    # num_op_a_xw_1 = utils.op_count_a_xw(adj_pruning, feats[1], wgt_1) * 0.2
    num_op_a_xw_1 = utils.op_count_a_xw(adj_pruning, feats[1], wgt_1)

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
    if num_op_norm_ax_w_pruning != 0.0 and num_op_norm_a_xw_pruning != 0.0:
        print(
            'layer_wo: {:.3f}\nlayer_ax_w: {:.3f} reduction: {:.2f}\nlayer_a_xw: {:.3f} reduction: {:.2f}'
            .format(num_op_norm_wo_pruning, num_op_norm_ax_w_pruning,
                    num_op_norm_wo_pruning / num_op_norm_ax_w_pruning, num_op_norm_a_xw_pruning,
                    num_op_norm_wo_pruning / num_op_norm_a_xw_pruning))

    # with open(log_file, 'wt') as f:
    #     print('layer_wo: {:.3f}\nlayer_ax_w: {:.3f}\nlayer_a_xw: {:.3f}'.format(
    #         num_op_norm_wo_pruning, num_op_norm_ax_w_pruning, num_op_norm_a_xw_pruning),
    #           file=f)

    # # Print weight and bais distribution
    # module = net_gcn
    # f_0 = net_gcn.feats[0]
    # f_1 = net_gcn.feats[1]
    # # print(f_0.cpu().numpy().tolist())
    # print()
    # print(f_1.cpu().numpy().tolist())
    # w_0 = getattr(module.net_layer[0], 'weight')
    # w_1 = getattr(module.net_layer[1], 'weight')
    # # bias_0 = getattr(module.net_layer[0], 'bias')
    # # bias_1 = getattr(module.net_layer[1], 'bias')
    # utils.plot_val_distribution(f_0, 'f_0')
    # utils.plot_val_distribution(f_1, 'f_1')
    # utils.plot_val_distribution(w_0, 'w_0')
    # utils.plot_val_distribution(w_1, 'w_1')
    # # utils.plot_val_distribution(bias_0, 'b_0')
    # # utils.plot_val_distribution(bias_1, 'b_1')

    time_cost = net_gcn.mm_time

    return best_val_acc['val_acc'], best_val_acc['test_acc'], best_val_acc[
        'epoch'], adj_spar, wei_spar, feat_spar, time_cost


# def run_get_mask(args, seed, imp_num, rewind_weight_mask=None):
def run_get_mask(args,
                 seed,
                 imp_num,
                 adj,
                 features,
                 labels,
                 idx_train,
                 idx_val,
                 idx_test,
                 n_classes,
                 edge_index,
                 rewind_weight_mask=None):

    if args['gpu'] < 0:
        cuda = False
    else:
        cuda = True
        gpu_id = args['gpu']
    device = th.device("cuda:" + str(gpu_id) if cuda else "cpu")

    pruning.setup_seed(seed)
    # # adj, features, labels, idx_train, idx_val, idx_test = load_data(args['dataset'])
    # adj, features, labels, idx_train, idx_val, idx_test, n_classes = utils.load_dataset(
    #     args['dataset'])

    # node_num = features.size()[0]
    # class_num = labels.numpy().max() + 1

    # adj = adj.cuda()
    # features = features.cuda()
    # labels = labels.cuda()
    adj = adj.to(device)
    features = features.to(device)
    labels = labels.to(device)
    edge_index = edge_index.to(device)
    loss_func = nn.CrossEntropyLoss()

    in_feats = features.shape[-1]
    n_hidden = args['n_hidden']
    embedding_dim = [in_feats]
    embedding_dim += [n_hidden] * (args['n_layer'] - 2)
    embedding_dim.append(n_classes)

    # net_gcn = net.net_gcn(embedding_dim=args['embedding_dim'], adj=adj)
    # net_gcn = net.net_gcn(embedding_dim=embedding_dim, adj=adj)
    if args['net'] == 'gcn':
        net_gcn = net.net_gcn(embedding_dim=embedding_dim, adj=adj)
        pruning.add_mask(net_gcn)
    elif args['net'] == 'graphsage':
        net_gcn = GraphSAGE(embedding_dim=embedding_dim, adj=adj)
        pruning_sage.add_mask(net_gcn)
    # pruning.add_mask(net_gcn)
    # net_gcn = net_gcn.cuda()
    net_gcn = net_gcn.to(device)

    if args['weight_dir']:
        print("load : {}".format(args['weight_dir']))
        encoder_weight = {}
        # cl_ckpt = torch.load(args['weight_dir'], map_location='cuda')
        cl_ckpt = torch.load(args['weight_dir'], map_location=device)
        encoder_weight['weight_orig_weight'] = cl_ckpt['gcn.fc.weight']
        ori_state_dict = net_gcn.net_layer[0].state_dict()
        ori_state_dict.update(encoder_weight)
        net_gcn.net_layer[0].load_state_dict(ori_state_dict)

    if rewind_weight_mask:
        net_gcn.load_state_dict(rewind_weight_mask)
        if not args['rewind_soft_mask'] or args['init_soft_mask_type'] == 'all_one':
            # pruning.soft_mask_init(net_gcn, args['init_soft_mask_type'], seed)
            if args['net'] == 'gcn':
                pruning.soft_mask_init(net_gcn, args['init_soft_mask_type'], seed)
            elif args['net'] == 'graphsage':
                pruning_sage.soft_mask_init(net_gcn, args['init_soft_mask_type'], seed)
        # # adj_spar, wei_spar = pruning.print_sparsity(net_gcn)
        # if args['net'] == 'gcn':
        #     adj_spar, wei_spar = pruning.print_sparsity(net_gcn)
        # elif args['net'] == 'graphsage':
        #     adj_spar, wei_spar = pruning_sage.print_sparsity(net_gcn)
    else:
        # pruning.soft_mask_init(net_gcn, args['init_soft_mask_type'], seed)
        if args['net'] == 'gcn':
            pruning.soft_mask_init(net_gcn, args['init_soft_mask_type'], seed)
        elif args['net'] == 'graphsage':
            pruning_sage.soft_mask_init(net_gcn, args['init_soft_mask_type'], seed)

    optimizer = torch.optim.Adam(net_gcn.parameters(),
                                 lr=args['lr'],
                                 weight_decay=args['weight_decay'])

    acc_test = 0.0
    best_val_acc = {'val_acc': 0, 'epoch': 0, 'test_acc': 0}
    rewind_weight = copy.deepcopy(net_gcn.state_dict())
    for epoch in range(args['total_epoch']):

        optimizer.zero_grad()
        # output = net_gcn(features, adj)
        if args['net'] == 'gcn':
            output = net_gcn(features, adj)
        elif args['net'] == 'graphsage':
            output = net_gcn(features, edge_index)

        loss = loss_func(output[idx_train], labels[idx_train])
        loss.backward()
        # pruning.subgradient_update_mask(net_gcn, args)  # l1 norm
        if args['net'] == 'gcn':
            pruning.subgradient_update_mask(net_gcn, args)  # l1 norm
        elif args['net'] == 'graphsage':
            pruning_sage.subgradient_update_mask(net_gcn, args)  # l1 norm
        optimizer.step()
        with torch.no_grad():
            # output = net_gcn(features, adj, val_test=True)
            if args['net'] == 'gcn':
                output = net_gcn(features, adj, val_test=True)
            elif args['net'] == 'graphsage':
                output = net_gcn(features, edge_index)
            acc_val = f1_score(labels[idx_val].cpu().numpy(),
                               output[idx_val].cpu().numpy().argmax(axis=1),
                               average='micro')
            acc_test = f1_score(labels[idx_test].cpu().numpy(),
                                output[idx_test].cpu().numpy().argmax(axis=1),
                                average='micro')
            if acc_val > best_val_acc['val_acc']:
                best_val_acc['test_acc'] = acc_test
                best_val_acc['val_acc'] = acc_val
                best_val_acc['epoch'] = epoch
                # best_epoch_mask = pruning.get_final_mask_epoch(
                #     net_gcn,
                #     adj_percent=args['pruning_percent_adj'],
                #     wei_percent=args['pruning_percent_wei'])
                if args['net'] == 'gcn':
                    best_epoch_mask = pruning.get_final_mask_epoch(
                        net_gcn,
                        adj_percent=args['pruning_percent_adj'],
                        wei_percent=args['pruning_percent_wei'],
                        feat_percent=args['pruning_percent_feat'])
                elif args['net'] == 'graphsage':
                    best_epoch_mask = pruning_sage.get_final_mask_epoch(
                        net_gcn,
                        adj_percent=args['pruning_percent_adj'],
                        wei_percent=args['pruning_percent_wei'],
                        feat_percent=args['pruning_percent_feat'])

            # print(
            #     "(Get Mask) Epoch:[{}] Val:[{:.2f}] Test:[{:.2f}] | Best Val:[{:.2f}] Test:[{:.2f}] at Epoch:[{}]"
            #     .format(epoch, acc_val * 100, acc_test * 100, best_val_acc['val_acc'] * 100,
            #             best_val_acc['test_acc'] * 100, best_val_acc['epoch']))

    # print('Mat_time: {:.4f}'.format(net_gcn.mm_time))

    time_cost = net_gcn.mm_time

    return best_epoch_mask, rewind_weight, time_cost, net_gcn


def parser_loader():
    parser = argparse.ArgumentParser(description='GLT')
    ###### Unify pruning settings #######
    parser.add_argument('--s1',
                        type=float,
                        default=0.0001,
                        help='[adj] scale sparse rate (default: 0.0001)')
    parser.add_argument('--s2',
                        type=float,
                        default=0.0001,
                        help='[weight] scale sparse rate (default: 0.0001)')
    parser.add_argument('--s3',
                        type=float,
                        default=0.0001,
                        help='[feature] scale sparse rate (default: 0.0001)')
    parser.add_argument('--total_epoch', type=int, default=300)
    parser.add_argument('--pruning_percent_wei', type=float, default=0.1)
    parser.add_argument('--pruning_percent_adj', type=float, default=0.1)
    parser.add_argument('--pruning_percent_feat', type=float, default=0.1)
    parser.add_argument('--weight_dir', type=str, default='')
    parser.add_argument('--rewind_soft_mask', action='store_true')
    parser.add_argument('--init_soft_mask_type',
                        type=str,
                        default='',
                        help='all_one, kaiming, normal, uniform')
    ###### Others settings #######
    parser.add_argument('--dataset', type=str, default='citeseer')
    parser.add_argument('--embedding-dim', nargs='+', type=int, default=[3703, 16, 6])
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--n-hidden', type=int, default=128)
    parser.add_argument('--n-layer', type=int, default=3)
    parser.add_argument("--gpu", type=int, default=-1, help="gpu")
    parser.add_argument('--net', type=str, default='')
    parser.add_argument('--graph-prune-ratio', type=float, default=0.5)
    return parser


if __name__ == "__main__":
    print('>> Task start time: ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    Task_time_start = time.perf_counter()

    parser = parser_loader()
    args = vars(parser.parse_args())
    # print(args)

    # from dgl.data import SBMMixtureDataset
    # dataset = SBMMixtureDataset(n_graphs=1, n_nodes=8396, n_communities=60, avg_deg=5.5)

    # args['net'] = 'gcn'
    # args['net'] = 'graphsage'

    # args['dataset'] = 'cora'
    # args['dataset'] = 'citeseer'
    # args['dataset'] = 'chameleon'
    # args['dataset'] = 'actor'
    # args['dataset'] = 'squirrel'
    # args['dataset'] = 'wikics'
    # args['dataset'] = 'pubmed'
    # args['dataset'] = 'reddit'
    # args['dataset'] = 'arxiv'
    # args['dataset'] = 'amazon_comp'

    # args['graph_prune_ratio'] = 0.8

    args['total_epoch'] = 200
    args['gpu'] = 0
    args['n_hidden'] = 512
    args['n_layer'] = 3
    # args['pruning_percent_wei'] = 0.5
    # args['pruning_percent_adj'] = 0.02
    args['init_soft_mask_type'] = 'all_one'

    if args['net'] == 'gcn':
        args['pruning_percent_wei'] = 0.5
        args['pruning_percent_feat'] = 0.3
    elif args['net'] == 'graphsage':
        args['pruning_percent_wei'] = 0.5
        args['pruning_percent_feat'] = 0.2

    # args['pruning_percent_wei'] = 0.1

    if args['dataset'] == 'cora':
        args['lr'] = 0.008
        args['weight_decay'] = 8e-5
        args['s1'] = 1e-2
        args['s2'] = 1e-2
        args['s3'] = 1e-2
    elif args['dataset'] == 'citeseer':
        args['lr'] = 0.01
        args['weight_decay'] = 5e-4
        args['s1'] = 1e-2
        args['s2'] = 1e-2
        args['s3'] = 1e-2
    elif args['dataset'] == 'pubmed':
        args['lr'] = 0.01
        args['weight_decay'] = 5e-4
        args['s1'] = 1e-6
        args['s2'] = 1e-3
        args['s3'] = 1e-6

    seed_dict = {
        'cora': 2377,
        'citeseer': 4428,
        'pubmed': 3333,
        'arxiv': 8956,
        'reddit': 9781,
        'amazon_comp': 8763,
        'SBM-10000': 6759,
        'aifb': 5896,
        'chameleon': 4869,
        'wikics': 7859,
        'squirrel': 6021,
        'actor': 2026
    }
    seed = seed_dict[args['dataset']]
    rewind_weight = None

    if args['gpu'] < 0:
        cuda = False
    else:
        cuda = True
        gpu_id = args['gpu']
    device = th.device("cuda:" + str(gpu_id) if cuda else "cpu")

    log_name = 'acc_' + args['net'] + '_' + args['dataset'] + '_' + time.strftime(
        "%m%d_%H%M", time.localtime()) + '.txt'
    log_file = '../results/accuracy/' + log_name
    # print(log_file)
    sys.stdout = logger.Logger(log_file, sys.stdout)
    print(args)

    adj, features, labels, idx_train, idx_val, idx_test, n_classes, g, edge_index = utils.load_dataset(
        args['dataset'])

    base_acc_val, base_acc_test, final_epoch_list, adj_spar, wei_spar, feat_spar, time_cost_base = run_base(
        args, adj, features, labels, idx_train, idx_val, idx_test, n_classes, edge_index)

    pruning_ratio = args['graph_prune_ratio']
    error_threshold = 1
    # error_threshold = 0.005

    # adj_pruned = GraphPruning.random(adj, pruning_ratio)
    # adj_pruned = GraphPruning.mlf_pruning(adj, pruning_ratio)
    adj_pruned = GraphPruning.edge_sim_pruning(g, adj, pruning_ratio)
    v_hasNgh = list(set(adj_pruned.indices()[0].tolist()))
    v_hasNgh.sort()
    nonzero_v_num = len(v_hasNgh)

    # nonzero_row = adj_pruned.indices()[0]
    # nonzero_row_uni = th.unique(nonzero_row)

    # print(utils.count_sparsity(features))

    adj_pruned_scale = adj_pruned.to_dense().byte()[v_hasNgh]
    adj_pruned_scale = adj_pruned_scale[:, v_hasNgh].to_sparse().float().to(device)

    print('Base acc:[{:.2f}], Base time:[{:.4f}]'.format(base_acc_test * 100, time_cost_base))

    time_pruning_total = 0
    for p in range(100):

        # final_mask_dict, rewind_weight = run_get_mask(args, seed, p, rewind_weight)
        # final_mask_dict, rewind_weight, time_pruning = run_get_mask(
        #     args, seed, p, adj, features, labels, idx_train, idx_val, idx_test, n_classes,
        #     edge_index, rewind_weight)
        final_mask_dict, rewind_weight, time_pruning, model = run_get_mask(
            args, seed, p, adj_pruned, features, labels, idx_train, idx_val, idx_test, n_classes,
            edge_index, rewind_weight)

        time_pruning_total += time_pruning

        # Dump SpMM time cost
        feat_pruned = model.feats[1]
        feat_pruned_scale = feat_pruned[v_hasNgh].to_sparse()

        t0 = time.time()
        aa = torch.sparse.mm(adj.to(device), feat_pruned.to_sparse())
        t_graph_full = time.time() - t0

        t0 = time.time()
        aa = torch.sparse.mm(adj_pruned_scale, feat_pruned_scale)
        t_graph_pruned = time.time() - t0
        ##

        if args['net'] == 'gcn':
            rewind_weight['adj_mask1_train'] = final_mask_dict['adj_mask']
            rewind_weight['adj_mask2_fixed'] = final_mask_dict['adj_mask']
            rewind_weight['feat_mask1_train'] = final_mask_dict['feat_mask']
            rewind_weight['feat_mask2_fixed'] = final_mask_dict['feat_mask']
            rewind_weight['net_layer.0.weight_mask_train'] = final_mask_dict['weight1_mask']
            rewind_weight['net_layer.0.weight_mask_fixed'] = final_mask_dict['weight1_mask']
            rewind_weight['net_layer.1.weight_mask_train'] = final_mask_dict['weight2_mask']
            rewind_weight['net_layer.1.weight_mask_fixed'] = final_mask_dict['weight2_mask']
        elif args['net'] == 'graphsage':
            rewind_weight['adj_mask1_train'] = final_mask_dict['adj_mask']
            rewind_weight['adj_mask2_fixed'] = final_mask_dict['adj_mask']
            rewind_weight['feat_mask1_train'] = final_mask_dict['feat_mask']
            rewind_weight['feat_mask2_fixed'] = final_mask_dict['feat_mask']
            rewind_weight['net_layer.0.lin_l.weight_mask_train'] = final_mask_dict['weight1_mask']
            rewind_weight['net_layer.0.lin_l.weight_mask_fixed'] = final_mask_dict['weight1_mask']
            # rewind_weight['net_layer.1.lin_l.weight_mask_train'] = final_mask_dict['weight2_mask']
            # rewind_weight['net_layer.1.lin_l.weight_mask_fixed'] = final_mask_dict['weight2_mask']
            rewind_weight['net_layer.1.weight_mask_train'] = final_mask_dict['weight2_mask']
            rewind_weight['net_layer.1.weight_mask_fixed'] = final_mask_dict['weight2_mask']

        # best_acc_val, final_acc_test, final_epoch_list, adj_spar, wei_spar = run_fix_mask(
        #     args, seed, rewind_weight)
        best_acc_val, final_acc_test, final_epoch_list, adj_spar, wei_spar, feat_spar, time_effect = run_fix_mask(
            args, seed, rewind_weight, adj, features, labels, idx_train, idx_val, idx_test,
            n_classes, edge_index)
        # best_acc_val, final_acc_test, final_epoch_list, adj_spar, wei_spar = run_fix_mask(
        #     args, seed, rewind_weight, adj_pruned, features, labels, idx_train, idx_val, idx_test,
        #     n_classes, edge_index)

        print("=" * 120)
        print(
            "syd : Sparsity:[{}], Best Val:[{:.2f}] at epoch:[{}] | Final Test Acc:[{:.2f}] Adj:[{:.2f}%] Wei:[{:.2f}%] Feat:[{:.2f}%]| Acc Drop:[{:.2f}]"
            .format(p + 1, best_acc_val * 100, final_epoch_list, final_acc_test * 100, adj_spar,
                    wei_spar, feat_spar, (base_acc_test - final_acc_test) * 100))
        print(
            'Base time:[{:.3f}], Pruning time:[{:.3f}], Effect time:[{:.3f}], Pruning cost:[{:.2f}]'
            .format(time_cost_base, time_pruning_total, time_effect,
                    (time_pruning_total + time_effect) / time_cost_base))
        # print('Pruning cost:[{:.2f}]'.format((time_pruning_total + time_effect) / time_cost_base))
        print(
            'Full graph pruning time:[{:.3f}], Subgraph pruning time:[{:.3f}], Reduction:[{:.3f}], Subgraph size:[{:.1f}%]'
            .format(t_graph_full * 1000, t_graph_pruned * 1000, t_graph_full / t_graph_pruned,
                    adj_pruned._nnz() / adj._nnz() * 100))
        print("=" * 120)

        # Accuracy threshold
        if (base_acc_val - final_acc_test) > error_threshold:
            print('>>> Acc declines over threshold >>>')
            break

        # Minimal sparsity threshold
        if args['net'] == 'gcn':
            # wgt_0 = model.net_layer[0].weight_mask_fixed.T
            wgt_1 = model.net_layer[1].weight_mask_fixed.T
        elif args['net'] == 'graphsage':
            # wgt_0 = model.net_layer[0].lin_l.weight_mask_fixed.T
            wgt_1 = model.net_layer[1].weight_mask_fixed.T

        sparsity_wgt_1 = utils.count_sparsity(wgt_1)
        if sparsity_wgt_1 <= 0.005:
            print('>>> Weight sparsity over threshold >>>')
            break

    print('\n>> Task {:s} execution time: {}'.format(
        args['dataset'], utils.time_format(time.perf_counter() - Task_time_start)))
