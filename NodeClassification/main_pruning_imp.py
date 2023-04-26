# import os
# import random
import argparse

import torch
import torch as th
import torch.nn as nn
import numpy as np

import net as net
# from utils import load_data
from sklearn.metrics import f1_score
# import pdb
import pruning
import copy
# from scipy.sparse import coo_matrix
import warnings
import utils
import time

warnings.filterwarnings('ignore')


# def run_fix_mask(args, seed, rewind_weight_mask):
def run_fix_mask(args, seed, rewind_weight_mask, adj, features, labels, idx_train, idx_val,
                 idx_test, n_classes):

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
    loss_func = nn.CrossEntropyLoss()

    in_feats = features.shape[-1]
    n_hidden = args['n_hidden']
    embedding_dim = [in_feats]
    embedding_dim += [n_hidden] * (args['n_layer'] - 2)
    embedding_dim.append(n_classes)

    # net_gcn = net.net_gcn(embedding_dim=args['embedding_dim'], adj=adj)
    net_gcn = net.net_gcn(embedding_dim=embedding_dim, adj=adj)
    pruning.add_mask(net_gcn)
    # net_gcn = net_gcn.cuda()
    net_gcn = net_gcn.to(device)
    net_gcn.load_state_dict(rewind_weight_mask)
    adj_spar, wei_spar = pruning.print_sparsity(net_gcn)

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
    wgt_0 = net_gcn.net_layer[0].weight_mask_fixed.T
    wgt_1 = net_gcn.net_layer[1].weight_mask_fixed.T
    feats = []

    print('Wgt density:', utils.count_sparsity(wgt_0), utils.count_sparsity(wgt_1))
    print()

    for epoch in range(args['total_epoch']):

        optimizer.zero_grad()
        # output = net_gcn(features, adj)
        output = net_gcn(features, adj)
        loss = loss_func(output[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            output = net_gcn(features, adj, val_test=True)
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
    adj_pruning = torch.mul(adj, adj_mask)
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

    return best_val_acc['val_acc'], best_val_acc['test_acc'], best_val_acc[
        'epoch'], adj_spar, wei_spar


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
    loss_func = nn.CrossEntropyLoss()

    in_feats = features.shape[-1]
    n_hidden = args['n_hidden']
    embedding_dim = [in_feats]
    embedding_dim += [n_hidden] * (args['n_layer'] - 2)
    embedding_dim.append(n_classes)

    # net_gcn = net.net_gcn(embedding_dim=args['embedding_dim'], adj=adj)
    net_gcn = net.net_gcn(embedding_dim=embedding_dim, adj=adj)
    pruning.add_mask(net_gcn)
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
            pruning.soft_mask_init(net_gcn, args['init_soft_mask_type'], seed)
        adj_spar, wei_spar = pruning.print_sparsity(net_gcn)
    else:
        pruning.soft_mask_init(net_gcn, args['init_soft_mask_type'], seed)

    optimizer = torch.optim.Adam(net_gcn.parameters(),
                                 lr=args['lr'],
                                 weight_decay=args['weight_decay'])

    acc_test = 0.0
    best_val_acc = {'val_acc': 0, 'epoch': 0, 'test_acc': 0}
    rewind_weight = copy.deepcopy(net_gcn.state_dict())
    for epoch in range(args['total_epoch']):

        optimizer.zero_grad()
        output = net_gcn(features, adj)
        loss = loss_func(output[idx_train], labels[idx_train])
        loss.backward()
        pruning.subgradient_update_mask(net_gcn, args)  # l1 norm
        optimizer.step()
        with torch.no_grad():
            output = net_gcn(features, adj, val_test=True)
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
                best_epoch_mask = pruning.get_final_mask_epoch(
                    net_gcn,
                    adj_percent=args['pruning_percent_adj'],
                    wei_percent=args['pruning_percent_wei'])

            # print(
            #     "(Get Mask) Epoch:[{}] Val:[{:.2f}] Test:[{:.2f}] | Best Val:[{:.2f}] Test:[{:.2f}] at Epoch:[{}]"
            #     .format(epoch, acc_val * 100, acc_test * 100, best_val_acc['val_acc'] * 100,
            #             best_val_acc['test_acc'] * 100, best_val_acc['epoch']))

    return best_epoch_mask, rewind_weight


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
    parser.add_argument('--total_epoch', type=int, default=300)
    parser.add_argument('--pruning_percent_wei', type=float, default=0.1)
    parser.add_argument('--pruning_percent_adj', type=float, default=0.1)
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
    return parser


if __name__ == "__main__":
    print('>> Task start time: ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    Task_time_start = time.perf_counter()

    parser = parser_loader()
    args = vars(parser.parse_args())
    print(args)

    # args['dataset'] = 'cora'
    # args['embedding_dim'] = [1433, 128, 7]
    # args['dataset'] = 'citeseer'
    # args['embedding_dim'] = [3703, 128, 6]
    # args['dataset'] = 'pubmed'
    # args['embedding_dim'] = [3703, 128, 6]

    # args['dataset'] = 'cora'
    # args['dataset'] = 'citeseer'
    args['dataset'] = 'pubmed'
    # args['dataset'] = 'reddit'
    # args['dataset'] = 'amazon_comp'

    args['total_epoch'] = 20
    args['gpu'] = -1
    args['n_hidden'] = 128
    args['n_layer'] = 3
    args['lr'] = 0.008
    args['weight_decay'] = 8e-5
    args['pruning_percent_wei'] = 0.3
    args['pruning_percent_adj'] = 0
    args['s1'] = 1e-2
    args['s2'] = 1e-2
    args['init_soft_mask_type'] = 'all_one'

    seed_dict = {
        'cora': 2377,
        'citeseer': 4428,
        'pubmed': 3333,
        'arxiv:': 8956,
        'reddit': 9781,
        'amazon_comp': 8763
    }
    seed = seed_dict[args['dataset']]
    rewind_weight = None

    adj, features, labels, idx_train, idx_val, idx_test, n_classes = utils.load_dataset(
        args['dataset'])

    for p in range(20):

        # final_mask_dict, rewind_weight = run_get_mask(args, seed, p, rewind_weight)
        final_mask_dict, rewind_weight = run_get_mask(args, seed, p, adj, features, labels,
                                                      idx_train, idx_val, idx_test, n_classes,
                                                      rewind_weight)

        rewind_weight['adj_mask1_train'] = final_mask_dict['adj_mask']
        rewind_weight['adj_mask2_fixed'] = final_mask_dict['adj_mask']
        rewind_weight['net_layer.0.weight_mask_train'] = final_mask_dict['weight1_mask']
        rewind_weight['net_layer.0.weight_mask_fixed'] = final_mask_dict['weight1_mask']
        rewind_weight['net_layer.1.weight_mask_train'] = final_mask_dict['weight2_mask']
        rewind_weight['net_layer.1.weight_mask_fixed'] = final_mask_dict['weight2_mask']

        # best_acc_val, final_acc_test, final_epoch_list, adj_spar, wei_spar = run_fix_mask(
        #     args, seed, rewind_weight)
        best_acc_val, final_acc_test, final_epoch_list, adj_spar, wei_spar = run_fix_mask(
            args, seed, rewind_weight, adj, features, labels, idx_train, idx_val, idx_test,
            n_classes)

        print("=" * 120)
        print(
            "syd : Sparsity:[{}], Best Val:[{:.2f}] at epoch:[{}] | Final Test Acc:[{:.2f}] Adj:[{:.2f}%] Wei:[{:.2f}%]"
            .format(p + 1, best_acc_val * 100, final_epoch_list, final_acc_test * 100, adj_spar,
                    wei_spar))
        print("=" * 120)

    print('\n>> Task {:s} execution time: {}'.format(
        args.dataset, utils.time_format(time.perf_counter() - Task_time_start)))
