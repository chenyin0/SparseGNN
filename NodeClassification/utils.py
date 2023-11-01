import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
# from scipy.sparse.linalg.eigen.arpack import eigsh
from scipy.sparse.linalg import eigsh
import sys
# import pdb
import torch
# import metis
import gc
import time

# import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset, RedditDataset, AmazonCoBuyComputerDataset, SBMMixtureDataset
from dgl.data import AsNodePredDataset
from dgl.data import ChameleonDataset
from dgl.data import WikiCSDataset
from dgl.data import SquirrelDataset
from dgl.data import ActorDataset
from dgl.data.rdf import AIFBDataset
from ogb.nodeproppred import DglNodePropPredDataset
from dgl import AddSelfLoop

import copy
import matplotlib.pyplot as plt


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("../dataset/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("../dataset/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    # preprocess feature
    features = preprocess_features(features)
    features = torch.tensor(features, dtype=torch.float32)
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    # preprocess adj
    adj = sparse_mx_to_torch_sparse_tensor(adj).to_dense()
    # adj = torch_normalize_adj(adj)
    # adj2 = preprocess_adj(adj)
    # adj2 = sparse_mx_to_torch_sparse_tensor(adj2).to_dense()
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    _, l_num = labels.shape
    labels = torch.tensor((labels * range(l_num)).sum(axis=1), dtype=torch.int64)

    idx_test = test_idx_range.tolist()
    idx_train = list(range(len(y)))
    idx_val = list(range(len(y), len(y) + 500))

    return adj, features, labels, idx_train, idx_val, idx_test


def load_dataset(dataset_str):
    transform = (AddSelfLoop()
                 )  # by default, it will first remove self-loops to prevent duplication
    if dataset_str == 'cora':
        dataset = CoraGraphDataset(raw_dir='../dataset', transform=transform)
    elif dataset_str == 'citeseer':
        dataset = CiteseerGraphDataset(raw_dir='../dataset', transform=transform)
    elif dataset_str == 'pubmed':
        dataset = PubmedGraphDataset(raw_dir='../dataset', transform=transform)
    elif dataset_str == 'reddit':
        dataset = RedditDataset(raw_dir='../dataset', transform=transform)
    elif dataset_str == 'arxiv':
        dataset = AsNodePredDataset(DglNodePropPredDataset('ogbn-arxiv', root='../dataset'))
    # elif dataset_str == 'ogbn-mag':
    #     dataset = DglNodePropPredDataset('ogbn-mag', root='../dataset')
    # elif dataset_str == 'ogbn-products':
    #     dataset = AsNodePredDataset(DglNodePropPredDataset('ogbn-products', root='../dataset'))
    elif dataset_str == 'amazon_comp':
        dataset = AmazonCoBuyComputerDataset(raw_dir='../dataset', transform=transform)
    elif dataset_str == 'SBM-10000':
        # dataset = SBMMixtureDataset(n_graphs=16, n_nodes=8396, n_communities=60, avg_deg=5.5)
        dataset = SBMMixtureDataset(n_graphs=1, n_nodes=10000, n_communities=20, avg_deg=6)
    elif dataset_str == 'aifb':
        # dataset = SBMMixtureDataset(n_graphs=16, n_nodes=8396, n_communities=60, avg_deg=5.5)
        dataset = AIFBDataset()
    elif dataset_str == 'chameleon':
        dataset = ChameleonDataset(raw_dir='../dataset', transform=transform)
    elif dataset_str == 'wikics':
        dataset = WikiCSDataset(raw_dir='../dataset', transform=transform)
    elif dataset_str == 'squirrel':
        dataset = SquirrelDataset(raw_dir='../dataset', transform=transform)
    elif dataset_str == 'actor':
        dataset = ActorDataset(raw_dir='../dataset', transform=transform)
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset_str))

    g = dataset[0]
    adj = g.adj()
    # adj = adj.to_dense()
    adj = adj.coo()
    v_num = g.number_of_nodes()
    indices = torch.tensor([adj[0].tolist(), adj[1].tolist()])
    values = torch.ones(adj[0].shape[0])
    a1 = adj[0]
    a2 = adj[1].tolist()
    print(max(a1), max(a2))
    adj = torch.sparse_coo_tensor(indices=indices, values=values, size=[v_num, v_num])

    features = g.ndata['feat']
    labels = g.ndata['label']
    if dataset_str == 'amazon_comp' or dataset_str == 'chameleon' or dataset_str == 'wikics' or dataset_str == 'squirrel' or dataset_str == 'actor' or dataset_str == 'arxiv' or dataset_str == 'reddit':
        n_classes = dataset.num_classes
    else:
        n_classes = dataset.num_labels

    if dataset_str == 'chameleon' or dataset_str == 'squirrel' or dataset_str == 'actor':
        train_mask = g.ndata['train_mask'][:, 0]
        val_mask = g.ndata['val_mask'][:, 0]
        test_mask = g.ndata['test_mask'][:, 0]
    elif dataset_str == 'wikics':
        train_mask = g.ndata['train_mask'][:, 0]
        val_mask = g.ndata['val_mask'][:, 0]
        test_mask = g.ndata['test_mask']
    else:
        train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']

    # if dataset_str == 'chameleon':
    #     idx_train = [n for n in range(0, 1092)]
    #     idx_val = [n for n in range(1092, 1821)]
    #     idx_test = [n for n in range(1821, 2277)]
    # elif dataset_str == 'wikics':
    #     idx_train = [n for n in range(0, 1092)]
    #     idx_val = [n for n in range(1092, 1821)]
    #     idx_test = [n for n in range(1821, 2277)]
    # else:
    #     idx_train = np.nonzero(train_mask).squeeze().tolist()
    #     idx_val = np.nonzero(val_mask).squeeze().tolist()
    #     idx_test = np.nonzero(test_mask).squeeze().tolist()

    idx_train = np.nonzero(train_mask).squeeze().tolist()
    idx_val = np.nonzero(val_mask).squeeze().tolist()
    idx_test = np.nonzero(test_mask).squeeze().tolist()

    edges = g.all_edges()
    edge_src = edges[0]
    edge_dst = edges[1]
    edge_index = torch.stack([edge_src, edge_dst], dim=0)

    return adj, features, labels, idx_train, idx_val, idx_test, n_classes, g, edge_index


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    #return sparse_to_tuple(features)
    return features.todense()


def torch_normalize_adj(adj):
    # adj = adj + torch.eye(adj.shape[0]).cuda()
    device = adj.device
    adj = adj.type(torch.int8).cpu()
    # adj = adj + torch.eye(adj.shape[0]).to_sparse().to(device)

    ind = [i for i in range(adj.size()[0])]
    indices = torch.tensor([ind, ind])
    values = torch.ones(adj.size()[0])
    diag = torch.sparse_coo_tensor(indices=indices, values=values, dtype=torch.int8)
    adj = adj + diag

    adj = adj.to_dense()
    rowsum = adj.sum(1)
    adj = adj.to_sparse().float()
    d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    # d_mat_inv_sqrt = torch.diag(d_inv_sqrt).cuda()
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    # return adj.mm(d_mat_inv_sqrt).t().mm(d_mat_inv_sqrt)

    # adj_sp = adj.to_sparse()
    # d_mat_inv_sqrt_sp = d_mat_inv_sqrt.to_sparse()
    # return torch.sparse.mm(torch.sparse.mm(adj_sp, d_mat_inv_sqrt_sp).t(), d_mat_inv_sqrt_sp)

    d_mat_inv_sqrt_sp = d_mat_inv_sqrt.to_sparse()
    res = torch.sparse.mm(torch.sparse.mm(adj, d_mat_inv_sqrt_sp).t(), d_mat_inv_sqrt_sp).to(device)

    return res


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    #return sparse_to_tuple(adj_normalized)
    return adj_normalized


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k + 1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_adj_raw(dataset_str):

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("../dataset/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("../dataset/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    adj_raw = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    return adj_raw


# def partition(adj_raw, n):

#     node_num = adj_raw.shape[0]
#     adj_list = [[] for _ in range(node_num)]
#     for i, j in zip(adj_raw.row, adj_raw.col):
#         if i == j:
#             continue
#         adj_list[i].append(j)

#     _, ss_labels =  metis.part_graph(adj_list, nparts=n, seed=0)
#     ss_labels = torch.tensor(ss_labels, dtype=torch.int64)

#     return ss_labels


def op_count_ax_w(adj, feat, weight):
    num_op = op_count(adj, feat)
    ax = torch.mm(adj, feat)
    num_op += op_count(ax, weight)
    # axw = torch.mm(ax, weight)

    return num_op


def op_count_a_xw(adj, feat, weight):
    num_op = op_count(feat, weight)
    xw = torch.mm(feat, weight)
    num_op += op_count(adj, xw)
    # axw = torch.mm(ax, weight)

    return num_op


# def op_count(mat_a, mat_b):
#     mat_a = mat_a.clone().detach().cpu().numpy()
#     mat_b = mat_b.clone().detach().cpu().numpy().T
#     nonzero_ind_mat_a = []
#     nonzero_ind_mat_b = []

#     for i in range(mat_a.shape[0]):
#         ind = np.nonzero(mat_a[i])[0].tolist()
#         nonzero_ind_mat_a.append(ind)

#     for i in range(mat_b.shape[0]):
#         ind = np.nonzero(mat_b[i])[0].tolist()
#         nonzero_ind_mat_b.append(ind)

#     num_mul = 0
#     num_add = 0

#     for i in range(len(nonzero_ind_mat_a)):
#         for j in range(len(nonzero_ind_mat_b)):
#             ind_a = nonzero_ind_mat_a[i]
#             ind_b = nonzero_ind_mat_b[j]
#             ind_nonzero = set(ind_a) & set(ind_b)
#             num_nonzero = len(ind_nonzero)
#             num_mul += num_nonzero
#             num_add += num_nonzero

#     return num_mul + num_add

# def op_count(mat_a, mat_b):
#     mat_a = mat_a.clone().detach()
#     mat_b = mat_b.clone().detach()

#     mat_a_sp = mat_a.to_dense().byte()
#     mat_b_sp = mat_b.to_dense().byte()

#     mat_a_zeros = torch.zeros_like(mat_a_sp, dtype=torch.uint8)
#     mat_a_ones = torch.ones_like(mat_a_sp, dtype=torch.uint8)
#     mat_a_nonzero = torch.where(mat_a_sp != 0, mat_a_ones, mat_a_zeros)

#     mat_b_zeros = torch.zeros_like(mat_b_sp, dtype=torch.uint8)
#     mat_b_ones = torch.ones_like(mat_b_sp, dtype=torch.uint8)
#     mat_b_nonzero = torch.where(mat_b_sp != 0, mat_b_ones, mat_b_zeros)

#     mat_b_nonzero = mat_b_nonzero.t()

#     num_mul = 0
#     num_add = 0

#     for i in range(mat_b_nonzero.shape[0]):
#         partial = torch.mul(mat_a_nonzero, mat_b_nonzero[i])
#         num_nonzero = partial.sum().item()
#         num_mul += num_nonzero
#         num_add += num_nonzero

#     return num_mul + num_add


def op_count(mat_a, mat_b):
    mat_a = mat_a.clone().detach()
    mat_b = mat_b.clone().detach()

    mat_a = mat_a.to_dense()
    mat_b = mat_b.to_dense()

    mat_a_zeros = torch.zeros_like(mat_a, dtype=torch.uint8)
    mat_a_ones = torch.ones_like(mat_a, dtype=torch.uint8)
    mat_a_nonzero = torch.where(mat_a != 0, mat_a_ones, mat_a_zeros)

    mat_b_zeros = torch.zeros_like(mat_b, dtype=torch.uint8)
    mat_b_ones = torch.ones_like(mat_b, dtype=torch.uint8)
    mat_b_nonzero = torch.where(mat_b != 0, mat_b_ones, mat_b_zeros)

    del mat_a_zeros, mat_a_ones, mat_b_zeros, mat_b_ones
    gc.collect()

    mat_a_nonzero = mat_a_nonzero.float()
    mat_b_nonzero = mat_b_nonzero.float()
    mat_res = torch.mm(mat_a_nonzero, mat_b_nonzero)
    op_num = mat_res.sum().item() * 2  # Regard #add = #mul

    return op_num


def count_sparsity(m):
    m_zeros = torch.zeros_like(m, dtype=torch.bool)
    m_ones = torch.ones_like(m, dtype=torch.bool)
    m_nonzero = torch.where(m != 0, m_ones, m_zeros)
    num_total = m_nonzero.numel()
    num_nonzero = m_nonzero.sum().item()
    sparsity = round(num_nonzero / num_total, 3)

    return sparsity


def time_format(sec):
    if sec > 3600:
        hour, tmp = divmod(sec, 3600)
        min, s = divmod(tmp, 60)
        time = str(int(hour)) + 'h' + str(int(min)) + 'm' + str(int(s)) + 's'
    elif sec > 60:
        min, s = divmod(sec, 60)
        time = str(int(min)) + 'm' + str(int(s)) + 's'
    else:
        s = round(sec, 2)
        time = str(s) + 's'

    return time


# def random_val():
#     import random

#     length = 128
#     up_bound = pow(2, length) - 1
#     low_bound = 0
#     num = 10000
#     a = []
#     for i in range(num):
#         val = random.randint(low_bound, up_bound)
#         val_bin = bin(val)
#         val_str = str(val_bin)[2:]
#         if len(val_str) < length:
#             val_str = '0' * (length - len(val_str)) + val_str
#         a.append(val_str)
#     print(a)

#     filepath = './' + str(length) + 'b_10000.txt'
#     with open(filepath, 'w') as f:
#         for i in a:
#             f.write(i)
#             f.write('\n')


def plot_val_distribution(data, file_name):
    data = data.view(data.numel())
    data = data.cpu().numpy().tolist()
    plt.hist(data, bins=100)
    plt.savefig('./figures/' + file_name + '.jpg')
    plt.close()


def time_count(a, b):
    t0 = time.time()
    x = torch.sparse.mm(a, b)
    t = time.time() - t0

    return t