import numpy as np
import scipy.sparse as sp
import torch
import sys
import pickle as pkl
import networkx as nx
import random
from normalization import fetch_normalization, row_normalize
from time import perf_counter


def get_A_r(adj, r):
    adj_label = adj.to_dense()
    if r == 1:
        adj_label = adj_label
    elif r == 2:
        adj_label = adj_label@adj_label
    elif r == 3:
        adj_label = adj_label@adj_label@adj_label
    elif r == 4:
        adj_label = adj_label@adj_label@adj_label@adj_label
    return adj_label


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def rmse(output, labels) :
    loss = torch.div(torch.abs(output-labels), labels)
    loss = torch.sum(loss, 0)
    loss = loss*100/output.shape[0]
    return torch.sum(loss)

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def preprocess_citation(adj, features, normalization="FirstOrderGCN"):
    adj_normalizer = fetch_normalization(normalization)
    
    adj = adj_normalizer(adj)  

    features = row_normalize(features)
    return adj, features

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def load_citation(dataset_str="cora", normalization="AugNormAdj", cuda=True):
    """
    Load Citation Networks Datasets.
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str.lower(), names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))


    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    source_nodes = []
    destination_nodes = []
    edge_labels = []
    for source in graph :
        for destination in graph[source]:
            if random.randint(1, 100) <= 5: 
                graph[source].remove(destination)
                source_nodes.append(source)
                destination_nodes.append(destination)
                edge_labels.append(1)
        while random.randint(1, 100) <= 5:
            destination = random.randint(0, 2708)
            while destination in graph[source] : 
                destination = random.randint(0, 2708)
            source_nodes.append(source)
            destination_nodes.append(destination)
            edge_labels.append(0)

    source_nodes = torch.tensor(source_nodes, dtype=torch.long)
    destination_nodes = torch.tensor(destination_nodes, dtype=torch.long)
    edge_labels = torch.tensor(edge_labels, dtype=torch.long)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_train = range(140)
    idx_val = range(140, 1700)
    idx_test = range(1700, 2708)

    adj_normalized, features = preprocess_citation(adj, features, normalization)

    # porting to pytorch
    features = torch.FloatTensor(np.array(features.todense())).float()
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    adj_normalized = sparse_mx_to_torch_sparse_tensor(adj_normalized).float()
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()

    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        adj_normalized = adj_normalized.cuda()
        labels = labels.cuda()
        source_nodes = source_nodes.cuda()
        destination_nodes = destination_nodes.cuda()
        edge_labels = edge_labels.cuda()

    return adj_normalized, adj, features, labels, source_nodes, destination_nodes, edge_labels, idx_train, idx_val



def load_citation_in_order(dataset = "cora", normalization="AugNormAdj", cuda = True) :
    dir = "data/lab_data/"
    features = np.genfromtxt(dir+"features.txt")
    labels = np.genfromtxt(dir+"labels.txt")
    adj = np.genfromtxt(dir+"adjacent.txt")
    adj = sp.coo_matrix(adj)
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj, features = preprocess_citation(adj, features, normalization)

    features = torch.FloatTensor(features)
    labels = torch.FloatTensor(labels)
    # adj = torch.FloatTensor(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    
    idx_train = range(1200)
    idx_train = torch.LongTensor(idx_train)
    idx_val = range(1200, 1300)
    idx_val = torch.LongTensor(idx_val)
    idx_test = range(1200, 1680)
    idx_test = torch.LongTensor(idx_test)

    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    return adj, features, labels, idx_train, idx_val, idx_test

