from __future__ import division
from __future__ import print_function
import random
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import cluster
import copy
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, CitationFull
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GCNConv
import os.path as osp

from models import GMLP, GRACE_cluster
from utils import load_citation, rmse, get_A_r, load_citation_in_order, accuracy, cal_f1_score
import warnings
from utils import enhance_sim_matrix, post_proC, err_rate, best_map
warnings.filterwarnings('ignore')

# Settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--epochs', type=int, default=400,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.6,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--data', type=str, default='Cora',
                    help='dataset to be used')
parser.add_argument('--alpha', type=float, default=2.0,
                    help='To control the ratio of Ncontrast loss')
parser.add_argument('--batch_size', type=int, default=400,
                    help='batch size')
parser.add_argument('--order', type=int, default=2,
                    help='to compute order-th power of adj')
parser.add_argument('--instance_tau', type=float, default=1,
                    help='temperature for Ncontrast loss')
parser.add_argument('--cluster_tau', type=float, default=5,
                    help='temperature for Ncontrast loss')
parser.add_argument('--theta', type=float, default=0.7,
                    help='threshold of adj matrix')
parser.add_argument('--drop_edge_rate_1', type=float, default=0.2)
parser.add_argument('--drop_edge_rate_2', type=float, default=0.4)
parser.add_argument('--drop_feature_rate_1', type=float, default=0.2)
parser.add_argument('--drop_feature_rate_2', type=float, default=0.4)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

## get data
adj, features, labels, idx_train, idx_val, idx_test = load_citation(args.data, 'AugNormAdj', args.cuda)
# adj = adj.to_dense()
# adj_label = torch.where(adj < args.theta, torch.zeros_like(adj), torch.ones_like(adj))
adj_label = get_A_r(adj, args.order)


## Model and optimizer
MLP_model = GMLP.Unsup_GMLP(nfeat=features.shape[1],
                       nhid=args.hidden,
                       dropout=args.dropout,
                       )
GCN_model = GMLP.GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                dropout=args.dropout)
CC_model = GRACE_cluster.CC(nfeat=features.shape[1],
                            nhid=args.hidden,
                            prohid=args.hidden,
                            nclass=labels.max().item() + 1,
                            instance_tau=args.instance_tau,
                            cluster_tau=args.cluster_tau)

MLP_optimizer = optim.Adam(MLP_model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
GCN_optimizer = optim.Adam(GCN_model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
CC_optimizer = optim.Adam(CC_model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    MLP_model.cuda()
    GCN_model.cuda()
    CC_model.cuda()
    features = features.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def Ncontrast(x_dis, adj_label, tau = args.instance_tau):
    """
    compute the Ncontrast loss
    """
    x_dis = torch.exp( tau * x_dis)
    x_dis_sum = torch.sum(x_dis, 1)
    x_dis_sum_pos = torch.sum(x_dis*adj_label, 1)
    loss = -torch.log(x_dis_sum_pos * (x_dis_sum**(-1))+1e-8).mean()
    return loss

def get_batch(batch_size):
    """
    get a batch of feature & adjacency matrix
    """
    rand_indx = torch.tensor(np.random.choice(np.arange(2700), batch_size)).type(torch.long).cuda()
    # rand_indx[0:len(idx_train)] = idx_train
    features_batch = features[rand_indx]
    adj_label_batch = adj_label[rand_indx,:][:,rand_indx]
    return features_batch, adj_label_batch

def get_neighbour_batch(cur) :
    batch_indx = torch.nonzero(adj_label[cur]).squeeze(1)
    # batch_indx = batch_indx[torch.nonzero(torch.where(batch_indx > cur, torch.zeros_like(batch_indx), batch_indx)).squeeze(1)]
    batch_indx = torch.cat((idx_train, batch_indx))
    features_batch = features[batch_indx]
    adj_label_batch = adj_label[batch_indx, :][:, batch_indx]
    return features_batch, adj_label_batch


# def get_neighbour_batch(cur):
#     batch_indx = torch.nonzero(adj_label[cur]).squeeze(1)
#     batch_indx = batch_indx[torch.nonzero(torch.where(batch_indx > cur, torch.zeros_like(batch_indx), batch_indx)).squeeze(1)]
#     rand_indx = torch.tensor(np.random.choice(np.arange(2000), args.batch_size)).type(torch.long).cuda()
#     rand_indx[0:len(idx_train)] = idx_train
#     rand_indx[-len(batch_indx):] = batch_indx
#     features_batch = features[rand_indx]
#     adj_label_batch = adj_label[rand_indx,:][:,rand_indx]
#     return features_batch, adj_label_batch

def train_unsup():
    features_batch, adj_label_batch = get_batch(batch_size=args.batch_size)
    MLP_model.train()
    MLP_optimizer.zero_grad()
    x, x_dis = MLP_model(features_batch)
    loss_Ncontrast = Ncontrast(x_dis, adj_label_batch, tau = args.tau)
    loss_Ncontrast.backward()
    MLP_optimizer.step()
    return 

def train_unsup_gcn():
    GCN_model.train()
    GCN_optimizer.zero_grad()
    x, x_dis = GCN_model(features, adj_label)
    loss_Ncontrast = Ncontrast(x_dis, adj_label, tau = args.tau)
    loss_Ncontrast.backward()
    GCN_optimizer.step()
    return 

def train_grace_cluster(model, x, edge_index):
    model.train()
    CC_optimizer.zero_grad()
    edge_index_1 = dropout_adj(edge_index, p=args.drop_edge_rate_1)[0]
    edge_index_2 = dropout_adj(edge_index, p=args.drop_edge_rate_2)[0]
    x_1 = GRACE_cluster.drop_feature(x, args.drop_feature_rate_1)
    x_2 = GRACE_cluster.drop_feature(x, args.drop_feature_rate_2)
    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)
    loss = model.loss(z1, z2, batch_size=0)
    loss.backward()
    CC_optimizer.step()

    return loss.item()


def train_classifier(embedding, classifier):
    classifier.train()
    classifier_optimizer.zero_grad()
    output = classifier(embedding)
    loss_nll = F.nll_loss(output[idx_train], labels[idx_train])
    loss_nll.backward()
    classifier_optimizer.step()

    val_f1 = cal_f1_score(output[idx_val], labels[idx_val])
    test_f1 = cal_f1_score(output[idx_test], labels[idx_test])
    return val_f1, test_f1

def test_spectral(c, labels, n_class):
    y_result = post_proC(c, n_class)
    print("Spectral Clustering Done.. Finding Best Fit..")
    scores = err_rate(labels.detach().cpu().numpy(), y_result)
    return scores

# def my_test():
#     model.eval()
#     output = model(features[idx_train])
#     output = torch.cat((output, model(features[idx_val])), dim = 0)
#     for i in idx_test :
#         features_batch, adj_label_batch = get_neighbour_batch(i)
#         model.train()
#         optimizer.zero_grad()
#         out, x_dis = model(features_batch)
#         loss_train_class = F.nll_loss(out[idx_train], labels[idx_train])
#         loss_Ncontrast = Ncontrast(x_dis, adj_label_batch, tau = 1)
#         loss_train = loss_train_class + loss_Ncontrast * args.alpha
#         loss_train.backward()
#         optimizer.step()

#         output = torch.cat((output, model(features[i].unsqueeze(0))), dim = 0)

#     acc_test = cal_f1_score(output[idx_test], labels[idx_test])
#     return acc_test

def print_pic(output, out, name) :
    plt.figure()
    plt.subplot(2,1,1)
    mx_idx = output.shape[0]
    plt.plot(range(mx_idx), output.detach().cpu().numpy()[:, 0], label='output')
    plt.plot(range(mx_idx), out.detach().cpu().numpy()[:, 0], label='true')
    plt.legend(loc=3)
    plt.subplot(2,1,2)
    plt.plot(range(mx_idx), output.detach().cpu().numpy()[:,1], label='output')
    plt.plot(range(mx_idx), out.detach().cpu().numpy()[:,1], label='true')
    plt.legend(loc=3)
    plt.savefig('./pics/'+name+'.jpg')

def get_dataset(path, name):
    assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP']
    name = 'dblp' if name == 'DBLP' else name

    return (CitationFull if name == 'dblp' else Planetoid)(
        path,
        name,
        pre_transform = T.NormalizeFeatures())

path = osp.join(osp.expanduser('~'), 'datasets', args.data)
dataset = get_dataset(path, args.data)
data = dataset[0]

if args.cuda:
    data = data.cuda()

print('\n'+'training configs', args)
for epoch in range(args.epochs):
    loss = train_grace_cluster(CC_model, data.x, data.edge_index)
    print("Epoch: %d, Loss: %f"%(epoch, loss))
    # train_unsup_gcn()

# # embedding, x_dis= GCN_model(features, adj_label)
embedding = CC_model(features, data.edge_index)
# embedding = embedding.detach()
pred = CC_model.getCluster(embedding)
pred = pred.detach()

# scores = test_spectral(embedding, labels, labels.max().item()+1)
scores = err_rate(labels.cpu().numpy(), pred.cpu().numpy())
print(scores)

# best_val_f1 = 0
# best_test_f1 = 0
# classifier = GMLP.Classifier(nhid=embedding.shape[1], nclass=labels.max().item() + 1)
# classifier_optimizer = optim.Adam(classifier.parameters(),
#                     lr=args.lr, weight_decay=args.weight_decay)

# if args.cuda:
#     classifier = classifier.cuda()

# for epoch in tqdm(range(args.epochs)):
#     val_f1, test_f1 = train_classifier(embedding, classifier)
#     if val_f1 > best_val_f1:
#         best_val_f1 = val_f1
#         best_test_f1 = test_f1
    
# print(best_val_f1)
# print(best_test_f1)


# log_file = open(r"log.txt", encoding="utf-8",mode="a+")  
# with log_file as file_to_be_write:  
#     print('tau', 'order', \
#             'batch_size', 'hidden', \
#                 'alpha', 'lr', \
#                     'weight_decay', 'data', \
#                         'test_acc', file=file_to_be_write, sep=',')
#     print(args.tau, args.order, \
#          args.batch_size, args.hidden, \
#              args.alpha, args.lr, \
#                  args.weight_decay, args.data, \
#                      test_acc.item(), file=file_to_be_write, sep=',')


