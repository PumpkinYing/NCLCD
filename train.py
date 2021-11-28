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
import copy

from models import Unsup_GMLP, Classifier
from utils import load_citation, rmse, get_A_r, load_citation_in_order, accuracy, cal_f1_score
import warnings
warnings.filterwarnings('ignore')

# Settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--epochs', type=int, default=400,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.6,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--data', type=str, default='cora',
                    help='dataset to be used')
parser.add_argument('--alpha', type=float, default=2.0,
                    help='To control the ratio of Ncontrast loss')
parser.add_argument('--batch_size', type=int, default=400,
                    help='batch size')
parser.add_argument('--order', type=int, default=2,
                    help='to compute order-th power of adj')
parser.add_argument('--tau', type=float, default=1.0,
                    help='temperature for Ncontrast loss')
parser.add_argument('--theta', type=float, default=0.7,
                    help='threshold of adj matrix')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

## get data
adj, features, labels, idx_train, idx_val, idx_test = load_citation(args.data, 'AugNormAdj', args.cuda)
# adj = adj.to_dense()
# adj_label = torch.where(adj < args.theta, torch.zeros_like(adj), torch.ones_like(adj))
adj_label = get_A_r(adj, args.order)


## Model and optimizer
model = Unsup_GMLP(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item()+1,
            dropout=args.dropout,
            )
classifier = Classifier(nhid=args.hidden, nclass = labels.max().item() + 1)

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

classifier_optimizer = optim.Adam(classifier.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    classifier.cuda()
    features = features.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def Ncontrast(x_dis, adj_label, tau = args.tau):
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
    rand_indx[0:len(idx_train)] = idx_train
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
    model.train()
    optimizer.zero_grad()
    x, x_dis = model(features_batch)
    loss_Ncontrast = Ncontrast(x_dis, adj_label_batch, tau = args.tau)
    loss_Ncontrast.backward()
    optimizer.step()
    return 

def train_classifier(embedding):
    model.eval()
    classifier.train()
    classifier_optimizer.zero_grad()
    output = classifier(embedding)
    loss_nll = F.nll_loss(output[idx_train], labels[idx_train])
    loss_nll.backward()
    classifier_optimizer.step()

    val_f1 = cal_f1_score(output[idx_val], labels[idx_val])
    test_f1 = cal_f1_score(output[idx_test], labels[idx_test])
    return val_f1, test_f1

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

print('\n'+'training configs', args)
for epoch in tqdm(range(args.epochs)):
    train_unsup()

best_val_f1 = 0
best_test_f1 = 0
embedding, x_dis= model(features)
embedding = embedding.detach()
for epoch in tqdm(range(args.epochs)):
    val_f1, test_f1 = train_classifier(embedding)
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_test_f1 = test_f1
    
print(best_val_f1)
print(best_test_f1)


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


