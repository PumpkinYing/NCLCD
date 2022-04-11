from __future__ import division
from __future__ import print_function
import random
import time
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import average_precision_score, roc_auc_score

from models import GMLP
from utils import load_citation, accuracy, get_A_r, load_citation_in_order, accuracy
from sklearn import cluster
import copy
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, CitationFull
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GCNConv
from sklearn.manifold import TSNE
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
parser.add_argument('--epochs', type=int, default=300,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.6,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--data', type=str, default='Cora',
                    help='dataset to be used')
parser.add_argument('--alpha', type=float, default=2.0,
                    help='To control the ratio of Ncontrast loss')
parser.add_argument('--batch_size', type=int, default=5000,
                    help='batch size')
parser.add_argument('--order', type=int, default=1,
                    help='to compute order-th power of adj')
parser.add_argument('--instance_tau', type=float, default=0.4,
                    help='temperature for Ncontrast loss')
parser.add_argument('--cluster_tau', type=float, default=0.4,
                    help='temperature for Ncontrast loss')
parser.add_argument('--theta', type=float, default=0.5,
                    help='threshold of adj matrix')
parser.add_argument('--entropy_weight', type=float, default=0.1,
                    help='threshold of adj matrix')
parser.add_argument('--seed', type=int, default=233)
parser.add_argument('--load', type=bool, default=False)
parser.add_argument('--model', default='mlp')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

## get data
adj, features, labels = load_citation(args.data, 'AugNormAdj', args.cuda)
# adj = adj.to_dense()
# adj_label = torch.where(adj < args.theta, torch.zeros_like(adj), torch.ones_like(adj))
adj_label = get_A_r(adj, args.order)

def Ncontrast(x_dis, adj_label, tau = args.instance_tau):
    """
    compute the Ncontrast loss
    """
    x_dis = torch.exp( tau * x_dis)
    x_dis_sum = torch.sum(x_dis, 1)
    x_dis_sum_pos = torch.sum(x_dis*adj_label, 1)
    loss = -torch.log(x_dis_sum_pos * (x_dis_sum**(-1))+1e-8).mean()
    return loss

def entropy(x, input_as_probabilities):
    """ 
    Helper function to compute the entropy over the batch 

    input: batch w/ shape [b, num_classes]
    output: entropy value [is ideally -log(num_classes)]
    """

    if input_as_probabilities:
        x_ =  torch.clamp(x, min = 1e-8)
        b =  x_ * torch.log(x_)
    else:
        b = F.softmax(x, dim = 1) * F.log_softmax(x, dim = 1)

    if len(b.size()) == 2: # Sample-wise entropy
        return -b.sum(dim = 1).mean()
    elif len(b.size()) == 1: # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' %(len(b.size())))

def get_batch(batch_size, features, adj_label):
    """
    get a batch of feature & adjacency matrix
    """
    rand_indx = torch.tensor(np.random.choice(np.arange(features.shape[0]), batch_size)).type(torch.long).cuda()
    # rand_indx[0:len(idx_train)] = idx_train
    features_batch = features[rand_indx]
    adj_label_batch = adj_label[rand_indx,:][:,rand_indx]
    return features_batch, adj_label_batch

def train_unsup_mlp():
    features_batch, adj_label_batch = get_batch(min(args.batch_size, features.shape[0]), features, adj_label)
    MLP_model.train()
    MLP_optimizer.zero_grad()
    x, x_dis = MLP_model(features_batch)
    loss_Ncontrast = Ncontrast(x_dis, adj_label_batch, tau = args.instance_tau)
    loss_Ncontrast.backward()
    MLP_optimizer.step()
    return loss_Ncontrast

def train_unsup_gcn():
    features_batch, adj_label_batch = get_batch(args.batch_size, features, adj_label)
    GCN_model.train()
    GCN_optimizer.zero_grad()
    # x, x_dis = GCN_model(features_batch, adj_label_batch)
    x, x_dis = GCN_model(features, adj_label)
    # loss_Ncontrast = Ncontrast(x_dis, adj_label_batch, tau = args.instance_tau)
    loss_Ncontrast = Ncontrast(x_dis, adj_label, tau = args.instance_tau)
    loss_Ncontrast.backward()
    GCN_optimizer.step()
    return loss_Ncontrast

def train_grace_cluster(model, x, edge_index):
    model.train()
    CC_optimizer.zero_grad()
    edge_index_1 = dropout_adj(edge_index, p=args.drop_edge_rate_1)[0]
    edge_index_2 = dropout_adj(edge_index, p=args.drop_edge_rate_2)[0]
    x_1 = GRACE_cluster.drop_feature(x, args.drop_feature_rate_1)
    x_2 = GRACE_cluster.drop_feature(x, args.drop_feature_rate_2)
    h1, c1 = model(x_1, edge_index_1)
    h2, c2 = model(x_2, edge_index_2)
    loss = model.loss(h1, h2, c1, c2, batch_size=0)
    loss.backward()
    CC_optimizer.step()

    return loss.item()


def train_unsup_classifier(cluster_adj_label, embedding, classifier):
    embedding_batch, adj_label_batch = get_batch(min(features.shape[0], args.batch_size), embedding, cluster_adj_label)
    adj_label_batch = torch.where(adj_label_batch > args.theta, torch.ones_like(adj_label_batch), torch.zeros_like(adj_label_batch))
    classifier.train()
    classifier_optimizer.zero_grad()
    output, z_dis = classifier(embedding_batch)
    loss_classification = Ncontrast(z_dis, adj_label_batch, tau = args.cluster_tau)
    cluster_entropy = entropy(torch.mean(output, axis=0), input_as_probabilities = True)
    # print(loss_classification, cluster_entropy)
    loss_classification -= args.entropy_weight*cluster_entropy
    loss_classification.backward()
    classifier_optimizer.step()
    return loss_classification

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
    y_result = post_proC(c, n_class, args.seed)
    print("Spectral Clustering Done.. Finding Best Fit..")
    scores = err_rate(labels.detach().cpu().numpy(), y_result)
    return scores


def print_pic(output, out, name) :
    plt.figure()
    mx_idx = output.shape[0]
    # print pic for two outputs
    # plt.subplot(2,1,1)
    # plt.plot(range(mx_idx), output.detach().cpu().numpy()[:, 0], label='output')
    # plt.plot(range(mx_idx), out.detach().cpu().numpy()[:, 0], label='true')
    # plt.legend(loc=3)
    # plt.subplot(2,1,2)
    # plt.plot(range(mx_idx), output.detach().cpu().numpy()[:,1], label='output')
    # plt.plot(range(mx_idx), out.detach().cpu().numpy()[:,1], label='true')
    # plt.legend(loc=3)

    #print pic for one pic
    plt.plot(range(mx_idx), output.detach().cpu().numpy(), label='output')
    plt.plot(range(mx_idx), out.detach().cpu().numpy(), label='true')
    plt.savefig('./pics/'+name+'.jpg')

def seed_it(seed):
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)


seed_it(args.seed)

## Model and optimizer
MLP_model = GMLP.Unsup_GMLP(nfeat=features.shape[1],
                       nhid=args.hidden,
                       dropout=args.dropout)
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

print('\n'+'training configs', args)
filepath = "saved_models/gcnmodel_{}_instance_tau_{}_seed_{}_lr_0.01.pkl".format(args.data, args.instance_tau, args.seed)

colors = ['b', 'g', 'r', 'c', 'm', 'olive', 'orange', 'tan']

if args.load :
    GCN_model.load_state_dict(torch.load(filepath))
else :
    tsne = TSNE(n_components=2, init='pca', perplexity=30)

    for epoch in range(args.epochs):
        # loss = train_grace_cluster(CC_model, data.x, data.edge_index)
        if args.model == "mlp":
            loss = train_unsup_mlp()
        else :
            loss = train_unsup_gcn()
        print("Epoch: %d, Loss: %f"%(epoch, loss))
        if(epoch == 0) :
            embedding, _ = GCN_model(features, adj_label)
            cur_embedding = embedding.detach().cpu().numpy()
            positions = tsne.fit_transform(cur_embedding)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.scatter(positions[:,0], positions[:, 1], c=labels.detach().cpu().numpy(), cmap=matplotlib.colors.ListedColormap(colors))
            plt.savefig("{}.eps".format(epoch), format="eps")
        if (epoch+1)%50 == 0:
            embedding, _ = GCN_model(features, adj_label)
            cur_embedding = embedding.detach().cpu().numpy()
            positions = tsne.fit_transform(cur_embedding)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.scatter(positions[:,0], positions[:, 1], c=labels.detach().cpu().numpy(), cmap=matplotlib.colors.ListedColormap(colors))
            plt.savefig("{}.eps".format(epoch), format="eps")
            plt.show()


    # torch.save(GCN_model.state_dict(), filepath)

if args.model == "mlp":
    embedding, x_dis= MLP_model(features)
else :
    embedding, x_dis= GCN_model(features, adj_label)

embedding = embedding.detach()
# cluster_adj_label = torch.where(x_dis > args.theta, torch.ones_like(x_dis), torch.zeros_like(x_dis))
print("Self training done, clustering start")

begin_time = time.time()
test_spectral(embedding, labels, labels.max().item()+1)
end_time = time.time()
print("Spectral time:", end_time-begin_time)
# scores = err_rate(data.y.cpu().numpy(), pred.cpu().numpy())
# print("Spectral clustering scores:")
# print(scores)

# filename = "log_{}_{}.txt".format(args.data, args.model)
# log_file = open(filename, encoding="utf-8",mode="a+")  
# with log_file as file_to_be_write:  
#     print("args",file=file_to_be_write)
#     print(args, file=file_to_be_write)
#     # print("spectral scores:", file=file_to_be_write)
#     # print(scores, file=file_to_be_write)

torch.cuda.empty_cache()
classifier = GMLP.Classifier(nhid=embedding.shape[1], nclass=labels.max().item() + 1)
classifier_optimizer = optim.Adam(classifier.parameters(),
                    lr=args.lr, weight_decay=args.weight_decay)

begin_time = time.time()
if args.cuda:
    classifier = classifier.cuda()
for epoch in range(args.epochs):
    loss = train_unsup_classifier(x_dis, embedding, classifier)
    # print("Classifier loss: %f"%loss)
end_time = time.time() 
print("Classifier time:", end_time-begin_time)

classifier.eval()
logic = classifier.get_classify_result(embedding)
pred = torch.argmax(logic, dim=1)
scores = err_rate(labels.cpu().numpy(), pred.cpu().numpy())
print("SS clustering scores:")
print(scores)

# filename = "heat_map.txt"
# log_file = open(filename, encoding="utf-8",mode="a")  
# with log_file as file_to_be_write:  
#     print("ss clustering scores:", file=file_to_be_write)
#     print(scores, file=file_to_be_write)


