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
from sklearn.cluster import DBSCAN

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

db = DBSCAN(eps=0.1, min_samples=10).fit(features)
print(db)