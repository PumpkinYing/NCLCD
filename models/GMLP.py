import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, Linear, LayerNorm
from torch_geometric.nn import GCNConv

class Mlp(nn.Module):
    def __init__(self, input_dim, hid_dim, dropout):
        super(Mlp, self).__init__()
        self.fc1 = Linear(input_dim, hid_dim)
        self.fc2 = Linear(hid_dim, hid_dim)
        self.act_fn = torch.nn.functional.gelu
        self._init_weights()

        self.dropout = Dropout(dropout)
        self.layernorm = LayerNorm(hid_dim, eps=1e-6)

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.layernorm(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def get_feature_dis(x):
    """
    x :           batch_size x nhid
    x_dis(i,j):   item means the similarity between x(i) and x(j).
    """
    # x_dis = x@x.T
    # mask = torch.eye(x_dis.shape[0]).cuda()
    # x_sum = torch.sum(x**2, 1).reshape(-1, 1)
    # x_sum = torch.sqrt(x_sum).reshape(-1, 1)
    # x_sum = x_sum @ x_sum.T
    # x_dis = x_dis*(x_sum**(-1))
    # x_dis = (1-mask) * x_dis
    # return x_dis

    x = F.normalize(x, dim=1)
    return torch.mm(x, x.t())

class GraphConvolutionLayer(torch.nn.Module) :
    def __init__(self, in_features, out_features, bias=True) :
        super(GraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias :
            self.bias = torch.nn.Parameter(torch.FloatTensor(out_features))
        else :
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) :
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv,stdv)
        if self.bias is not None :
            self.bias.data.uniform_(-stdv,stdv)

    def forward(self, input, adj) :
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None :
            return output + self.bias
        else :
            return output

class GCN(torch.nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolutionLayer(nfeat, nhid)
        self.gc2 = GraphConvolutionLayer(nhid, nhid)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, adj):
        x = self.relu(self.gc1(x, adj))
        x = self.dropout(x)
        x = self.gc2(x, adj)
        x_dis = get_feature_dis(x)
        return x, x_dis


class Classifier(nn.Module):
    def __init__(self, nhid, nclass):
        super(Classifier, self).__init__()
        self.nhid = nhid
        self.classifier = Linear(self.nhid, nclass)
    
    def forward(self, x):
        class_feature = self.classifier(x)
        class_logits = F.softmax(class_feature, dim=1)
        Z = class_logits
        z_dis = get_feature_dis(Z)
        if self.training :
            return class_logits, z_dis
        else :
            return class_logits

    def get_classify_result(self, x):
        class_feature = self.classifier(x)
        class_logits = F.softmax(class_feature, dim=1)
        return class_logits
        

class GMLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GMLP, self).__init__()
        self.nhid = nhid
        self.mlp = Mlp(nfeat, self.nhid, dropout)
        self.classifier = Linear(self.nhid, nclass)

    def forward(self, x):
        x = self.mlp(x)

        feature_cls = x
        Z = x

        if self.training:
            x_dis = get_feature_dis(Z)

        class_feature = self.classifier(feature_cls)
        class_logits = F.log_softmax(class_feature, dim=1)

        if self.training:
            return class_logits, x_dis
        else:
            return class_logits

class Unsup_GMLP(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(Unsup_GMLP, self).__init__()
        self.nhid = nhid
        self.mlp = Mlp(nfeat, self.nhid, dropout)

    def forward(self, x):
        x = self.mlp(x)
        Z = x
        x_dis = get_feature_dis(Z)
        return x, x_dis