import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, Linear, LayerNorm

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
    x_dis = x@x.T
    mask = torch.eye(x_dis.shape[0]).cuda()
    x_sum = torch.sum(x**2, 1).reshape(-1, 1)
    x_sum = torch.sqrt(x_sum).reshape(-1, 1)
    x_sum = x_sum @ x_sum.T
    x_dis = x_dis*(x_sum**(-1))
    x_dis = (1-mask) * x_dis
    return x_dis

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

        # class_feature = self.classifier(feature_cls)
        batch_size = x.shape[0]
        feature_concated = torch.cat((feature_cls.repeat(1, batch_size).reshape(batch_size*batch_size, -1), feature_cls.repeat(batch_size, 1)), dim=1)
        class_feature = self.decoder(feature_concated)
        class_logits = F.sigmoid(class_feature)

        if self.training:
            return class_logits, x_dis
        else:
            return class_logits

    def edge_prediction(self, source_nodes_feature, destination_nodes_feature):
        source_feature_cls = self.mlp(source_nodes_feature)
        destination_feature_cls = self.mlp(destination_nodes_feature)
        feature_concated = torch.cat((source_feature_cls, destination_feature_cls), dim=1)
        class_prob = self.decoder(feature_concated)
        return F.sigmoid(class_prob)

    
class Decoder(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(Decoder, self).__init__()
        self.nhid = nhid
        self.mlp = Mlp(nfeat, self.nhid, dropout)
        self.last_layer = Linear(self.nhid, 1)
    
    def forward(self, x):
        x = self.mlp(x)