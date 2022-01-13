import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, Linear, LayerNorm
from torch_geometric.nn import GCNConv

class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation=F.relu,
                 base_model=GCNConv, k: int = 2):
        super(Encoder, self).__init__()
        self.base_model = base_model

        assert k >= 2
        self.k = k
        self.conv = [base_model(in_channels, 2 * out_channels)]
        for _ in range(1, k-1):
            self.conv.append(base_model(2 * out_channels, 2 * out_channels))
        self.conv.append(base_model(2 * out_channels, out_channels))
        self.conv = nn.ModuleList(self.conv)

        self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        for i in range(self.k):
            x = self.activation(self.conv[i](x, edge_index))
        return x


class CC(torch.nn.Module):
    def __init__(self, nfeat : int, nhid: int, prohid: int, nclass : int,
                 instance_tau: float = 0.5, cluster_tau : float = 2):
        super(CC, self).__init__()
        encoder = Encoder(nfeat, nhid)
        self.encoder = encoder
        self.num_hidden = nhid
        self.num_proj_hidden = prohid
        self.class_num = nclass
        self.instance_tau: float = instance_tau
        self.cluster_tau: float = cluster_tau
        self.instance_projector = nn.Sequential(
            nn.Linear(self.num_hidden, self.num_proj_hidden),
            nn.ReLU(),
            nn.Linear(self.num_proj_hidden, self.num_hidden)
        )
        self.cluster_projector = nn.Sequential(
            nn.Linear(self.num_hidden, self.num_proj_hidden),
            nn.ReLU(),
            nn.Linear(self.num_proj_hidden, self.class_num),
            nn.Softmax(dim=1)
        )


    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x, edge_index)
        z = F.normalize(self.instance_projector(h), dim=1)
        c = self.cluster_projector(h)
        return z, c

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def clu_sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = z1.t()
        z2 = z2.t()
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def instance_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        # N = 2 * z1.size(0)
        # z = torch.cat((z1, z2), dim=0)

        # sim = torch.matmul(z, z.T) / self.instance_tau
        # sim_i_j = torch.diag(sim, z1.size(0))
        # sim_j_i = torch.diag(sim, z1.size(0))

        # mask = self.mask_correlated_clusters(z1.size(0))
        # positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        # negative_samples = sim[mask].reshape(N, -1)

        # criterion = nn.CrossEntropyLoss(reduction="sum")
        # labels = torch.zeros(N).to(positive_samples.device).long()
        # logits = torch.cat((positive_samples, negative_samples), dim=1)
        # loss = criterion(logits, labels)
        # loss /= N

        # return loss

        f = lambda x: torch.exp(x / self.instance_tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def cluster_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.cluster_tau)
        refl_sim = f(self.clu_sim(z1, z1))
        between_sim = f(self.clu_sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

        p1 = z1.sum(0).view(-1)
        p1 /= p1.sum()
        ne1 = math.log(p1.size(0)) + (p1 * torch.log(p1)).sum()
        p2 = z2.sum(0).view(-1)
        p2 /= p2.sum()
        ne2 = math.log(p2.size(0)) + (p2 * torch.log(p2)).sum()
        ne_loss = ne1 + ne2

        z1 = z1.t()
        z2 = z2.t()
        N = 2 * self.class_num
        c = torch.cat((z1, z2), dim=0)

        similarity_f = nn.CosineSimilarity(dim=2)
        criterion = nn.CrossEntropyLoss(reduction="sum")
        sim = similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.cluster_tau
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        mask = self.mask_correlated_clusters(self.class_num)
        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = criterion(logits, labels)
        loss /= N

        return loss + ne_loss

        f = lambda x: torch.exp(x / self.cluster_tau)
        refl_sim = f(self.sim_clust(z1, z1))
        between_sim = f(self.sim_clust(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag())).sum()

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, h1, h2, c1, c2,
             mean: bool = True, batch_size: int = 0):
        if batch_size == 0:
            l1 = self.instance_loss(h1, h2)
            l2 = self.instance_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)
        

        ins_loss = (l1 + l2) * 0.5
        # ins_loss = ins_loss.mean() if mean else ins_loss.sum()

        clu_loss_1 = self.cluster_loss(c1, c2)
        clu_loss_2 = self.cluster_loss(c2, c1)
        clu_loss = (clu_loss_1 + clu_loss_2) * 0.5
        # clu_loss = clu_loss.mean() if mean else clu_loss.sum()

        return ins_loss + clu_loss

    def getCluster(self, x, edge_index) :
        x = self.encoder(x, edge_index)
        pred = self.cluster_projector(x)
        return torch.argmax(pred, dim = 1)


def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x


