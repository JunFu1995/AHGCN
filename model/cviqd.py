from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

import networkx as nx
import numpy as np 
import walker
import math
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features,  use_fc, dp=0.5, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.use_fc = use_fc
        self.d = nn.Dropout(dp)
    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight.data)

    def forward(self, input, adj):
        input = self.d(input)
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support) if not self.use_fc else support
        return output 

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCNNet(nn.Module):
    def __init__(self, k, use_hgcn, use_ms, use_fc):
        super(GCNNet, self).__init__()
        if use_ms:
            self.gc1 = GraphConvolution(256*4, 256, use_fc)
        else:
            self.gc1 = GraphConvolution(512, 256, use_fc) 
                       
        self.bn1 = nn.BatchNorm1d(20, eps=1e-05, momentum=0.1, affine=True)
        self.gc2 = GraphConvolution(256, 128, use_fc)
        self.bn2 =  nn.BatchNorm1d(20, eps=1e-05, momentum=0.1, affine=True)
        self.gc3 = GraphConvolution(128, 64, use_fc)
        self.bn3 =  nn.BatchNorm1d(20, eps=1e-05, momentum=0.1, affine=True)
        self.gc4 = GraphConvolution(64, 32, use_fc)
        self.bn4 = nn.BatchNorm1d(20, eps=1e-05, momentum=0.1, affine=True) # training 0.1 else 0 
        self.gc5 = GraphConvolution(32, 1, use_fc,dp=0.1)
        self.relu = nn.Softplus()
        self.k = k 
        self.use_hgcn = use_hgcn 
        
    def para_init(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)

    def norm_adj(self, matrix):
        D = torch.diag_embed(matrix.sum(2))
        D = D ** 0.5
        D = D.inverse()
        # D(-1/2) * A * D(-1/2)
        normal = D.bmm(matrix).bmm(D)
        return normal.detach()

    def norm_H(self, H):
        """
        H shape : B, N, H
        """
        #print(H.shape)
        b, n, e = H.shape 
        W = torch.ones(b, e).to(H.device)
        DV = H.sum(-1)
        DE = H.sum(1)
        invDE = torch.diag_embed(torch.pow(DE, -1))
        DV2 = torch.diag_embed(torch.pow(DV, -0.5))
        W = torch.diag_embed(W) 
        HT = H.permute(0,2,1)
        #print(DV2.shape, H.shape, W.shape, invDE.shape, DV2.shape)
        G = DV2.bmm(H).bmm(W).bmm(invDE).bmm(HT).bmm(DV2)
        return G 

    # def constructH(self, feature, A, k):
    #     # random walk
    #     H = []
    #     for i in range(A.shape[0]):
    #         testA = A[i] - torch.diag_embed(torch.diag(A[i]))
    #         G = nx.from_numpy_array(testA.cpu().numpy())
    #         X = walker.random_walks(G, n_walks=1, walk_len=30)
    #         Y = []
    #         for x in X:
    #             y = np.zeros(20)
    #             y[x] = 1
    #             Y.append(y)
    #         Y = np.vstack(Y).transpose()
    #         h = torch.tensor(Y).to(A.device).float().unsqueeze(0)
    #         H.append(h)

    #     H = torch.cat(H, dim=0)
    #     H = torch.cat((A, H), dim=-1)
    #     return H 

    def constructH(self, feature, A, k):

        b, n, c = feature.shape 
        fn = feature / (torch.norm(feature, p=2, dim=-1).unsqueeze(-1).detach() + 1e-12)
        cos = fn.bmm(fn.permute(0,2,1))
        ind = torch.argsort(cos, dim=-1, descending=False) #cvqid ascending
        superedge = torch.zeros(b,n,n).to(feature.device)

        topk = ind[:,:,:k]
        superedge.scatter_(-1, topk, 1) #c[:,:,:k])
        superedge = superedge.permute(0,2,1)
        #print(topk+1)
        H = torch.cat((A, superedge), dim=-1)
        return H

    def forward(self, feature, A, f):
        if self.use_hgcn:
            if self.k == 0:
                adj = self.norm_H(A)
            else:
                H = self.constructH(f, A, self.k)
                adj = self.norm_H(H)     
        else:
            adj = self.norm_adj(A)
        gc1 = self.gc1(feature, adj)
        gc1 = self.bn1(gc1)
        gc1 = self.relu(gc1)
        gc2 = self.gc2(gc1, adj)
        gc2 = self.bn2(gc2)
        gc2 = self.relu(gc2)
        gc3 = self.gc3(gc2, adj)
        gc3 = self.bn3(gc3)
        gc3 = self.relu(gc3)
        gc4 = self.gc4(gc3, adj)
        gc4 = self.bn4(gc4)
        gc4 = self.relu(gc4)
        gc5 = self.gc5(gc4, adj)
        gc5 = self.relu(gc5)
        return gc5

class OIQANet(nn.Module):
    def __init__(self, model, k, use_hgcn, use_ms, use_fc, inplace=False):
        super(OIQANet, self).__init__()

        self.resnet = nn.Sequential(*list(model.children())[:-2])
        self.GCN = GCNNet(k, use_hgcn, use_ms, use_fc)
        self.maxpool = nn.MaxPool2d(8)
        self.k = k 
        self.use_hgcn = use_hgcn
        self.use_ms = use_ms
        self.maxpool = nn.MaxPool2d(8)
        # local distortion aware module
        lda_out_channels = 256

        self.lda1_pool = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.MaxPool2d(8, stride=8), 
        )
        self.lda1_fc = nn.Linear(16 * 64, lda_out_channels)

        self.lda2_pool = nn.Sequential(
            nn.Conv2d(128, 16, kernel_size=1, stride=1, padding=0, bias=False),   
            nn.MaxPool2d(4, stride=4),
        )
        self.lda2_fc = nn.Linear(16 * 64, lda_out_channels)

        self.lda3_pool = nn.Sequential(
            nn.Conv2d(256, 16, kernel_size=1, stride=1, padding=0, bias=False),            
            nn.MaxPool2d(2, stride=2),
        )
        self.lda3_fc = nn.Linear(16 * 64, lda_out_channels)

        self.lda4_pool = nn.Sequential(
            nn.Conv2d(512, 16, kernel_size=1, stride=1, padding=0, bias=False), 
            nn.MaxPool2d(1, stride=1), # using maxpool2d
        )
        self.lda4_fc = nn.Linear(16 * 64, lda_out_channels)

        self.rankloss = nn.MarginRankingLoss(0.5)
    def extract_feat(self, x):
        #l0 = self.resnet[:3](x)
        l1 = self.resnet[:5](x)
        l2 = self.resnet[5](l1)
        l3 = self.resnet[6](l2)
        l4 = self.resnet[7](l3)

        # the same effect as lda operation in the paper, but save much more memory
        #lda_0 = self.lda0_fc(self.lda0_pool(l0).view(x.size(0), -1))
        lda_1 = self.lda1_fc(self.lda1_pool(l1).view(x.size(0), -1))
        lda_2 = self.lda2_fc(self.lda2_pool(l2).view(x.size(0), -1))
        lda_3 = self.lda3_fc(self.lda3_pool(l3).view(x.size(0), -1))
        lda_4 = self.lda4_fc(self.lda4_pool(l4).view(x.size(0), -1))

        feat = torch.cat((lda_1, lda_2, lda_3, lda_4), dim=1) #torch.cat((l1,l2,l3,l4),dim=1) # 1024
        return feat, feat

    def para_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def loss_build(self, x_hat, x):
        distortion =  F.mse_loss(x_hat, x, size_average=True)
        return distortion

    def forward(self, x, label, A, requires_loss):
        batch_size = x.size(0)
        y = x.view(-1, 3, 256, 256)

        if self.use_ms:
            all_feature, f = self.extract_feat(y)
        else:
            all_feature = self.resnet(y)
            all_feature = self.maxpool(all_feature)

        feature = all_feature.view(batch_size, 20, -1)
        gc5 = self.GCN(feature, A, feature)
        fc_in = gc5.view(gc5.size()[0], -1)
        score = torch.mean(fc_in, dim=1).unsqueeze(1)

        if requires_loss:
            return score, label, self.loss_build(score, label)
        else:
            return score, label

if __name__ == '__main__':
    print(models.resnet18(pretrained=False))

    import networkx as nx
    import walker

    # create a random graph
    G = nx.random_partition_graph([1000] * 15, .01, .001)

    # generate random walks
    X = walker.random_walks(G, n_walks=50, walk_len=25)