import torch
import torch.nn as nn
import torch
import torch.nn.functional as F
from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        # input("enter")
        # print(adj)
        # input('enter to continue')
        x = torch.tanh(self.gc1(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        node_embeddings = x
        x = self.gc2(x, adj)
        
        # return output of softmax layer when NLLLoss is used
        return nn.LogSoftmax(dim=1)(x), node_embeddings
