import torch
import torch.nn as nn
import torch
import torch.nn.functional as F
from layers import GraphConvolution


# class GCN(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_class, dropout):
#         super(GCN, self).__init__()

#         self.gc1 = GraphConvolution(input_dim, hidden_dim)
#         self.gc2 = GraphConvolution(hidden_dim, hidden_dim)
#         self.out = nn.Linear(in_features=hidden_dim, out_features=num_class)
#         self.dropout = dropout

#     def forward(self, x, adj):
#         x = torch.tanh(self.gc1(x, adj))
#         # x = F.dropout(x, self.dropout, training=self.training)
#         x = self.gc2(x, adj)
#         x = torch.tanh(x)
#         node_embeddings = x
#         # x = F.dropout(x, self.dropout, training=self.training)
#         x = self.out(x)

#         # return output of softmax layer when NLLLoss is used
#         return nn.LogSoftmax(dim=1)(x), node_embeddings


class GCN(nn.Module):
    def __init__(self, input_dim, num_hid_layers, hidden_dim, num_class, dropout):
        super(GCN, self).__init__()

        self.gcn = nn.ModuleList()
        self.gcn.append(GraphConvolution(input_dim, hidden_dim))
        for k in range(num_hid_layers):
            self.gcn.append(GraphConvolution(hidden_dim, hidden_dim))
            # Enable dropout if necessary
            # self.gcn.append(F.dropout(y, self.dropout, training=self.training))
        self.out = nn.Linear(hidden_dim, num_class)

    def forward(self, x, adj_w):
        y = x
        for i in range(len(self.gcn)):
            y = torch.tanh(self.gcn[i](y, adj_w))
        node_embeddings = y
        y = self.out(y)

        return nn.LogSoftmax(dim=1)(y), node_embeddings
