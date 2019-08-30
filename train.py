from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy
from models import GCN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
# adj, features, labels, idx_train, idx_val, idx_test = load_data()
# ----------------------------------------------
# Train data
train_loc = torch.FloatTensor(100, 2).uniform_(0, 1)
train_depot = torch.FloatTensor(2).uniform_(0, 1)
train_prize_ = (train_depot[None, :] - train_loc).norm(p=2, dim=-1)
train_prize = (1 + (train_prize_ / train_prize_.max(dim=-1, keepdim=True)[0] * 99).int()).float() / 100.

train_features = torch.cat([train_loc, torch.unsqueeze(torch.tensor(train_prize), dim=1)], dim=1)
train_labels = torch.from_numpy(np.random.choice([0, 1], size=(100, 1), p=[0.5, 0.5]))
train_adj = torch.from_numpy(np.random.rand(100,100)) # torch.ones(size=(100, 100))

# Val data
val_loc = torch.FloatTensor(100, 2).uniform_(0, 1)
val_depot = torch.FloatTensor(2).uniform_(0, 1)
val_prize_ = (val_depot[None, :] - val_loc).norm(p=2, dim=-1)
val_prize = (1 + (val_prize_ / val_prize_.max(dim=-1, keepdim=True)[0] * 99).int()).float() / 100.

val_features = torch.cat([train_loc, torch.unsqueeze(torch.tensor(train_prize), dim=1)], dim=1)
val_labels = torch.from_numpy(np.random.choice([0, 1], size=(100, 1), p=[0.3, 0.7]))
val_adj = torch.ones(size=(100, 100))


# test data
test_loc = torch.FloatTensor(100, 2).uniform_(0, 1)
test_depot = torch.FloatTensor(2).uniform_(0, 1)
test_prize_ = (test_depot[None, :] - test_loc).norm(p=2, dim=-1)
test_prize = (1 + (test_prize_ / test_prize_.max(dim=-1, keepdim=True)[0] * 99).int()).float() / 100.

test_features = torch.cat([train_loc, torch.unsqueeze(torch.tensor(train_prize), dim=1)], dim=1)
test_labels = torch.from_numpy(np.random.choice([0, 1], size=(100, 1), p=[0.8, 0.2]))
test_adj = torch.ones(size=(100, 100))
# -----------------------------------------------


# Model and optimizer
model = GCN(nfeat=train_features.shape[1],
            nhid=args.hidden,
            nclass=1,
            dropout=args.dropout)
# optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
optimizer = optim.SGD(model.parameters(), lr=args.lr)

if args.cuda:
    model.cuda()
    train_features = train_features.cuda()
    train_adj = train_adj.cuda().type(torch.cuda.FloatTensor)
    train_labels = train_labels.cuda().type(torch.cuda.FloatTensor)

    val_features = val_features.cuda()
    val_adj = val_adj.cuda()
    val_labels = val_labels.cuda().type(torch.cuda.FloatTensor)

    test_features = val_features.cuda()
    test_adj = val_adj.cuda()
    test_labels = val_labels.cuda().type(torch.cuda.FloatTensor)

    # idx_train = idx_train.cuda()
    # idx_val = idx_val.cuda()
    # idx_test = idx_test.cuda()



def compute_weighted_adj(node_coordinates):
    return adj_w

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    train_output = model(train_features, train_adj)
    # input('enter for after output')
    # print(train_output)
    # input('enter')
    # print(train_labels)
    # loss = torch.nn.BCELoss()
    print(train_output)
    loss_train = F.binary_cross_entropy(train_output, train_labels)
    acc_train = accuracy(train_output, train_labels)
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        val_output = model(val_features, val_adj)

    loss_val = F.binary_cross_entropy(val_output, val_labels)
    acc_val = accuracy(val_output, val_labels)
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    test_output = model(test_features, test_adj)
    loss_test = F.binary_cross_entropy(test_output, test_labels)
    acc_test = accuracy(test_output, test_labels)
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
