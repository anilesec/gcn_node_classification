from __future__ import division
from __future__ import print_function

import time
import os
import argparse
import numpy as np

import torch
# import wandb
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy, compute_weighted_adj, log_values
from tensorboard_logger import Logger as TbLogger
from models import GCN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-1,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.8,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--no_tensorboard', action='store_true', 
                    help='Disable logging TensorBoard files')
parser.add_argument('--run_name', default='run', 
                    help='Name to identify the run')
parser.add_argument('--log_dir', default='logs', 
                    help='Directory to write TensorBoard information to')
parser.add_argument('--dataset_size', type=int, default=100, 
                    help='total numnber of instances in training set')                    

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

train_features = torch.cat([train_loc, torch.unsqueeze(torch.detach(train_prize), dim=1)], dim=1)
train_labels = torch.from_numpy(np.random.choice([0, 1], size=(100, 1), p=[0.5, 0.5]))
train_adj = torch.from_numpy(compute_weighted_adj(train_loc))
# train_adj = torch.from_numpy(np.random.rand(100,100)) # torch.ones(size=(100, 100))


# Val data
val_loc = torch.FloatTensor(100, 2).uniform_(0, 1)
val_depot = torch.FloatTensor(2).uniform_(0, 1)
val_prize_ = (val_depot[None, :] - val_loc).norm(p=2, dim=-1)
val_prize = (1 + (val_prize_ / val_prize_.max(dim=-1, keepdim=True)[0] * 99).int()).float() / 100.

val_features = torch.cat([train_loc, torch.unsqueeze(torch.detach(train_prize), dim=1)], dim=1)
val_labels = torch.from_numpy(np.random.choice([0, 1], size=(100, 1), p=[0.5, 0.5]))
val_adj = torch.from_numpy(compute_weighted_adj(val_loc))
# val_adj = torch.ones(size=(100, 100))


# test data
test_loc = torch.FloatTensor(100, 2).uniform_(0, 1)
test_depot = torch.FloatTensor(2).uniform_(0, 1)
test_prize_ = (test_depot[None, :] - test_loc).norm(p=2, dim=-1)
test_prize = (1 + (test_prize_ / test_prize_.max(dim=-1, keepdim=True)[0] * 99).int()).float() / 100.

test_features = torch.cat([train_loc, torch.unsqueeze(torch.detach(train_prize), dim=1)], dim=1)
test_labels = torch.from_numpy(np.random.choice([0, 1], size=(100, 1), p=[0.5, 0.5]))
test_adj = torch.from_numpy(compute_weighted_adj(test_loc))
# test_adj = torch.ones(size=(100, 100))
# -----------------------------------------------


# # initialize wandb
# wandb.init(project='gcn_node_classification')
# # load all arguments to config to save as hyperparameters
# wandb.config.update(args)



# Optionally configure tensorboard
args.run_name = "{}_{}".format(args.run_name, time.strftime("%Y%m%dT%H%M%S"))
tb_logger = None
if not args.no_tensorboard:
    tb_logger = TbLogger(os.path.join(args.log_dir, args.run_name))


# Model and optimizer
model = GCN(nfeat=train_features.shape[1],
            nhid=args.hidden,
            nclass=1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
# optimizer = optim.SGD(model.parameters(), lr=args.lr)

# # save pytorch model and track all of the gradients and optionally parameters
# wandb.watch(model, log='all')  # "gradients", "parameters", "all", or None.

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



def train(epoch, train_features, train_labels, train_adj):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    train_output = model(train_features, train_adj)
    loss_train = F.binary_cross_entropy_with_logits(train_output, train_labels)
    acc_train = accuracy(train_output, train_labels)
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        val_output = model(val_features, val_adj)

    loss_val = F.binary_cross_entropy_with_logits(val_output, val_labels)
    acc_val = accuracy(val_output, val_labels)
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


    # tensorboard logging
    log_values(epoch, loss_train, loss_val, acc_train, acc_val, tb_logger)

    # wandb logging
    # wandb.log({
    #     "train_loss": loss_train,
    #     "val_loss": loss_val,
    #     "train_accu": acc_train,
    #     "val_accu": acc_val
    # })
    
    return loss_train, loss_val, acc_train, acc_val


def test():
    model.eval()
    test_output = model(test_features, test_adj)
    loss_test = F.binary_cross_entropy_with_logits(test_output, test_labels)
    acc_test = accuracy(test_output, test_labels)
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
loss_train_lst = []
loss_val_lst = []
acc_train_lst = []
acc_val_lst = []
for epoch in range(args.epochs):
    running_loss_train = 0
    running_loss_val = 0
    running_acc_train = 0
    running_acc_val = 0
    for i in range(args.dataset_size):
        train_loc = torch.FloatTensor(100, 2).uniform_(0, 1)
        train_depot = torch.FloatTensor(2).uniform_(0, 1)
        train_prize_ = (train_depot[None, :] - train_loc).norm(p=2, dim=-1)
        train_prize = (1 + (train_prize_ / train_prize_.max(dim=-1, keepdim=True)[0] * 99).int()).float() / 100.

        train_features = torch.cat([train_loc, torch.unsqueeze(torch.detach(train_prize), dim=1)], dim=1)
        train_labels = torch.from_numpy(np.random.choice([0, 1], size=(100, 1), p=[0.5, 0.5]))
        train_adj = torch.from_numpy(compute_weighted_adj(train_loc))
        # train_adj = torch.from_numpy(np.random.rand(100,100)) # torch.ones(size=(100, 100))

        if args.cuda:
            model.cuda();
            train_features = train_features.cuda();
            train_adj = train_adj.cuda().type(torch.cuda.FloatTensor);
            train_labels = train_labels.cuda().type(torch.cuda.FloatTensor);

        loss_train, loss_val, acc_train, acc_val = train(epoch, train_features, train_labels, train_adj)
        running_loss_train += loss_train.item()
        running_loss_val += loss_val.item()
        running_acc_train += acc_train.item()
        running_acc_val += acc_val.item()
    
    loss_train_lst.append(running_loss_train / args.dataset_size)
    loss_val_lst.append(running_loss_val / args.dataset_size)
    acc_train_lst.append(running_acc_train / args.dataset_size)
    acc_val_lst.append(running_acc_val / args.dataset_size)

print("Optimization Finished!")
import matplotlib.pyplot as plt
fig1 = plt.figure()
plt.plot(loss_train_lst)
plt.plot(loss_val_lst)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'], loc='upper left')
plt.title('Loss vs Epoch')
fig1.savefig('images/model_train_val_loss.png')


fig2 = plt.figure()
plt.plot(acc_train_lst)
plt.plot(acc_val_lst)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train', 'val'], loc='upper_left')
plt.title('Accuracy vs Epoch')
fig2.savefig('images/model_train_val_acc.png')

print("plots saved in 'images' folder")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
