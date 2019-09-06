from __future__ import division
from __future__ import print_function

import time
import os
import argparse
import numpy as np

import torch
from tqdm import tqdm
import wandb
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from utils import load_data, accuracy, compute_weighted_adj, log_values, load_dataset, save_dataset
from tensorboard_logger import Logger as TbLogger
from models import GCN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=60,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-6,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.2,
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

# Load OP data
# load created dataset from path
train_dataset = load_dataset(filename='/nfs/team/mlo/aswamy/code/learn_comb_opt_op/data/op/op_k_sols_dist100_trainN10000_seed1111.pkl')
val_dataset = load_dataset(filename='/nfs/team/mlo/aswamy/code/learn_comb_opt_op/data/op/op_k_sols_dist100_valN10000_seed2222.pkl')

train_dataset = train_dataset[0:1000]
val_dataset = val_dataset[0:500]
# ----------------------------------------------
# Train data generated and with random labels
# train_loc = torch.FloatTensor(100, 2).uniform_(0, 1)
# train_depot = torch.FloatTensor(2).uniform_(0, 1)
# train_prize_ = (train_depot[None, :] - train_loc).norm(p=2, dim=-1)
# train_prize = (1 + (train_prize_ / train_prize_.max(dim=-1, keepdim=True)[0] * 99).int()).float() / 100.

# train_features = torch.cat([train_loc, torch.unsqueeze(torch.detach(train_prize), dim=1)], dim=1)
# train_labels = torch.from_numpy(np.random.choice([0, 1], size=(100, 1), p=[0.5, 0.5]))
# train_adj = torch.from_numpy(compute_weighted_adj(train_loc))

# Val data with random labels
# val_loc = torch.FloatTensor(100, 2).uniform_(0, 1)
# val_depot = torch.FloatTensor(2).uniform_(0, 1)
# val_prize_ = (val_depot[None, :] - val_loc).norm(p=2, dim=-1)
# val_prize = (1 + (val_prize_ / val_prize_.max(dim=-1, keepdim=True)[0] * 99).int()).float() / 100.

# val_features = torch.cat([val_loc, torch.unsqueeze(torch.detach(val_prize), dim=1)], dim=1)
# val_labels = torch.from_numpy(np.random.choice([0, 1], size=(100, 1), p=[0.5, 0.5]))
# val_adj = torch.from_numpy(compute_weighted_adj(val_loc))


# test data with random label
# test_loc = torch.FloatTensor(100, 2).uniform_(0, 1)
# test_depot = torch.FloatTensor(2).uniform_(0, 1)
# test_prize_ = (test_depot[None, :] - test_loc).norm(p=2, dim=-1)
# test_prize = (1 + (test_prize_ / test_prize_.max(dim=-1, keepdim=True)[0] * 99).int()).float() / 100.

# test_features = torch.cat([test_loc, torch.unsqueeze(torch.detach(test_prize), dim=1)], dim=1)
# test_labels = torch.from_numpy(np.random.choice([0, 1], size=(100, 1), p=[0.5, 0.5]))
# test_adj = torch.from_numpy(compute_weighted_adj(test_loc))
# -----------------------------------------------


# initialize wandb
wandb.init(project='gcn_node_classification')
# load all arguments to config to save as hyperparameters
wandb.config.update(args)


# Optionally configure tensorboard
args.run_name = "{}_{}".format(args.run_name, time.strftime("%Y%m%dT%H%M%S"))
tb_logger = None
if not args.no_tensorboard:
    tb_logger = TbLogger(os.path.join(args.log_dir, args.run_name))

# Model and optimizer
model = GCN(nfeat=train_dataset[0].x.shape[1],
            nhid=args.hidden,
            nclass=2,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
# optimizer = optim.SGD(model.parameters(), lr=args.lr)

# # save pytorch model and track all of the gradients and optionally parameters
wandb.watch(model, log='all')  # "gradients", "parameters", "all", or None.

# if args.cuda:
#     model.cuda()
#     # train_features = train_features.cuda()
#     # train_adj = train_adj.cuda().type(torch.cuda.FloatTensor)
#     # train_labels = train_labels.cuda().type(torch.cuda.LongTensor)

#     # val_features = val_features.cuda()
#     # val_adj = val_adj.cuda()
#     # val_labels = val_labels.cuda().type(torch.cuda.LongTensor)

#     test_features = test_features.cuda()
#     test_adj = test_adj.cuda()
#     test_labels = test_labels.cuda().type(torch.cuda.LongTensor)



def train(model, train_features, train_labels, train_adj):
    model.train()
    optimizer.zero_grad()
    train_output, _ = model(train_features, train_adj)

    # loss when binary classification (probability threshold based)
    # loss_train = F.binary_cross_entropy_with_logits(train_output, train_labels)

    # loss when softmax is used
    loss_train = nn.NLLLoss()(train_output, train_labels.squeeze(1))

    acc_train = accuracy(train_output, train_labels)
    loss_train.backward()
    optimizer.step()

    # if not args.fastmode:
    # Evaluate validation set performance separately,
    # deactivates dropout during validation run.
    # model.eval()
    # val_output, _ = model(val_features, val_adj)
    # loss_val =  nn.NLLLoss()(val_output, val_labels.squeeze(1))
    # acc_val = accuracy(val_output, val_labels)
    
    return loss_train, acc_train


def eval(model, val_features, val_labels, val_adj):
    model.eval()
    val_output, _ = model(val_features, val_adj)
    loss_val =  nn.NLLLoss()(val_output, val_labels.squeeze(1))
    acc_val = accuracy(val_output, val_labels)
    
    return loss_val, acc_val


# Train model
t_total = time.time()
loss_train_lst = []
loss_val_lst = []
acc_train_lst = []
acc_val_lst = []
for epoch in range(args.epochs):
    running_loss_train = 0
    running_acc_train = 0
    for i in tqdm(range(len(train_dataset))):
        # data with random labels
        # train_loc = torch.FloatTensor(100, 2).uniform_(0, 1)
        # train_depot = torch.FloatTensor(2).uniform_(0, 1)
        # train_prize_ = (train_depot[None, :] - train_loc).norm(p=2, dim=-1)
        # train_prize = (1 + (train_prize_ / train_prize_.max(dim=-1, keepdim=True)[0] * 99).int()).float() / 100.

        # train_features = torch.cat([train_loc, torch.unsqueeze(torch.detach(train_prize), dim=1)], dim=1)
        # train_labels = torch.from_numpy(np.random.choice([0, 1], size=(100, 1), p=[0.5, 0.5]))
        # train_adj = torch.from_numpy(compute_weighted_adj(train_loc))

        # data with labels based on approximate solutions
        train_features = train_dataset[i].x
        # val_features = val_dataset[i].x
        # convert true score into labels
        train_labels = train_dataset[i].y > 0
        # val_labels = val_dataset[i].y > 0
        # create weighted adj using only coordinates
        train_adj = torch.from_numpy(compute_weighted_adj(train_dataset[i].x[:, :-2]))
        # val_adj = torch.from_numpy(compute_weighted_adj(val_dataset[i].x[:, :-2]))

        if args.cuda:
            model.cuda()
            train_features = train_features.cuda()
            train_adj = train_adj.cuda().type(torch.cuda.FloatTensor)
            train_labels = train_labels.cuda().type(torch.cuda.LongTensor)

            # val_features = val_features.cuda()
            # val_adj = val_adj.cuda().type(torch.cuda.FloatTensor)
            # val_labels = val_labels.cuda().type(torch.cuda.LongTensor)

        loss_train, acc_train = train(model, train_features, train_labels, train_adj)
        
        running_loss_train += loss_train.item()
        running_acc_train += acc_train.item()
        
    running_loss_val = 0
    running_acc_val = 0
    for i in tqdm(range(len(val_dataset))):
        val_features = val_dataset[i].x
        val_labels = val_dataset[i].y > 0
        val_adj = torch.from_numpy(compute_weighted_adj(val_dataset[i].x[:, :-2]))

        if args.cuda:
            model.cuda()
            val_features = val_features.cuda()
            val_adj = val_adj.cuda().type(torch.cuda.FloatTensor)
            val_labels = val_labels.cuda().type(torch.cuda.LongTensor)
        
        loss_val, acc_val = eval(model, val_features, val_labels, val_adj)

        running_loss_val += loss_val.item()
        running_acc_val += acc_val.item()


    if epoch % 5 == 0:
        print('Epoch: {:04d}'.format(epoch+1),
            'loss_train: {:.4f}'.format(running_loss_train / len(train_dataset)),
            'acc_train: {:.4f}'.format(running_acc_train / len(train_dataset)),
            'loss_val: {:.4f}'.format(running_loss_val / len(train_dataset)),
            'acc_val: {:.4f}'.format(running_acc_val / len(train_dataset)))

    # tensorboard logging
    # log_values(epoch, loss_train, loss_val, acc_train, acc_val, tb_logger)

    # wandb logging
    wandb.log({
        "train_loss": running_loss_train / len(train_dataset),
        "val_loss": running_loss_val / len(val_dataset),
        "train_accu": running_acc_train / len(train_dataset),
        "val_accu": running_acc_val / len(val_dataset)
    })

    # loss_train_lst.append(running_loss_train / args.dataset_size)
    # loss_val_lst.append(running_loss_val / args.dataset_size)
    # acc_train_lst.append(running_acc_train / args.dataset_size)
    # acc_val_lst.append(running_acc_val / args.dataset_size)

print("Optimization Finished!")
# import matplotlib.pyplot as plt
# fig1 = plt.figure()
# plt.plot(loss_train_lst)
# plt.plot(loss_val_lst)
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.legend(['train', 'val'], loc='upper left')
# plt.title('Loss vs Epoch')
# fig1.savefig('images/model_train_val_loss.png')


# fig2 = plt.figure()
# plt.plot(acc_train_lst)
# plt.plot(acc_val_lst)
# plt.xlabel('epoch')
# plt.ylabel('accuracy')
# plt.legend(['train', 'val'], loc='upper_left')
# plt.title('Accuracy vs Epoch')
# fig2.savefig('images/model_train_val_acc.png')

# print("plots saved in 'images' folder")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
# test()
print('plotting embeddings')
import itertools
# get and plot embeddings
with torch.no_grad():
    graph_list = [1, 10, 20]
    for i in tqdm(graph_list):
        train_adjaceny = torch.from_numpy(compute_weighted_adj(train_dataset[i].x[:, :-2]))
        train_labels = list(itertools.chain(*((train_dataset[i].y>0).cpu().numpy())))
        
        if args.cuda:
            train_adjaceny = train_adjaceny.cuda().type(torch.cuda.FloatTensor)
            train_features = train_dataset[i].x.cuda()
            

        _, node_embeds = model(train_features, train_adjaceny)
        save_dataset(node_embeds, "images/ins_"+str(i+1)+"embeds.pkl")

        # plot tsne
        import matplotlib
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        all_embeddings = node_embeds.cpu().numpy()

        for perp in list([5, 10, 15, 20]):
            tsne = TSNE(n_components=2, perplexity = perp, learning_rate = 10,
                        n_iter = 1000, random_state=5)
            transform_2d = tsne.fit_transform(all_embeddings)
        #     plt.figure(figsize=(8,8))
            # input('enter')
            # print(train_labels.shape, transform_2d.shape)
            plt.scatter(transform_2d[0:20, 0], transform_2d[0:20,1], c=train_labels[0:20])
            for i, score in enumerate(train_labels[0:20]):
                plt.annotate(score,(transform_2d[:, 0][i], transform_2d[:,1][i]))
            plt.title("ins_node_embeddings(perp="+str(perp)+')')
            plt.xlabel('dim_1')
            plt.ylabel('dim_2')
            plt.savefig('images/ins'+str(i+1)+'_node_embeddings_label_binary_perp_'+str(perp))
            plt.show()