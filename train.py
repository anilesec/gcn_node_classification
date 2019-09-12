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

from utils import load_data, accuracy, compute_weighted_adj
from utils import log_values, load_dataset, save_dataset
from tensorboard_logger import Logger as TbLogger
from models import GCN


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, epoch):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, str(epoch))
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, str(epoch))
            self.counter = 0

    def save_checkpoint(self, val_loss, model, model_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'pretrained/checkpoint.pt')
        self.val_loss_min = val_loss


def train(model, train_features, train_labels, train_adj):
    model.train()
    optimizer.zero_grad()
    train_output, _ = model(train_features, train_adj)

    # loss when softmax is used
    loss_train = nn.NLLLoss()(train_output, train_labels.squeeze(1))
    acc_train = accuracy(train_output, train_labels)
    loss_train.backward()
    optimizer.step()

    return loss_train, acc_train


def eval(model, val_features, val_labels, val_adj):
    model.eval()
    val_output, _ = model(val_features, val_adj)
    loss_val =  nn.NLLLoss()(val_output, val_labels.squeeze(1))
    acc_val = accuracy(val_output, val_labels)
    
    return loss_val, acc_val


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    # parser.add_argument('--fastmode', action='store_true', default=False,
    #                     help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Initial learning rate.')
    # parser.add_argument('--weight_decay', type=float, default=0.000001,
    #                     help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Number of hidden units.')
    parser.add_argument('--num_hid_layers', type=int, default=2,
                        help='Number of hidden layers.')
    parser.add_argument('--es_patience', type=int, default=100,
                        help='early stopping buffer to check for metric improvement')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--no_tensorboard', action='store_true', 
                        help='Disable logging TensorBoard files')
    parser.add_argument('--run_name', default='run', 
                        help='Name to identify the run')
    parser.add_argument('--log_dir', default='logs', 
                        help='Directory to write TensorBoard information to')
    # parser.add_argument('--dataset_size', type=int, default=100, 
    #                     help='total numnber of instances in training set')                    

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

    # taking only limited samples for trainnig and validation
    train_dataset = train_dataset[0:10000]
    val_dataset = val_dataset[0:5000]

    # disable sync
    # os.environ['WANDB_MODE'] = 'dryrun'

    # initialize wandb
    wandb.init(project='gcn_node_classification_after_tuning')
    # load all arguments to config to save as hyperparameters
    wandb.config.update(args)

    # Optionally configure tensorboard
    args.run_name = "{}_{}".format(args.run_name, time.strftime("%Y%m%dT%H%M%S"))
    tb_logger = None
    if not args.no_tensorboard:
        tb_logger = TbLogger(os.path.join(args.log_dir, args.run_name))

    # model
    model = GCN(input_dim=train_dataset[0].x.shape[1],
                num_hid_layers=args.num_hid_layers,
                hidden_dim=args.hidden_dim,
                num_class=2,
                dropout=args.dropout)
    # optimizer
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    # save pytorch model and track all of the gradients and optionally parameters
    wandb.watch(model, log='all')  # "gradients", "parameters", "all", or None.

    # Train model
    t_total = time.time()
    
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=args.es_patience, verbose=True)

    for epoch in range(args.epochs):
        running_loss_train = 0
        running_acc_train = 0
        for i in tqdm(range(len(train_dataset))):
            # data with labels based on approximate solutions
            # features are centered to depot location
            train_features = train_dataset[i].x - torch.cat([train_dataset[i].x[0][:2], torch.tensor([0.0, 0.0])], dim=0)
            # convert true score into labels
            train_labels = train_dataset[i].y > 0
            # create weighted adj using only coordinates
            train_adj = torch.from_numpy(compute_weighted_adj(train_dataset[i].x[:, :-2]))

            if args.cuda:
                model.cuda()
                train_features = train_features.cuda()
                train_adj = train_adj.cuda().type(torch.cuda.FloatTensor)
                train_labels = train_labels.cuda().type(torch.cuda.LongTensor)

            loss_train, acc_train = train(model, train_features, train_labels, train_adj)
            
            running_loss_train += loss_train.item()
            running_acc_train += acc_train.item()
            
        running_loss_val = 0
        running_acc_val = 0
        for i in tqdm(range(len(val_dataset))):
            val_features = val_dataset[i].x - torch.cat([val_dataset[i].x[0][:2], torch.tensor([0.0, 0.0])], dim=0)
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

        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping((running_loss_val / len(val_dataset)), model, epoch)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
        if epoch % 5 == 0:
            print('Epoch: {:04d}'.format(epoch+1),
                'loss_train: {:.4f}'.format(running_loss_train / len(train_dataset)),
                'acc_train: {:.4f}'.format((running_acc_train / len(train_dataset) * 100)),
                'loss_val: {:.4f}'.format(running_loss_val / len(val_dataset)),
                'acc_val: {:.4f}'.format((running_acc_val / len(val_dataset) * 100)))
        
        # tensorboard logging
        log_values(epoch, loss_train, loss_val, acc_train, acc_val, tb_logger)

        # wandb logging
        wandb.log({
            "train_loss": running_loss_train / len(train_dataset),
            "val_loss": running_loss_val / len(val_dataset),
            "train_accu": (running_acc_train / len(train_dataset)) * 100,
            "val_accu": (running_acc_val / len(val_dataset)) * 100
        })

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

"""
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
                plt.scatter(transform_2d[0:20, 0], transform_2d[0:20,1], c=train_labels[0:20])
                for i, score in enumerate(train_labels[0:20]):
                    plt.annotate(score,(transform_2d[:, 0][i], transform_2d[:,1][i]))
                plt.title("ins_node_embeddings(perp="+str(perp)+')')
                plt.xlabel('dim_1')
                plt.ylabel('dim_2')
                plt.savefig('images/ins'+str(i+1)+'_node_embeddings_label_binary_perp_'+str(perp))
                # plt.show()

                """