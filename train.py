from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import os
import glob
from torch.autograd import Variable

import gcn_utils
from utils import accuracy
from models import GAT, GCN

"""
python3 train.py --model GCN --dataset cora --epochs 10000 --lr 0.01 --weight_decay 5e-4 --hidden 16 --dropout 0.5
python3 train.py --model GCN --dataset citeseer --epochs 10000 --lr 0.01 --weight_decay 5e-4 --hidden 16 --dropout 0.5
python3 train.py --model GCN --dataset pubmed --epochs 10000 --lr 0.01 --weight_decay 5e-4 --hidden 16 --dropout 0.5
"""

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='GAT', help='GAT or GCN.')
parser.add_argument('--dataset', type=str, default='mr', help='cora citeseer pubmed')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=7, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')

# args = parser.parse_args([])
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
if args.dataset in 'cora citeseer pubmed'.split():
    adj, features, labels, idx_train, idx_val, idx_test = gcn_utils.load_data(args.dataset)
if args.dataset == 'mr':
    adj, features, labels, idx_train, idx_val, idx_test = gcn_utils.gcn_text_mr_load_data()
else:
    adj, features, labels, idx_train, idx_val, idx_test = gcn_utils.load_corpus(args.dataset)

# Model and optimizer
if args.model == 'GAT':
    model = GAT(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=int(labels.max()) + 1,
                dropout=args.dropout,
                nheads=args.nb_heads,
                alpha=args.alpha)
elif args.model == 'GCN':
    model = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=int(labels.max()) + 1,
                dropout=args.dropout)
else:
    raise ValueError("Model {} not registered".format(args.model))

optimizer = optim.Adadelta(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

features, adj, labels = Variable(features), Variable(adj), Variable(labels)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_val.data.item()


# Verify which device is in use
# print(f"Device: {torch.cuda.get_device_name(0)}")

# Train model
t_total = time.time()
loss_values = []
bad_counter = 0
best = np.inf
best_epoch = 0
for epoch in range(args.epochs):
    loss_values.append(train(epoch))

    if loss_values[-1] < best:
        if args.cuda:
            model.cpu()
        torch.save(model.state_dict(), '{}_{}_{}.pkl'.format(model._get_name(), args.dataset, epoch))
        if args.cuda:
            model.cuda()
        best = loss_values[-1]
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        print("Patience {0} exceeded. Best in last {0} epochs is {1:.4f}.".format(args.patience, best))
        break

    files = glob.glob('{}_{}_*.pkl'.format(model._get_name(), args.dataset))
    for file in files:
        epoch_nb = int(''.join(filter(str.isdigit, file)))
        if epoch_nb < best_epoch:
            os.remove(file)

print('Saved model {}_{}_{}.pkl'.format(model._get_name(), args.dataset, best_epoch))

files = glob.glob('{}_{}_*.pkl'.format(model._get_name(), args.dataset))
for file in files:
    epoch_nb = int(''.join(filter(str.isdigit, file)))
    if epoch_nb > best_epoch:
        os.remove(file)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))


def compute_test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data.item()),
          "accuracy= {:.4f}".format(acc_test.data.item()))


compute_test()
