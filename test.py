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

from utils import load_data, accuracy
from models import GAT, GCN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='GAT', help='GAT or GCN.')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()

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

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

features, adj, labels = Variable(features), Variable(adj), Variable(labels)


def compute_test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data.item()),
          "accuracy= {:.4f}".format(acc_test.data.item()))


# Load model
filename = glob.glob("{}_*.pkl".format(model._get_name()))[0]
if args.cuda:
    model_location = 'cuda'
else:
    model_location = 'cpu'
model.load_state_dict(torch.load(filename, map_location=model_location))
model.eval()
print('Loaded {}.'.format(filename))

# Testing
compute_test()
