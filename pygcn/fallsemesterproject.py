
# Commented out IPython magic to ensure Python compatibility.
import os
import logging
import sys

import zipfile
import scipy.io

import time
import argparse

import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

from pygcn.layers import GraphConvolution
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import scipy.sparse as sp
import numpy as np

from pygcn.utils import load_data, accuracy, load_data_pubmed
from pygcn.models import GCN
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
from torch.backends import cudnn

from pygcn.train import make_model, train_iterator, test,  multiple_run_scores 

# %matplotlib inline
import matplotlib.pyplot as plt



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
parser.add_argument('--earlyStop', type=int, default=10,
                    help='windows of the early stopping.')

import sys
sys.argv=['']
del sys
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

def plot_func(epochs, accuracies, name, title):
  fig = plt.gcf()
  fig.set_size_inches(12.5, 7.5)
  x = range(epochs+1)
  # print(len(x))
  # print(len(accuracies))
  fig = plt.figure
  plt.plot(x, accuracies)

  plt.ylabel('score', fontsize = 16)
  plt.xlabel('# Epoch', fontsize = 16)

  plt.title(title, fontsize = 20)

  plt.grid()

  plt.savefig(name, dpi=100)
  plt.show()


# Load data
adj, features, labels = load_data(dataset= 'cora')
# load the indexes
idx_train, idx_val, idx_test = find_indexes(features, d = 0.052)
# the dataset is not balanced regarding the label of the nodes

"""###*Hyper parameter tuning (cora)*"""

# validation accuracies
accuracies = [] 
param_grid = {
    'lr' : [1e-2, 5e-2, 8e-3],
    'weight_decay': [5e-4, 5e-3, 9e-4],
    'epoch': [200, 250, 300]
}
for params in ParameterGrid(param_grid):
  args.lr = params['lr']
  args.weight_decay = params['weight_decay']
  args.epochs = params['epoch']
  print('learning rate = {}, weight decay = {}, epochs = {}'.format(args.lr, args.weight_decay, args.epochs))
  tmp_acc, _, _, _ = train_iterator(args, features, labels, adj, idx_train, idx_val)
  accuracies.append(tmp_acc)

id = np.argmax(accuracies)
best_params = ParameterGrid(param_grid)[id]
print(best_params)
args.lr = best_params['lr']
args.weight_decay = best_params['weight_decay']
args.epochs = best_params['epoch']
_, best_model, val_acc_list, eff_epoch = train_iterator(args, features, labels, adj, idx_train, idx_val)

plot_func(eff_epoch, val_acc_list, 'cora_valAcc_trend', 'Model Learning Curve(Cora)')

# Testing the best model
test(best_model,features, adj, labels, idx_test)

"""###*Rand split*

"""

list_acc = multiple_run_scores(args, features, labels, adj, idx_train, idx_val, idx_test, d= 0.052)

"""###*2) CITESEER Data Set*"""

# Load data
adj2, features2, labels2 = load_data(dataset= 'citeseer')
# load the indexes
idx_train2, idx_val2, idx_test2 = find_indexes(features, d = 0.036)
# the dataset is not balanced regarding the label of the nodes

"""###*Hyper parameter tuning (citeseer)*"""

# validation accuracies
accuracies2 = [] 
param_grid = {
    'lr' : [1e-2, 5e-2, 8e-3],
    'weight_decay': [5e-4, 5e-3, 9e-4],
    'epoch': [200, 250, 300]
}
for params in ParameterGrid(param_grid):
  args.lr = params['lr']
  args.weight_decay = params['weight_decay']
  args.epochs = params['epoch']
  print('learning rate = {}, weight decay = {}, epochs = {}'.format(args.lr, args.weight_decay, args.epochs))
  tmp_acc, _, _, _ = train_iterator(args, features2, labels2, adj2, idx_train2, idx_val2)
  accuracies2.append(tmp_acc)

id2 = np.argmax(accuracies2)
best_params2 = ParameterGrid(param_grid)[id2]
print(best_params2)
args.lr = best_params2['lr']
args.weight_decay = best_params2['weight_decay']
args.epochs = best_params2['epoch']
_, best_model2, val_acc_list2, eff_epoch2 = train_iterator(args, features2, labels2, adj2, idx_train2, idx_val2)

plot_func(eff_epoch2, val_acc_list2, 'citeseer_valAcc_trend', 'Model Learning Curve(Citeseer)')

# Testing the best model
test(best_model2,features2, adj2, labels2, idx_test2)

list_acc2 = multiple_run_scores(args, features2, labels2, adj2, idx_train2, idx_val2, idx_test2, d=0.036)

"""###*3) PUBMED DataSet*"""

# Load data
# set the flag to false for the first run
adj3, features3, labels3 = load_data_pubmed(flag= False)
# load the indexes
idx_train3, idx_val3, idx_test3 = find_indexes(features3, d = 0.003)
# the dataset is not balanced regarding the label of the nodes

"""####*Hyper parameter Tuning(Pubmed)*"""

# validation accuracies
accuracies3 = [] 
param_grid = {
    'lr' : [1e-2, 5e-2, 8e-3],
    'weight_decay': [5e-4, 5e-3, 9e-4],
    'epoch': [200, 250, 300]
}
for params in ParameterGrid(param_grid):
  args.lr = params['lr']
  args.weight_decay = params['weight_decay']
  args.epochs = params['epoch']
  print('learning rate = {}, weight decay = {}, epochs = {}'.format(args.lr, args.weight_decay, args.epochs))
  tmp_acc, _, _, _ = train_iterator(args, features3, labels3, adj3, idx_train3, idx_val3)
  accuracies3.append(tmp_acc)

id3 = np.argmax(accuracies3)
best_params3 = ParameterGrid(param_grid)[id3]
print(best_params3)
args.lr = best_params3['lr']
args.weight_decay = best_params3['weight_decay']
args.epochs = best_params3['epoch']
_, best_model3, val_acc_list3, eff_epoch3 = train_iterator(args, features3, labels3, adj3, idx_train3, idx_val3)

plot_func(eff_epoch3, val_acc_list3, 'pubmed_valAcc_trend', 'Model Learning Curve(Pubmed)')

# Testing the best model
test(best_model3,features3, adj3, labels3, idx_test3)

list_acc3 = multiple_run_scores(args, features3, labels3, adj3, idx_train3, idx_val3, idx_test3, d = 0.003)

"""####*Plot in the same Figure*"""

def plot_3(epoch_c, cora, epoch_ci, citeseer, epoch_p, pubmed):
  x = range(epoch_c)
  fig = plt.gcf()
  fig.set_size_inches(10.5, 7.5)  
  fig = plt.figure
  plt.plot(x, cora, label='cora_valAcc_trend')
  x = range(epoch_ci)
  plt.plot(x, citeseer, label='citeseer_valAcc_trend')
  x = range(epoch_p)
  plt.plot(x, pubmed, label='pubmed_valAcc_trend')

  plt.ylabel('Score', fontsize = 16)
  plt.xlabel('# Epochs', fontsize = 16)

  plt.title('Models Learning curves', fontsize = 20)

  plt.legend(fontsize = 16)
  plt.grid()

  plt.savefig('summary.png', dpi=100)
  plt.show()

plot_3(best_params['epoch'], val_acc_list, best_params2['epoch'], val_acc_list2, best_params3['epoch'], val_acc_list3 )
