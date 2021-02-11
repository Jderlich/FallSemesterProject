from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from pygcn.utils import load_data, accuracy
from pygcn.models import GCN

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

import sys
sys.argv=['']
del sys
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()

# Model and optimizer
def make_model(args, features, labels):
  if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
  # Model and optimizer
  model = GCN(nfeat=features.shape[1],
              nhid=args.hidden,
              nclass=labels.max().item() + 1,
              dropout=args.dropout)
  optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
  return model, optimizer


# Train the given model
def train(epoch, model, optimizer, adj, labels, idx_train, idx_val, features):
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
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
    return loss_val.item(), acc_train.item(), acc_val.item()

 # to train different model
def train_iterator(args, features, labels, adj, idx_train, idx_val):
  #  early stopping counter
  val_acc_list = []
  stop_counter = 0;
  val_loss = 1;
  train_acc = 0;
  eff_epoch = 0;

  # instantiate the model
  model, optimizer = make_model(args, features, labels)

  # Train model
  t_total = time.time()
  for epoch in range(args.epochs):
    tmp_val, train_acc, val_acc = train(epoch, model, optimizer, adj, labels, idx_train, idx_val, features)
    val_acc_list.append(val_acc)
    eff_epoch = epoch
    if ( val_loss == tmp_val):
        stop_counter += 1
    else:
        val_loss = tmp_val
        stop_counter = 0
  
    if (stop_counter == args.earlyStop):
        print('EARLY STOPPING at Epoch {}'.format(epoch))
        eff_epoch = epoch
        break
  
  print("Optimization Finished!")
  print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
  return val_acc, model, val_acc_list, eff_epoch;


# test the given model
def test(model,features, adj, labels, idx_test):
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return acc_test.item()

# for multiple runs
def multiple_run_with_rand_split(args, features, labels, adj, idx_train, idx_val, idx_test, d = 1):
  # load random indexes
  # args already contained the best parameters found previously
  idx_train, idx_val, idx_test = find_indexes(features, d)
  _, model, _, _ = train_iterator(args, features, labels, adj, idx_train, idx_val)
  # Testing the best model
  acc_test = test(model,features, adj, labels, idx_test )
  return acc_test

def multiple_run_scores(args, features, labels, adj, idx_train, idx_val, idx_test, d = 1):
  acc_test_list = []
  for i in range(10):
    acc_test = multiple_run_with_rand_split(args, features, labels, adj, idx_train, idx_val, idx_test, d)
    acc_test_list.append(acc_test)
  acc_test_array = np.array(acc_test_list)
  print('Cora dataset')
  print('accuracy confidence interval with mean accuracy = {:.2f}% and std = {:.4f}'.format(np.mean(acc_test_array)*100,np.std(acc_test_array) ))
  return acc_test_list


