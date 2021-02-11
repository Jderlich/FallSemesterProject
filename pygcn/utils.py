import numpy as np
import scipy.sparse as sp
import torch

import zipfile
import scipy.io
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

# load Pubmed
def load_data_pubmed(path="FallSemesterProject/data/", dataset="pubmed", flag = False):
    """Load citation network dataset (cora only for now)"""
    path = path+dataset+'/'
    print('Loading {} dataset...'.format(dataset))

    idx_labels = np.genfromtxt("{}{}.label".format(path, dataset),
                                        dtype=np.dtype(str))

    labels = encode_onehot(idx_labels[:, -1])

    #unzip the file containing the features if no yet done
    if (flag == True):
      with zipfile.ZipFile("{}{}.feature.zip".format(path, dataset), 'r') as zip_ref:
        zip_ref.extractall("./dataset_unzipped/")
    #load the features
    mat = scipy.io.loadmat("./dataset_unzipped/pubmed.feature.mat")
    features = mat['feature']
    features[features != 0] = 1
    features = sp.csr_matrix(features, dtype=np.float32)

    # build graph
    idx = np.array(idx_labels[:, 0], dtype=np.int64)
    idx_map = {j: i for i, j in enumerate(idx)}

    edges_unordered = np.genfromtxt("{}{}.edgelist".format(path, dataset),
                                    dtype=np.int32)
      
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj, features, labels

# load the data (cora, citesseer)
def load_data(path="FallSemesterProject/data/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    path = path+dataset+'/'
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    if dataset =='citeseer':
      idx_features_labels, mappa = map_docID_to_int(idx_features_labels)

    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int64)
    idx_map = {j: i for i, j in enumerate(idx)}
    if dataset == 'citeseer':
      edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.dtype(str))
      edges_unordered= map_edges(mappa,edges_unordered)
    else:
      edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
      
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))


    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, features, labels


def map_docID_to_int(features):
  mappa = dict()
  counter = 0;
  for num in range(len(features)):
    idx = features[num,0]
    mappa[idx] = counter
    features[num,0] = counter
    counter += 1
  return features, mappa

# map everynode in each edge in the edges array
def map_edges(mappa,edges ):
  res = []
  lung = len(edges)
  for id in range(lung):
    tmp2 = []

    tmp = mappa.get(str(edges[id,0]))
    if tmp is None:
      continue
    tmp2.append(tmp)

    tmp = mappa.get(str(edges[id,1]))
    if tmp is None:
      continue
    tmp2.append(tmp)
    res.append(tmp2)
  return np.array(res, dtype= np.int32)

# indexes for train, test and validation
def find_indexes(features, d = 1):

  idx = range(features.shape[0])
  idx_train, idx_test, y_train, _ = train_test_split(
    idx, idx, test_size=1000, random_state=1)
  idx_train, idx_val, y_train, _ = train_test_split(
    idx_train, y_train, test_size=500, random_state=1)
  d = 1
  if d != 1:
    num = int((features.shape[0] * d) // 1) +1
    idx_train, _, _, _ = train_test_split(
      idx_train, y_train, train_size = num, random_state=1)
  
  idx_train = torch.LongTensor(idx_train)
  idx_val = torch.LongTensor(idx_val)
  idx_test = torch.LongTensor(idx_test)

  return idx_train, idx_val, idx_test

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
