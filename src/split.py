"""
This code is used to divide the training and test sets for the 5-fold cross validation
It is the dataset that needs to be divided before running the 5-fold cross validation code
"""

import argparse
import copy

import numpy as np
import torch

from utils import _norm, init_seed


device = torch.device('cuda')
# device = torch.device('cpu')

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix




sdf = pd.read_csv("../Data/C_D.csv")
node_list = pd.read_csv("../Data/node_list.csv")


n_nodes = node_list['Node'].unique()


CIDs = []
DIDs = []
for i, values in enumerate(sdf.values):
    CIDs.append(values[0])
    DIDs.append(values[1])

name2features = {}
for i, values in enumerate(node_list.values):
    idx = node_list.index[i]
    name = values[0]
    name2features[name] = idx

row_indexes = []
for cid in CIDs:
    idx = name2features[cid]
    row_indexes.append(idx)

col_indexes = []
for did in DIDs:
    idx = name2features[did]
    col_indexes.append(idx)

data = np.ones_like(col_indexes)
node_num = len(n_nodes)
A_CD = csr_matrix((data, (row_indexes, col_indexes)), shape=(node_num, node_num))
A_DC = A_CD.transpose()


pos_edges = [A_CD, A_DC]
edge_index = []

for i, edge in enumerate(pos_edges):
    edge_tmp = torch.from_numpy(np.vstack((edge.nonzero()[1], edge.nonzero()[0]))).type(torch.cuda.LongTensor)
    value_tmp = torch.ones(edge_tmp.shape[1]).type(torch.cuda.FloatTensor)
    edge_index.append((edge_tmp, value_tmp))

# split 5-cv dataset
def split_dataset(adj,saving_path):

    edge_index1 = adj[1][0]
    edge_indx = edge_index1
    pos_edge_label_index = copy.deepcopy(edge_indx)
    pos_edge_label = torch.ones(pos_edge_label_index.shape[1])

    # read negativeSample
    negativeSample = pd.read_csv("../Data/NegativeSample.csv", header=None)
    neg_edge_index = torch.tensor(negativeSample.values, dtype=torch.long).T.to(device)
    neg_edge_label = torch.zeros(neg_edge_index.shape[1])

    edge_label_index = torch.cat([pos_edge_label_index, neg_edge_index], dim=-1)
    edge_label = torch.cat([pos_edge_label, neg_edge_label], dim=0)

    edge_label = edge_label.to(device)
    edge_label_index = edge_label_index.to(device)


    print(edge_label.shape)
    print(edge_label_index.shape)

    total_num_edges = edge_label_index.shape[1]
    # shuffle
    num_train = int(total_num_edges * 0.8)
    step_length = int(total_num_edges * 0.2)

    index_shuffled = torch.randperm(edge_label_index.size(1))
    index = index_shuffled.cpu().numpy().tolist()
    edge_label = edge_label[index_shuffled]
    edge_label_index = edge_label_index[:, index_shuffled]

    for i in range(5):
        test_dataset_start = 0 + i * step_length
        test_dataset_end = (i + 1) * step_length
        test_index = torch.tensor(index[test_dataset_start:test_dataset_end], dtype=torch.long).to(device)
        train_index = torch.tensor(index[0:test_dataset_start] + index[test_dataset_end:len(index)],
                                   dtype=torch.long).to(device)
        test_edge_label = edge_label[test_index]
        test_edge_label_index = edge_label_index[:, test_index]
        train_edge_label = edge_label[train_index]
        train_edge_label_index = edge_label_index[:, train_index]

        train_label = np.array(train_edge_label.cpu().numpy())
        test_label = np.array(test_edge_label.cpu().numpy())
        train_edge_index = np.array(train_edge_label_index.cpu().numpy())
        test_edge_index = np.array(test_edge_label_index.cpu().numpy())
        train_edge = np.vstack((train_edge_index, train_label))
        test_edge = np.vstack((test_edge_index, test_label))

        DDI_train = pd.DataFrame(train_edge.T)
        DDI_train.to_csv(saving_path + 'DDI_train' + str(i) + '.csv')
        DDI_test = pd.DataFrame(test_edge.T)
        DDI_test.to_csv(saving_path + 'DDI_test' + str(i) + '.csv')

    return train_edge_label, train_edge_label_index, test_edge_label, test_edge_label_index






if __name__ == '__main__':
    init_seed(seed=222)
    """
        Specify the path where the 5-cv dataset divisions will be saved folder-name is the set folder name, which can be changed.
        The 1-th division is saved in folder-name01/, The 2-th division is saved in folder-name02/
        The 3-th division is saved in folder-name03/...in a similar fashion
        The saving_path can be specified by yourself
    """
    saving_path = f'../Data/five_cvdata/folder-name01/' # Location of the folder to be saved to
    split_dataset(edge_index, saving_path)

