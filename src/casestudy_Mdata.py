"""
This code is the code that does the casestudy prediction,
the code structure is the same as the code for cross-validation
"""


# -*- coding:utf-8 -*-
import argparse
import copy

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, average_precision_score
from sklearn.metrics import precision_score
from torch import nn
from torch_geometric.utils import add_self_loops, negative_sampling
from tqdm import tqdm
import torch.nn.functional as F

from model import FastGTNs
from network import PygGTNs_LP
from utils import _norm, init_seed

import time
import os
import os.path as osp
from methods import average_list, sum_list


import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy import sparse

device = torch.device('cuda')
# device = torch.device('cpu')

sdf = pd.read_csv("../Data/C_D.csv")
node_list = pd.read_csv("../Data/node_list.csv")
n_nodes = node_list['Node'].unique()

# 3、generate A_CD sparse matrix
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

# 4、generate A_CG sparse matrix
sdf2 = pd.read_csv("../Data/C_G.csv")
CIDs2 = []
GIDs = []
for i, values in enumerate(sdf2.values):
    CIDs2.append(values[0])
    GIDs.append(values[1])

row_indexes2 = []
for cid in CIDs2:
    idx = name2features[cid]
    row_indexes2.append(idx)

col_indexes2 = []
for gid in GIDs:
    idx = name2features[gid]
    col_indexes2.append(idx)

data2 = np.ones_like(col_indexes2)
A_CG = csr_matrix((data2, (row_indexes2, col_indexes2)), shape=(node_num, node_num))
A_GC = A_CG.transpose()

# 5、generate A_GD sparse matrix
sdf3 = pd.read_csv("../Data/G_D.csv")
GIDs2 = []
DIDs2 = []
score = []
for i, values in enumerate(sdf3.values):
    GIDs2.append(values[0])
    DIDs2.append(values[1])
    score.append(values[2])

row_indexes3 = []
for gid in GIDs2:
    idx = name2features[gid]
    row_indexes3.append(idx)

col_indexes3 = []
for did in DIDs2:
    idx = name2features[did]
    col_indexes3.append(idx)

data3 = np.ones_like(col_indexes3)

A_GD = csr_matrix((data3, (row_indexes3, col_indexes3)), shape=(node_num, node_num))
A_DG = A_GD.transpose()
edges = [A_CG, A_GC, A_GD, A_DG]

# 5、Node Feasures
fea1 = pd.read_csv("../Data/drug_feature.csv", index_col=0)
matrix1 = fea1.values

fea2 = pd.read_csv("../Data/gene_feature.csv", index_col=0)
fea2.fillna(0, inplace=True)
matrix2 = fea2.values

fea5 = pd.read_csv("../Data/disease_feature.csv", index_col=0)
fea5.fillna(0, inplace=True)
matrix5 = fea5.values

node_faeture = np.concatenate((matrix1, matrix2, matrix5))

# pos edge index
pos_edges = [A_CD, A_DC]

edge_index = []

for i, edge in enumerate(pos_edges):
    edge_tmp = torch.from_numpy(np.vstack((edge.nonzero()[1], edge.nonzero()[0]))).type(torch.cuda.LongTensor)
    value_tmp = torch.ones(edge_tmp.shape[1]).type(torch.cuda.FloatTensor)
    edge_index.append((edge_tmp, value_tmp))


def test(args, model, edge_label_index, j):
    SCORE = []
    PRED = []
    model.eval()
    with torch.no_grad():
        out = model(node_features, A, num_nodes, edge_label_index)

        _, pred = out.max(dim=1)
        sco = F.softmax(out, dim=1)
        scores = sco[:, 1]
        SCORE.extend(scores)
        PRED.extend(pred)

    SCO = [x.item() for x in SCORE]
    SCORE.clear()
    a = np.array(SCO)
    SCO.clear()

    ss = pd.DataFrame(a.T)
    """
        Specify the path where the score will be saved folder-name is the set folder name, which can be changed.
    """
    output1 = 'folder-name/score' + str(j) + '.csv'
    ss.to_csv(output1)

    PRE = [x.item() for x in PRED]
    PRED.clear()
    P = np.array(PRE)
    PRE.clear()
    PP = pd.DataFrame(P.T)
    """
        Specify the path where the prediction will be saved folder-name is the set folder name, which can be changed.
    """
    output2 = 'folder-name/pred_label' + str(j) + '.csv'
    PP.to_csv(output2)
    print('Complete！')


def train(args, train_index, train_label):
    epoch_list = []
    lossTr_list = []

    model = PygGTNs_LP(args, num_edge_type, node_features, num_nodes).to(device)
    model.init()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    weight = torch.FloatTensor([0.5, 0.5])
    criterion = nn.CrossEntropyLoss(weight=weight).to(device)
    model.train()
    for epoch in tqdm(range(args.epoch)):
        optimizer.zero_grad()
        out = model(node_features, A, num_nodes, train_index)

        loss = criterion(out, train_label.long())

        epoch_list.append(epoch)
        lossTr_list.append(loss.cpu().detach().numpy())

        loss.backward()
        optimizer.step()

        print('epoch {:03d} train_loss {:.8f}'.format(epoch, loss.item()))

    """
        Specify the path where the model will be saved folder-name is the set folder name, which can be changed.
    """
    torch.save(model, "./folder-name/model.pth")
    return lossTr_list, epoch_list


if __name__ == '__main__':
    init_seed(seed=111)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='FastGTNs',
                        help='Model')
    parser.add_argument('--dataset', type=str, default='data0212',
                        help='Dataset')
    parser.add_argument('--epoch', type=int, default=100,
                        help='Training Epochs')
    parser.add_argument('--node_dim', type=int, default=128,
                        help='hidden dimensions')
    parser.add_argument('--num_channels', type=int, default=2,
                        help='number of channels')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='l2 reg')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of GT/FastGT layers')
    parser.add_argument('--runs', type=int, default=10,
                        help='number of runs')
    parser.add_argument("--channel_agg", type=str, default='mean')
    parser.add_argument("--remove_self_loops", action='store_true', help="remove_self_loops")
    # Configurations for FastGTNs
    parser.add_argument("--non_local", action='store_true', help="use non local operations")
    parser.add_argument("--non_local_weight", type=float, default=0,
                        help="weight initialization for non local operations")
    parser.add_argument("--beta", type=float, default=0, help="beta (Identity matrix)")
    parser.add_argument('--K', type=int, default=3,
                        help='number of non-local negibors')
    parser.add_argument("--pre_train", action='store_true', help="pre-training FastGT layers")
    parser.add_argument('--num_FastGTN_layers', type=int, default=1,
                        help='number of FastGTN layers')
    parser.add_argument("--trainingName", default='data0324', help="the name of this training")
    parser.add_argument("--crossValidation", type=int, default=1, help="do cross validation")
    parser.add_argument("--foldNumber", type=int, default=1, help="fold number of cross validation")

    parser.add_argument('--savedir', default="", help="directory to save the loss picture")

    args = parser.parse_args()
    print(args)

    epochs = args.epoch
    node_dim = args.node_dim
    num_channels = args.num_channels
    lr = args.lr
    weight_decay = args.weight_decay
    num_layers = args.num_layers

    edges = edges
    node_features = node_faeture
    num_nodes = edges[0].shape[0]

    args.num_nodes = num_nodes
    # build adjacency matrices for each edge type
    A = []
    num_edges = []
    for i, edge in enumerate(edges):
        edge_tmp = torch.from_numpy(np.vstack((edge.nonzero()[1], edge.nonzero()[0]))).type(torch.cuda.LongTensor)
        # print(edge_tmp)
        # if i > 1:
        #     value_tmp = torch.from_numpy(data3).type(torch.cuda.FloatTensor)
        # else:
        #     value_tmp = torch.ones(edge_tmp.shape[1]).type(torch.cuda.FloatTensor)
        value_tmp = torch.ones(edge_tmp.shape[1]).type(torch.cuda.FloatTensor)
        num_edges.append(edge_tmp.size(1))
        # normalize each adjacency matrix
        edge_tmp, value_tmp = add_self_loops(edge_tmp, edge_attr=value_tmp,
                                             fill_value=1e-20, num_nodes=num_nodes)
        deg_inv_sqrt, deg_row, deg_col = _norm(edge_tmp.detach(), num_nodes, value_tmp.detach())
        value_tmp = deg_inv_sqrt[deg_row] * value_tmp
        A.append((edge_tmp, value_tmp))

    # no weight
    edge_tmp = torch.stack((torch.arange(0, num_nodes), torch.arange(0, num_nodes))).type(torch.cuda.LongTensor)
    value_tmp = torch.ones(num_nodes).type(torch.cuda.FloatTensor)
    A.append((edge_tmp, value_tmp))

    print('init:')
    for x, y in A:
        print(x.shape, y.shape)
    print(node_features.shape)


    print('----------')
    num_edge_type = len(A)
    node_features = torch.from_numpy(node_features).type(torch.cuda.FloatTensor)
    print(node_features.shape)
    print('num_nodes=', num_nodes)

    train_data = pd.read_csv('../Data/casestudy/DDI_ALL_train.csv', index_col=0)

    drdi = train_data.iloc[:, [0, 1]]
    drdi_lab = train_data.iloc[:, 2]


    train_index = torch.tensor(drdi.values, dtype=torch.long).T.to(device)
    train_label = torch.tensor(drdi_lab.values, dtype=torch.long).T.to(device)


    print('Data reading complete！')

    T = train(args, train_index, train_label)

    """
    Specify here the folder where the results of this code will be saved, 
    if it already exists, a prompt will be output,
    if not, a folder with the corresponding name will be generated.
    The name and path are set by the runner
    """

    saving_path = f'folder-name/result'
    if osp.exists(saving_path):
        print('There is already a training of the same name')
    else:
        os.makedirs(saving_path)

    """
        The path here needs to be changed to the path of the model saved in the train function.
    """
    model = torch.load("./folder-name/model.pth")
    for i in range(0, 7):
        j = i + 1
        input = '../Data/casestudy/Nega_case' + str(j) + '.csv'
        print(input)
        nega_case = pd.read_csv(input, header=None)
        test_edge_label_index = torch.tensor(nega_case.values, dtype=torch.long).T.to(device)
        test(args, model, test_edge_label_index, j)
