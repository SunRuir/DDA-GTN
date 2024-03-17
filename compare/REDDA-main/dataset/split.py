
import argparse
import copy

import numpy as np
import torch

from utils import init_seed


device = torch.device('cuda')
# device = torch.device('cpu')

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix


node_num = 894 + 20561 + 454


# 5、生成A_CD稀疏矩阵
sdf5 = pd.read_csv("associations/KFCdataset.csv")
CIDs2 = []
DIDs2 = []

for i, values in enumerate(sdf5.values):
    CIDs2.append(values[0])
    d2 = values[1]
    DIDs2.append(d2)

data5 = np.ones_like(CIDs2)
A_CD = csr_matrix((data5, (CIDs2, DIDs2)), shape=(node_num, node_num))
A_DC = A_CD.transpose()

# 正边索引

pos_edges = [A_CD, A_DC]
edge_index = []

for i, edge in enumerate(pos_edges):
    edge_tmp = torch.from_numpy(np.vstack((edge.nonzero()[1], edge.nonzero()[0]))).type(torch.cuda.LongTensor)
    value_tmp = torch.ones(edge_tmp.shape[1]).type(torch.cuda.FloatTensor)
    edge_index.append((edge_tmp, value_tmp))


def split_dataset(adj,saving_path):
    # adj p-a a-p p-s s-p
    # edge_index1 = graph['paper', 'to', 'author'].edge_index
    # edge_index2 = graph['author', 'to', 'paper'].edge_index

    edge_index1 = adj[1][0]
    # edge_index2 = adj[1][0]
    # edge_indx = torch.cat((edge_index1, edge_index2), dim=1)
    edge_indx = edge_index1
    pos_edge_label_index = copy.deepcopy(edge_indx)
    pos_edge_label = torch.ones(pos_edge_label_index.shape[1])

    # 复边读进来
    # negativeSample = pd.read_csv("../Data/NegativeSample0829.csv", header=None)
    negativeSample = pd.read_csv("NegativeSampleDRDI.csv", header=None)
    neg_edge_index = torch.tensor(negativeSample.values, dtype=torch.long).T.to(device)
    neg_edge_label = torch.zeros(neg_edge_index.shape[1])

    edge_label_index = torch.cat([pos_edge_label_index, neg_edge_index], dim=-1)
    edge_label = torch.cat([pos_edge_label, neg_edge_label], dim=0)

    edge_label = edge_label.to(device)
    edge_label_index = edge_label_index.to(device)


    print(edge_label.shape)
    print(edge_label_index.shape)

    total_num_edges = edge_label_index.shape[1]
    # 打乱再取811
    num_train = int(total_num_edges * 0.9)
    step_length = int(total_num_edges * 0.1)

    index_shuffled = torch.randperm(edge_label_index.size(1))
    index = index_shuffled.cpu().numpy().tolist()
    edge_label = edge_label[index_shuffled]
    edge_label_index = edge_label_index[:, index_shuffled]

    for i in range(10):
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
    init_seed(seed=777)
    saving_path = f'Siridataset05/'
    split_dataset(edge_index,saving_path)

