
"""
This code is to turn the raw data into sparse matrices and tensor data that can be put into the model
"""


import pandas as pd
import numpy as np
import torch
import copy
from torch_geometric.utils import add_self_loops
from scipy.sparse import csr_matrix
from utils import _norm

device = torch.device('cuda')
# device = torch.device('cpu')

# Read in the raw data and turn it into a sparse matrix
def data_preprocessing(sdf, node_list):

    sdf = pd.read_csv(sdf)
    node_list = pd.read_csv(node_list)
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
    matrix2 = fea2.values


    # pos_edge_index
    pos_edges = [A_CD, A_DC]
    edge_index = []

    for i, edge in enumerate(pos_edges):
        edge_tmp = torch.from_numpy(np.vstack((edge.nonzero()[1], edge.nonzero()[0]))).type(torch.cuda.LongTensor)
        value_tmp = torch.ones(edge_tmp.shape[1]).type(torch.cuda.FloatTensor)
        edge_index.append((edge_tmp, value_tmp))

    return edges, matrix1, matrix2, edge_index


# Turning the adjacency matrix of a heterogeneous network into a tensor
def get_adjacency_matrix(edges, num_edges, num_nodes, A):
    for i, edge in enumerate(edges):
        edge_tmp = torch.from_numpy(np.vstack((edge.nonzero()[1], edge.nonzero()[0]))).type(torch.cuda.LongTensor)
        print(edge_tmp)
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

    # ues weight
    #
    # edge_tmp = torch.from_numpy(np.vstack((col.values, row.values))).type(torch.cuda.LongTensor)
    # value_tmp = torch.from_numpy(sc.values).type(torch.cuda.FloatTensor)
    # A.append((edge_tmp, value_tmp))

    return A, num_edges


