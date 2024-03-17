
"""
# This code is a sample code to perform one time 5-cross-validation
# At the end of the run, the log of the 5-cross-validation and the prediction results will be saved in the specified folder.
# saving_path is the location of the specified saving folder
"""


# -*- coding:utf-8 -*-
import torch.nn.functional as F
import os.path as osp
import pandas as pd
import numpy as np
import argparse
import torch
import time
import os

from data_preprocessing import data_preprocessing, get_adjacency_matrix
from network import PygGTNs_LP

from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
from sklearn.metrics import precision_score
from methods import average_list, sum_list
from scipy.sparse import csr_matrix
from utils import _norm, init_seed

from tqdm import tqdm
from torch import nn

device = torch.device('cuda')
# device = torch.device('cpu')

sdf = "../Data/C_D.csv"
node_list = "../Data/node_list.csv"
edges, matrix1, matrix2, edge_index = data_preprocessing(sdf, node_list)

def test(args, model, test_edge_label_index, test_edge_label, epoch, lossVal_list, node_features, A, num_nodes):
    model.eval()
    with torch.no_grad():
        out = model(node_features, A, num_nodes, test_edge_label_index)
        # Val Loss
        criterion = nn.CrossEntropyLoss(weight=weight).to(device)
        val_loss = criterion(out, test_edge_label.long())
        lossVal_list.append(val_loss)
        _, pred = out.max(dim=1)

        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for index in range(len(pred)):
            if pred[index] == 1 and test_edge_label[index] == 1:
                TP += 1
            elif pred[index] == 1 and test_edge_label[index] == 0:
                FP += 1
            elif pred[index] == 0 and test_edge_label[index] == 1:
                FN += 1
            else:
                TN += 1
        print('TP=', TP)
        print('FP=', FP)
        print('FN=', FN)
        print('TN=', TN)

        sco = F.softmax(out, dim=1)
        scores = sco[:, 1]
        if (epoch + 1) == args.epoch:
            SCORE.extend(scores)
            LABEL.extend(test_edge_label)
            PRED.extend(pred)
        model.train()

    test_roc, test_aupr = (roc_auc_score(test_edge_label.cpu().numpy(), scores.cpu().numpy()),
                           average_precision_score(test_edge_label.cpu().numpy(), scores.cpu().numpy()))
    print('val_loss {:.8f} test_aupr {:.4f} test_roc {:.4f}'.format(val_loss.item(), test_aupr, test_roc))
    return TP, FP, FN, TN


def train(args, train_index, train_label, test_index, test_label, lossTr_list, lossVal_list, epoch_list, node_features, A, num_nodes):
    model = PygGTNs_LP(args, num_edge_type, node_features, num_nodes).to(device)
    model.init()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=weight).to(device)
    model.train()
    for epoch in tqdm(range(args.epoch)):
        optimizer.zero_grad()
        out = model(node_features, A, num_nodes, train_index)
        loss = criterion(out, train_label.long())
        lossTr_list.append(loss)
        epoch_list.append(epoch)
        print('epoch {:03d} train_loss {:.8f} '.format(epoch, loss.item()))
        # -------------- each epoch begin test ------------------------#
        TP, FP, FN, TN = test(args, model, test_index, test_label, epoch, lossVal_list, node_features, A, num_nodes)
        # -------------------------------------------------------------#
        loss.backward()
        optimizer.step()
    return TP, FP, FN, TN


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default='666',
                        help='Seed')
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
    parser.add_argument('--num_FastGTN_layers', type=int, default=2,
                        help='number of FastGTN layers')

    parser.add_argument("--trainingName", default='data0324', help="the name of this training")
    parser.add_argument("--crossValidation", type=int, default=1, help="do cross validation")
    parser.add_argument("--foldNumber", type=int, default=5, help="fold number of cross validation")

    parser.add_argument('--savedir', default="Siridataset/", help="directory to save the loss picture")
    args = parser.parse_args()
    print(args)

    init_seed(seed=args.seed)

    epochs = args.epoch
    node_dim = args.node_dim
    num_channels = args.num_channels
    lr = args.lr
    weight_decay = args.weight_decay
    num_layers = args.num_layers

    edges = edges
    num_nodes = edges[0].shape[0]
    args.num_nodes = num_nodes

    SCORE = []
    LABEL = []
    PRED = []
    weight = torch.FloatTensor([0.5, 0.5])

    # build adjacency matrices for each edge type
    A = []
    num_edges = []
    A, num_edges = get_adjacency_matrix(edges, num_edges, num_nodes, A)
    print('-------------------------')
    num_edge_type = len(A)

    print('num_nodes=', num_nodes)

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
    if args.crossValidation == 1:
        # k-fold cross validation
        k = args.foldNumber
        step_length = len(edge_index[0][0][0]) // k

        num_of_epoch = args.epoch
        # prepare log
        log_path = saving_path + '/log.txt'
        result_file = open(file=log_path, mode='w')

        result_file.write(f'5-fold cross validation--GPU\n')
        result_file.write(f'seed = {args.seed}\n')
        result_file.write(f'num_nodes = {num_nodes}\n')
        result_file.write(f'{k}-fold cross validation\n')
        result_file.write(f'number of epoch : {num_of_epoch}\n')
        result_file.write(f'learn rate: initial = {lr}\n')
        result_file.write(f'weight decay = {weight_decay}\n')
        result_file.write(f'node_dim = {node_dim}\n')

        # Record start time
        start_time = time.time()
        print(start_time)
        result_file.write(f'start_time = {start_time}\n')

        TP_list = []
        FP_list = []
        FN_list = []
        TN_list = []

        # Perform k-fold cross validation
        for i in range(k):

            # Update the feature matrix
            fea5 = pd.read_csv('../Data/feature_generate/feature03/disease_feature' + str(i) + '.csv', index_col=0)
            fea5.fillna(0, inplace=True)
            matrix5 = fea5.values

            node_faeture = np.concatenate((matrix1, matrix2, matrix5))
            node_features = torch.from_numpy(node_faeture).type(torch.cuda.FloatTensor)

            model = PygGTNs_LP(args, num_edge_type, node_features, num_nodes).to(device)

            # Read in the partitioned training and test sets
            train_data = pd.read_csv('../Data/five_cvdata/Siridataset05/DDI_train' + str(i) + '.csv', index_col=0)
            test_data = pd.read_csv('../Data/five_cvdata/Siridataset05/DDI_test' + str(i) + '.csv', index_col=0)

            drdi = train_data.iloc[:, [0, 1]]
            drdi_lab = train_data.iloc[:, 2]
            drdi2 = test_data.iloc[:, [0, 1]]
            drdi2_lab = test_data.iloc[:, 2]

            train_index = torch.tensor(drdi.values, dtype=torch.long).T.to(device)
            train_label = torch.tensor(drdi_lab.values, dtype=torch.long).T.to(device)
            test_index = torch.tensor(drdi2.values, dtype=torch.long).T.to(device)
            test_label = torch.tensor(drdi2_lab.values, dtype=torch.long).T.to(device)

            # ------------------------ begin train ------------------------------ #
            epoch_list = []
            lossTr_list = []
            lossVal_list = []
            TP, FP, FN, TN = train(args, train_index, train_label, test_index, test_label,
                                    lossTr_list, lossVal_list, epoch_list,
                                   node_features, A, num_nodes)
            TP_list.append(TP)
            FP_list.append(FP)
            FN_list.append(FN)
            TN_list.append(TN)

        # End of cross-validation
        end_time = time.time()
        print('Time consuming:' + str(end_time - start_time))

        result_file.write(f'TP_list = {TP_list}\n')
        result_file.write(f'FP_list = {FP_list}\n')
        result_file.write(f'TN_list = {TN_list}\n')
        result_file.write(f'FN_list = {FN_list}\n')

        # Calculate the sum of K-fold TP, FP, FN, TN

        test_TP = sum_list(TP_list)
        test_FP = sum_list(FP_list)
        test_FN = sum_list(FN_list)
        test_TN = sum_list(TN_list)

        # Calculation of evaluation indicators
        LAB = [x.item() for x in LABEL]
        B = np.array(LAB)
        SCO = [x.item() for x in SCORE]
        a = np.array(SCO)
        roc, aupr = roc_auc_score(B, a), average_precision_score(B, a)
        print('Final result sum_TP:', test_TP)
        print('Final result sum_FP:', test_FP)
        print('Final result sum_FN :', test_FN)
        print('Final result sum_TN :', test_TN)
        print('Final result ROC {:.4f}'.format(roc))
        print('Final result Aupr {:.4f}'.format(aupr))

        result_file.write(f'roc = {roc}\n')
        result_file.write(f'aupr = {aupr}\n')

        if (test_TP + test_TN + test_FP + test_FN) != 0:
            Accuracy = (test_TP + test_TN) / (test_TP + test_TN + test_FP + test_FN)
        else:
            Accuracy = 0
        if (test_TP + test_FP) != 0:
            Precision = (test_TP) / (test_TP + test_FP)
        else:
            Precision = 0
        if (test_TP + test_FN) != 0:
            Sensitivity = (test_TP) / (test_TP + test_FN)
        else:
            Sensitivity = 0
        if (test_FP + test_TN) != 0:
            Specificity = test_TN / (test_FP + test_TN)
        else:
            Specificity = 0

        F1 = 2 * (Precision * Sensitivity) / (Precision + Sensitivity)

        print('Final result Accuracy {:.4f}'.format(Accuracy))
        print('Final result Precision {:.4f}'.format(Precision))
        print('Final result Recall {:.4f}'.format(Sensitivity))
        print('Final result Specificity {:.4f}'.format(Specificity))
        print('Final result F1 {:.4f}'.format(F1))

        result_file.write(f'Accuracy = {Accuracy}\n')
        result_file.write(f'Precision = {Precision}\n')
        result_file.write(f'Sensitivity = {Sensitivity}\n')
        result_file.write(f'Specificity = {Specificity}\n')
        result_file.write(f'F1 = {F1}\n')
        result_file.write('Time consuming:' + str(end_time - start_time))

        # Save prediction labels and prediction scores
        SCO = [x.item() for x in SCORE]
        a = np.array(SCO)
        ss = pd.DataFrame(a.T)
        save_file_path_score = saving_path + 'score.csv'
        ss.to_csv(save_file_path_score)

        LAB = [x.item() for x in LABEL]
        B = np.array(LAB)
        LL = pd.DataFrame(B.T)
        save_file_path_label = saving_path + 'true_label.csv'
        LL.to_csv(save_file_path_label)

        PRE = [x.item() for x in PRED]
        P = np.array(PRE)
        PP = pd.DataFrame(P.T)
        save_file_path_pre = saving_path + 'pred_label.csv'
        PP.to_csv(save_file_path_pre)
