import os
import numpy as np
import pandas as pd
import torch as th
from warnings import simplefilter
from model import Model
from sklearn.model_selection import KFold
from load_data import load, remove_graph
from utils import get_metrics_auc, set_seed, plot_result_auc,\
    plot_result_aupr, EarlyStopping, get_metrics
from args import args

import time
import os
import os.path as osp


def train():
    simplefilter(action='ignore', category=FutureWarning)
    print(args)
    set_seed(args.seed)
    # try:
    #     os.mkdir(args.saved_path)
    # except:
    #     pass

    saving_path = args.saved_path
    if osp.exists(saving_path):
        # raise Exception('There is already a training of the same name')
        print('There is already a training of the same name')
    else:
        os.makedirs(saving_path)


    if args.device_id:
        print('Training on GPU')
        device = th.device('cuda:{}'.format(args.device_id))
    else:
        print('Training on CPU')
        device = th.device('cpu')

    df = pd.read_csv('./dataset/{}.csv'.format(args.dataset), header=None).values
    # data 405876行3列
    data = np.array([[i, j, df[i, j]] for i in range(df.shape[0]) for j in range(df.shape[1])])
    data = data.astype('int64')
    # data_pos 2704，3
    data_pos = data[np.where(data[:, -1] == 1)[0]]


    # balance
    data_neg = data[np.where(data[:, -1] == 0)[0]]
    random_rows = np.random.choice(data_neg.shape[0],size=data_pos.shape[0],replace=False)
    data_neg = data_neg[random_rows,:]

    df_neg = pd.DataFrame(data_neg)
    neg_path = saving_path + '/negative.csv'
    df_neg.to_csv(neg_path,index=False)

    # # data_neg 403172，3
    # data_neg = data[np.where(data[:, -1] == 0)[0]]
    # assert len(data) == len(data_pos) + len(data_neg)
    assert len(data_pos) == len(data_neg)

    set_seed(args.seed)
    kf = KFold(n_splits=args.nfold, shuffle=True, random_state=args.seed)
    fold = 1
    pred_result = np.zeros(df.shape)


    log_path = saving_path + '/log.txt'
    result_file = open(file=log_path, mode='w')
    # result_file.write(f'database:{args.interactionDatasetName}\n')
    result_file.write(f'REDDA--GPU\n')
    result_file.write(f'{args.nfold}-fold cross validation\n')
    result_file.write(f'number of epoch : {args.epoch}\n')
    result_file.write(f'learn rate: initial = {args.learning_rate}\n')
    result_file.write(f'weight decay = {args.weight_decay}\n')

    #
    start_time = time.time()
    print(start_time)
    result_file.write(f'start_time = {start_time}\n')

    for w in range(10):
        print('{}-Cross Validation: Fold {}'.format(args.nfold, fold))


        readpath = 'dataset/Siridataset05/'
        train_idx = pd.read_csv(readpath+f'DDI_train{str(w)}.csv', index_col=0)
        test_idx = pd.read_csv(readpath + f'DDI_test{str(w)}.csv', index_col=0)
        train_pos_idx = train_idx[train_idx['2'] == 1]
        train_neg_idx = train_idx[train_idx['2'] == 0]
        train_pos_id = train_pos_idx.values
        train_neg_id = train_neg_idx.values

        test_pos_idx = test_idx[test_idx['2'] == 1]
        test_neg_idx = test_idx[test_idx['2'] == 0]
        test_pos_id = test_pos_idx.values.astype('int64')
        test_neg_id = test_neg_idx.values.astype('int64')


        # train_pos_id, test_pos_id = data_pos[train_pos_idx], data_pos[test_pos_idx]
        # # train_pos_id、ndarray\2433,3    test_pos_id:ndarray\271,3
        # train_neg_id, test_neg_id = data_neg[train_neg_idx], data_neg[test_neg_idx]
        # # train_neg_id:ndarray\362854,3   test_neg_id:ndarray\40318,3

        train_pos_idx = [tuple(train_pos_id[:, 0]), tuple(train_pos_id[:, 1])]
        # train_pos_idx：list 2，2433
        test_pos_idx = [tuple(test_pos_id[:, 0]), tuple(test_pos_id[:, 1])]
        train_neg_idx = [tuple(train_neg_id[:, 0]), tuple(train_neg_id[:, 1])]
        test_neg_idx = [tuple(test_neg_id[:, 0]), tuple(test_neg_id[:, 1])]

        g = load()
        g = remove_graph(g, test_pos_id[:, :-1]).to(device)
        feature = {'drug': g.nodes['drug'].data['h'], 'disease': g.nodes['disease'].data['h']}
        # mask_label = th.tensor(np.ones(df.shape))
        mask_label = np.ones(df.shape)
        mask_label[test_pos_idx] = 0
        mask_train = np.where(mask_label == 1)
        mask_train = [tuple(mask_train[0]), tuple(mask_train[1])]
        mask_label[test_neg_idx] = 0
        mask_test = np.where(mask_label == 0)
        mask_test = [tuple(mask_test[0]), tuple(mask_test[1])]
        print('Number of total training samples: {}, pos samples: {}, neg samples: {}'.format(len(mask_train[0]),
                                                                                              len(train_pos_idx[0]),
                                                                                              len(train_neg_idx[0])))
        print('Number of total testing samples: {}, pos samples: {}, neg samples: {}'.format(len(mask_test[0]),
                                                                                             len(test_pos_idx[0]),
                                                                                             len(test_neg_idx[0])))
        assert len(mask_test[0]) == len(test_neg_idx[0]) + len(test_pos_idx[0])
        label = th.tensor(df).float().to(device)

        model = Model(etypes=g.etypes, ntypes=g.ntypes,
                      in_feats=feature['drug'].shape[1],
                      hidden_feats=args.hidden_feats,
                      num_heads=args.num_heads,
                      dropout=args.dropout)
        model.to(device)

        optimizer = th.optim.Adam(model.parameters(),
                                  lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
        optim_scheduler = th.optim.lr_scheduler.CyclicLR(optimizer,
                                                         base_lr=0.1 * args.learning_rate,
                                                         max_lr=args.learning_rate,
                                                         gamma=0.995,
                                                         step_size_up=20,
                                                         mode="exp_range",
                                                         cycle_momentum=False)
        criterion = th.nn.BCEWithLogitsLoss(pos_weight=th.tensor(len(train_neg_idx[0]) / len(train_pos_idx[0])))
        print('Loss pos weight: {:.3f}'.format(len(train_neg_idx[0]) / len(train_pos_idx[0])))
        stopper = EarlyStopping(patience=args.patience, saved_path=args.saved_path)

        for epoch in range(1, args.epoch + 1):
            model.train()
            score = model(g, feature)
            pred = th.sigmoid(score)
            loss = criterion(score[mask_train].cpu().flatten(),
                             label[mask_train].cpu().flatten())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            optim_scheduler.step()
            model.eval()
            AUC_, _ = get_metrics_auc(label[mask_train].cpu().detach().numpy(),
                                      pred[mask_train].cpu().detach().numpy())
            early_stop = stopper.step(loss.item(), AUC_, model)

            if epoch % 50 == 0:
                AUC, AUPR = get_metrics_auc(label[mask_test].cpu().detach().numpy(),
                                            pred[mask_test].cpu().detach().numpy())
                print('Epoch {} Loss: {:.3f}; Train AUC {:.3f}; AUC {:.3f}; AUPR: {:.3f}'.format(epoch, loss.item(),
                                                                                                 AUC_, AUC, AUPR))
                print('-' * 50)
                if early_stop:
                    break

        stopper.load_checkpoint(model)
        model.eval()
        pred = th.sigmoid(model(g, feature)).cpu().detach().numpy()
        pred_result[test_pos_idx] = pred[test_pos_idx]
        pred_result[test_neg_idx] = pred[test_neg_idx]
        # SCORE.extend(score)
        # LABEL.extend(label)
        fold += 1

    AUC, aupr, acc, f1, pre, rec, spec = get_metrics(label.cpu().detach().numpy().flatten(), pred_result.flatten())
    print(
        'Overall: AUC {:.3f}; AUPR: {:.3f}; Acc: {:.3f}; F1: {:.3f}; Precision {:.3f}; Recall {:.3f}'.
            format(AUC, aupr, acc, f1, pre, rec))


    end_time = time.time()
    print(end_time)
    result_file.write(f'end_time = {end_time}\n')
    print('Time consuming:' + str(end_time - start_time))
    result_file.write('Time consuming:' + str(end_time - start_time))

    result_file.write(f'AUC = {AUC}\n')
    result_file.write(f'aupr = {aupr}\n')
    result_file.write(f'acc = {acc}\n')
    result_file.write(f'f1 = {f1}\n')
    result_file.write(f'pre = {pre}\n')
    result_file.write(f'rec = {rec}\n')
    result_file.write(f'spec = {spec}\n')


    pd.DataFrame(pred_result).to_csv(os.path.join(args.saved_path,
                                                  'result.csv'), index=False, header=False)

if __name__ == '__main__':
    train()
