import numpy as np
import scipy.sparse as sp
import tensorflow as tf
import gc
import random
from clac_metric import cv_model_evaluate
from utils import *
from model import GCNModel
from opt import Optimizer
import time
import os
import os.path as osp
import pandas as pd
tf.compat.v1.disable_eager_execution()





def PredictScore(train_drug_dis_matrix, drug_matrix, dis_matrix, seed, epochs, emb_dim, dp, lr,  adjdp):
    np.random.seed(seed)
    # tf.reset_default_graph()
    tf.compat.v1.reset_default_graph()
    # tf.set_random_seed(seed)
    tf.compat.v1.set_random_seed(seed)
    adj = constructHNet(train_drug_dis_matrix, drug_matrix, dis_matrix)
    adj = sp.csr_matrix(adj)
    association_nam = train_drug_dis_matrix.sum()
    X = constructNet(train_drug_dis_matrix)
    features = sparse_to_tuple(sp.csr_matrix(X))
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]
    adj_orig = train_drug_dis_matrix.copy()
    adj_orig = sparse_to_tuple(sp.csr_matrix(adj_orig))

    adj_norm = preprocess_graph(adj)
    adj_nonzero = adj_norm[1].shape[0]
    placeholders = {
        # 'features': tf.sparse_placeholder(tf.float32),
        # 'adj': tf.sparse_placeholder(tf.float32),
        # 'adj_orig': tf.sparse_placeholder(tf.float32),
        # 'dropout': tf.placeholder_with_default(0., shape=()),
        # 'adjdp': tf.placeholder_with_default(0., shape=())

        'features': tf.compat.v1.sparse_placeholder(tf.float32),
        'adj': tf.compat.v1.sparse_placeholder(tf.float32),
        'adj_orig': tf.compat.v1.sparse_placeholder(tf.float32),
        'dropout': tf.compat.v1.placeholder_with_default(0.4, shape=()),
        'adjdp': tf.compat.v1.placeholder_with_default(0.6, shape=())
    }
    model = GCNModel(placeholders, num_features, emb_dim,
                     features_nonzero, adj_nonzero, train_drug_dis_matrix.shape[0], name='LAGCN')
    with tf.name_scope('optimizer'):
        opt = Optimizer(
            preds=model.reconstructions,
            labels=tf.reshape(tf.compat.v1.sparse_tensor_to_dense(
                placeholders['adj_orig'], validate_indices=False), [-1]),
            model=model,
            lr=lr, num_u=train_drug_dis_matrix.shape[0], num_v=train_drug_dis_matrix.shape[1], association_nam=association_nam)
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    for epoch in range(epochs):
        feed_dict = dict()
        feed_dict.update({placeholders['features']: features})
        feed_dict.update({placeholders['adj']: adj_norm})
        feed_dict.update({placeholders['adj_orig']: adj_orig})
        feed_dict.update({placeholders['dropout']: dp})
        feed_dict.update({placeholders['adjdp']: adjdp})
        _, avg_cost = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)
        if epoch % 100 == 0:
            feed_dict.update({placeholders['dropout']: 0.4})
            feed_dict.update({placeholders['adjdp']: 0.6})
            res = sess.run(model.reconstructions, feed_dict=feed_dict)
            print("Epoch:", '%04d' % (epoch + 1),
                  "train_loss=", "{:.5f}".format(avg_cost))
    print('Optimization Finished!')
    feed_dict.update({placeholders['dropout']: 0.4})
    feed_dict.update({placeholders['adjdp']: 0.6})
    res = sess.run(model.reconstructions, feed_dict=feed_dict)
    sess.close()
    return res


def cross_validation_experiment(drug_dis_matrix, drug_matrix, dis_matrix, seed, epochs, emb_dim, dp, lr, adjdp,saving_path):
    index_matrix = np.mat(np.where(drug_dis_matrix == 1))
    # print(index_matrix.shape)
    # input()

    association_nam = index_matrix.shape[1]
    random_index = index_matrix.T.tolist()
    random.seed(seed)
    random.shuffle(random_index)
    k_folds = 10
    CV_size = int(association_nam / k_folds)
    temp = np.array(random_index[:association_nam - association_nam %
                                 k_folds]).reshape(k_folds, CV_size,  -1).tolist()
    temp[k_folds - 1] = temp[k_folds - 1] + \
        random_index[association_nam - association_nam % k_folds:]
    # for _ in temp:
    #     print(len(_))

    random_index = temp
    metric = np.zeros((1, 7))
    print("seed=%d, evaluating drug-disease...." % (seed))
    for k in range(k_folds):
        print("------this is %dth cross validation------" % (k+1))
        train_matrix = np.matrix(drug_dis_matrix, copy=True)
        train_matrix[tuple(np.array(random_index[k]).T)] = 0
        drug_len = drug_dis_matrix.shape[0]
        dis_len = drug_dis_matrix.shape[1]
        drug_disease_res = PredictScore(
            train_matrix, drug_matrix, dis_matrix, seed, epochs, emb_dim, dp, lr,  adjdp)
        print(drug_disease_res.shape)
        predict_y_proba = drug_disease_res.reshape(drug_len, dis_len)

        df_score = pd.DataFrame(predict_y_proba)
        df_score.to_csv(saving_path + 'predict_y_proba' + str(k) + '.csv')

        df_train_matrix = pd.DataFrame(train_matrix)
        df_train_matrix.to_csv(saving_path + 'train_matrix' + str(k) + '.csv')


        metric_tmp = cv_model_evaluate(
            drug_dis_matrix, predict_y_proba, train_matrix,k,saving_path)

        print(metric_tmp)
        metric += metric_tmp
        del train_matrix
        gc.collect()
    print(metric / k_folds)
    metric = np.array(metric / k_folds)
    return metric


if __name__ == "__main__":
    drug_sim = np.loadtxt('../Ldata/drug_sim.csv', delimiter=',')
    dis_sim = np.loadtxt('../Ldata/dis_sim.csv', delimiter=',')
    drug_dis_matrix = np.loadtxt('../Ldata/drug_dis.csv', delimiter=',')

    epoch = 4000
    emb_dim = 64
    lr = 0.01
    adjdp = 0.6
    dp = 0.4
    simw = 6
    result = np.zeros((1, 7), float)
    average_result = np.zeros((1, 7), float)
    circle_time = 1

    saving_path = f'result_KFCdata/Siridataset05/'
    if osp.exists(saving_path):
        print('There is already a training of the same name')
        # raise Exception('There is already a training of the same name')

    else:
        os.makedirs(saving_path)
    log_path = saving_path + '/log.txt'
    result_file = open(file=log_path, mode='w')
    result_file.write(f'on LAGCN\n')
    start_time = time.time()
    print(start_time)

    for i in range(circle_time):
        # result += cross_validation_experiment(
        #     drug_dis_matrix, drug_sim*simw, dis_sim*simw, i, epoch, emb_dim, dp, lr, adjdp, saving_path)
        result += cross_validation_experiment(
            drug_dis_matrix, drug_sim * simw, dis_sim * simw, 5, epoch, emb_dim, dp, lr, adjdp, saving_path)
    average_result = result / circle_time
    print(average_result)

    end_time = time.time()
    print('Time consuming:' + str(end_time - start_time))
    result_file.write('Time consuming:' + str(end_time - start_time))
