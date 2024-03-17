import numpy as np
import pandas as pd


def get_metrics_by_mask(real_score, predict_score, k, saving_path, row_mask, col_mask):
    real_score

def get_metrics(real_score, predict_score,k,saving_path):
    sorted_predict_score = np.array(
        sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(
        sorted_predict_score_num*np.arange(1, 1000)/1000)]

    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))

    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)

    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1

    # df_predlabel = pd.DataFrame(predict_score_matrix)
    # df_predlabel.to_csv(saving_path+'predict_y_label' + str(k) + '.csv')



    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1)-TP
    FN = real_score.sum()-TP
    TN = len(real_score.T)-TP-FP-FN

    fpr = FP/(FP+TN)
    tpr = TP/(TP+FN)
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    auc = 0.5*(x_ROC[1:]-x_ROC[:-1]).T*(y_ROC[:-1]+y_ROC[1:])

    recall_list = tpr
    precision_list = TP/(TP+FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack(
        (recall_list, precision_list)).tolist())).T
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5*(x_PR[1:]-x_PR[:-1]).T*(y_PR[:-1]+y_PR[1:])

    f1_score_list = 2*TP/(len(real_score.T)+TP-TN)
    accuracy_list = (TP+TN)/len(real_score.T)
    specificity_list = TN/(TN+FP)

    max_index = np.argmax(f1_score_list)


    max_thresholds = thresholds[0, max_index]


    print('-----max_thresholds:',max_thresholds)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    return [aupr[0, 0], auc[0, 0], f1_score, accuracy, recall, specificity, precision]






def cv_model_evaluate(interaction_matrix, predict_matrix, train_matrix,k,saving_path):
    test_index = np.where(train_matrix == 0)
    train_index = np.where(train_matrix == 1)

    #size1 = train_index[0].shape[0]

    # # balance
    # size1 = train_index[0].shape[0]
    # random_rows = np.random.choice(test_index[0].shape[0], size=size1)
    # select_rows = (test_index[0][random_rows],test_index[1][random_rows])
    # test_index = select_rows

    # # balance
    # df = pd.read_csv('../Kdata/NegativeSampleDRDI.csv',header=None)
    # random_rows = df.sample(n=size1)
    # # data_neg = df[random_rows, :]
    # array1 = random_rows[0].values
    # array2 = random_rows[1].values
    # combined_tuple = (array1,array2)
    # test_index = combined_tuple

    # random_rows = np.random.choice(test_index[0].shape[0], size=size1)
    # select_rows = (test_index[0][random_rows],test_index[1][random_rows])
    # test_index = select_rows
    #

    real_score = interaction_matrix[test_index]
    predict_score = predict_matrix[test_index]

    # predict_matrix.tofile(f"pm{k}.bin")
    # test_index.tofile(f"ti")
    return get_metrics(real_score, predict_score,k,saving_path)
