
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, recall_score, confusion_matrix, precision_score
import numpy as np

def calculateMetric(real_score, predict_score, edge_label_index, edge_label, thresholdPercent):
    real_score = real_score.reshape((real_score.shape[0],))
    predict_score = predict_score.reshape((predict_score.shape[0],))

    topPredictedPairs(predict_score, edge_label_index, edge_label)

    thresholds = thresholdCalculation(predict_score, thresholdPercent)
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]
    # ----------------------------- #
    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
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
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    return [aupr[0, 0], auc[0, 0], f1_score, accuracy, recall, specificity, precision]

def labelBasedMetrics(label, pred_label):
    tn, fp, fn, tp = confusion_matrix(label, pred_label).ravel()
    f1 = f1_score(label, pred_label)
    acc = accuracy_score(label, pred_label)
    recall = recall_score(label, pred_label)
    precision = precision_score(label, pred_label)
    specificity = tn / (tn + fp)
    return f1, acc, recall, specificity, precision, 0, 0


def thresholdCalculation(predict_score, percent):
    # _fpr, _tpr, thresholds = roc_curve(real_score, predict_score)
    # thresholds = np.mat(thresholds)
    # thresholds_num = thresholds.shape[1]
    sorted_predict_score = np.array(sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    numberOfThresholds = round(sorted_predict_score_num * percent/100)
    if numberOfThresholds == 0:
        return sorted_predict_score[0]
    if numberOfThresholds < 1000: 
        numberOfThresholds = 1000
    print('number of thresholds: ', numberOfThresholds)
    steps = sorted_predict_score_num // numberOfThresholds
    indexes = steps * np.arange(numberOfThresholds)
    thresholds = sorted_predict_score[indexes]
    return thresholds

def topPredictedPairs(predictScoreArray, edge_label_index, edge_label):
    dic = {}
    for i in range(len(predictScoreArray)):
        dic.update({ i: predictScoreArray[i] })

    sortedPredictionScores = np.sort(predictScoreArray)
    for i in range(1, 11):
        predictionScore = sortedPredictionScores[-i]
        for key, value in dic.items():
            if value == predictionScore:
                drug = edge_label_index[0][key]
                disease = edge_label_index[1][key]
                print(i, ': ', 'drug: ', drug)
                print(i, ': ', 'disease: ', disease)
                print(i, ': label: ', edge_label[key])
                print('------------------------------')
                dic.pop(key)
                break

