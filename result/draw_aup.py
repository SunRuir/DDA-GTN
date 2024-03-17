import seaborn
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import matplotlib
plt.rc('font', family='Times New Roman')
matplotlib.rcParams.update({'font.size': 14}) # Change all font sizes, change other properties similarly
plt.rc('font', weight='bold')
from sklearn import metrics

def plot_pr(y, score):
    plt.rc('font', family='Times New Roman')
    font = 'Times New Roman'
    precision_dtc, recall_dtc, thresholds_dtc = precision_recall_curve(y, score)
    # aupr= metrics.auc(recall_dtc, precision_dtc)  # calculate AUPR
    plt.plot(recall_dtc,precision_dtc,  color='darkorange',label='AUPR=0.887±0.008')
    # precision_dtc2, recall_dtc2, thresholds_dtc2 = precision_recall_curve(y2, score2)
    # aupr2 = metrics.auc(recall_dtc2, precision_dtc2)  # calculate AUPR
    # plt.plot(precision_dtc2, recall_dtc2, color='steelblue',label='without weights')
    # plt.title('Precision-Recall curve')
    # plt.plot([1, 0], [0, 1], 'r--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.5, 1.05])
    plt.ylabel('Precision',weight = 'bold')
    plt.xlabel('Recall',weight = 'bold')
    # plt.grid(linestyle='-.')
    # plt.grid(True)
    # plt.legend(loc='upper right')
    plt.legend(loc='upper left')
    plt.savefig("Mdata_AUPR.png", dpi=300)
    plt.savefig("Mdata_AUPR_600.png", dpi=600)
    plt.savefig('Mdata_AUPR_svg_600.svg', format='svg', dpi=600)  # save picture
    plt.show()


#-------First 5-cv result
K_score = pd.read_csv('case3_result/score00.csv', index_col=0)
K_label = pd.read_csv('case3_result/true_label00.csv', index_col=0)
y_true = np.array(K_label)
y_score = np.array(K_score)

#-------Second 5-cv result
K_score2 = pd.read_csv('case3_result/score01.csv', index_col=0)
K_label2 = pd.read_csv('case3_result/true_label01.csv', index_col=0)
y_true2 = np.array(K_label2)
y_score2 = np.array(K_score2)

#-------Third 5-cv result
K_score3 = pd.read_csv('case3_result/score02.csv', index_col=0)
K_label3 = pd.read_csv('case3_result/true_label02.csv', index_col=0)
y_true3 = np.array(K_label3)
y_score3 = np.array(K_score3)

#-------Fourth 5-cv result
K_score4 = pd.read_csv('case3_result/score03.csv', index_col=0)
K_label4 = pd.read_csv('case3_result/true_label03.csv', index_col=0)
y_true4 = np.array(K_label4)
y_score4 = np.array(K_score4)

#-------Fifth 5-cv result
K_score5 = pd.read_csv('case3_result/score04.csv', index_col=0)
K_label5 = pd.read_csv('case3_result/true_label04.csv', index_col=0)
y_true5 = np.array(K_label5)
y_score5 = np.array(K_score5)

# concatenate
label = np.concatenate((y_true, y_true2, y_true3, y_true4, y_true5), axis=0)
score = np.concatenate((y_score, y_score2, y_score3, y_score4, y_score5), axis=0)

plot_pr(label,score)
# plot_roc(label,score)
