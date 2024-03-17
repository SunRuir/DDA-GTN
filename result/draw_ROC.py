import seaborn
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 14}) # Change all font sizes, change other properties similarly
plt.rc('font', family='Times New Roman')
plt.rc('font', weight='bold')
from sklearn import metrics

def plot_roc(y, score):
    # y = [0 1 0 1 0 1...] score = [[0.1 0.9], [0.4, 0.6]...] numpy

    fpr, tpr, thresholds = roc_curve(y, score)
    auc = metrics.auc(fpr, tpr)
    # fpr2, tpr2, thresholds2 = roc_curve(y2, score2)
    # auc2 = metrics.auc(fpr2, tpr2)
    # plot
    # plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, color='darkorange', label='AUROC=0.891±0.005')
    # plt.plot(fpr2, tpr2, color='steelblue', label='AUROC=0.844±0.012')
    # plt.plot(fpr, tpr, color='darkorange', label='with weights ROC curve (area = %0.2f)' % auc)
    # plt.plot(fpr2, tpr2, color='steelblue', label='without weights ROC curve (area = %0.2f)' % auc2)

    # plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.05, 1.1])
    plt.ylim([-0.05, 1.1])
    plt.ylabel('True Positive Rate',weight = 'bold')
    plt.xlabel('False Positive Rate',weight = 'bold')
    # plt.grid(linestyle='-.')
    plt.legend(loc='upper left')
    # plt.subplots_adjust(top=1, bottom=0, right=0.93, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    # plt.grid(True)
    plt.savefig("Mdata_ROC.png", dpi=300, bbox_inches="tight",pad_inches = -0.1)
    plt.savefig("Mdata_ROC600.png", dpi=600)
    plt.savefig('Mdata_ROC_svg_600.svg', format='svg', dpi=600)  # save picture
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

label = np.concatenate((y_true, y_true2, y_true3, y_true4, y_true5), axis=0)
score = np.concatenate((y_score, y_score2, y_score3, y_score4, y_score5), axis=0)



# plot_result_aupr(y_true,y_scores)
plot_roc(label,score)