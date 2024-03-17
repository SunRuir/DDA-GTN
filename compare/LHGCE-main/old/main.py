import torch
import prepareData
import numpy as np
from linear.FNN import trainFNN, testFNN
from metrics import calculateMetric, labelBasedMetrics
import argparse
from sklearn.svm import OneClassSVM

# For reproducibility
torch.manual_seed(0)
np.random.seed(0)
torch.use_deterministic_algorithms(True)

# Command line options
parser = argparse.ArgumentParser(description='Options')
parser.add_argument('--emb', help='embedding method for drug features', type=str, default='matrix')
parser.add_argument('--feature-list', help='the feature list to include', type=str, nargs="+")
parser.add_argument('--folds', help='number of folds for cross-validation',type=int,  default=5)
parser.add_argument('--batchsize', help='batch-size for DNN',type=int, default=1000)
parser.add_argument('--epoch', help='number of epochs to train in model',type=int, default=20)
parser.add_argument('--thr-percent', help='the threshold percentage with respect to batch size',type=int, default=30)
parser.add_argument('--dropout', help='dropout probability for DNN',type=float, default=0.3)
parser.add_argument('--lr', help='learning rate for DNN',type=float, default=0.001)
parser.add_argument('--agg', help='aggregation method for DNN input', type=str, default='concatenate')
parser.add_argument('--clf', help='classifier to use for prediction', type=str, default='MAFCN')

args = parser.parse_args()
print(args)

# Setting the global variables
FEATURE_LIST = args.feature_list
EPOCHS = args.epoch
BATCHSIZE = args.batchsize
EMBEDDING = args.emb
FOLDS = args.folds
DROPOUT = args.dropout
THRESHOLD_PERCENT = args.thr_percent
LEARNING_RATE= args.lr
CLASSIFIER = args.clf
AGGREGATION = args.agg
DRUG_NUMBER = 269
DISEASE_NUMBER = 598
ENZYME_NUMBER = 108
STRUCTURE_NUMBER = 881
PATHWAY_NUMBER = 258
TARGET_NUMBER = 529
INTERACTIONS_NUMBER = 18416
NONINTERACTIONS_NUMBER = 142446

def crossValidation(drugDic, diseaseSim, totalInteractions, totalNonInteractions):
    metrics = np.zeros(7)

    sizeOfInteractions = totalInteractions.shape[0]
    sizeOfNonInteractions = totalNonInteractions.shape[0]

    totalInteractionIndices = np.random.permutation(sizeOfInteractions)
    totalNonInteractionIndices = np.random.permutation(sizeOfNonInteractions)

    interactionsIndicesFolds = totalInteractionIndices.reshape(FOLDS, sizeOfInteractions // FOLDS)
    nonInteractionsIndicesFolds = totalNonInteractionIndices.reshape(FOLDS, sizeOfNonInteractions // FOLDS)

    for k in range(FOLDS):
        testInteractionsIndex = interactionsIndicesFolds[k]
        trainInteractionsIndex = np.setdiff1d(interactionsIndicesFolds.flatten(), testInteractionsIndex, assume_unique=True)

        testNonInteractionsIndex = nonInteractionsIndicesFolds[k]
        trainNonInteractionsIndex = np.setdiff1d(nonInteractionsIndicesFolds.flatten(), testNonInteractionsIndex, assume_unique=True)

        allDataDic = {}
        for featureIndex in range(len(FEATURE_LIST)):
            involvedDiseases = []
            XTrain = []
            YTrain = []
            for drugIndex, diseaseIndex in totalInteractions[trainInteractionsIndex]:
                drug = drugDic[FEATURE_LIST[featureIndex]][drugIndex]
                XTrain.append(drug)
                involvedDiseases.append(diseaseSim[diseaseIndex])
                YTrain.append([1])

            interactions = len(YTrain)
            print('train: interactions: ', interactions)
                
            for drugIndex, diseaseIndex in totalNonInteractions:
                drug = drugDic[FEATURE_LIST[featureIndex]][drugIndex]
                XTrain.append(drug)
                involvedDiseases.append(diseaseSim[diseaseIndex])
                YTrain.append([0])
            print('train: nonInteractions: ', len(YTrain) - interactions)

            allDataDic[FEATURE_LIST[featureIndex]] = np.array(XTrain)
        
        YTrain = np.array(YTrain)
        allDataDic['diseases'] = np.array(involvedDiseases)
        XTrain = makeX(allDataDic, AGGREGATION)
        allDataDic = {} 
        allDataDic['y'] = YTrain
        allDataDic['X'] = XTrain

        if CLASSIFIER == 'MAFCN':
            trainedModel = trainFNN(allDataDic, EPOCHS, BATCHSIZE, DROPOUT, LEARNING_RATE)
        if CLASSIFIER == 'OCC':
            model = OneClassSVM(gamma='scale', nu=0.01)
            model.fit(allDataDic['X'])

        # TESTING
        allDataDic = {}
        for featureIndex in range(len(FEATURE_LIST)):
            XTest = []
            YTest = []
            involvedDiseases = []
            for drugIndex, diseaseIndex in totalInteractions[testInteractionsIndex]:
                drug = drugDic[FEATURE_LIST[featureIndex]][drugIndex]
                XTest.append(drug)
                involvedDiseases.append(diseaseSim[diseaseIndex])
                YTest.append([1])

            interactions = len(YTest)

            for drugIndex, diseaseIndex in totalNonInteractions:
                drug = drugDic[FEATURE_LIST[featureIndex]][drugIndex]
                XTest.append(drug)
                involvedDiseases.append(diseaseSim[diseaseIndex])
                YTest.append([0])

            nonInteractions = len(YTest) - interactions
            print('test: interactions: ', interactions, 'non-interactions: ', nonInteractions)

            allDataDic[FEATURE_LIST[featureIndex]] = np.array(XTest)

        YTest = np.array(YTest)
        allDataDic['diseases'] = np.array(involvedDiseases)
        XTest = makeX(allDataDic, AGGREGATION)
        allDataDic = {} 
        allDataDic['X'] = XTest

        if CLASSIFIER == 'MAFCN':
            y_pred_prob = testFNN(trainedModel, allDataDic).detach().numpy()
            metric = np.array(calculateMetric(YTest, y_pred_prob, THRESHOLD_PERCENT))
        if CLASSIFIER == 'OCC':
            pred_label = model.predict(allDataDic['X'])
            pred_label[pred_label == -1] = 0
            metric = labelBasedMetrics(YTest, pred_label)

        metrics += metric
        print('metric: ', metric)

    return metrics

def makeX(dataDic, aggregation='concatenate'):
    diseases = torch.from_numpy(dataDic['diseases'])
    del dataDic['diseases']

    X = torch.tensor([], dtype=float)
    for index, featureKey in enumerate(dataDic):
        tensorred = torch.from_numpy(dataDic[featureKey])
        if aggregation == 'concatenate':
            if index == 0:
                X = tensorred
            else:
                X = torch.cat((X, tensorred), 1)
        # make sure the dimensions are the same
        elif aggregation == 'sum':
            if index == 0:
                X = tensorred
            else:
                X = X + tensorred
        elif aggregation == 'mul':
            if index == 0:
                X = tensorred
            else:
                X = X * tensorred
        elif aggregation == 'avg':
            if index == 0:
                X = tensorred
            else:
                X = (X + tensorred)/2
        else:
            exit('please use a known aggregation method')
    # don't convert X to numpy array since this is the input to our model
    X = torch.cat((X, diseases), 1)
    return X

def makePlotData(drugDic, totalInteractions, totalNonInteractions):
    for featureIndex in range(len(FEATURE_LIST)):
        positives = []
        negatives = []
        labels = []
        X = []
        Y = []
        for drugIndex, diseaseIndex in totalInteractions:
            drug = drugDic[FEATURE_LIST[featureIndex]][drugIndex]
            positives.append(drug)
        labels.append('positives')
        positives = np.array(positives)
        X.append(positives[:, 0])
        Y.append(positives[:, 1])

        for drugIndex, diseaseIndex in totalNonInteractions:
            drug = drugDic[FEATURE_LIST[featureIndex]][drugIndex]
            negatives.append(drug)
        labels.append('negatives')
        negatives = np.array(negatives)
        X.append(negatives[:, 0])
        Y.append(negatives[:, 1])

        X = np.array(X)
        Y = np.array(Y)

        prepareData.plotAndSave(X, Y, labels, FEATURE_LIST[featureIndex])
    
    return X, Y, labels

def main():
    print('nDrugs: ', DRUG_NUMBER)
    print('nDisease: ', DISEASE_NUMBER)
    drugDic = prepareData.prepareDrugData(FEATURE_LIST, EMBEDDING)
    drugDisease = np.loadtxt('./data/drug_dis.csv', delimiter=',')
    diseaseSim = np.loadtxt('./data/dis_sim.csv', delimiter=',')
    diseaseSim = 0.6 * diseaseSim
    totalInteractions = np.array(np.mat(np.where(drugDisease == 1)).T) #(18416, 2)
    totalNonInteractions = np.array(np.mat(np.where(drugDisease == 0)).T) #(142446, 2)
    
    # makePlotData(drugDic, totalInteractions, totalNonInteractions)
    selectedInteractions, selectedNonInteractions = prepareData.splitter(100, 100, totalInteractions, totalNonInteractions)

    results = crossValidation(drugDic, diseaseSim, selectedInteractions, selectedNonInteractions)
    print('results: ', results / FOLDS)

main()
