# import matplotlib
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
import torch
from torch_geometric.data import HeteroData
import pandas as pd
import json

FOLDS = 10
saving_path = f'KFCdata_result05/'
# def Cosine(matrix)
def Jaccard(matrix):
    matrix = np.mat(matrix)
    numerator = matrix * matrix.T
    denominator = np.ones(np.shape(matrix)) * matrix.T + matrix * np.ones(np.shape(matrix.T)) - matrix * matrix.T
    return np.array(numerator / denominator)

def extractId():
    chemicalDic = {}
    diseaseDic = {}

    data = sio.loadmat('./data/LAGCN/SCMFDD_Dataset.mat')

    chemicals = data['chemical_list']
    diseases = data['disease_list']

    for i in range(len(chemicals)):
        chemical = chemicals[i][0][0]
        chemicalDic.update({chemical: i})

    for i in range(len(diseases)):
        disease = diseases[i][0][0]
        diseaseDic.update({disease: i})
        
    with open('./data/LAGCN/diseaseIndex.json', 'w') as convert_file:
        convert_file.write(json.dumps(diseaseDic))

    with open('./data/LAGCN/chemicalIndex.json', 'w') as convert_file:
        convert_file.write(json.dumps(diseaseDic))


def readFromMat():
    data = sio.loadmat('./data/LAGCN/SCMFDD_Dataset.mat')

    np.savetxt('./drugDrug_feature_matrix.txt', np.array(data['drug_drug_interaction_feature_matrix']))
    np.savetxt('./structure_feature_matrix.txt', np.array(data['structure_feature_matrix']))
    np.savetxt('./target_feature_matrix.txt', np.array(data['target_feature_matrix']))
    np.savetxt('./enzyme_feature_matrix.txt', np.array(data['enzyme_feature_matrix']))
    np.savetxt('./pathway_feature_matrix.txt', np.array(data['pathway_feature_matrix']))


def splitEdgesBasedOnFolds(interactionsIndicesFolds, k):
    testSuperVisionEdgesIndex = interactionsIndicesFolds[k]
    messageEdgesIndex = np.setdiff1d(interactionsIndicesFolds.flatten(), testSuperVisionEdgesIndex, assume_unique=True)
    trainSuperVisionEdgesIndex = messageEdgesIndex
    return messageEdgesIndex, trainSuperVisionEdgesIndex, testSuperVisionEdgesIndex

def splitter(sameSize, interactions, nonInteractions, INTERACTIONS_NUMBER, NONINTERACTIONS_NUMBER):
    # remove some samples to be dividable by the folds
    interactionSelectionSize = INTERACTIONS_NUMBER - (INTERACTIONS_NUMBER % FOLDS)
    nonInteractionSelectionSize = NONINTERACTIONS_NUMBER - (NONINTERACTIONS_NUMBER % FOLDS)

    # choose randomly
    interactionsIndices = np.random.choice(INTERACTIONS_NUMBER, interactionSelectionSize, replace=False)
    if sameSize:
        nonInteractionsIndices = np.random.choice(NONINTERACTIONS_NUMBER, interactionSelectionSize, replace=False)
    else:
        nonInteractionsIndices = np.random.choice(NONINTERACTIONS_NUMBER, nonInteractionSelectionSize, replace=False)

    selectedInteractionsPairs = interactions[interactionsIndices]
    selectedNonInteractionsPairs = nonInteractions[nonInteractionsIndices]
    
    test_df = pd.DataFrame(selectedNonInteractionsPairs)
    test_df.to_csv(saving_path+'Negsample.csv')

    return selectedInteractionsPairs, selectedNonInteractionsPairs

def foldify(totalInteractions, totalNonInteractions):
    sizeOfInteractions = totalInteractions.shape[0]
    sizeOfNonInteractions = totalNonInteractions.shape[0]

    totalInteractionIndices = np.random.permutation(sizeOfInteractions)
    totalNonInteractionIndices = np.random.permutation(sizeOfNonInteractions)

    interactionsIndicesFolds = totalInteractionIndices.reshape(FOLDS, sizeOfInteractions // FOLDS)
    nonInteractionsIndicesFolds = totalNonInteractionIndices.reshape(FOLDS, sizeOfNonInteractions // FOLDS)

    return interactionsIndicesFolds, nonInteractionsIndicesFolds

def makePosEdgeIndex(dataset, name, delimiter=',', percent = 100, dataframe=False):
    if dataframe:
        matrix = pd.read_csv('./data/' + dataset + '/' + name, delimiter=delimiter, header=None).drop(columns=0, index=0).to_numpy(dtype=np.integer)
    else:
        if dataset == 'Mdata':
            sdf2 = pd.read_csv("./data/Mdata/A_CG.csv", index_col=0)
            matrix = sdf2.values
        elif dataset == 'KFCdata':
            sdf2 = pd.read_csv('./data/KFCdata' + '/' + name + '.csv', index_col=0)
            matrix = sdf2.values
        elif dataset == 'Bdata':
            sdf2 = pd.read_csv('./data/Bdata' + '/' + name + '.csv', index_col=0)
            matrix = sdf2.values
        else:
            matrix = np.loadtxt('./data/' + dataset + '/' + name, delimiter=delimiter)


    result = np.where(matrix == 1)
    edgeIndex = [[], []]
    choseIndex = np.random.choice(result[0].shape[0], int(result[0].shape[0] * percent / 100), replace=False)
    for index in choseIndex:
        edgeIndex[0].append(result[0][index]) 
        edgeIndex[1].append(result[1][index])

    edgeIndex = torch.tensor(edgeIndex, dtype=torch.long)
    return edgeIndex


def metadata(dataset):
    if dataset == 'LAGCN' or dataset == 'LAGCN-therapeutic':
        numbers = {
            'drug': 269,
            'disease': 598,
            'pathway': 258,
            'enzyme': 108,
            'target': 529,
            'structure': 881,
            'interactions': 18416 if dataset == 'LAGCN' else 6244,
            'nonInteractions': 142446 if dataset == 'LAGCN' else 154618,
        }
        if dataset == 'LAGCN':
            drugDisease = np.loadtxt('./data/' + 'LAGCN' + '/drug_disease.csv', delimiter=',')
            totalInteractions = np.array(np.mat(np.where(drugDisease == 1)).T) #(18416, 2)
            totalNonInteractions = np.array(np.mat(np.where(drugDisease == 0)).T) #(142446, 2)
        else:
            drugDisease = np.loadtxt('./data/' + 'LAGCN' + '/therapeutic.txt', delimiter=' ')
            totalInteractions = np.array(np.mat(np.where(drugDisease == 1)).T) #(6244, 2)
            totalNonInteractions = np.array(np.mat(np.where(drugDisease < 1)).T) #(154618, 2)
    
    elif dataset == 'deepDR':
        numbers = {
            'drug': 1519,
            'disease': 1229,
            'protein': 1025,
            'sideEffect': 12904,
            'interactions': 6677,
            'nonInteractions': 1860174,
        }
        drugDisease = np.loadtxt('./data/' + 'deepDR' + '/drugDisease.txt', delimiter='\t')
        totalInteractions = np.array(np.mat(np.where(drugDisease == 1)).T) #(6677, 2)
        totalNonInteractions = np.array(np.mat(np.where(drugDisease == 0)).T) #(1860174, 2)

    elif dataset == 'Mdata':
        numbers = {
            'drug': 5975,
            'disease': 2447,
            'protein': 12852,
            'interactions': 35146,
            'nonInteractions': 14585679,
        }
        sdf = pd.read_csv("./data/Mdata/A_CD.csv", index_col=0)
        drugDisease = sdf.values
        totalInteractions = np.array(np.mat(np.where(drugDisease == 1)).T) #(35146, 2)
        totalNonInteractions = np.array(np.mat(np.where(drugDisease == 0)).T) #(14585679, 2)
        
    elif dataset == 'KFCdata':
        numbers = {
            'drug': 894,
            'disease': 454,
            'protein': 18877,
            'gene': 20561,
            'pathway': 314,
            'interactions': 2558,
            'nonInteractions': 401971,
        }
        sdf = pd.read_csv("./data/KFCdata/KFCdata_A_DD.csv", index_col=0)
        drugDisease = sdf.values
        totalInteractions = np.array(np.mat(np.where(drugDisease == 1)).T) #(35146, 2)
        totalNonInteractions = np.array(np.mat(np.where(drugDisease == 0)).T) #(14585679, 2)
        
    elif dataset == 'Bdata':
        numbers = {
            'drug': 269,
            'disease': 598,
            'protein': 6040,
            'interactions': 18416,
            'nonInteractions': 142446,
        }
        sdf = pd.read_csv("./data/Bdata/Bdataset_baseline.csv", header=None)
        drugDisease = sdf.values
        totalInteractions = np.array(np.mat(np.where(drugDisease == 1)).T) #(35146, 2)
        totalNonInteractions = np.array(np.mat(np.where(drugDisease == 0)).T) #(14585679, 2)

    return totalInteractions, totalNonInteractions, numbers.get('interactions'), numbers.get('nonInteractions')

def prepareDrugData(featureList, embeddingMethod):
    featureMatrixDic = {}
    finalDic = {}

    for feature in featureList:
        matrix = np.loadtxt('./data/lagcn/'+ feature+ '_feature_matrix.txt')
        featureMatrixDic[feature] = matrix
    
    if embeddingMethod == 'AE' or embeddingMethod == 'matrix':
        finalDic = featureMatrixDic;

    elif embeddingMethod == 'jaccardGCN':
        for feature, matrix in featureMatrixDic.items():
            jacMatrix = Jaccard(matrix)
            # remove self loops
            embedding = GCNEmbedding(jacMatrix, matrix, 50, 0.001)
            finalDic[feature] = embedding

    elif embeddingMethod == 'jaccard':
        for feature, matrix in featureMatrixDic.items():
            finalDic[feature] = Jaccard(matrix)
            
    elif embeddingMethod == 'PCA':
        for feature, matrix in featureMatrixDic.items():
            pca = PCA(n_components=2,)
            transformed = pca.fit_transform(matrix)
            finalDic[feature] = transformed

    elif embeddingMethod == 'diseaseGCN':
        diseaseSim = np.loadtxt('./data/lagcn/dis_sim.csv', delimiter=',')
        drugDisease = np.loadtxt('./data/lagcn/drug_dis.csv', delimiter=',')
        drugDisease = np.transpose(drugDisease)
        embedding = GCNEmbedding(diseaseSim, drugDisease, 300, 0.0001)
        np.savetxt('./diseaseGCN_feature_matrix.txt', np.array(embedding))
        exit()

    else:
        exit('please provide a known embedding method')
    # elif similarity == 'cosine':
    return finalDic


