from prepareData import makePosEdgeIndex
from torch_geometric.data import HeteroData
import numpy as np 
import torch
import pandas as pd

def dataloader(dataset):
    print('dataset: ', dataset)
    data = HeteroData()
    if dataset == 'LRSSL':
        numbers = {
            'drug': 763,
            'disease': 681,
            'go': 4447,
            'target': 1426,
            'structure': 623,
            'interactions': 3051,
            'nonInteractions': 516552,
        }

        drugDisease = pd.read_csv('./data/' + dataset + '/drug_dis_mat.txt', sep='\t', header=None).drop(columns=0, index=0).to_numpy(dtype=np.integer)
        totalInteractions = np.array(np.mat(np.where(drugDisease == 1)).T) #(3051, 2)
        totalNonInteractions = np.array(np.mat(np.where(drugDisease == 0)).T) #(516552, 2)
        
        data['drug'].x = torch.eye(numbers.get('drug'), dtype=torch.float)
        data['disease'].x = torch.tensor(pd.read_csv('./data/' + dataset + '/disease_similarity.txt', sep='\t', header=None).drop(columns=0, index=0).to_numpy(dtype=np.float64), dtype=torch.float)        
        data['go'].x = torch.eye(numbers.get('go'), dtype=torch.float)
        data['target'].x = torch.eye(numbers.get('target'), dtype=torch.float)
        data['structure'].x = torch.eye(numbers.get('structure'), dtype=torch.float)
      
        data['drug', 'edge', 'go'].edge_index = makePosEdgeIndex(dataset, 'drug_target_go_mat.txt', '\t', dataframe=True)
        data['drug', 'edge', 'target'].edge_index = makePosEdgeIndex(dataset, 'drug_target_domain_mat.txt', '\t', dataframe=True)
        data['drug', 'edge', 'structure'].edge_index = makePosEdgeIndex(dataset, 'drug_pubchem_mat.txt', '\t', dataframe=True)

    elif dataset == 'LAGCN' or dataset == 'LAGCN-therapeutic':
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

        data['drug'].x = torch.eye(numbers.get('drug'), dtype=torch.float)
        # data['disease'].x = torch.tensor(np.loadtxt('./data/LAGCN/dis_sim.csv', delimiter=','), dtype=torch.float)        
        data['disease'].x = torch.eye(numbers.get('disease'), dtype=torch.float)
        data['pathway'].x = torch.eye(numbers.get('pathway'), dtype=torch.float)
        data['enzyme'].x = torch.eye(numbers.get('enzyme'), dtype=torch.float)
        data['target'].x = torch.eye(numbers.get('target'), dtype=torch.float)
        data['structure'].x = torch.eye(numbers.get('structure'), dtype=torch.float)
      
        data['drug', 'edge', 'pathway'].edge_index = makePosEdgeIndex('LAGCN', 'drug_pathway.txt', ' ')
        data['drug', 'edge', 'enzyme'].edge_index = makePosEdgeIndex('LAGCN', 'drug_enzyme.txt', ' ')
        data['drug', 'edge', 'target'].edge_index = makePosEdgeIndex('LAGCN', 'drug_target.txt', ' ')
        data['drug', 'edge', 'structure'].edge_index = makePosEdgeIndex('LAGCN', 'drug_structure.txt', ' ')
    
    elif dataset == 'deepDR':
        numbers = {
            'drug': 1519,
            'disease': 1229,
            'protein': 1025,
            'sideEffect': 12904,
            'interactions': 6677,
            'nonInteractions': 1860174,
        }

        data['drug'].x = torch.eye(numbers.get('drug'), dtype=torch.float)
        data['disease'].x = torch.eye(numbers.get('disease'), dtype=torch.float)
        data['protein'].x = torch.eye(numbers.get('protein'), dtype=torch.float)
        data['sideEffect'].x = torch.eye(numbers.get('sideEffect'), dtype=torch.float)

        data['drug', 'edge', 'protein'].edge_index = makePosEdgeIndex('deepDR', 'drugProtein.txt', '\t')
        data['drug', 'edge', 'sideEffect'].edge_index = makePosEdgeIndex('deepDR', 'drugsideEffect.txt', '\t')

    elif dataset == 'MGATRx':
        numbers = {
            'drug': 4008,
            'disease': 2958,
            'pathway': 2308,
            'meshcat': 2124,
            'target': 8722,
            'sideEffect': 11744,
            'substructure': 881,
            'interactions': 8957,
            'nonInteractions': 11846707,
        }
        drugDisease = np.loadtxt('./data/' + 'MGATRx' + '/drug-disease.txt', delimiter=' ')
        totalInteractions = np.array(np.mat(np.where(drugDisease == 1)).T) #(8957, 2)
        totalNonInteractions = np.array(np.mat(np.where(drugDisease == 0)).T) #(11846707, 2)

        data['drug'].x = torch.eye(numbers.get('drug'), dtype=torch.float)
        data['disease'].x = torch.eye(numbers.get('disease'), dtype=torch.float)        
        data['pathway'].x = torch.eye(numbers.get('pathway'), dtype=torch.float)
        data['meshcat'].x = torch.eye(numbers.get('meshcat'), dtype=torch.float)
        data['target'].x = torch.eye(numbers.get('target'), dtype=torch.float)
        data['sideEffect'].x = torch.eye(numbers.get('sideEffect'), dtype=torch.float)
        data['substructure'].x = torch.eye(numbers.get('substructure'), dtype=torch.float)

        data['drug', 'edge', 'substructure'].edge_index = makePosEdgeIndex(dataset, 'drug-chemfp.txt', ' ')
        data['drug', 'edge', 'meshcat'].edge_index = makePosEdgeIndex(dataset, 'drug-meshcat.txt', ' ')
        data['drug', 'edge', 'pathway'].edge_index = makePosEdgeIndex(dataset, 'drug-pathways.txt', ' ')
        data['drug', 'edge', 'sideEffect'].edge_index = makePosEdgeIndex(dataset, 'drug-se.txt', ' ')
        data['drug', 'edge', 'target'].edge_index = makePosEdgeIndex(dataset, 'drug-targets.txt', ' ')
        data['disease', 'edge', 'pathway'].edge_index = makePosEdgeIndex(dataset, 'disease-pathways.txt', ' ')
        data['disease', 'edge', 'target'].edge_index = makePosEdgeIndex(dataset, 'disease-targets.txt', ' ')
        
        
    elif dataset == 'Mdata':
        numbers = {
            'drug': 5975,
            'disease': 2447,
            'protein': 12852,
            'interactions': 35146,
            'nonInteractions': 14585679,
        }
        data['drug'].x = torch.eye(numbers.get('drug'), dtype=torch.float)
        data['disease'].x = torch.eye(numbers.get('disease'), dtype=torch.float)
        data['protein'].x = torch.eye(numbers.get('protein'), dtype=torch.float)
        # data['sideEffect'].x = torch.eye(numbers.get('sideEffect'), dtype=torch.float)

        data['drug', 'edge', 'protein'].edge_index = makePosEdgeIndex('Mdata', 'drugProtein.txt', '\t')
        # data['drug', 'edge', 'sideEffect'].edge_index = makePosEdgeIndex('deepDR', 'drugsideEffect.txt', '\t')
        
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
        data['drug'].x = torch.eye(numbers.get('drug'), dtype=torch.float)
        data['disease'].x = torch.eye(numbers.get('disease'), dtype=torch.float)
        data['protein'].x = torch.eye(numbers.get('protein'), dtype=torch.float)
        data['gene'].x = torch.eye(numbers.get('gene'), dtype=torch.float)
        data['pathway'].x = torch.eye(numbers.get('pathway'), dtype=torch.float)
              
        data['drug', 'edge', 'protein'].edge_index = makePosEdgeIndex('KFCdata', 'KFCdata_A_DP', ' ')
        data['protein', 'edge', 'gene'].edge_index = makePosEdgeIndex('KFCdata', 'KFCdata_A_PG', ' ')
        data['gene', 'edge', 'pathway'].edge_index = makePosEdgeIndex('KFCdata', 'KFCdata_A_GP', ' ')
        data['pathway', 'edge', 'disease'].edge_index = makePosEdgeIndex('KFCdata', 'KFCdata_A_PD', ' ') 
        
    elif dataset == 'Bdata':
        numbers = {
            'drug': 269,
            'disease': 598,
            'protein': 6040,
            'interactions': 18416,
            'nonInteractions': 142446,
        }
        data['drug'].x = torch.eye(numbers.get('drug'), dtype=torch.float)
        data['disease'].x = torch.eye(numbers.get('disease'), dtype=torch.float)
        data['protein'].x = torch.eye(numbers.get('protein'), dtype=torch.float)
        # data['sideEffect'].x = torch.eye(numbers.get('sideEffect'), dtype=torch.float)

        data['drug', 'edge', 'protein'].edge_index = makePosEdgeIndex('Bdata', 'drug_protein', '\t')
        data['protein', 'edge', 'disease'].edge_index = makePosEdgeIndex('Bdata', 'protein_disease', '\t')

    return data
