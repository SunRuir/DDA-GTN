DDA-GTN: large-scale drug repurposing on drug-gene-disease heterogenous association networks using graph transformers
==

In this work, we first present a benchmark dataset that includes three entities: drugs, genes, and diseases that form a three-layer heterogeneous network, and introduce Graph Transformers Networks to learn the low-dimensional embedded representations of drugs-diseases in the heterogeneous network as a way to predict drug-disease associations. We named this method DDA-GTN.

# 1.Platform and Dependency

## 1.1 Platform

- ubuntu 18.04

- RTX 3090(24GB)

## 1.2 Dependency

| Requirements      | Release                                |
| --------- | ----------------------------------- |
| CUDA     | 11.3                     |
| Python     | 3.9.13                     |
| torch     | 1.11.0                     |
| torch_geometric     | 2.1.0.post1                     |
| torch-scatter     | 1.6.0                     |
| torch-sparse     | 0.6.15                     |
| torch-cluster     | 1.6.0                     |
| pandas     | 1.4.4                     |
| scikit-learn     | 1.1.2                     |
| matplotlib     | 3.6.0                     |

# 2. Project Catalog Structure

## 2.1 src

> This folder stores the code files.

## 2.2 result

> This folder contains the logs of the five five-fold cross-validations, the model parameters and predictions of the fifth cross-validation, the code for calculating the results and standard deviation of the average five cross-validations, and the code for drawing the AUC and ROC curves.

## 2.3 compare

> This file contains the running code for LAGCN, LHGCE, and REDDA, which were originally sourced from the github repositories published by the corresponding publications.

# 3. Workflow

## 3.1 Configuration environment

```
pip install -r request.txt
```

## 3.2 Download Mdataset

Mdataset is the benchmarking dataset of DDA-GTN. It can be downloaded from [Zenodo](https://zenodo.org/records/10826915).

Put the dataset directory under the root directory of this project. Your project should look like this:

```
/DDA-GTN
  |-Data
  |-src
  |-result
  |-compare
```

### Data directory preparation

The data is in zenodo, which contains the input data for Mdataset's 5 cross validation on DDA-GTN.

The Data folder in zenodo should be stored side by side with the src folder.

| Folder name      | Descriptions                                |
| --------- | ----------------------------------- |
| C_D.csv     | drug-disease association <br>  CTD IDs -- MeSH IDs                     |
| C_G.csv     | drug-gene association <br>  CTD IDs -- Gene Symbol                     |
| G_D.csv     | gene-disease association <br>  Gene Symbol -- MeSH IDs -- InferenceScore                    |
| disease_feature.csv     | disease feature matrix 2447*881 matrices                     |
| drug_feature.csv     | drug feature matrix 5975*881 matrices                     |
| gene_feature.csv     | gene feature matrix 12582*881 matrices                     |
| node_list.csv     | It contains all the nodes in the heterogeneous network in the order of drug(CTD IDs), gene(Gene Symbol), and disease(MeSH IDs), and the positions corresponding to the nodes are the indexes that end up in the sparse matrix                    |
| NegativeSample0829.csv     | Randomly select as many negative samples as positive samples from the drug-disease association matrix <br> drug index -- disease index |

## 3.3 Split dataset

> python src/split.py seed **

To run this code you need to set the random seed and the save path, the path needs to be created manually before you run it








## 3.4 Cross Validation and Prediction

### 3.4.1 Cross Validation

> python src/MdataNW.py

This will save models and logs in Siridataset/models and result/log.txt, respectively.

#### Optional parameters

- epoch: Default=100. The number of training epoch.

- lr: Default=0.005. The initial learning rate.

- weight_decay: Default=5e-4. The weight decay for this training.

- node_dim: Default=128. The dim for node feature matrix.

### 3.4.2 Prediction

All results are deposited in Zenodo for reproducing purpose. The raw logs can also be found in [Zenodo](https://zenodo.org/records/10826915).

> python src/casestudy_Mdata.py

> This will save models and logs in Siridataset/models and result/log.txt, respectively.








