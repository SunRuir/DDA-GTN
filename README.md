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

## 3.3 Cross Validation and Prediction
### 3.3.1 Cross Validation
> python src/MdataNW.py
This will save models and logs in Siridataset/models and result/log.txt, respectively.
#### Optional parameters
- epoch: Default=100. The number of training epoch.
- lr: Default=0.005. The initial learning rate.
- weight_decay: Default=5e-4. The weight decay for this training.
- node_dim: Default=128. The dim for node feature matrix.

### 3.3.2 Prediction
All results are saved in zenodo
```
https://zenodo.org/records/10826915
```
> python src/casestudy_Mdata.py
> This will save models and logs in Siridataset/models and result/log.txt, respectively.








