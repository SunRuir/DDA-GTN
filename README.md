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

## 2.2 Data
The data is in zenodo, which contains the input data for Mdataset's 5 cross validation on DDA-GTN.

## 2.3 result
- records
  >This folder contains the log files from the five cross-validations.

# 3. Workflow
## 3.1 Configuration environment
```
pip install -r request.txt
```
## 3.2 Download Mdataset
```
[zenodo链接](https://zenodo.org/records/10826915)
```
## 3.3 Cross Validation and Prediction
### 3.3.1 Cross Validation
> python src/MdataNW.py
This will save models and logs in Siridataset/models and result/log.txt, respectively.
#### Optional parameters
- epoch: Default=100. The number of training epoch.
- lr: Default=0.001. The initial learning rate.
- weight_decay: Default=5e-4. The weight decay for this training.
- node_dim: Default=64. The dim for node feature matrix.

### 3.3.2 Prediction
Downloading casestudy data from zenodo
```
zenodo链接
```
> python src/casestudy_Mdata.py
> This will save models and logs in Siridataset/models and result/log.txt, respectively.
#### Optional parameters
- epoch: Default=100. The number of training epoch.
- lr: Default=0.001. The initial learning rate.
- weight_decay: Default=5e-4. The weight decay for this training.
- node_dim: Default=64. The dim for node feature matrix.









