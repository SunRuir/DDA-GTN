# Note

This folder contains the running code for DDA-GTN, divided into is one time 5-fold cross-validation, fivetimes 5-fold cross-validation, casestudy, data processing, and model.

# Contents

1 **MdataNW.py**：This file containing the code for running one time 5-cross validation for DDA-GTN. During the running of the code, log files are generated and stored in the saving_path/logi.txt.

2 **MdataNW_5cross.py**：This file containing the code for running 5 times 5-cross validation for DDA-GTN. During the running of the code, log files are generated and stored in the saving_path/logi.txt.

3 **casestudy_Mdata.py**：This file contains code for predicting new drug-disease associations.

4 **model.py**：This file contains the code for the Fast-GTN model.

5 **utils.py**：This file contains the function in the model implementation.

6 **inits.py**：This file contains the custom function in the model implementation.

7 **methods.py**：This file contains the function that calculates the average indicator value for the 50% discount cross validation.

8 **split.py**：This file generates the dataset used in training. The generated files will be stored in the Siridataset{i} folder.

9 **network.py**：This file contains the code for the DDA-GTN model.

10 **data_preprocessing.py**：This file generates the dataset used in training. The generated files will be stored in the dataset folder.

11 **negativesample.ipynb**：This file holds the process of randomly generating negative samples, and the results are saved in Data\NegativeSample.csv

12 **feature_ge_Cycle.py**: This file contains a loop to generate five five-fold cross-validated disease feature.The generated files will be stored in the feature0{i} folder. feature0{i} is a folder name that can be set when running the code, it can be changed to another name.

D
## Data
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
