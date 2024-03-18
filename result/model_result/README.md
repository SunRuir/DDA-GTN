# Note

This file contains the model parameters and results for the fifth  5-cross-validation, so you can import the model directly and test it without training, but please note that this is the fifth 5-cross-validation, and you need to change the folder where you read in the data to the fifth one.

# Contens

1 **fastGTN_{i}.pth**: Models preserved in the ith fold

2 **pred_label.csv**: The predict label of the drug-disease association pair, the first column is the row index, the second column is the 0/1 label, the first row is the column index, the header and the index are saved together during the saving process. 0 means there is no association, 1 means there is an association.

3 **true_label.csv**: The real label of the drug-disease association pair, the first column is the row index, the second column is the 0/1 label, the first row is the column index, the header and the index are saved together during the saving process. 0 means there is no association, 1 means there is an association.
