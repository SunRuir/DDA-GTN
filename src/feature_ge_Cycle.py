import pandas as pd
import numpy as np

if __name__ == '__main__':


    node_list = pd.read_csv("../Data/node_list.csv")
    sdf = pd.read_csv("../Data/C_D.csv")
    feature = pd.read_csv("../Data/drug_feature.csv")


    # Setting the five folder paths to be read in
    read_paths = ['../Data/five_cvdata/Siridataset01/', '../Data/five_cvdata/Siridataset02/', '../Data/five_cvdata/Siridataset03/',
                    '../Data/five_cvdata/Siridataset04/', '../Data/five_cvdata/Siridataset05/']

    # Cyclically read in five file paths
    for r, folder_path in enumerate(read_paths):
        # Build the name of each folder that holds the disisease feature five times
        """
            Specify the path where the feature will be saved folder-name is the set folder name, which can be changed.
            In this experiment, the saved file name is set to be feature0{r + 1}, 
            and r+1 denotes the location where the disease feature is saved for the r+1th time
            
            ！！Note: The folder where the features are stored needs to be created in a blank folder in the specified path before running the code.
            The folder created needs to be exactly the same as the folder name and path set by save_path in order to successfull work.
        """
        save_path = f'../Data/feature_generate/feature0{r + 1}/'
        print(f'Saved DataFrame to: {save_path}')

        # Five cycles of read-in of the 5-cv
        for i in range(5):
            read_file_path = folder_path + f'DDI_test{str(i)}.csv'
            print(f'Read DataFrame to: {read_file_path}')
            df_test = pd.read_csv(read_file_path, index_col=0)

            # Replace the elements in test.csv with the values at the corresponding positions in node.csv
            df_test['0'] = node_list.loc[df_test['0']]['Node'].values
            df_test['1'] = node_list.loc[df_test['1']]['Node'].values

            new_columns = {'0': 'ChemicalID', '1': 'DiseaseID'}
            df_selected = df_test[['0', '1']].rename(columns=new_columns)

            # Merge A_CD.csv with df_selected
            merged = pd.merge(sdf, df_selected, on=['ChemicalID', 'DiseaseID'], how='left', indicator=True)
            # Delete the rows to be deleted in A_CD.csv
            rows_to_drop = merged[merged['_merge'] == 'left_only']

            chemical = node_list.loc[:5972]
            gene = node_list.loc[5973:18824]
            disease = node_list.loc[18825:21271]

            n_chemicals = np.array(chemical['Node'])
            n_diseases = np.array(disease['Node'])
            n_genes = np.array(gene['Node'])

            df = pd.DataFrame(index=n_chemicals, columns=n_diseases)

            for q in range(len(rows_to_drop)):
                row_i = rows_to_drop.iloc[[q]]
                user_id = row_i['ChemicalID'].tolist()[0]
                item_name = row_i['DiseaseID'].tolist()[0]
                df.at[user_id, item_name] = 1
            df.fillna(0, inplace=True)

            matrix1 = feature.values[:, 1:]
            print(np.shape(matrix1))
            matrix2 = df.T

            print('-----------{:.4f}time matrix multiplication complete！---------'.format(i))
            d = np.dot(matrix2.values, matrix1)
            dfd = pd.DataFrame(d)

            save_file_path = save_path + f'disease_feature{str(i)}.csv'
            print(f'Save Disease Feature to: {save_file_path}')
            dfd.to_csv(save_file_path)
            print(d)












