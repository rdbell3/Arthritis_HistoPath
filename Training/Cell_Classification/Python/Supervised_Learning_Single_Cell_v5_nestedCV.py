# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 12:11:44 2021

@author: richa
"""

#%% Clustering Imports

import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime 
import pandas as pd
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from datetime import datetime


from sklearn.impute import SimpleImputer

#%% Paths

root_path = 'Path\to\output\folder'

results_root = root_path + '\\Output\\results_excels'
images_path = root_path + '\\Images'
save_path = root_path + '\\Output\\Cluster_Labels'
plots_save_path = root_path + '\\Output\\Plots'
data_save_path = root_path + '\\Output\\Data'
models_save_path = root_path + '\\Output\\Models-Performance'



#%% Import Data

all_data = pd.read_csv(f'{data_save_path}\\cell_by_feature_Martrix.csv')#, sep="\t", header = 0 , index_col = 0)

all_data = all_data.drop(columns='Unnamed: 0')


#%%
# Cell_Labels_Dictionary = {'Synovial_Fibroblasts' : 1, 'Synovial_Lining_Cells' : 2,'Synovial_Vessel_Cells' : 3,
#                           'Lymphocytes' : 4, 'Synovial_Sub-Lining_Cells' : 5, 'Synovial_Fat_Cells' : 6,
#                           'Growth_Plate_Chondrocytes' : 7, 'Bone_Cell' : 8 }

Cell_Labels_Dictionary = ['Synovial_Fibroblasts', 'Synovial_Lining_Cells','Synovial_Vessel_Cells',
                          'Lymphocytes',  'Synovial_Fat_Cells', 'Growth_Plate_Chondrocytes', 'Bone_Cell']

cell_labels = all_data['Cell_Label'].values.astype(np.int8).tolist()

for idx1, i in enumerate(cell_labels):
    if i == 0:
        cell_labels[idx1] = 'No_Label'
        continue
    else:
        for idx2, k in enumerate(Cell_Labels_Dictionary):
            if i == idx2 + 1:
                cell_labels[idx1] = k
                continue

cell_labels = pd.DataFrame(cell_labels, columns=['Cell_Label_Names'])
#cell_labels = pd.DataFrame(cell_labels, columns=['Cell_Label'])
print(cell_labels.value_counts())

Tissue_Labels_Dictionary = ['Synovium', 'Muscle-Tendon','Artifact',
                          'Growth_Plate',  'Bone_Marrow', 'Cortical_Bone',
                          'Trabecular_Bone', 'Meniscus-Cartilage', 'Fat']

tissue_labels = all_data['Tissue_Label'].values.astype(np.int8).tolist()

for idx1, i in enumerate(tissue_labels):
    if i == 0:
        tissue_labels[idx1] = 'No_Label'
        continue
    else:
        for idx2, k in enumerate(Tissue_Labels_Dictionary):
            if i == idx2 + 1:
                tissue_labels[idx1] = k
                continue

tissue_labels = pd.DataFrame(tissue_labels, columns=['Tissue_Label'])



slide_names = all_data['hemo-nucleus_Name']

slide_names_truncated = [x.split('[')[0][:-1] for x in slide_names]

slide_names_truncated = pd.DataFrame(slide_names_truncated, columns = ['Slide_Name'])





#%%

#all_data = all_data.drop(columns=['Cell_Label', 'Tissue_Label'])

all_data = all_data.drop(columns=['Tissue_Label'])

# all_data.reset_index(drop=True, inplace=True)
# cell_labels.reset_index(drop=True, inplace=True)
# tissue_labels.reset_index(drop=True, inplace=True)


all_data = pd.concat([all_data, cell_labels, tissue_labels,slide_names_truncated ], axis = 1)

cells_with_labels = all_data.loc[all_data['Cell_Label_Names'] != 'No_Label']
cells_withOut_labels = all_data.loc[all_data['Cell_Label_Names'] == 'No_Label'] 


#cells_with_labels.to_csv(f'{data_save_path}//only_cells_with_labels.csv')




#%% K-Folds

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support, classification_report


from xgboost import XGBClassifier

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

feature_used = cells_with_labels.drop(columns = ['hemo-nucleus_X', 'hemo-nucleus_Y', 'hemo-nucleus_XM',
                                           'hemo-nucleus_YM', 'hemo-nucleus_Name', 'Cell_Label','Cell_Label_Names', 'Tissue_Label', 'Slide_Name']).columns.tolist()



k_folds = 5

#kf = KFold(n_splits = k_folds)

skf = StratifiedKFold(n_splits = k_folds)

unique_slide_names = np.unique(slide_names_truncated, return_counts=False)

#cells_with_labels.reset_index(drop=True, inplace=True)


#cells_with_labels = pd.concat([cells_with_labels, slide_names_truncated], axis = 1)

X = cells_with_labels.drop(columns = ['hemo-nucleus_X', 'hemo-nucleus_Y', 'hemo-nucleus_XM',
                                           'hemo-nucleus_YM', 'hemo-nucleus_Name', 'Cell_Label','Cell_Label_Names', 'Tissue_Label', 'Slide_Name']).values

#cells_with_labels = cells_with_labels.drop(columns = ['Slide_Name'])


stratifier = cells_with_labels['Slide_Name']

y = cells_with_labels['Cell_Label'].values

y = y - 1 # make y start at 0 for XGBOOST

Cell_Labels_Dictionary.sort()
Cell_Labels_Dictionary_pd = pd.DataFrame(Cell_Labels_Dictionary, columns = ['Cell Labels_Names'] )


fmax = np.finfo(np.float64).max
pinf = float('+inf')
ninf = float('-inf')

X[X >= fmax] = 0
X[X >= pinf] = 0
X[X <= ninf] = 0


performance = []
con_mat = []
models = []
count = 0
for train_index, test_index in skf.split(X, stratifier):
    
    # Record the starting time
    start_time = datetime.now()
    print(start_time)
    count +=1
    # print(stratifier.iloc[train_index].describe())
    # print(stratifier.iloc[train_index].value_counts())
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    imputer = SimpleImputer(missing_values=np.nan,
                       strategy='mean')#'constant', fill_value = 0)
    imputer.fit(X_train)
    
    X_train = np.array(imputer.transform(X_train), dtype=np.float32)
    X_test = np.array(imputer.transform(X_test), dtype=np.float32)
    
   
    cv_inner = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # need Cuda GPU set up
    clf = XGBClassifier( objective='multi:softmax', num_class = 7,
                        tree_method='hist', device = 'cuda',
                        eval_metric = 'merror') 
    
    space = {
        "learning_rate": [0.05, 0.1, 0.2],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "subsample": [0.25, 0.5],
        "max_depth": [6, 12],
        "n_estimators": [10, 100, 200, 400]
        "gamma": [0, 0.1, 0.3],
        "min_child_weight": [1, 5, 10]
        }

    # define search
    search = GridSearchCV(clf, space, scoring='accuracy', cv=cv_inner, refit=True)
   
    result = search.fit(X_train, y_train)
    
    # get the best performing model fit on the whole training set
    best_model = result.best_estimator_
	# evaluate model on the hold out dataset
    yhat = best_model.predict(X_test)
    
    performance_current = precision_recall_fscore_support(y_test, yhat, average = None)
    
    models.append(best_model)  
    
    print(classification_report(y_test, yhat))

    #print(best_model.get_params())
    model_params = best_model.get_params()
    
    # Remove key-value pairs where the value is None
    keys_to_remove = [key for key, value in model_params.items() if value is None]
    
    for key in keys_to_remove:
        del model_params[key]
    
    if count == 1:
        model_params_df = pd.DataFrame(list(model_params.values())).T
        model_params_df.columns = model_params.keys()
    else:
        temp = pd.DataFrame(list(model_params.values())).T
        temp.columns = model_params.keys()
        model_params_df = pd.concat([model_params_df, temp])
    
    print(model_params)




	# report progress

    model_number = pd.DataFrame([f'Model_{count}']*len(Cell_Labels_Dictionary), columns = ['Model_Number'])
                                                  
    performance_current = pd.DataFrame(np.column_stack(precision_recall_fscore_support(y_test, yhat, average = None)),
                     columns = ['Precision', 'Recall', 'F1', 'Support'])

    performance_current = pd.concat([model_number, Cell_Labels_Dictionary_pd, performance_current], axis = 1)


    
    if count == 1:
        performance = performance_current
        
    else:
        performance = pd.concat([performance, performance_current])    
    
    # Record the ending time
    end_time = datetime.now()
    # Calculate the duration
    duration = (end_time - start_time)
    print(end_time)
    print(f"The loop ran for {duration} Mins.")

    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    indices_top20 = indices[0:60]
    top20_features_used = []
    
    for f in range(0,60):
        a = feature_used[indices_top20[f]]
        top20_features_used.append(a)
    
    
    # print("Feature importances ranking:")
    # for f in range(0,19):
    #     print(f)
    #     print('{0:.2f}%'.format(importances[indices_top20[f]]*100).rjust(6, ' '),
    #           'feature %d: %s ' % (indices_top20[f], top20_features_used[indices_top20[f]]))
    
    
    # Plot the feature importances of the forest
    plt.figure(figsize=(10, 8))
    plt.ylabel("Feature importances")
    plt.bar(range(len(top20_features_used)), importances[indices_top20], align="center")
    plt.xticks(range(len(top20_features_used)), top20_features_used, rotation=90)
    plt.show()
    
    
    cm = confusion_matrix(y_test, yhat)
    con_mat.append(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=best_model.classes_)
    disp.plot()
    plt.xticks(rotation = 90)
    plt.show()
   


    
    
#%%Save Model
import pickle
from datetime import datetime
import os
import csv


now = datetime.now()


new_model_folder_Name = now.strftime('%F-%H-%M')

if not os.path.isdir(f'{models_save_path}//{new_model_folder_Name}//'):
    os.mkdir(f'{models_save_path}//{new_model_folder_Name}//')



for idx, i in enumerate(models):

    filename = f'{models_save_path}//{new_model_folder_Name}//model_{idx + 1}.sav'
    cm_current = pd.DataFrame(con_mat[idx], columns = Cell_Labels_Dictionary)
    
    cm_current.to_csv(f'{models_save_path}//{new_model_folder_Name}//model_{idx + 1}.csv')
    pickle.dump(i, open(filename, 'wb'))


performance.to_csv(f'{models_save_path}//{new_model_folder_Name}//performace_all_Models.csv')
model_params_df.to_csv(f'{models_save_path}//{new_model_folder_Name}//model_params_all_Models.csv')



    
