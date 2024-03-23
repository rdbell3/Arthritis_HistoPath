# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 12:11:44 2021

@author: richa
"""

#%% Imports

import numpy as np
from matplotlib import pyplot as plt

import pandas as pd
from tqdm import tqdm

from datetime import datetime

from sklearn.impute import SimpleImputer
import pickle
import os

#%% Paths


root_path = 'Path\\'

data_path = f'{root_path}xls'

# results_root = root_path + '\\Output\\results_excels'
# images_path = root_path + '\\Images'


save_infer_path = root_path + '\\Path\\Cell_Inference'
#plots_save_path = root_path + '\\Plots'
#data_save_path = root_path + '\\Data'


model_location =  'path\to\Models'


loaded_model = pickle.load(open(model_location + '\\model_1.sav', 'rb'))

#%% Cells to Analyze

cells_to_Analyze = 1

#%% Import Data

xls = os.listdir(data_path)


for xl in tqdm(xls):

        
    data = pd.read_csv(f'{data_path}\\{xl}')#, sep="\t", header = 0 , index_col = 0)
    data = data.loc[data['Tissue_Label'] == cells_to_Analyze] 

    data = data.drop(columns='Unnamed: 0')
    data.reset_index(drop=True, inplace=True)


    #%% Predict on the rest of the data


    
    
    X_predict = data.drop(columns = ['hemo-nucleus_X', 'hemo-nucleus_Y', 'hemo-nucleus_XM',
                                               'hemo-nucleus_YM', 'hemo-nucleus_Name', 'Tissue_Label',
                                               ]).values #,'DBSCAN_Clusters']
   
    fmax = np.finfo(np.float64).max
    pinf = float('+inf')
    ninf = float('-inf')
    
    X_predict[X_predict >= fmax] = 0
    X_predict[X_predict >= pinf] = 0
    X_predict[X_predict <= ninf] = 0
    
    imputer = SimpleImputer(missing_values=np.nan,
                            strategy='mean')#, fill_value = mean)
    imputer.fit(X_predict)
    
    X_predict = imputer.transform(X_predict)
    
    y_pred_probs = loaded_model.predict_proba(X_predict)
    
    max_prob = np.amax(y_pred_probs, axis = 1)
    one_hot = np.argmax(y_pred_probs, axis = 1)
    
    predicted_cell_label = []
    #Mouse
    Cell_Labels_Dictionary = ['Synovial_Fibroblasts', 'Synovial_Lining_Cells','Synovial_Vessel_Cells',
                              'Lymphocytes',  'Synovial_Fat_Cells', 'Growth_Plate_Chondrocytes', 'Bone_Cell']
    #Human
    Cell_Labels_Dictionary = ['Stromal-Connective', 'Fibroblasts', 'Macrophages-Histocytes',
                          'Synovial Lining Cell', 'Vascular Endothelial Cell',
                           'Plasma Cell', 'Lymphocytes']
    

    
    for idx, label in enumerate(one_hot):
        #if 0.85 > max_prob[idx] >= 0.6 :
            #predicted_cell_label.append(f'Lo_{Cell_Labels_Dictionary_2[label]}')
         #   continue
        if max_prob[idx] >= 0.75 :
            predicted_cell_label.append(f'{Cell_Labels_Dictionary[label]}')
            continue
        else:
            predicted_cell_label.append('No_Label')
    
    predicted_cell_label = pd.DataFrame(predicted_cell_label, columns = ['Predicted_Cell_Labels'])
    #print(predicted_cell_label.value_counts())
    pred_cell_counts = predicted_cell_label.value_counts()
    pred_cell_counts = pred_cell_counts.to_frame().reset_index()
    
    pred_cell_counts.rename(columns = {0:'Counts'},
          inplace = True)
    
    slide_names = data['hemo-nucleus_Name']

    short_names = [x.split('[')[0][:-1] for x in slide_names]
    
    short_names = pd.DataFrame(short_names, columns = ['Slide_Name'])
    short_names.reset_index(drop=True, inplace=True)
    pred_cell_counts_with_names = pd.concat([short_names, pred_cell_counts], axis = 1)
    
    
    if xl == xls[0]:
        Summarized_Pred_Cells = pred_cell_counts_with_names
        
    else:
        Summarized_Pred_Cells = pd.concat([Summarized_Pred_Cells, 
                                           pred_cell_counts_with_names], axis = 0)
    

    x_y_Preds = pd.concat([data['hemo-nucleus_X'], data['hemo-nucleus_Y'], predicted_cell_label], axis = 1)
    

    Group = x_y_Preds['Predicted_Cell_Labels'].values
    nuc_loc_values = x_y_Preds[['hemo-nucleus_X', 'hemo-nucleus_Y']].values

    # Set DPI to 600
    plt.figure(dpi=600)

    unique_labels, lanel_cnts = np.unique(Group, return_counts=True)
    
    for l in unique_labels:  
        plt.scatter(nuc_loc_values[Group == l, 0], nuc_loc_values[Group == l, 1], 
                    s=0.01, # marker size
                    alpha=1, # transparency
                    label = l, # label
                    marker='o'  # set marker type to a solid point
                    )
    plt.legend(bbox_to_anchor=(1,1), loc="upper left", markerscale = 10)
    plt.title(xl)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    
    ### Save Preds
    Preds = pd.concat([data['hemo-nucleus_Name'], predicted_cell_label], axis = 1)
    short_name = short_names['Slide_Name'].iloc[0]
    Preds.to_csv(f'{save_infer_path}\\PredictedCellType_{short_name}.csv')
    
    
Summarized_Pred_Cells.to_csv(f'{save_infer_path}\\All_Summarized_Pred_Cells.csv')   
    

