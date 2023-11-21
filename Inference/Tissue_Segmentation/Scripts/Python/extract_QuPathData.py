# -*- coding: utf-8 -*-
"""
Created on Wed May 31 21:15:07 2023

@author: bellr
"""

import pandas as pd
import os

path = 'C:\\Users\\bellr\\Documents\\Richard Data\\TNF-Tg_12mo\\QP_TNF-Tg_12Mo\\annotation_data\\'

files = os.listdir(path)

cnt = 0

for file in files:
    if cnt == 0:
        current_file = pd.read_csv(path + file, sep = '\t')
        cnt =+1
    else:
        temp = pd.read_csv(path + file, sep = '\t')
        current_file = pd.concat([current_file, temp], axis = 0)



cols_to_add = []

for name in current_file['Image']:
    
    short_name = name.split('-')
    
    cols_to_add.append([short_name[0], short_name[1].split(' ')[1], short_name[1].split(' ')[2],
                        short_name[1].split(' ')[3].split('.')[0]])
    

cols_to_add = pd.DataFrame(cols_to_add, columns=['Mouse_ID', 'Age', 'Sex', 'Genotype'])


current_file = current_file.reset_index(drop = True)


current_file = pd.concat([current_file, cols_to_add], axis = 1)

    
#current_file.to_csv(path + '12moMale_vsOther_Annotation_Data.csv')

#%% Reshape
current_file['Sex_Genotype_Age'] = current_file['Sex'] +'_'+ current_file['Genotype'] +'_'+  current_file['Age'] 
current_file['Sex_Genotype'] = current_file['Sex'] +'_'+ current_file['Genotype']


data_reshape_area = current_file.pivot(index= ['Mouse_ID', 'Sex', 'Age', 'Genotype', 'Sex_Genotype_Age'],
                                  columns=['Class'], values = 'Area µm^2').reset_index()

data_reshape_detections = current_file.pivot(index= ['Mouse_ID', 'Sex', 'Age', 'Genotype', 'Sex_Genotype_Age'],
                                  columns=['Class'], values = 'Num Detections').reset_index()


features_to_graph = data_reshape_area.drop(columns = ['Mouse_ID', 'Sex', 'Age',
                                                      'Genotype', 'Sex_Genotype_Age']).columns.tolist()




#%% Graph
import seaborn as sns
from matplotlib import pyplot as plt

for idx, feat in enumerate(features_to_graph):
    ax = sns.boxplot(x='Sex_Genotype_Age',
                     y=data_reshape_detections[feat],
                     #hue = 'Sex',
                     # order= ['PBS_L_3 hr', 'LPS_L_3 hr', 'PBS_R_3 hr', 'LPS_R_3 hr',
                     #         'PBS_L_Day 2', 'LPS_L_Day 2', 'PBS_R_Day 2', 'LPS_R_Day 2'],
                     
                     # order= ['B6', 'WT', 'Tg8'],
                     
                      # order= ['WT_L', 'Tg8_L',
                      #         'WT_R', 'Tg8_R'],
                     
                      # order= [
                      #         'M_WT', 'F_WT',
                      #         'M_Tg8', 'F_Tg8'],
                     
                     # order= ['M_PBS_L','F_PBS_L', 'M_PBS_R','F_PBS_R',
                     #         'M_LPS_L','F_LPS_L', 'M_LPS_R', 'F_LPS_R'],
                     
                     
                     order= ['F_WT_3', #'M_WT_3',
                             'F_WT_5.5', 'M_WT_5.5',
                             'M_WT_12',
                             
                             'F_TNF_3', 'F_TNF_5.5',
                             'M_TNF_5.5', 'M_TNF_12'],
                     
                     
                     data=data_reshape_detections)
    #ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=30)
    ax.set_title(features_to_graph[idx])
    sns.stripplot(x = "Sex_Genotype_Age",
              y = data_reshape_detections[feat],
              #hue = 'Sex',
              
              # order= ['B6', 'WT', 'Tg8'],             
              
              # order= ['M_WT', 'F_WT',
              #    'M_Tg8', 'F_Tg8'],
               # order= [ 'M_WT', 'F_WT',
               #                'M_Tg8', 'F_Tg8'],
              
              # order= ['WT_L', 'Tg8_L',
              #         'WT_R', 'Tg8_R'],
              
                    order= ['F_WT_3', #'M_WT_3',
                            'F_WT_5.5', 'M_WT_5.5',
                            'M_WT_12',
                            
                            'F_TNF_3', 'F_TNF_5.5',
                            'M_TNF_5.5', 'M_TNF_12'],
              
              color = 'black',
              data = data_reshape_detections)
    plt.ylabel('Cells Counts')
    #plt.yscale('log')
    
    plt.savefig(f'{path}//Graphs//{features_to_graph[idx]}_Counts.png')
    plt.show()
    #time.sleep(0.05)


for idx, feat in enumerate(features_to_graph):
    ax = sns.boxplot(x='Sex_Genotype_Age',
                     y=data_reshape_area[feat],
                     #hue = 'Sex',
                     # order= ['PBS_L_3 hr', 'LPS_L_3 hr', 'PBS_R_3 hr', 'LPS_R_3 hr',
                     #         'PBS_L_Day 2', 'LPS_L_Day 2', 'PBS_R_Day 2', 'LPS_R_Day 2'],
                     
                     # order= ['B6', 'WT', 'Tg8'],
                     
                      # order= ['WT_L', 'Tg8_L',
                      #         'WT_R', 'Tg8_R'],
                     
                       order= ['F_WT_3', #'M_WT_3',
                               'F_WT_5.5', 'M_WT_5.5',
                               'M_WT_12',
                               
                               'F_TNF_3', 'F_TNF_5.5',
                               'M_TNF_5.5', 'M_TNF_12'],
                     
                     # order= ['M_PBS_L','F_PBS_L', 'M_PBS_R','F_PBS_R',
                     #         'M_LPS_L','F_LPS_L', 'M_LPS_R', 'F_LPS_R'],
                     data=data_reshape_area)
    #ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=30)
    ax.set_title(features_to_graph[idx])
    sns.stripplot(x = "Sex_Genotype_Age",
              y = data_reshape_area[feat],
              #hue = 'Sex',
              
              # order= ['B6', 'WT', 'Tg8'],             
              
              # order= ['M_WT', 'F_WT',
              #    'M_Tg8', 'F_Tg8'],
               # order= [ 'M_WT', 'F_WT',
               #                'M_Tg8', 'F_Tg8'],
              
              # order= ['WT_L', 'Tg8_L',
              #         'WT_R', 'Tg8_R'],
              
                order= ['F_WT_3', #'M_WT_3',
                        'F_WT_5.5', 'M_WT_5.5',
                        'M_WT_12',
                        
                        'F_TNF_3', 'F_TNF_5.5',
                        'M_TNF_5.5', 'M_TNF_12'],
              
              color = 'black',
              data = data_reshape_area)
    plt.ylabel('Area')
    #plt.yscale('log')
    
    plt.savefig(f'{path}//Graphs//{features_to_graph[idx]}_Area.png')
    plt.show()
    #time.sleep(0.05)


