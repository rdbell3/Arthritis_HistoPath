# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 17:42:54 2021

@author: richa
"""
#%% Imports
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

from sklearn.preprocessing import StandardScaler

from skimage.feature import hog, local_binary_pattern
from skimage import data, exposure

import pandas as pd
from tqdm import tqdm



from sklearn import cluster, mixture, preprocessing

#import snf
#from snf import compute
#import seaborn as sns

#import scipy.cluster.hierarchy as hc
import scipy.spatial as sp

from scipy import stats

#from scipy import spatial


#import networkx as nx

from sklearn.neighbors import NearestNeighbors


from scipy.spatial import KDTree
from scipy.spatial import ConvexHull, convex_hull_plot_2d
#from netneurotools import cluster
from sklearn.linear_model import LinearRegression
from sklearn.cluster import DBSCAN

from sklearn.utils import resample
from datetime import datetime

from numpy.random import RandomState

from skimage import io

import warnings

warnings.filterwarnings('ignore')


#%% Paths

root_path = 'Path\to\Output\Folder\'

results_root = root_path + '\\Output\\results_excels'
images_path = root_path + '\\Images\\'
save_path = root_path + '\\Output\\Cluster_Labels'
plots_save_path = root_path + '\\Output\\Plots'
data_save_path = root_path + '\\Output\\Data'


tissue_mask_path = root_path + '\\Tissue_Masks'
cell_mask_path = root_path + '\\Cell_Masks'

mask_downsample = 1


#%% Hyperparameters and Empty Lists

img_names = os.listdir(images_path)
img_names = [k[:-4] for k in img_names]

downsample = 'none' #10000

scan_data = []
df_roi_list = []

##Distance  150 and 300 for 40x
##Distance 75 and 150 for 20x

nuclei_distance_measures = [150, 300]#, 600]

now = datetime.now()
now = now.strftime('%F')


#%% Functions

def density_degree(dist_matix):
    
    for row in range(len(dist_matix[:, 0])):
        for col in range(len(dist_matix[0, :])):
        
            if dist_matix[row, col] <= 75:
                dist_matix[row, col] = 100
                continue
            
            if 150 >= dist_matix[row, col] > 75:
                dist_matix[row, col] = 10
                continue
            
            if 300 >= dist_matix[row, col] > 150:
                dist_matix[row, col] = 1
                continue
            
            if dist_matix[row, col] > 300:
                dist_matix[row, col] = 0
                continue
        
    return dist_matix

def unique_image_names(file_names_list, parser = '_', parse_index = 3):
    
    unique_names = []
    
    for files in file_names_list:
        name_split = files.split(parser)
        
        if name_split[parse_index] not in unique_names:
            unique_names.append(name_split[parse_index])
    
    return unique_names
    


#%% Computational Loop
import time

unique_scan_names = unique_image_names(img_names, parser = '.', parse_index = 0)

t0 = time.time()

count_1 = 0

for num, unique_scan in enumerate(unique_scan_names): 
    print('Slide Name:' + unique_scan)
    print(str(num + 1) +' of ' + str(len(unique_scan_names)))
    
    
    for img in tqdm(img_names):
        img_name_split = img.split('.')
                
        if img_name_split[0] == unique_scan:
            # count_1 += 1
            # # print(count_1)

            # if count_1 == 50:
            #     break
            
            excel_names = ['hemo-nucleus', 'eosin-nucleus', 'hemo-cyto', 'eosin-cyto']# 'residual-nucleus', 'residual-cyto'
            
            # only works for specific QuPath Names
            offsets = img_name_split[1].split(',')
            x_offset = offsets[0].split('=')
            x_offset = int(x_offset[1])
            y_offset = offsets[1].split('=')
            y_offset = int(y_offset[1])
            
            w = offsets[2].split('=')
            w = int(w[1])
            
            img_size = offsets[2].split('=')
            img_size = int(img_size[1])
            
            # excel import loop
            for xl in excel_names:
                if xl == excel_names[0]:
                    img_data = pd.read_csv(f'{results_root}\\{img}_{xl}.txt', sep="\t", header = 0, index_col = False)
                    img_data = img_data.drop(columns = [' ','FeretX','FeretY', 'IntDen','RawIntDen',
                                                        '%Area', 'MinThr', 'MaxThr'])

                    x = img_data['X'].values
                    y = img_data['Y'].values
                    
                    # Add offset to x and y coodinates
                    
                    # not sure which way is faster
                    #x = np.add(x, x_offset).tolist()
                    #y = np.add(y, y_offset).tolist()
                    
                    # or this
                    x = [f + x_offset for f in x]
                    y = [f + y_offset for f in y]
                    
                    
                    img_data.loc[:, 'X'] = x
                    img_data.loc[:, 'Y'] = y                    
                    
                    #img_data = img_data[['X', 'Y', 'Mean']]
                    
                    img_data = img_data.add_prefix(f'{xl}_')
                    
                else:
                    temp = pd.read_csv(f'{results_root}\\{img}_{xl}.txt', sep="\t", header = 0)# , index_col = 0)
                    # temp = temp.drop(columns = ['Area', 'Mean', 'StdDev',
                    #                             'Mode', 'Min', 'Max',
                    #                             'X', 'Y', 'XM', 'YM',
                    #                             'Perim.', 'Circ.', 'Feret',                                               
                    #                             'IntDen', 'Median', 'Skew',
                    #                             'Kurt', '%Area', 'RawIntDen',
                    #                             'FeretX', 'FeretY',
                    #                             'MinFeret', 'AR',
                    #                             'FeretAngle', 'Round', 'Solidity',
                    #                              'MinThr', 'MaxThr', 'Name'])
                    temp = temp.drop(columns = [' ', 'Area',
                                                'X', 'Y', 'XM', 'YM',
                                                'Perim.', 'Circ.', 'Feret',                                               
                                                'IntDen', 
                                                 '%Area', 'RawIntDen',
                                                'FeretX', 'FeretY',
                                                'MinFeret', 'AR',
                                                 'Round', 'Solidity',
                                                  'MinThr', 'MaxThr', 'Name'])                   
                    #temp = temp[['Mean']]
                    temp = temp.add_prefix(f'{xl}_')
                    img_data = pd.concat([img_data, temp], axis = 1)
                    del(temp)
                
                
                    
            ##### remove cells at the edge of images
            img_nuc_loc = img_data[['hemo-nucleus_X', 'hemo-nucleus_Y']].values
            
            file = f'{img}.tif'       
            
            # Add Tissue Label
            if os.path.exists(f'{tissue_mask_path}/{img_name_split[0]}.vsi - 40x') == True:
                
                Tissue_Mask_Folder_Name = f'{img_name_split[0]}.vsi - 40x'
                
                #current_TissueMask_tiles = os.listdir(f'{tissue_mask_path}/{Tissue_Mask_Folder_Name}')
                #current_TissueMask_tiles = [ x for x in current_TissueMask_tiles if ".png" not in x ]
    
                nans = 0
                maskNum = np.empty((0,1))
                im = io.imread(f'{tissue_mask_path}/{Tissue_Mask_Folder_Name}/{file}')    
                for index, cell_loc in enumerate(img_nuc_loc):
                    xcor = cell_loc[0]
                    ycor = cell_loc[1]
                    
                    #for tile in current_TissueMask_tiles:
                        # name_split_1 = tile.split('=')
                        # #print(name_split_1)
                        # x_ = name_split_1[2].split(',')
                        # #print(x_[0])
                        # x_ = int(x_[0])
                        # y_ = name_split_1[3].split(',')
                        # y_ = int(y_[0])
                        
                        # #im = io.imread(f'{tissue_mask_path}/{Tissue_Mask_Folder_Name}/{tile}')
    
                    
                    
                        # if x_ <= xcor < x_ + 1024 and y_ <= ycor < y_ + 1024:
                    #print(f'{Tissue_Mask_Folder_Name}/{file}')


                
                    found = False
                    for mask in range(len(im)):
                          if im[mask][round(ycor)-y_offset][round(xcor)-x_offset] == 255:
                              maskNum = np.append(maskNum, [[mask]], axis=0)
                              #print(index)
                              found = True
                              break
                          
                    if not found:
                        maskNum = np.append(maskNum, [[0]], axis=0)
                        #print(index)
                        nans += 1
                            
                    #break
                                
            Tissue_Label = pd.DataFrame(maskNum, columns = ['Tissue_Label'])#.shift(periods=1, axis = 0)
            
            #Tissue_Label = Tissue_Label.iloc[1: , :]
            img_data = pd.concat([img_data, Tissue_Label], axis = 1)


            
            # Add Cell Label
            if os.path.exists(f'{cell_mask_path}//{file}') == True:
            
                im = io.imread(f'{cell_mask_path}//{file}')
                im = im[0:-1, :, :]
                nans = 0
                maskNum = np.empty((0,1))
    
                for index, cell_loc in enumerate(img_nuc_loc):
                    xcor = cell_loc[0]
                    ycor = cell_loc[1]
                    
                    found = False
                    for mask in range(len(im)):
                         if im[mask][round(ycor)-y_offset][round(xcor)-x_offset] == 255:
                             maskNum = np.append(maskNum, [[mask]], axis=0)
                             found = True
                             break
                    if not found:
                        maskNum = np.append(maskNum, [[0]], axis=0)
                        nans += 1
                        
                Cell_Label = pd.DataFrame(maskNum, columns = ['Cell_Label'])#.shift(periods=1, axis = 0)
               # Cell_Label.reset_index(drop=True, inplace=True)
                
                #Cell_Label = Cell_Label.iloc[1: , :]
                img_data = pd.concat([img_data, Cell_Label], axis = 1)

            cells_to_remove = []
            
            pix_to_remove = 32
            
            for idx, cell_loc in enumerate(img_nuc_loc):
            
                if x_offset < cell_loc[0] <  x_offset + pix_to_remove:
                    cells_to_remove.append(idx)
                    continue
                
                if (img_size - pix_to_remove) + x_offset < cell_loc[0] <  img_size + x_offset:
                    cells_to_remove.append(idx)
                    continue
                
                if y_offset < cell_loc[1] <  y_offset + pix_to_remove:
                    cells_to_remove.append(idx)
                    continue
                    
                if (img_size - pix_to_remove ) + y_offset < cell_loc[1] < img_size + y_offset:
                    cells_to_remove.append(idx)
                    continue
                    
            img_data = img_data.drop(index = cells_to_remove)
            
            #img_nuc_loc = img_data[['hemo-nucleus_X', 'hemo-nucleus_Y']].values
            


            
            if 'slide_data' not in locals():
                slide_data = img_data
                
            else:
                slide_data = pd.concat([slide_data, img_data], axis = 0)
    
    
    
    
    # 

    
    
    # potential to downsample here if computation is to much
    #if slide_data.shape[0] > downsample:
        #slide_data = resample(slide_data, n_samples = downsample)       
    
    #slide_data = resample(slide_data, n_samples = 500)
    
# Start Computation loop
    ## Select Tissue Type
    #slide_data = slide_data[slide_data.Tissue_Label == 1]
    



    

    # nuc_loc_values = nuc_loc.values   

    # slide_data['Cell_Label'].fillna(0,inplace=True)
    # slide_data['Tissue_Label'].fillna(0,inplace=True)

    # Group = slide_data['Tissue_Label'].values


    # unique_labels, lanel_cnts = np.unique(Group, return_counts=True)
    
    # for l in unique_labels:  
    #     plt.scatter(nuc_loc_values[Group == l, 0], nuc_loc_values[Group == l, 1], 
    #                 s=0.05, # marker size
    #                 alpha=1, # transparency
    #                 label = l, # label
    #                 )
    # plt.legend(bbox_to_anchor=(1,1), loc="upper left", markerscale = 2)
    # plt.xlabel('UMAP_0')
    # plt.ylabel('UMAP_1')
    # plt.show()


    # plt.scatter(nuc_loc_values[:,0], nuc_loc_values[:,1],
    #     s = 0.05)

    # plt.show()
    
    # slide_data = slide_data[slide_data.Tissue_Label != 5]
    # slide_data = slide_data[slide_data.Tissue_Label != 0]


    #slide_data_dwn = resample(slide_data, n_samples = 1000)
    nuc_loc = slide_data[['hemo-nucleus_X', 'hemo-nucleus_Y']]
    nuc_loc_values = nuc_loc.values
    
    
    #tree = KDTree(nuc_loc)
    
    # for i in range(len(slide_data)):
        
    #     d, i = tree.query([img_nuc_loc[i][0], img_nuc_loc[i][1]], k = 50, distance_upper_bound = nuclei_dist)
    Group = slide_data['Tissue_Label'].values
    #nuc_loc_values = slide_data[['hemo-nucleus_X', 'hemo-nucleus_Y']].values

    unique_labels, lanel_cnts = np.unique(Group, return_counts=True)
    
    for l in unique_labels:  
        plt.scatter(nuc_loc_values[Group == l, 0], nuc_loc_values[Group == l, 1], 
                    s=0.05, # marker size
                    alpha=1, # transparency
                    label = l, # label
                    )
    plt.legend(bbox_to_anchor=(1,1), loc="upper left", markerscale = 2)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    
    # a = nuc_loc[0:200].values
    
    # t = KDTree(a, leafsize = 10)
    
    slide_date_no_loc = slide_data.drop(columns = ['hemo-nucleus_X', 'hemo-nucleus_Y', 'hemo-nucleus_XM',
                                                    'hemo-nucleus_YM', 'hemo-nucleus_Name','Tissue_Label', 'Cell_Label']) 
    col_names = slide_date_no_loc.columns
    
    tree = KDTree(nuc_loc)
    
    nuc_loc = slide_data[['hemo-nucleus_X', 'hemo-nucleus_Y']]
    nuc_loc_values = nuc_loc.values
    
    # find_close_cells_parent = sp.distance_matrix(nuc_loc.values, nuc_loc.values)
    
    # dist_matrix = find_close_cells_parent.copy()
    
    #find_close_cells = find_close_cells_parent.copy()
    
    #For distance hyper paramerter    
    for idx, nuclei_dist in enumerate(nuclei_distance_measures):
        
        print('Current Distance Measurment:')
        print(nuclei_dist)

        for i in tqdm(range(len(slide_data))):
            
            d, ix = tree.query([nuc_loc_values[i][0], nuc_loc_values[i][1]], k = 100, distance_upper_bound = nuclei_dist)
            

            current_close_cells = np.delete(ix, np.where(d > nuclei_dist))
            current_close_cells_distance = np.delete(d, np.where(d > nuclei_dist))
            
            avg_dist = np.mean(current_close_cells_distance[1:])
            degree = np.sum(np.reciprocal(current_close_cells_distance[1:]))
            number_close_cells = len(current_close_cells)
            
            close_cells = []
            for h in range(len(current_close_cells)):
               
                close_cells.append(nuc_loc.iloc[current_close_cells[h]].values)
        
            close_cells_coords = np.array(close_cells)
            
            
            
            
            r_squared_current_close_cells = np.zeros([1,1])
            
            if len(close_cells_coords)>5:
                x, y = close_cells_coords[:, 0], close_cells_coords[:, 1]
                
                # Reshaping
                x, y = x.reshape(-1,1), y.reshape(-1, 1)
                
                # Linear Regression Object and Fitting linear model to the data
                lin_regression = LinearRegression().fit(x,y)
                
                r_squared_current_close_cells = np.array(lin_regression.score(x,y))
            
            for cells in current_close_cells:
                
                if cells == current_close_cells[0]:
                    close_cell_data = slide_date_no_loc.iloc[[cells]].values
                else:    
                    close_cell_data = np.append(close_cell_data, slide_date_no_loc.iloc[[cells]].values, axis = 0)
          
            

            # Calculate interesting features
            if close_cell_data.shape[0] < 2:
                avg = close_cell_data.reshape((1,-1))
                #sd = np.zeros([1, close_cell_data.shape[1]])
                #Zscore = np.zeros([1, close_cell_data.shape[1]])
                
            else:
                avg = close_cell_data.mean(axis = 0)
                
            sd = close_cell_data.std(axis = 0)
            
            #Z-Score
            Zscore = np.divide((slide_date_no_loc.iloc[[i]].values - avg), sd)
            Zscore = np.nan_to_num(Zscore, nan = 0)
            
            #Ratio Change
            ratio_change = np.divide((slide_date_no_loc.iloc[[i]].values - avg), avg)
            
            
            # # From Scipy Packag https://docs.scipy.org/doc/scipy/reference/stats.html?highlight=stats#module-scipy.stats
            skew = stats.skew(close_cell_data, axis = 0)
            kurtosis = stats.kurtosis(close_cell_data, axis = 0)
            skew = stats.skew(close_cell_data, axis = 0)
            moderesult = stats.mode(close_cell_data, axis = 0)
            mode = moderesult[0]
            iqr = stats.iqr(close_cell_data, axis = 0)
            kurtosis = stats.kurtosis(close_cell_data, axis = 0)
            sem = stats.sem(close_cell_data, axis = 0)
            
            
            entropy = stats.entropy(abs(close_cell_data), axis = 0)
                   
            
            
            #Computing Distance of String of Closests Cells
            visited = dict([(i,i)])
            
            dist, ix = tree.query([nuc_loc_values[i][0], nuc_loc_values[i][1]], k=[2], distance_upper_bound = nuclei_dist)
            points = np.empty((0,2))
            points = np.append(points, [[nuc_loc_values[i][0], nuc_loc_values[i][1]]], axis = 0)
            total_distance = 0
            
            if(dist != float("inf")):
                total_distance = dist[0]
                row = ix[0]
                points = np.append(points, [[nuc_loc_values[row][0], nuc_loc_values[row][1]]], axis=0)
                visited[row] = row
                
                for j in range(0,30):
                    #print(dist, ix)
                    count = 0
                    while(ix[0] in visited):
                        dist, ix = tree.query([nuc_loc_values[row][0], nuc_loc_values[row][1]], k=[count + 2], distance_upper_bound = nuclei_dist)
                        count += 1
                        if(dist == float("inf")):
                            break
                    if(dist == float("inf")):
                            break
                    total_distance += dist[0]                                   
                    row = ix[0]
                    points = np.append(points, [[nuc_loc_values[row][0], nuc_loc_values[row][1]]], axis=0)
                    visited[row] = row


            #Area of Convex Hull
            if(points.shape[0] > 2):        
                polygon = ConvexHull(points)
                area = polygon.area
            else:
                area = 0
            
            
            r_squared_string_cells = np.zeros([1,1])
            
            if len(points)>6:
                x, y = points[:, 0], points[:, 1]
                
                # Reshaping
                x, y = x.reshape(-1,1), y.reshape(-1, 1)
                
                # Linear Regression Object and Fitting linear model to the data
                lin_regression = LinearRegression().fit(x,y)
                
                r_squared_string_cells = np.array(lin_regression.score(x,y))



            total_distance = np.array(total_distance)
            area = np.array(area)
            
            
            # Add all new calculations to this list
            data_to_join = [avg, sd, skew, kurtosis, Zscore, ratio_change,  iqr, sem, entropy]
            data_to_join_names = ['avg', 'sd', 'skew', 'kurtosis', 'Zscore', 'ratio_change','iqr', 'sem', 'entropy'] 
            
            for idx_2, joining_data in enumerate(data_to_join):
                
                if idx_2 == 0:
                    new_data_temp1 = pd.DataFrame(joining_data.reshape((1,-1)), columns = col_names)
                    #name_prefix = data_to_join_names[idx_2]
                    new_data_temp1 = new_data_temp1.add_prefix(f'{nuclei_dist}_{data_to_join_names[idx_2]}_')
                else:
                    temp = pd.DataFrame(joining_data.reshape((1,-1)), columns = col_names)
                    temp = temp.add_prefix(f'{nuclei_dist}_{data_to_join_names[idx_2]}_')
                    new_data_temp1 = pd.concat([new_data_temp1, temp], axis = 1)
            

            new_data_temp1 = pd.concat([new_data_temp1, pd.DataFrame(r_squared_current_close_cells.reshape(1,1),
                                                                      columns = [f'{nuclei_dist}_r_squared']),
                                        
                                        pd.DataFrame(r_squared_string_cells.reshape(1,1),
                                                                      columns = [f'{nuclei_dist}_string_r_squared']),
                                        
                                        pd.DataFrame(total_distance.reshape(1,1),
                                                     columns = [f'{nuclei_dist}_string_total_distance']),
                                        
                                        pd.DataFrame(area.reshape(1,1),
                                                     columns = [f'{nuclei_dist}_string_total_distance']),
                                        
                                        pd.DataFrame(avg_dist.reshape(1,1),
                                                     columns = [f'{nuclei_dist}_avg_neigbor_distance']),
                                        
                                        pd.DataFrame(degree.reshape(1,1),
                                                     columns = [f'{nuclei_dist}_degree']),
                                        
                                        pd.DataFrame([[number_close_cells]], #.reshape(1,1),
                                                     columns = [f'{nuclei_dist}_number_close_cells'])
                                        
                                        ], axis = 1)
            
    
            if i == 0:
                new_data = new_data_temp1
            else: 
                new_data = pd.concat([new_data, new_data_temp1], axis = 0)

        if idx == 0:
            new_data_final = new_data
        else: 
            new_data_final = pd.concat([new_data_final, new_data], axis = 1)            


    new_data_final.reset_index(drop=True, inplace=True)
    slide_data.reset_index(drop=True, inplace=True)
    
    # new_data_final = pd.concat([new_data_final, nuc_degree], axis = 1) #, spectral_labels, gmm_labels
    slide_data = pd.concat([slide_data, new_data_final], axis = 1)
    slide_data['Tissue_Label'].fillna('0',inplace=True)
    
    save_name = '_'.join(str(e) for e in nuclei_distance_measures)
    
    slide_data.to_csv(f'{data_save_path}//{now}_{save_name}_{unique_scan}_data.csv')
    del(slide_data)
    
    


