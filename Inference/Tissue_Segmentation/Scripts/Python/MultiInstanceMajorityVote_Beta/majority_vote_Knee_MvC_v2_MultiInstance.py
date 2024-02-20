# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 13:31:19 2021

@author: richard + matt
"""

#majority_vote_of Predictions of overlapping tiles

import os
import numpy as np
import skimage.io as io
import tifffile
from tqdm import tqdm
from scipy.stats import mode
import cv2
import matplotlib.pyplot as plt
from skimage.transform import rescale, resize

def findMinDiff(arr, n):
 
    # Sort array in non-decreasing order
    arr = sorted(arr)
 
    # Initialize difference as infinite
    diff = 10**20
 
    # Find the min diff by comparing adjacent
    # pairs in sorted array
    for i in range(n-1):
        if arr[i+1] - arr[i] < diff:
            diff = arr[i+1] - arr[i]
 
    # Return min diff
    return diff



def majority_vote(path, folders, n_classes, size, im_size):

    #print(path)
    #print(folders)
    #print(n_classes)
    #print(size)
    
    if n_classes == 2:
        Class_Names = ['Meniscus', 'Cartilage']
    
    
    if n_classes == 7:
        Class_Names = ['Synovium', 'Muscle-Tendon','Artifact','Growth Plate', 'Bone Marrow', 'Bone',
                         'Fat']
    
    if n_classes == 9:
        Class_Names = ['Synovium', 'Muscle-Tendon','Artifact','Growth Plate', 'Bone Marrow', 'Cortical Bone',
                       'Trabecular Bone', 'Meniscus-Cartilage',  'Fat']
        
    if n_classes == 10:
        Class_Names = ['Synovium', 'Muscle-Tendon','Artifact','Growth Plate', 'Bone Marrow', 'Cortical Bone',
                       'Trabecular Bone', 'Meniscus', 'Cartilage',  'Fat']
    
    if n_classes == 11:    
        Class_Names = ['Synovium', 'Muscle-Tendon','Artifact','Growth Plate', 'Bone Marrow', 'Cortical Bone',
                       'Trabecular Bone', 'Meniscus', 'Cartilage',  'Fat', 'Bone Marrow Fat']
    
    



    save_path = f'{path}\\Majority_Vote_Results\\'
    root_path = f'{path}\\Results_Temp\\'
    og_fold = 'OG'
    
    #print(og_fold)
    if not os.path.isdir(f'{save_path}'):
        os.mkdir(f'{save_path}')
    
    #folders = os.listdir(f'{root_path}\\{og_fold}')
    print(folders)
    
    for slides in tqdm(folders):
        
        images = os.listdir(f'{root_path}\\OG\{slides}')
        format_name = images[0].split('=')
        if "Thumbs.db" in images:
            images.remove("Thumbs.db")
    
        x = []
        y = []
        #print(len(images))
        for i in images: 
            name_split_1 = i.split('=')
            #print(name_split_1)
            x_ = name_split_1[2].split(',')
            #print(x_[0])
            x.append(int(x_[0]))
            y_ = name_split_1[3].split(',')
            y.append(int(y_[0]))
            #print(y_)
        
    
        x = list(set(x))
        y = list(set(y))
        max_x = max(x)
        max_y = max(y)
    
        min_x = min(x)
        min_y = min(y)
        #print(max_x, max_y)
        
        n = len(x)
        spacing = findMinDiff(x, n)
        print('start')
        
    
        pix = int((spacing*im_size)/size)
        
        for ii in range(min_x, max_x + 1 + spacing, spacing):
            for jj in range(min_y, max_y + 1 + spacing, spacing):
         
                range_interest_x = [ww for ww in x if ii - size/2 <= ww <=ii]
                range_interest_y = [ww for ww in y if jj - size/2 <= ww <=jj]
    
                preds= []
                    
                for aa in range_interest_x:
                    for bb in range_interest_y:
    
                                
                        format_name_main = format_name.copy()
    
                        format_name_main[2] = f'{aa},y'
                        format_name_main[3] = f'{bb},w'
    
                        current_image = ('=').join(format_name_main)
    
                        if range_interest_x == []:
                            aa = ii 
             
                        if range_interest_y == []:
                            bb = jj 
    
    
                        dis_ii = (ii-aa)/spacing
                        dis_jj = (jj-bb)/spacing
    
                        try:
                            prediction_current = io.imread(f'{root_path}//{og_fold}//{slides}//{current_image}')
                                  
                            prediction_current = prediction_current[:,int(pix*dis_jj):int(pix*dis_jj)+int(pix),int(pix*dis_ii):int(pix*dis_ii)+int(pix)]
        
                            if (prediction_current.shape[1]) < pix or (prediction_current.shape[2] < pix):
                                prediction_current = resize(prediction_current, (prediction_current.shape[0],pix,pix),anti_aliasing=True)
        
                            prediction_current[prediction_current < 75] = 0
        
                            prediction_background = np.sum(prediction_current,axis = 0)
        
                            prediction_background[prediction_background != 0 ] = 255
                            
                            prediction_background = 255 - prediction_background.reshape((1,pix,pix))
        
                            prediction_current = np.append(prediction_current, prediction_background, axis = 0)
        
        
                            pred_thresh = np.argmax(prediction_current,axis = 0)
                            
                            preds.append(pred_thresh)
                            
                        except:
                            pass
                            
    
                if len(preds) >= 1:
                    fin_pred = np.stack(preds,axis = 0)
    
                    if fin_pred.all() != 9:
                        
                        done = mode(fin_pred, axis = 0)
    
                        done = done[0].reshape((pix, pix))
                        done[done == 9] = 10 
                        done[done == 8] = 9
    
                        if 7 in done: 
                            
                            preds= []
                            for aa in range_interest_x:
                                for bb in range_interest_y:
    
                                    format_name_main = format_name.copy()
                
                                    format_name_main[2] = f'{aa},y'
                                    format_name_main[3] = f'{bb},w'
                
                                    current_image = ('=').join(format_name_main)
                
                                    if range_interest_x == []:
                                        aa = ii 
                         
                                    if range_interest_y == []:
                                        bb = jj 
                
                
                                    dis_ii = (ii-aa)/spacing
                                    dis_jj = (jj-bb)/spacing
                
    
                                    try:
                                        prediction_current_MvC = io.imread(f'{root_path}//FT//{slides}//{current_image}')
                     
                                        prediction_current_MvC = prediction_current_MvC[:,int(pix*dis_jj):int(pix*dis_jj)+int(pix),int(pix*dis_ii):int(pix*dis_ii)+int(pix)]
                    
                                        if (prediction_current_MvC.shape[1]) < pix or (prediction_current_MvC.shape[2] < pix):
                                            prediction_current_MvC = resize(prediction_current_MvC, (prediction_current_MvC.shape[0],pix,pix),anti_aliasing=True)
                    
                    
                                        prediction_current_MvC[prediction_current_MvC < 75] = 0
                    
                                        prediction_background = np.sum(prediction_current_MvC,axis = 0)
                    
                    
                                        prediction_background[prediction_background != 0 ] = 255
                                        
                    
                                        prediction_background = 255 - prediction_background.reshape((1,pix,pix))
                                        #print(prediction_current.min(), prediction_current.max(),prediction_background.min(), prediction_background.max())
                    
                                        prediction_current_MvC = np.append(prediction_current_MvC, prediction_background, axis = 0)
                    
                    
                                        pred_thresh = np.argmax(prediction_current_MvC,axis = 0)
                                        
                                        preds.append(pred_thresh)
                                    except:
                                        pass
                                    
                                    
                            if len(preds) >= 1:
                                fin_pred_MvC = np.stack(preds,axis = 0)
                
                                #if fin_pred_MvC.all() != 9:
                                    
                                done_MvC = mode(fin_pred_MvC, axis = 0)
            
                                done_MvC = done_MvC[0].reshape((pix, pix))
                                
                                done_MvC[done_MvC == 2] = 10
                                done_MvC[done_MvC == 1] = 8
                                done_MvC[done_MvC == 0] = 7
    
                            where_is_Seven = np.full((pix, pix), False)
                            where_is_Seven[done == 7] = True
                            np.copyto(done, done_MvC, where = where_is_Seven)
        
                        if not os.path.isdir(f'{save_path}\\{slides}'):
                            os.mkdir(f'{save_path}\\{slides}')
    
                        for iclass in range(n_classes):
        
                            idx = np.argwhere(done == iclass)
                            #print(idx)
        
                            done_copy = np.zeros((pix,pix)).astype(np.uint8)
        
                            done_copy[idx[:,0], idx[:,1]] = 255
        
                            new_format_name = format_name.copy()
        
                            test_name = new_format_name[0].split('x')[0][:-3]
                            
                            # new_new_format_name = []
                            
                            # new_format_name[0] = f'{test_name}x {Class_Names[iclass]} [x'
                            # new_format_name[1] = f'{ii},y'
                            # new_format_name[2] = f'{jj},w'
                            # new_format_name[3] = f'{spacing},h'
                            # new_format_name[4] = f'{spacing}].tif'
        
                            # current_image_fin = ('=').join(new_format_name)
                            done_copy = done_copy.astype(np.uint8)
        
                            # Need to modify name to make easy for qupath
                            new_new_format_name = f'{test_name} 40x {Class_Names[iclass]} [x={ii},y={jj},w={spacing},h={spacing}].tif'
    
        
                            if np.sum(done_copy) > 0:
                                cv2.imwrite(f'{save_path}\\{slides}\\{new_new_format_name}', done_copy)                




#majority_vote('Z:\\QP_Test\\', ['83.vsi - 40x'], 10, 2048, 512)

