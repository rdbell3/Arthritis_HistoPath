# -*- coding: utf-8 -*-
"""
Created on Wed May  5 08:38:20 2021

@author: richa
"""

import os
import shutil
from tqdm import tqdm
import cv2
import numpy as np

def move_files_for_inference(root_dir):
    
    #%%
    #root_dir = "R:\\MK Anti-TNF 2nd Batch"
    path = root_dir + "\\tiles"
    
    if not os.path.isdir(f'{root_dir}\\Org_for_Inference'):
            os.mkdir(f'{root_dir}\\Org_for_Inference')
    
    folder_names = os.listdir(path)
    
    #%%
    
    
    for i in tqdm(folder_names):
        # if count>7:
        #     break
        
        names = os.listdir(path + "\\" + i)
        
    
        for j in names:
            
            img = cv2.imread(f'{path}\\{i}\\{j}')
            
            #img = cv2.imread(f'{path}\\{i}\\ES 4164 - L3.vsi - 40x [x=0,y=13760,w=2048,h=2048].jpg')
            #img2 = cv2.imread(f'{path}\\{i}\\ES 4164 - L3.vsi - 40x [x=0,y=6880,w=2048,h=2048].jpg')
            
            
            
            #avg = np.mean(img)
            try:
                sd = np.std(img)
                
                #avg2 = np.mean(img2)
                #sd2 = np.std(img2)
                
                if  sd > 2:    
                    shutil.move(f'{path}\\{i}\\{j}',
                                f'{root_dir}\\Org_for_Inference\\{j}')
            except :
                continue

