# -*- coding: utf-8 -*-
"""
Created on Thu May  6 10:31:17 2021

@author: richa
"""
import os
import numpy as np
import skimage.io as io
import tifffile
from tqdm import tqdm
import shutil


def organize_for_majority_vote(root_path):
        
    
    if not os.path.isdir(f'{root_path}\\Results_Temp'):
        os.mkdir(f'{root_path}\\Results_Temp')
    
    save_path = root_path + '\\Results_Temp'
    root_path = root_path + '\\Results_from_Inference'
    
    folds = os.listdir(root_path)
    for j in folds:
        images = os.listdir(root_path + '\\' + j)
        
        if not os.path.isdir(f'{save_path}\\{j}'):
                os.mkdir(f'{save_path}\\{j}')
        
        img_names = []
        
        for k in images:
            img_name = k.split('[')
            img_name = img_name[0][:-1]

            #img_name = k.split('_')[3]
            
            # img_name = k.split(' ')
            # img_name = img_name[0]#[:-4]
            
            if img_name not in img_names:
                img_names.append(img_name)
        
        
        for slides in tqdm(img_names):
               
            if not os.path.isdir(f'{save_path}\\{j}\\{slides}'):
                os.mkdir(f'{save_path}\\{j}\\{slides}')
             
            for i in images:
                
                img_name = i.split('[')
                img_name = img_name[0][:-1]
                
                
               #img_name = i.split('_')[3]

                
                # img_name = i.split(' ')
                # img_name = img_name[0]#[:-4]
                
                if slides == img_name:
                    shutil.move(f'{root_path}\\{j}\\{i}',
                         f'{save_path}\\{j}\\{slides}\\{i}')
                
                


