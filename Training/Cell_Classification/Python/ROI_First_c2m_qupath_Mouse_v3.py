import cv2
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import roifile
import read_roi

from tqdm import tqdm

root = 'D:\\Bell, Brendal et al\\Single_Cell\\'
inf_path = f'{root}MVF_scHisto_Inf_12-19-23\\Cell_Inference\Slide_Specific_Inf\\'
save_path = f'{root}MVF_scHisto_Inf_12-19-23\\Cell_Prediction_Masks\\' #save path

#file_names = os.listdir(f'{root}\\Images\\') #image folder
#file_names = os.listdir(f'{root}\\Images\\')



def getList(dict):
    
    return [*dict]

infs = os.listdir(inf_path)

cnt = 1

for inf in infs:

    a = len(infs)
    print(inf.split('_')[-2])
    print(f'{cnt} of {a}')
    cnt += 1
    
    clust = pd.read_csv(f'{inf_path}{inf}', index_col = 0, header = 0) #label csv
    
    #clust = pd.read_csv('D:\\a.csv', index_col = 0, header = 0) #label csv
    
    #a = os.listdir('D:\\')
    
    #save_path = f'{root}Output\\Cell_Prediction_Masks_2-28-23_60%\\' #save path
    
    
    if not os.path.isdir(f'{save_path}'):
        os.mkdir(f'{save_path}')
    
    
    
    rois = list(clust['hemo-nucleus_Name'])
    imgs = [k.split(']')[0] for k in rois]
    
    imgs = [k + ']' for k in imgs]
    
    unique_imgs = set(imgs)
    
    
    
    
    
    
    for file_name in tqdm(unique_imgs):
    
        sh_fname_roi = f'{root}\MvF_Slides_For_SingleCell\\Output_Final\\rois\\{file_name}\\'  #path of ROI folder
        roi_files = os.listdir(sh_fname_roi)
    
        roi_files = sorted(roi_files)
    
        #img_fname = f'{root}\\Images\\{file_name}.png'
    
    
        
        #img = cv2.imread(img_fname,cv2.IMREAD_COLOR)
    
        class_name = list(clust.Predicted_Cell_Labels.unique())
        
    
        
        clust = clust[['hemo-nucleus_Name', 'Predicted_Cell_Labels']]
    
        #img_mask = [np.zeros((img.shape[0], img.shape[1])) for xyz in range(len(class_name))]
    
        img_mask = [np.zeros((1024, 1024)) for xyz in range(len(class_name))]
    
        counter = 0
    
        slide_name = file_name.split('[')[0][:-1]
    
        if not os.path.isdir(save_path + slide_name):
            os.mkdir(save_path + slide_name)
    
        for count, j in enumerate(roi_files):
    
            try:
                roi2 = read_roi.read_roi_file(sh_fname_roi+j)
    
    
                roi2['coordinates'] = np.column_stack((roi2[getList(roi2)[0]]['x'],roi2[getList(roi2)[0]]['y'] ))
                
                idx_file = clust.loc[clust['hemo-nucleus_Name'] == j]
                assert idx_file.shape[0] ==1
    
                cv2.fillPoly(img_mask[int(class_name.index(idx_file.Predicted_Cell_Labels.values))], pts =  np.array([roi2['coordinates']]), color = (255,255,255))
    
                #cv2.imshow('test',img_mask[int(class_name.index(idx_file.Cell_Label.values))])
                #cv2.waitKey(0)
            except:
                #print(j)
                pass
    
    
        # for abc in class_name:
        #     if not os.path.isdir(save_path + slide_name+ f'\\{abc}\\'):
        #         os.mkdir(save_path + slide_name+ f'\\{abc}\\')      
          
        for counting, p in enumerate(img_mask):
            slice = class_name[counting]
    
            name_beg = (' ').join(file_name.split(' ')[:3])
            name_end = (' ').join(file_name.split(' ')[3:])
            p = p.astype(np.uint8)
            if sum(sum(p)) > 0:
                cv2.imwrite(f'{save_path}\\{slide_name}\\{name_beg} {slice} {name_end}.tif', p)
    
    
    
