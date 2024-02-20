import cv2
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shutil
import roifile

from tqdm import tqdm

# Location of the original images to get names



root_path = 'D:\\AMP\\All_40x_Not_Annotated\\'

image_path = root_path + '\\Images_5'

output_path = root_path + 'Output_5'

output_path_new = 'D:\\AMP\\All_40x_Not_Annotated\\Output_5\\'

file_names = os.listdir(output_path + '//rois')
json_names = os.listdir(output_path_new + '//json')

a = file_names
#a = [x[:-4] for x in file_names]
b = [x[:-5] for x in json_names]
#jsons_to_do = [x for x in a + b if x not in a or x not in b]
jsons_to_do = [x for x in b if x not in a]

# for i in missing_jsons:
#     shutil.move(f'{image_path}//{i}.png', f'{output_path}\\missing_json\\{i}.png' )

# file_names = os.listdir(image_path)
# file_names = [k[:-4] for k in file_names]
json_names = [x[:-5] for x in json_names]


# missing_json_log = []

for file_name in tqdm(json_names):

    sh_fname = f'{output_path_new}//json//{file_name}.json'
    # try: 
    with open(sh_fname, 'r') as f:
        sh_json = json.load(f)
        
    counter = 0

    if not os.path.isdir(f'{output_path_new}//rois//{file_name}//'):
        os.mkdir(f'{output_path_new}//rois//{file_name}//')

    for key, value in sh_json['nuc'].items():

        if len(value['contour']) > 1:
    
            roi = roifile.ImagejRoi.frompoints(value['contour'])
            

                
            roi.tofile(f'{output_path_new}//rois//{file_name}//{file_name}_{counter}.roi')

            counter +=1

#     except:
#         missing_json_log.append(file_name)
    


#     #print(roi)


# #%%
# import csv

# with open(f'{root_path}\\missing_json.txt', 'w') as f:
#     f.write(str(missing_json_log)) 
    














