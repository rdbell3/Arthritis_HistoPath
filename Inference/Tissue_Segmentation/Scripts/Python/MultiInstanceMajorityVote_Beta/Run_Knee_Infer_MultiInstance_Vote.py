# -*- coding: utf-8 -*-
"""
Created on Tue May 30 15:17:23 2023

@author: bellr
"""


import argparse
import os
from move_files_for_inference_v2 import move_files_for_inference
from organize_inference_for_Majority_vote_MvC_v2 import organize_for_majority_vote 
from Knee_Load_Weights_And_Infer_Hiearchical_Model_v2 import infer_knee
from majority_vote_Knee_MvC_v2_MultiInstance import majority_vote
import multiprocessing


def split_folders(folder_list, num_splits):
    avg_split = len(folder_list) // num_splits
    remaining = len(folder_list) % num_splits

    split_folders = []
    start = 0
    for i in range(num_splits):
        end = start + avg_split + (1 if i < remaining else 0)
        split_folders.append(folder_list[start:end])
        start = end

    return split_folders



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Run_Knee_Infer',
                    description='This program infers knee tissue segmentation based on Bell et al models'
                    )
    parser.add_argument('QuPath_Folder_Path', nargs = '?', default = "Z:\\QP_Test\\",
                        help='Please set the path (String) to the QuPath Project Folder', type = str)
    parser.add_argument('Model_Path', nargs = '?', default = "C:\\Users\\richa\\Documents\\Scripts\\Run_Knee_Infer_Package\\Models\\",
                        help='Please set the path (String) to the Models', type = str)
    parser.add_argument('Delete_Temp_Files', nargs = '?',  default = True,
                        help='Indicate (True or False) if you want to delete the Temp Files Created (Default = True) which include the folders : tiles, Results_Temp, Results_from_Inference, Org_for_Inference')
    parser.add_argument('Infer_Batch_Size', nargs = '?', default = 40,
                        help='Indicate How Many (Integer) Images to Process at a Time - Dependent on GPU VRAM - Default = 10', type = int)
    parser.add_argument('Tile_Size', nargs = '?', const = 1, default = 512,
                        help='Indicate the tile size (Integer) exported by QuPath - Default = 512', type = int)
    parser.add_argument('Downsample', nargs = '?', const = 1, default = 4,
                        help='Indicate the Downsample rate (Integer) used in the export by QuPath - Default = 4', type = int)
    parser.add_argument('Instances', nargs = '?', type=int, default=2,
                        help='Number of instances to run in parallel')
    
    
    args = parser.parse_args()
    
    path = args.QuPath_Folder_Path
    model_path = args.Model_Path
    delete_temp_files = args.Delete_Temp_Files
    infer_batch_size = args.Infer_Batch_Size
    tile_size = args.Tile_Size
    downsample = args.Downsample
    
    orig_size = downsample * tile_size
    n_classes = 10
    instances = args.Instances
    
    print('Running 1/4: Move Files For Inference')
    #move_files_for_inference(path)
    print('')
    print('Running 2/4: Inference')
    #infer_knee(path, infer_batch_size, model_path)
    print('')
    
    print('Running 3/4: Organize for Majority Vote')
    #organize_for_majority_vote(path)
    
    
    folders = os.listdir(f'{path}\\Results_Temp\\OG')
    
    if "Thumbs.db" in folders:
        folders.remove("Thumbs.db")
    
    print('')
    print('Running 4/4: Majority Vote')

    
    
    # if delete_temp_files == True:
        
    #     import shutil
        
    #     shutil.rmtree(path + 'tiles')
    #     shutil.rmtree(path + 'Results_Temp')
    #     shutil.rmtree(path + 'Results_from_Inference')
    #     shutil.rmtree(path + 'Org_for_Inference')
    


    folder_splits = split_folders(folders, instances)

    processes = []
    for folders in folder_splits:
        process = multiprocessing.Process(target=majority_vote, args=(path, folders, n_classes, orig_size, tile_size))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()











