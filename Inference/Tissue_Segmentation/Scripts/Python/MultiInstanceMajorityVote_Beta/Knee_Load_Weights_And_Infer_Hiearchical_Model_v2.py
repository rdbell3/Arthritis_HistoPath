'''https://github.com/usuyama/pytorch-unet'''

#%% Imports

from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torch
from PIL import Image
import cv2
import random
from tqdm import tqdm 
from torch import nn
import transforms_inference as T
import tifffile
import segmentation_models_pytorch as smp


def infer_knee(path, batch_size, model_path):
    
    #%%## Set Hyperparamerters and Paths
    
    
    saved_weight_path_1 = model_path + 'Final_Model_9Class_b5_UNET++.pth'
    saved_weight_path_2 = model_path + 'Final_Model_MvC_2Class_b5_UNET++.pth'
    
    
    data_path = path
    root_path = data_path
    
    
    tiles = 'Org_For_Inference'
    folder_name = 'Results_from_Inference'
    
    if not os.path.isdir(f'{root_path}\\Results_from_Inference'):
        os.mkdir(f'{root_path}\\Results_from_Inference')
    

    root_path_T = data_path + tiles

    batch_size = batch_size
    n_classes = 9
    achitecture = 'efficientnet-b5'

    
    
    ## Make sure the model is the same as the loaded model
    # 1 = unet, 2 = unet++, 3 = PSPnet, 4 = deeplavV3
    model_id =  2

    if model_id == 2:
        model_name = 'UNET++'

    
    #%%
    
    
    def seed_everything(seed=1234):                                                  
        random.seed(seed)                                                            
        torch.manual_seed(seed)                                                      
        torch.cuda.manual_seed_all(seed)                                             
        np.random.seed(seed)                                                         
        os.environ['PYTHONHASHSEED'] = str(seed)                                     
        torch.backends.cudnn.deterministic = True                                    
        torch.backends.cudnn.benchmark = False 
    
    seed_everything(1234) 
    
    # Contruct DataLoader
    class KneeDataset(object):
        def __init__(self, root, imaug, transforms):
            self.root = root
            #self.transforms = transforms
            # load all image files, sorting them to
            # ensure that they are aligned
            self.path = os.listdir(root)
            self.masks = []
            self.origs = []
            self.subdirs = []
            self.filenames = []
            self.transforms = transforms
            self.imaug = imaug
            # no_duplicates = os.listdir('R:\\MK Anti-TNF Study\\Results_From_Inference\\')
            # no_duplicates = [x[:-4] for x in no_duplicates]
            # no_duplicates = [x + '.jpg' for x in no_duplicates]
            
            if "Thumbs.db" in self.path:
                self.path.remove("Thumbs.db")
            
            #Unique to the way QuPath exports the tiles with the names
            self.path_img = [ x for x in self.path if ".tif" not in x ]
            # self.path_img = [ x for x in self.path_img if x not in no_duplicates] 
            
    
        def __getitem__(self, idx):
    
            files = self.path_img[idx]
    
            img = cv2.imread(self.root+files)
    
    
            img = Image.fromarray(img, mode = 'RGB')
            
            
            if self.transforms is not None:
                img  = self.transforms(img)
    
    
            return img.float(), files
            
    
        def __len__(self):
            return int(len(self.path_img)) # Cut len in half bc we have the masks in the same path
    
    #%%
    
    def get_transform(train):
        transforms = []
        if train:
            transforms.append(T.ToTensor())
            transforms.append(T.Normalization([.858, .670, .823], [.098, .164, .050]))
        else:
            transforms.append(T.ToTensor())
            transforms.append(T.Normalization([.858, .670, .823], [.098, .164, .050]))  
        return T.Compose(transforms)
    
    #%%
    
    dataset_inference = KneeDataset(f'{root_path_T}\\', None, get_transform(train=False))
    
    # dataloaders = {
    
    #     'inference': DataLoader(dataset_inference, batch_size=batch_size, shuffle=True, num_workers=0)
    # }
    
    inference_loader = DataLoader(dataset_inference, batch_size=batch_size, shuffle=True, num_workers=0)
    
    #%%
    
    device = torch.device( 'cuda:0' if torch.cuda.is_available() else "cpu")
    #device = torch.device('cpu')
    
    
    
    
    if model_id == 1:
        model = smp.Unet(achitecture, classes = n_classes, in_channels=3, activation = None)
    
    if model_id == 2:
        model_1 = smp.UnetPlusPlus(achitecture, classes = n_classes, in_channels=3, activation = None)
    
    if model_id == 3:
        #model = smp.PSPNet(achitecture, encoder_weights='imagenet', classes = n_classes, in_channels=3, activation = None)
    
        model = smp.PSPNet(achitecture, classes = n_classes, in_channels=3, activation = None)
    
    if model_id == 4:
        model = smp.DeepLabV3Plus(achitecture, encoder_weights='imagenet', classes= n_classes, in_channels=3, activation = None)
    
    # check to make sure of model compatibility
    #saved_weight_path = 'C:\\Users\\richa\\Documents\\Machine_Learning\\Histomorphometry\\Tiles\\DWS_4_512_0.66-overlap\\2021-03-22-19-34-05_b0_PSPnet_dice_Eval_Test\\model_weights_b0_PSPnet_trained.pth'
    
    model_1 = nn.DataParallel(model_1)
    model_1.load_state_dict(torch.load(saved_weight_path_1))
    model_1.eval()
    
    
    model_1 = model_1.to(device)
    
    
    
    
    model_2 = smp.UnetPlusPlus(achitecture, classes = 2, in_channels=3, activation = None)
    
    model_2 = nn.DataParallel(model_2)
    model_2.load_state_dict(torch.load(saved_weight_path_2))
    model_2.eval()
    
    
    model_2 = model_2.to(device)
    
    
    
    
            
    #%%
    from os.path import exists
    
    with torch.no_grad():
              
        for data_fin in tqdm(inference_loader):
            inputs, name = data_fin[0].to(device), data_fin[1] #,  data_fin[2]
          
            outputs_1 = model_1(inputs)   
            #outputs = F.sigmoid(outputs)
            outputs_1 = torch.sigmoid(outputs_1)
            #print(outputs.shape)   
            outputs_2 = model_2(inputs)
            outputs_2 = torch.sigmoid(outputs_2)
            #concat_output = torch.cat((outputs_1[:, :7, :, :], outputs_2[:, :, :, :], outputs_1[:, 8:, :, :]), 1) 
            
            
            
            for j in range(outputs_1.shape[0]):
    
                current_name = name[j]
                current_name = current_name[:-4]        
                # if len(os.listdir(f'{root_path}\\{ach}_{model_name}_Eval_Test\\')) > 5:
                #      break
               
                # if outputs_1[j, 7, :, :].sum() > 1:
                #     outputs_2 = model_2(inputs)
                #     outputs_2 = torch.sigmoid(outputs_2)
                #     concat_output = torch.cat((outputs_1[j, :7, :, :], outputs_2[j, :, :, :], outputs_1[j, 8:, :, :]), 0) 
                #     a = outputs_1[j, :7, :, :]
                #     b = outputs_2[j, :, :, :]
                #     c =  outputs_1[j, 8:, :, :]
                    
                preds_1 = outputs_1[j,:,:,:]*255             
                preds_1 = preds_1.cpu().numpy().astype(np.uint8)
                
                if preds_1.sum() > 0:
                    if not os.path.isdir(f'{root_path}\\{folder_name}\\OG\\'):
                        os.mkdir(f'{root_path}\\{folder_name}\\OG\\')
        
                    tifffile.imwrite(f'{root_path}\\{folder_name}\\OG\\{current_name}.tif', preds_1, photometric='minisblack')
                
                
                preds_2 = outputs_2[j,:,:,:]*255             
                preds_2 = preds_2.cpu().numpy().astype(np.uint8)
                
                if preds_2.sum() > 0:
                    if not os.path.isdir(f'{root_path}\\{folder_name}\\FT\\'):
                        os.mkdir(f'{root_path}\\{folder_name}\\FT\\')
        
                    tifffile.imwrite(f'{root_path}\\{folder_name}\\FT\\{current_name}.tif', preds_2, photometric='minisblack')
                    
                    
#infer_knee('Z:\\QP_Test\\', 40, 'C:\\Users\\richa\\Documents\\Scripts\\Run_Knee_Infer_Package\\Models\\')
