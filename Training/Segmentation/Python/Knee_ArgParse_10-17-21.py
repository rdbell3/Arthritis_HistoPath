'''https://github.com/usuyama/pytorch-unet'''

#%% Imports

import copy
import time
from torchvision import transforms, datasets, models
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torch
from PIL import Image
import cv2
import random
from tqdm import tqdm 
from torch import nn
import transforms as T
#import utils
from imgaug import augmenters as iaa
#from imgaug.augmentables.segmaps import SegmentationMapOnImage
import torchvision
import skimage.io as io
from datetime import datetime
from collections import defaultdict
import torch.nn.functional as F
import csv

#import pandas as pd





#%% Set Hyperparamerters and Paths



root = '\\Path\\To\\Tiles\\'

root_path = root #+ 'Data//'
results_path = root + 'Results'

now = datetime.now()
now = now.strftime('%F-%H-%M-%S')

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('Batch_Size', nargs = '?', const = 1, default = 10, help='Indicate Batch Size', type = int)
parser.add_argument('Number_of_Epochs', nargs = '?', const = 1, default = 5, help='Indicate # of Epocs', type = int)
parser.add_argument('N_Classes', nargs = '?', const = 1, default = 11, help='Indicate How Many Classes To Predict', type = int)
parser.add_argument('Net_Params_Size', nargs = '?', const = 1, default = 0, help='Pick 0-7, 0 = least dense net 7=very dense', type = int)
parser.add_argument('Model_ID', nargs = '?', const = 1, default = 3, help='Model ID: 1=UNET, 2=UNET++, 3=PSPnet, 4=deeplabV3',
                    type = int)
# parser.add_argument('Weighted_Loss', nargs = '?', const = 1, default = True, help='Use a Weight Loss? True or False',
#                     type = bool)
parser.add_argument('Augment_Percent', nargs = '?', const = 1, default = 0.5,
                    help='How frequent does an augment occur? Ratio (float) between 0-1',
                    type = float)

parser.add_argument('Augment_Num_Low', nargs = '?', const = 1, default = 3,
                    help='Of the 11 augmentation styles, what is the lowest potential number to do each time a augment occurs?',
                    type = int)

parser.add_argument('Augment_Num_High', nargs = '?', const = 1, default = 7,
                    help='Of the 11 augmentation styles, what is the highest potential number to do each time a augment occurs?',
                    type = int)


args = parser.parse_args()

downsample = 4

batch_size = args.Batch_Size
num_epochs = args.Number_of_Epochs
n_classes = args.N_Classes

weighted_loss_switch = False

aug_percent =args.Augment_Percent
aug_low = args.Augment_Num_Low
aug_high = args.Augment_Num_High

thresh_percent = 0.5
thresh = 255 * thresh_percent

achitecture = f'efficientnet-b{args.Net_Params_Size}'

if weighted_loss_switch == True:
    loss_used = 'weighted_dice'
else:    
    loss_used = "dice" 

# 1 = unet, 2 = unet++, 3 = PSPnet, 4 = deeplabV3
model_id =  args.Model_ID

# Learning Parameters

LR_start = 0.025
LR_step_size = 2
LR_gamma = 0.5



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



ach = achitecture[-2:]

if model_id == 1:
    model_name = 'UNET'
if model_id == 2:
    model_name = 'UNET++'
if model_id == 3:
    model_name = 'PSPnet'
if model_id == 4:
    model_name = 'DeepLabV3'


folder_name = f'{now}_{model_name}_Results'

meta_data  = []
meta_data.append(['Batch Size', "Number of Epochs", 'Number of Classes', 'Threshold',
                  'Achitecture', 'Model', 'Loss Used', 'Learn Rate Start', 'Learn Rate Step Size', 'Learn Rate Gamma'])

meta_data.append([batch_size, num_epochs, n_classes, thresh_percent, achitecture,
                  model_name, loss_used, LR_start, LR_step_size, LR_gamma])


if not os.path.isdir(f'{results_path}//{folder_name}//'):
    os.mkdir(f'{results_path}//{folder_name}//')


with open(f'{results_path}//{folder_name}//{folder_name}_Meta_Data.csv', 'w', newline='') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerows(meta_data)  



log_file = []
log_file.append(['Epoch', 'Phase', 'Metric', 'Data']) 


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
    def __init__(self, root, imaug, transforms ):
        self.root = root
        #self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.path = os.listdir(root)
        self.transforms = transforms
        self.imaug = imaug

        #Unique to the way QuPath exports the tiles with the names
        self.path_img = [ x for x in self.path if ".tif" not in x ]
        self.path_img
        
        
    def __getitem__(self, idx):
        files = self.path_img[idx]

        img = cv2.imread(self.root+files)
        #print(img.mode)
        mask = io.imread(self.root+files[:-4]+'.tif') #Unique name from qupath
        #print(mask.min(), mask.max())
        mask = mask[1:,:,:]
        #print(mask.shape)
        

        if n_classes == 7:
            # Put Fat and BM Fat Together
            mask[9,:,:] = mask[9,:,:] + mask[10,:,:]
            mask = np.delete(mask, 10, axis = 0)

            # Put Cortical, Trabecular, Meniscus and Cartilage Together
            mask[5,:,:] = mask[5,:,:] + mask[6,:,:] + mask[7,:,:] + mask[8,:,:]
            mask = np.delete(mask, [6,7,8], axis = 0)


        if n_classes == 9:
            # Put Fat and BM Fat Together
            mask[9,:,:] = mask[9,:,:] + mask[10,:,:]
            mask = np.delete(mask, 10, axis = 0)
            
            # Put Meniscus and Cartilage Together
            mask[7,:,:] = mask[7,:,:] + mask[8,:,:] 
            mask = np.delete(mask, 8, axis = 0)


        if n_classes == 10:
            # Put Fat and BM Fat Together
            mask[9,:,:] = mask[9,:,:] + mask[10,:,:]
            mask = np.delete(mask, 10, axis = 0)
            
            # # Put Meniscus and Cartilage Together
            # mask[7,:,:] = mask[7,:,:] + mask[8,:,:] 
            # mask = np.delete(mask, 8, axis = 0)



        
        if np.amax(mask) > 256:
            print(np.amax(mask))
        
        #print(np.amax(mask))
        
        # Need to have the layes in the last dimesion for the data augmentations
        mask = np.swapaxes(mask, 0, 1)
        #print(mask.shape)
        mask = np.swapaxes(mask, 1, 2)
        #print(mask.shape)
        
        mask[mask>1] = 1
        mask = mask.astype('bool')

        #mask = SegmentationMapsOnImage(mask, shape=mask.shape)

        if self.imaug is not None:
            img, mask = self.imaug(image = img, segmentation_maps = [mask])
            mask = mask[0]
        
        cv2.imwrite(f'{results_path}//{folder_name}//{files}_aug.jpg', img)
        
        # Put the layes back in the first dimesion
        #print(masks.shape)

        masks = np.swapaxes(mask,1,2)
        #print(masks.shape)
        masks = np.swapaxes(masks,0,1)
        #print(masks.shape)


        masks = torch.as_tensor(masks, dtype=torch.uint8)
        img = Image.fromarray(img, mode = 'RGB')

        # Not sure what this does
        if self.transforms is not None:
            img, masks = self.transforms(img, masks)

        return img.float(), masks.float(), files
        

    def __len__(self):
        return int(len(self.path)/2) # Cut len in half bc we have the masks in the same path


#######################################################################################


def cosh_dice_loss(pred, target, smooth = 1):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    fin_loss = torch.log((torch.exp(loss) + torch.exp(-loss))/2.0)

    return fin_loss.mean()

#######################################################################################
#############################################################
#
def dice_loss(pred, target, smooth = 1):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    
    if weighted_loss_switch == True:
    
        if n_classes == 7:
            # Class_Names = ['Synovium', 'Muscle-Tendon','Artifact','Growth Plate', 'Bone Marrow', 'Bone',
            #              'Fat']
            w = torch.tensor([0.0866, 0.012,  0.52, 0.17, 0.018, 0.023, 0.153]).to(device)
            
            
        if n_classes == 9:
            # Class_Names = ['Synovium', 'Muscle-Tendon','Artifact','Growth Plate', 'Bone Marrow', 'Cortical Bone',
            #                'Trabecular Bone', 'Meniscus', 'Cartilage',  'Fat']
            
            w = torch.tensor([0.069, 0.01,  0.42, 0.142, 0.014, 0.037, 0.052, 0.127, 0.12]).to(device)    
        
            
        if n_classes == 10:
            # Class_Names = ['Synovium', 'Muscle-Tendon','Artifact','Growth Plate', 'Bone Marrow', 'Cortical Bone',
            #                'Trabecular Bone', 'Meniscus', 'Cartilage',  'Fat']
            
            w = torch.tensor([0.047, 0.007,  0.286, 0.096, 0.009, 0.025, 0.035, 0.283, 0.124, 0.0835]).to(device)
            
        if n_classes == 11:    
            # Class_Names = ['Synovium', 'Muscle-Tendon','Artifact','Growth Plate', 'Bone Marrow', 'Cortical Bone',
            #                'Trabecular Bone', 'Meniscus', 'Cartilage',  'Fat', 'Bone Marrow Fat']
            
            w = torch.tensor([0.026, 0.003,  0.157, 0.053, 0.005, 0.014, 0.019, 0.156, 0.068, 0.051, 0.457]).to(device)

        # batch by class tensor
        weight_loss = torch.sum(torch.mul(loss, w), dim=1)

        # return weighted loss
        
        loss_by_class = torch.mean(loss, 0)
        
        return loss_by_class, loss.mean(), weight_loss.mean()
    
    else:
        
        loss_by_class = torch.mean(loss, 0)
        
        return loss_by_class, loss.mean(), False

#%%
##########################################################################################
def calc_loss(pred, target, metrics, bce_weight=0.5):
    bceloss = nn.BCEWithLogitsLoss()#pos_weight=torch.tensor(3))
    
    bce =  bceloss(pred, target)
    #btl =  BinaryTverskyLossV2()
    #btl = btl(pred,target)
    #pred = torch.tanh(pred)
    pred = torch.sigmoid(pred)
    #shl = shape_loss(pred,target)
    #dice_weighted, 
    loss_by_class, dice, dice_weighted  = dice_loss(pred, target)
    cosh_dice = cosh_dice_loss(pred,target)
    #loss1 = (bce * bce_weight) + (dice * ((1 - bce_weight)*(1/3))) +  (btl * ((1 - bce_weight)*(2/3)))
    #loss =(bce * bce_weight) + (dice * (1 - bce_weight))+ cel    
    loss = (bce * bce_weight) + (dice * (1 - bce_weight))
    #loss = (.5*cel)+ (0.5*shape)    


    loss_weighted = (bce * bce_weight) + (dice_weighted * (1 - bce_weight))

    #metrics['shl'] += shl.data.cpu().numpy() * target.size(0)
    metrics['cosh_dice'] += cosh_dice.data.cpu().numpy() * target.size(0) 

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    #metrics['btl'] += btl.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    metrics['loss_weighted'] += loss_weighted.data.cpu().numpy() * target.size(0)
    metrics['loss_by_class'] += loss_by_class.data.cpu().numpy() * target.size(0)
    
    if weighted_loss_switch == True:
        return loss_weighted
    else:  
        return loss
#############################################################################################
def print_metrics(metrics, epoch_samples, phase, Class_Names):
    outputs = []
    for k in metrics.keys():
        
        if k == "loss_by_class":
            for i in range(len(metrics[k])):
                outputs.append("{}: {:.3f}".format(Class_Names[i], metrics[k][i] / epoch_samples))
        else:
            outputs.append("{}: {:.4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))

def build_log_file(log_file, metrics, epoch, phase, epoch_samples, Class_Names):
        
    for k in metrics.keys():
        if k == "loss_by_class":
            for i in range(len(metrics[k])):
                log_file.append([epoch, phase, f'{Class_Names[i]}', metrics[k][i] / epoch_samples])    
        else:
            log_file.append([epoch, phase, k, metrics[k] / epoch_samples])
        
        
    return log_file
    
    

def train_model(model, optimizer, scheduler, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        # now = datetime.now()
        # print(now)
        
        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            for data in tqdm(dataloaders[phase]):
                inputs, labels, files  = data[0].to(device), data[1].to(device),  data[2]
  
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    
                    loss = calc_loss(outputs, labels, metrics)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
            

                # statistics
                epoch_samples += inputs.size(0)
                
            if phase =='train':    
                scheduler.step()
                
            print_metrics(metrics, epoch_samples, phase, Class_Names)
            epoch_loss = metrics['loss'] / epoch_samples
            
            build_log_file(log_file, metrics, epoch, phase, epoch_samples, Class_Names)


            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                
                
                if not os.path.isdir(f'{results_path}//{folder_name}//'):
                    os.mkdir(f'{results_path}//{folder_name}//')
                
                
                torch.save(best_model_wts, f'{results_path}//{folder_name}//model_weights_{ach}_{model_name}_trained.pth')
                
            with open(f'{results_path}//{folder_name}//{folder_name}_log_file.csv', 'w', newline='') as file:
                writer = csv.writer(file, delimiter=',')
                writer.writerows(log_file)

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        
        now = datetime.now()
        print(now)

    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
###########################################################################################

# transform the RGB values to get closer to whate the pretrained weights from 
# imagenet expect

def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.ToTensor())
        transforms.append(T.Normalization([.858, .670, .823], [.098, .164, .050]))
    else:
        transforms.append(T.ToTensor())
        transforms.append(T.Normalization([.858, .670, .823], [.098, .164, .050]))  
    return T.Compose(transforms)



# apply agument sometimes = 50%; a random order of these 11, and between 4 and 8 each time

augmentation = iaa.Sometimes(aug_percent, iaa.Sequential([iaa.SomeOf((aug_low, aug_high), [
    iaa.Fliplr(0.5, random_state = 123),
    iaa.CoarseDropout(0.1, size_percent=0.2, random_state = 123),
    #iaa.Flipud(0.5, random_state = 123),
    iaa.OneOf([iaa.Affine(rotate=90, random_state = 123),
                iaa.Affine(rotate=180, random_state = 123),
                iaa.Affine(rotate=270, random_state = 123)]), #rotations
    #iaa.LinearContrast((0.5, 2.0),  random_state = 123),      #contrast adjustments      
    #iaa.Multiply((0.8, 1.5), random_state = 123), #linear transform
    iaa.AdditiveGaussianNoise(scale=(0,0.2*255), per_channel=False, random_state = 123),
    iaa.GaussianBlur(sigma=(0.0, 1.5), random_state = 123),
    #iaa.GaussianBlur(sigma=(0,3.0))
    iaa.WithHueAndSaturation([
        iaa.WithChannels(0, iaa.Add((-30, 10))),
        iaa.WithChannels(1, [
            iaa.Multiply((0.5, 1.5)),
            iaa.LinearContrast((0.5, 2))
            ])
        ]),
    iaa.AddToBrightness((-30, 30)),
    iaa.ChangeColorTemperature((3000, 8000)),
    iaa.MultiplyHueAndSaturation(mul_hue=(0.5, 1.5)),
    iaa.MultiplyHueAndSaturation(mul_saturation=(0.5, 1.5)),
    iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30))
    
])], random_order=True))





######################################################################################

#%%


dataset = KneeDataset(f'{root_path}//Test//', augmentation, get_transform(train=True))
dataset_evalu = KneeDataset(f'{root_path}//Test//', None, get_transform(train=False))

image_datasets = {
    'train': dataset, 'val': dataset_evalu
}


dataloaders = {
    'train': DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0),
    'val': DataLoader(dataset_evalu, batch_size=batch_size, shuffle=True, num_workers=0)
}



##############################################################################
###############################################################################

device = torch.device( 'cuda' if torch.cuda.is_available() else "cpu")


import segmentation_models_pytorch as smp


if model_id == 1:
    model = smp.Unet(achitecture, encoder_weights='imagenet', classes = n_classes, in_channels=3, activation = None)

if model_id == 2:
    model = smp.UnetPlusPlus(achitecture, encoder_weights='imagenet', classes = n_classes, in_channels=3, activation = None)

if model_id == 3:
    model = smp.PSPNet(achitecture, encoder_weights='imagenet', classes = n_classes, in_channels=3, activation = None)

if model_id == 4:
    model = smp.DeepLabV3Plus(achitecture, encoder_weights='imagenet', classes= n_classes, in_channels=3, activation = None)

model = nn.DataParallel(model)
model = model.to(device)


##########################################################################
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=LR_start,
                            momentum=0.9, weight_decay=0.0001)

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=LR_step_size,
                                                gamma=LR_gamma)

### Can change lr, step_size, gamma


class NormalizeInverse(torchvision.transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())

model = train_model(model, optimizer, lr_scheduler, num_epochs=num_epochs)
        
print('That is it!')


#%%
def dice_coeff_calc(pred, target, smooth = 1):
    #pred = pred.contiguous()
    #target = target.contiguous()    

    intersection = (pred * target).sum().sum()

    union = pred.sum().sum() + target.sum().sum()
    
    dice_coeff = (2.* intersection + 1) / (union + 1)
    dice_coeff = dice_coeff #.cpu().numpy()
    
    return dice_coeff




#%%
##########################################################

# can do for loop with train_loader
# import imageio
# import tifffile 
test_loader = DataLoader(dataset_evalu, batch_size = batch_size, shuffle=False, num_workers=0)

tile_dice_score_log = []
tile_dice_score_log.append(['Name','Location', 'Class', 'Dice Score', 'Area Prediction','Area Target']) 


 
with torch.no_grad():
          
    for data_fin in test_loader:
        inputsv, labelsv, name = data_fin[0].to(device),data_fin[1].to(device),  data_fin[2]
      
        outputs = model(inputsv)   
        outputs = F.sigmoid(outputs)
        #print(outputs.shape)   
    
        for j in range(outputs.shape[0]):
            
            
            current_name = name[j]
            current_name = current_name[:-4]
                    
            # if len(os.listdir(f'{results_path}//{folder_name}//')) > 500:
            #       break

            if outputs[j].shape[0] > 0:
                # if not os.path.isdir(f'{results_path}//{folder_name}//{current_name}//'):
                #     os.mkdir(f'{results_path}//{folder_name}//{current_name}//')
                    
                # pred = outputs[j,:,:,:]
                # target = labelsv[j,:,:,:]    
                
                # dice_coeff_2 = dice_coeff_calc_2(pred, target)
                
    
                temp = outputs[j,:,:,:]*255             
                temp = temp.cpu().numpy().astype(np.uint8)
                
                temp_max = np.amax(temp, 0)
                temp_idx = np.argmax(temp, 0)
                
                temp_max[temp_max > thresh] = 255
                temp_max[temp_max <= thresh] = 0
                
                mask_thresh = np.zeros([len(outputs[1,1,:,1]), len(outputs[1,1,1,:])])
                mask_thresh_disp = np.zeros([len(outputs[1,1,:,1]), len(outputs[1,1,1,:])])
                
                
                for i in range(len(temp_max[0,:])):
                    for f in range(len(temp_max[:,0])):
                        if temp_max[f, i] == 255:
                            mask_thresh[f,i] = temp_idx[f,i] + 1


                # orig = labelsv[j,:,:,:]*255             
                # orig = orig.cpu().numpy().astype(np.uint8)
                
                # orig_max = np.amax(orig, 0)
                # orig_idx = np.argmax(orig, 0)
                
                # orig_max[orig_max > thresh] = 255
                # orig_max[orig_max <= thresh] = 0
                
                # orig_thresh = np.zeros([len(outputs[1,1,:,1]), len(outputs[1,1,1,:])])
                # orig_thresh_disp = np.zeros([len(outputs[1,1,:,1]), len(outputs[1,1,1,:])])
                
                
                # for i in range(len(orig_max[0,:])):
                #     for f in range(len(orig_max[:,0])):
                #         if orig_max[f, i] == 255:
                #             orig_thresh[f,i] = orig_idx[f,i] + 1                
                
                
                
                

                
                
                
                # orig = Image.fromarray(orig_thresh)
                # orig.save(f'{results_path}//{folder_name}//{current_name}_GT.tif')
                
                # mask_thresh_1 = Image.fromarray(mask_thresh)
                # mask_thresh_1.save(f'{results_path}//{folder_name}//{current_name}_mask_thresh.tif')
             
                
                for k in range(outputs.shape[1]):
                    

                    mask_result = outputs[j,k,:,:]*255
                    
                    mask_thresh_2 = np.zeros([len(outputs[1,1,:,1]), len(outputs[1,1,1,:])])
                    
                    mask_thresh_2[mask_thresh-1 == k] = 1

                    original_name_split = name[j].split('[')
                    short_name = original_name_split[0]
                    location_name = original_name_split[1].split(',')
                    H_W = mask_thresh_2[0].size * downsample
                    x = location_name[0]
                    y = location_name[1]
                    x = x[2:]
                    y = y[2:]
                    
                    gt = labelsv[j,k,:,:]*255
                    
                    pred = outputs[j,k,:,:]
                    
                    m = nn.Threshold(thresh_percent, 1)
                    
                    pred = m(pred)
                    
                    # pred[pred > thresh_percent] = 1
                    # pred[pred <= thresh_percent] = 0
                    
                    # pred = torch.as_tensor(pred)
                    
                    target = labelsv[j,k,:,:].cpu().numpy()
                    
                    dice_coeff = dice_coeff_calc(mask_thresh_2, target)
                    
                    area_pred = np.sum(mask_thresh_2)
                    area_target = np.sum(target)
                    
                    input_dice_log = [short_name, location_name, Class_Names[k], dice_coeff, area_pred, area_target]
                    tile_dice_score_log.append(input_dice_log)
                    
                    
                    
                    # a = mask_result.cpu().numpy().astype(np.uint8)
                    # b = gt.cpu().numpy().astype(np.uint8)
                    # c = mask_thresh_2.astype(np.uint8)
                    

            
                    # example = Image.fromarray(a)#.convert('RGB')
                    # #example.show()
                    # example_gt = Image.fromarray(b)#.convert('RGB')
                    # #example_gt.show()
                    # #example_thresh = Image.fromarray(c)
                    # example_thresh_pred = Image.fromarray(c)
                    
                    # example_thresh_pred.save(f'{root_path}//{folder_name}//{current_name}//{short_name}{Class_Names[k]}[x={x},y={y},w={H_W},h={H_W}]-labelled.tif')
                    
                    # example.save(f'{root_path}//{folder_name}//{current_name}//{current_name}_result_{Class_Names[k]}.jpg')
                    # example_gt.save(f'{root_path}//{folder_name}//{current_name}//{current_name}_gt_{Class_Names[k]}.jpg')
                    # #example_thresh.save(f'{root_path}\\Eval_Test\\{str(name[j])}\\{name[j]}_mask_thresh_{Class_Names[k]}.jpg')                   
                    
                    # un_norm =  NormalizeInverse([.858, .670, .823], [.098, .164, .050])
                    # image_match = un_norm(torch.squeeze(inputsv[j,:,:,:],0))
                    # #image_match = torch.squeeze(inputsv[j,:,:,:],0)
                    # #print(image_match.shape)
                    
                    # image_match = (image_match.data.cpu().numpy()*255).astype(np.uint8)
                    # image_match = np.swapaxes(image_match,0,1)
                    # image_match = np.swapaxes(image_match,1,2)
                    
                
                    
                    # if k == 0:
                    #     cv2.imwrite(f'{results_path}//{folder_name}//{current_name}_original_{k}.jpg', image_match)


#%% Construct Slide level Metrics
# slide_dice_score_log = []
# slide_dice_score_log.append(['Name','Location', 'Class', 'Dice Score']) 

   







#%%            
######################### Write Log File #################

#log_file = pd.Dataframe(log_file)



    

with open(f'{results_path}//{folder_name}//{folder_name}_Tile_Dice_Score.csv', 'w', newline='') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerows(tile_dice_score_log)                   
