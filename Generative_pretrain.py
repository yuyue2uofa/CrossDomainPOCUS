import torch
import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from skimage.io import imread
from skimage.transform import resize

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data._utils.collate import default_collate
import torch.distributed as dist

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random

import cv2 as cv

import glob

import pickle
import re
from sklearn.metrics import confusion_matrix

from TransUNet.vit_seg_modeling import *
from TransUNet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg 
from TransUNet.vit_seg_modeling import VisionTransformer as ViT_seg
from SimMIM_model import SimMIM_TransUNet


from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

parser = argparse.ArgumentParser(description='MIM_TransUNet')
parser.add_argument('--epochs', default=1200, type=int, metavar='N',
                    help='number of total epochs to run') 
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--dataset_path', default = '/folder_path/', type=str) 
parser.add_argument('--train_csv', default='train.csv', type=str)
parser.add_argument('--save_dir', default = './results/', type=str) 
parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size') 
parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)') 
parser.add_argument('--lr', type=float, default=0.0005,
                        help='learning rate') 
parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
parser.add_argument('--mask_ratio', default=0.6, type=float,
                        help='Masking ratio (percentage of removed patches).')
parser.add_argument('--noise_ratio', default=0.2, type=float,
                        help='Masking ratio (percentage of removed patches from masked patches).') 
parser.add_argument('--loss_function', default="default", type=str,
                        help='loss function: default/rmse+mae') 
parser.add_argument('--output_channel', default=3, type=int,
                        help='output reconstruction channel') 
parser.add_argument('--transunet', default='R50-ViT-B_16', type=str,
                        help='transunet selection: R50-ViT-B_16/None; None: model will be UNet') 
parser.add_argument('--use_pretrained_transunet', default=None, type=bool) 
parser.add_argument('--save_frequency', default=400, type=int) 


args = parser.parse_args() 

if torch.cuda.is_available():
    device = "cuda"
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    device = "cpu"
    print('We will use CPU')



print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
if os.path.exists(args.save_dir):
    pass
else:
    os.mkdir(args.save_dir)


class MaskGenerator:
    def __init__(self, input_size=224, mask_patch_size=32, model_patch_size=16, 
                 mask_ratio=0.6, noise_ratio = 0):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        
        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        
        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        
        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
        
        self.noise_ratio = noise_ratio
        
        
    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        
        remaining_mask_ratio = 1 - self.noise_ratio
        noisy = mask_idx[:int(len(mask_idx)*(self.noise_ratio))]
        fully_masked_idx = mask_idx[int(len(mask_idx)*(self.noise_ratio)):]
    
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1 #1: with mask; 0: without mask; mask: all mask, masked, noise, blur; same as mask_fullsize
        fully_masked = np.zeros(self.token_count, dtype=int)
        fully_masked[fully_masked_idx] = 1 #1: with mask; 0: without mask; mask: all mask, masked, noise, blur; same as mask_fullsize
        #mask[masked] = 1
        
        
        noise = np.zeros(self.token_count, dtype=int)
        noise[noisy] = 1
        
        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1) #2D expand by 2(scale: 32//16)
        
        fully_masked = fully_masked.reshape((self.rand_size, self.rand_size))
        fully_masked = fully_masked.repeat(self.scale, axis=0).repeat(self.scale, axis=1) #2D expand by 2(scale: 32//16)
        
        noise = noise.reshape((self.rand_size, self.rand_size))
        noise = noise.repeat(self.scale, axis=0).repeat(self.scale, axis=1) #2D expand by 2(scale: 32//16)
        
        # new mask, size same as image size
        mask = np.repeat(mask, 16, axis=0).repeat(16, axis=1)
        fully_masked = np.repeat(fully_masked, 16, axis=0).repeat(16, axis=1)
        mask_fullsize = np.expand_dims(mask, axis=0) # 3 channel
        mask_fullsize = np.repeat(mask_fullsize, 3, axis=0) 
        
        noise = np.repeat(noise, 16, axis=0).repeat(16, axis=1)     
        
        return fully_masked, noise, mask_fullsize
        #fully_masked: masked region; noise: speckle noise region; mask_fullsize: mask+speckle noise region
        # 1: with mask/noise; 0: without mask/noise


def speckle_noise(image):
    row, col = image.shape
    gauss = np.random.randn(row,col)       
    noisy = image + image * gauss
    noisy = np.clip(noisy, a_min = 0, a_max=255)
    return noisy
  
class SegmentationDataSet(data.Dataset):
    def __init__(self, dataset_path: str, df_input:str, mask_ratio:float, noise_ratio:float, transform=None):
        self.dataset_path = dataset_path
        self.input_path = os.path.join(self.dataset_path, 'Images/')
        self.output_path = os.path.join(self.dataset_path, 'Masks/')
        self.df = pd.read_csv(df_input)
        self.images_list = list(self.df['filename'])
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.float32
        self.transform = transform
        

         
        self.mask_generator = MaskGenerator(
            input_size=224,
            mask_patch_size=32,
            model_patch_size=16,
            mask_ratio=mask_ratio, 
            noise_ratio=noise_ratio
        )
            
    def __len__(self):
        return len(self.images_list)
    def __getitem__(self, index: int):
        # Select the sample
        image_filename = self.images_list[index]
        # Load input and target
        image = cv.imread(os.path.join(self.input_path, image_filename),0)
        
        # padding
        width = max(image.shape) - image.shape[1]  # pad 0 on width
        height = max(image.shape) - image.shape[0]  # pad 0 on height
        image = np.pad(image, ((0, height), (0, width)), 'constant', constant_values=(0, 0))
       
        # Preprocessing
        image = cv.resize(image, (224,224))
        fully_masked, noise, mask_fullsize = self.mask_generator()
        mask_fullsize = torch.from_numpy(mask_fullsize)
        
        noise_image = speckle_noise(image)        
        image_noisy = (1-noise)*image + noise*noise_image

        # add: 3 channel
        image = np.repeat(image[None,...], 3, axis=0).transpose(1, 2, 0)
        # add transform
        if self.transform:
            image = self.transform(np.uint8(image))
            
        image_noisy = np.repeat(image_noisy[None,...], 3, axis=0).transpose(1, 2, 0)
        if self.transform:
            image_noisy = self.transform(np.uint8(image_noisy))

        fully_masked = torch.from_numpy(fully_masked)
        image_masked = (1-fully_masked)*image_noisy 
        

        return image, image_masked, mask_fullsize


# model
config_vit = CONFIGS_ViT_seg[args.transunet]
config_vit.n_classes = args.output_channel
model = SimMIM_TransUNet(config_vit, in_channels = 3, loss_function = args.loss_function)
model.to(device)
if args.use_pretrained_transunet:
    model.load_from(weights=np.load(CONFIGS_ViT_seg[args.transunet].pretrained_path))



PATH = args.save_dir + 'latest_model.pth' #change
if os.path.exists(PATH): #if path exist
    model.load_state_dict(torch.load(PATH))


# Training function

def train_model(dataloader, optimizer, model):
    train_loss = 0
    model.train()

    for images, images_masked, masks_full in dataloader: #image, image_masked, mask, mask_full
        images = images.cuda() #change cpu/cuda
        images_masked = images_masked.cuda()
        masks_full = masks_full.cuda()

        loss, rec = model(images_masked, masks_full, images)
        optimizer.zero_grad()
        loss.backward() # Calculate Gradients
        optimizer.step() # Update Weights

        train_loss =+ loss.item()


    return train_loss, rec, masks_full, model

    
train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([224, 224]),
        transforms.ToTensor()])

# Create training dataset
training_dataset = SegmentationDataSet(dataset_path = args.dataset_path, df_input = args.train_csv, mask_ratio = args.mask_ratio,
                                       noise_ratio = args.noise_ratio, transform = train_transform)
# Initialization

training_dataloader = data.DataLoader(dataset=training_dataset,
                                      batch_size=args.batch_size,
                                      shuffle = True)


print('------------------------------------------------------------------------')
print('Epochs:', args.epochs)
print('Batch size:', args.batch_size)
print('Learning rate:', args.lr)
print('')

with open(args.save_dir + 'training_result.txt', 'a') as f:
    f.write('mask ratio:'+str(args.mask_ratio)+ ', noise ratio:'+str(args.noise_ratio)+"\n")
with open(args.save_dir + 'training_result.txt', 'a') as f:
    f.write('loss function:'+str(args.loss_function)+"\n")
with open(args.save_dir + 'training_result.txt', 'a') as f:
    f.write('training data:'+str(args.train_csv)+"\n")
with open(args.save_dir + 'training_result.txt', 'a') as f:
    f.write('Batch size:'+str(args.batch_size)+'Learning rate:'+str(args.lr)+"\n")



optimizer = optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
torch.set_grad_enabled(True)

result_path = args.save_dir + "training_result.pickle"


if not os.path.exists(result_path): #if results do not exit
    results = {'train_loss': []}
else:
    results_ = open(result_path,'rb')
    results = pickle.load(results_)
    results_.close()
print(results)



for epoch in range(args.start_epoch, args.epochs):
    train_loss, pred, mask, model= train_model(training_dataloader, optimizer, model)  

    if epoch % 1 == 0:
        print("(epoch "+str(epoch)+")", 
              "\t"+"train loss: "+str(train_loss))
        torch.save(model.state_dict(),args.save_dir + 'latest_model.pth') #change
        with open(args.save_dir + 'training_result.txt', 'a') as f:
            f.write("(epoch "+str(epoch)+")"+ 
                    "\t"+"train loss: "+str(train_loss)+"\n")
    results['train_loss'].append(train_loss)

    pickle.dump(results, open(args.save_dir + "training_result.pickle", "wb")) #change
    if epoch % args.save_frequency == 0:
        torch.save(model.state_dict(),args.save_dir + 'latest_model_'+str(epoch)+'.pth') #change 