#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torchvision
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cv2 as cv

import glob

import pickle
import re
from sklearn.metrics import confusion_matrix
import segmentation_models_pytorch as smp

from TransUNet.vit_seg_modeling import *
from TransUNet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg 
from TransUNet.vit_seg_modeling import VisionTransformer as ViT_seg

parser = argparse.ArgumentParser(description='TransUNet/UNet')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run') 
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--dataset_path', default = '/folder_path/', type=str) 
parser.add_argument('--test_csv', default='test.csv', type=str)
parser.add_argument('--save_dir_finetune_MIM', default = './results_finetune_generative/', type=str) 
parser.add_argument('--save_dir_finetune_contrastive', default = './results_finetune_contrastive/', type=str)
parser.add_argument('--save_dir_finetune_image', default = './ensemble/1/', type=str) 
parser.add_argument('--vit_name', default = 'R50-ViT-B_16', type=str , metavar='MODEL',
                    help='Name of model to train: R50-ViT-B_16/None; if None, model will be UNet') 
parser.add_argument('--model_name_MIM', default = 'latest_model.pth', type=str, help = 'Pretrained model name') 
parser.add_argument('--model_name_contrastive', default = 'latest_model.pth', type=str, help = 'Pretrained model name') 
parser.add_argument('--batch_size', type=int, default=1, help='batch size (default: 16)') 
parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)') 
parser.add_argument('--lr', type=float, default=0.002, 
                        help='learning rate') 
parser.add_argument('--finetune_option', default = 'encoder_decoder', type=str, help = 'encoder/encoder_decoder/all') 
parser.add_argument('--output_channel', default = 1, type=int, help = 'output prediction: 1/4') 
parser.add_argument('--save_figure', action='store_true') 
parser.add_argument('--SE_confidence', action='store_true')
parser.add_argument('--top2_confidence', action='store_true')
parser.add_argument('--device_gpu', default='cuda:0', type=str, help='specify the gpu you are using')

args = parser.parse_args() #change


if torch.cuda.is_available():
    device = args.device_gpu
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    device = "cpu"
    print('We will use CPU')

    
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
os.makedirs(args.save_dir_finetune_image, exist_ok=True)


from segmentation_models_pytorch.losses.dice import DiceLoss
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True, threshold = 0.5):
        super(DiceBCELoss, self).__init__()
        self.thredshold = threshold
        self.diceloss = DiceLoss('binary', from_logits = False)
    def forward(self, inputs, targets, smooth=1): 
        dice_loss = self.diceloss(inputs, targets) 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        return Dice_BCE
    
def evaluation_metrics(y_true, y_pred, smooth = 1):
    y_true = y_true.detach().numpy()
    y_pred = y_pred.detach().numpy()
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    cm1 = confusion_matrix(y_true_f, y_pred_f)
    intersection = np.sum(y_true_f * y_pred_f)

    dice = (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    jaccard = (intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth - intersection)
    return(dice, jaccard)       

        
class SegmentationDataSet(data.Dataset):
    def __init__(self, dataset_path: str, df_input:str, image_transform1=None, image_transform=None, output_channel = 4):
        self.dataset_path = dataset_path
        self.input_path = os.path.join(self.dataset_path, 'Images/')
        self.output_path = os.path.join(self.dataset_path, 'Masks/')
        self.df = pd.read_csv(df_input)
        self.images_list = list(self.df['filename'])
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.float32
        self.image_transform1 = image_transform1
        self.image_transform = image_transform
        self.output_channel = output_channel
        
    def __len__(self):
        return len(self.images_list)
    def __getitem__(self, index: int):
        # Select the sample
        image_filename = self.images_list[index]
        # Load input and target
        image = cv.imread(os.path.join(self.input_path, image_filename),0)
        mask = cv.imread(os.path.join(self.output_path, image_filename),0)
        
        # padding
        width = max(image.shape) - image.shape[1]  # pad 0 on width
        height = max(image.shape) - image.shape[0]  # pad 0 on height
        image = np.pad(image, ((0, height), (0, width)), 'constant', constant_values=(0, 0))
        mask = np.pad(mask, ((0, height), (0, width)), 'constant', constant_values=(0, 0))
        
        if self.output_channel == 4:
            #for 4 channel output
            mask1 = mask.copy()
            mask2 = mask.copy()
            mask3 = mask.copy()

            mask1[mask1 == 1] = 255
            mask1[mask1 == 4] = 255
            mask2[mask2 == 2] = 255
            #mask2[mask2 == 4] = 255
            mask3[mask3 == 3] = 255
            mask[mask == 4] = 255
            mask = np.dstack((mask1, mask2, mask3, mask))
        
            mask = cv.resize(mask, (224,224)) #256, 256, 3
            mask[mask<128] = 0
            mask[mask>=128] = 1
            mask = np.transpose(mask, (2,0,1))
        else: #1 channel
            #for 1 channel output: bony region/background
            mask[mask >= 1] = 1        
            mask = cv.resize(mask, (224,224)) #256, 256, 3
            mask[mask>=0.5] = 1
            mask[mask<0.5] = 0
            mask = np.expand_dims(mask, axis=0)
            #mask = np.transpose(mask, (2,0,1))


        # add: 3 channel
        image = np.repeat(image[None,...], 3, axis=0).transpose(1, 2, 0)
        # add transform
        if self.image_transform:
            image1 = self.image_transform1(np.uint8(image))
            image = self.image_transform(np.uint8(image))
              
        # Typecasting
        mask = torch.from_numpy(mask).type(self.targets_dtype)

        return image1, mask, image_filename, image 
if args.vit_name: #TransUNet
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes= args.output_channel #add
    model_MIM = ViT_seg(config_vit, img_size=224, num_classes=args.output_channel)
    model_contrastive = ViT_seg(config_vit, img_size=224, num_classes=args.output_channel)
else:
    print("Check vit_name")

############## Notice: remember to add updated weights to model

PATH_MIM = args.save_dir_finetune_MIM + args.model_name_MIM
if os.path.exists(PATH_MIM):
    state_dict = torch.load(PATH_MIM, map_location='cpu')
    new_state_dict = {}

    for k, v in state_dict.items():
        # remove "module."
        if k.startswith("module."):
            new_key = k[len("module."):]
        else:
            new_key = k
        new_state_dict[new_key] = v

    model_MIM.load_state_dict(new_state_dict, strict=False)


PATH_contrastive = args.save_dir_finetune_contrastive + args.model_name_contrastive

if os.path.exists(PATH_contrastive):
    state_dict = torch.load(PATH_contrastive, map_location='cpu')
    new_state_dict = {}

    for k, v in state_dict.items():
        # remove "module."
        if k.startswith("module."):
            new_key = k[len("module."):]
        else:
            new_key = k
        new_state_dict[new_key] = v

    model_contrastive.load_state_dict(new_state_dict, strict=False)


def binary_entropy_confidence(prob, eps=1e-8):
    # prob is sigmoid output, shape [B,1,H,W]
    p = prob.clamp(min=eps, max=1-eps)

    H = - (p * torch.log(p) + (1-p) * torch.log(1-p))  # [B,1,H,W]
    H_norm = H / (math.log(2) + eps)
    c = 1 - H_norm
    return c  # [B,1,H,W]

def top2_gap_confidence_binary(p):
    # p: sigmoid probability for bone, [B,1,H,W]
    c = torch.abs(2 * p - 1)  # maps p=0.5 -> c=0, p=1 or 0 -> c=1
    return c

# Inference phase
def eval_model(dataloader, model_MIM1, model_contrastive1, save_path = args.save_dir_finetune_image):
    eval_loss = 0
    dices = []
    jaccards = []   
    dices_MIM = []
    jaccards_MIM = []
    dices_contrastive = []
    jaccards_contrastive = []
    detailed_results = []
    model_MIM1.eval()
    model_contrastive1.eval()
    with torch.no_grad():
        i = 0
        for images, masks, image_filename, image_gt in dataloader:
            i += 1
            images = images.to(device)
            image_gt = image_gt.to(device)
            masks = masks.to(device)
            
            preds_MIM = model_MIM1(images)
            preds_MIM = preds_MIM.to(device)
            preds_contrastive = model_contrastive1(images)
            preds_contrastive = preds_contrastive.to(device)
            #print(preds.shape, masks.shape, images.shape)
            #loss = DiceBCELoss().forward(preds, masks)
            #eval_loss += loss.item()
            masks = masks.cpu()
            images = images.cpu()
            image_gt = image_gt.cpu()
            preds_MIM = preds_MIM.cpu()
            preds_contrastive = preds_contrastive.cpu()
            if args.top2_confidence:
                c_g = top2_gap_confidence_binary(preds_MIM)
                c_c = top2_gap_confidence_binary(preds_contrastive)
                c_g = (c_g - c_g.min()) / (c_g.max() - c_g.min() + 1e-6)
                c_c = (c_c - c_c.min()) / (c_c.max() - c_c.min() + 1e-6)

            if args.SE_confidence:
                c_g = binary_entropy_confidence(preds_MIM) #MIM
                c_c = binary_entropy_confidence(preds_contrastive) #contrastive
                c_g = (c_g - c_g.min()) / (c_g.max() - c_g.min() + 1e-6)
                c_c = (c_c - c_c.min()) / (c_c.max() - c_c.min() + 1e-6)
            else:
                c_g = 1
                c_c = 1

            
            preds = (preds_MIM * c_g + preds_contrastive * c_c) / (c_g + c_c + 1e-6)

            #preds = (preds_MIM+preds_contrastive)/2
            preds[preds>=0.5] = 1 
            preds[preds<0.5] = 0
            
            preds_MIM[preds_MIM>=0.5] = 1 
            preds_MIM[preds_MIM<0.5] = 0
            
            preds_contrastive[preds_contrastive>=0.5] = 1 
            preds_contrastive[preds_contrastive<0.5] = 0
                    
            
            dice, jaccard  = evaluation_metrics(masks, preds)
            dices.append(dice)
            jaccards.append(jaccard)
            
            
            dice_MIM, jaccard_MIM  = evaluation_metrics(masks, preds_MIM)
            dices_MIM.append(dice_MIM)
            jaccards_MIM.append(jaccard_MIM)
            
            
            dice_contrastive, jaccard_contrastive= evaluation_metrics(masks, preds_contrastive)
            dices_contrastive.append(dice_contrastive)
            jaccards_contrastive.append(jaccard_contrastive)
            
            detailed_results.append({
                'image_filename': image_filename[0],
                'dice': dice,
                'jaccard': jaccard,
                'dice_MIM': dice_MIM,
                'jaccard_MIM': jaccard_MIM,
                'dice_contrastive': dice_contrastive,
                'jaccard_contrastive': jaccard_contrastive
            })
            
            
            
            filename_pred = image_filename[0].split('.')[0]+'_pred.png'
            filename_pred_MIM = image_filename[0].split('.')[0]+'_MIM_pred.png'
            filename_pred_contrastive= image_filename[0].split('.')[0]+'_contrastive_pred.png'
            filename_gt = image_filename[0].split('.')[0]+'_gt.png'
            filename_conf_MIM = image_filename[0].split('.')[0]+'_conf_MIM.png'
            filename_conf_contrastive= image_filename[0].split('.')[0]+'_conf_contrastive.png'
            filename_conf_diff = image_filename[0].split('.')[0]+'_conf_diff.png'
            filename_weight_MIM = image_filename[0].split('.')[0]+'_weight_MIM.png'

            if i%1 == 0 and args.save_figure:  
                #cv.imwrite(save_path+filename_pred, preds[0].permute(1, 2, 0).cpu().numpy() * 77 + image_gt[0].permute(1, 2, 0).cpu().numpy() * 178) #save prediction
                #cv.imwrite(save_path+filename_gt, masks[0].permute(1, 2, 0).cpu().numpy() * 77 + image_gt[0].permute(1, 2, 0).cpu().numpy() * 178) #save gt mask
                cv.imwrite(save_path+filename_pred, preds[0].permute(1, 2, 0).cpu().numpy() * 255) #save prediction
                cv.imwrite(save_path+filename_gt, masks[0].permute(1, 2, 0).cpu().numpy() * 255) #save gt mask
                cv.imwrite(save_path+filename_pred_MIM, preds_MIM[0].permute(1, 2, 0).cpu().numpy() * 255) #save prediction
                cv.imwrite(save_path+filename_pred_contrastive, preds_contrastive[0].permute(1, 2, 0).cpu().numpy() * 255) #save gt mask
                # ===== save confidence maps =====
                if isinstance(c_g, torch.Tensor):
                    conf_MIM_np = c_g[0].permute(1, 2, 0).cpu().numpy()
                    conf_MIM_np = (conf_MIM_np * 255).astype('uint8')
                    cv.imwrite(save_path + filename_conf_MIM, conf_MIM_np)
                    
                if isinstance(c_c, torch.Tensor):
                    conf_contrastive_np = c_c[0].permute(1, 2, 0).cpu().numpy()
                    conf_contrastive_np = (conf_contrastive_np * 255).astype('uint8')
                    cv.imwrite(save_path + filename_conf_contrastive, conf_contrastive_np)
                # ===== save difference map (who dominates) =====
                if isinstance(c_g, torch.Tensor) and isinstance(c_c, torch.Tensor):
                    diff_map = c_g - c_c
                    diff_np = diff_map[0].permute(1, 2, 0).cpu().numpy()
                    diff_np = ((diff_np - diff_np.min()) / (diff_np.max() - diff_np.min() + 1e-6) * 255).astype('uint8')
                    cv.imwrite(save_path + filename_conf_diff, diff_np)
                # ===== save fusion weight (MOST IMPORTANT) =====
                if isinstance(c_g, torch.Tensor) and isinstance(c_c, torch.Tensor):
                    w_g = c_g / (c_g + c_c + 1e-6)
                    weight_np = w_g[0].permute(1, 2, 0).cpu().numpy()
                    weight_np = (weight_np * 255).astype('uint8')
                    cv.imwrite(save_path + filename_weight_MIM, weight_np)

    
    df = pd.DataFrame(detailed_results)    
    return sum(dices)/len(dices), sum(jaccards)/len(jaccards), df

train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([224, 224]),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),])

# Create test dataset
test_dataset = SegmentationDataSet(dataset_path = args.dataset_path, df_input = args.test_csv,
                                        image_transform1=train_transform, image_transform=train_transform, output_channel = args.output_channel)
# Initialization


test_dataloader = data.DataLoader(dataset=test_dataset,
                                      batch_size=args.batch_size,
                                      shuffle = False, num_workers=4)     

# Run evaluation cycles

model_MIM = model_MIM.to(device)
model_contrastive = model_contrastive.to(device)

dice_eval, jaccard_eval, df_results = eval_model(test_dataloader, model_MIM, model_contrastive)
print("eval dice: "+str(dice_eval)+"\t"+"eval jaccard: "+str(jaccard_eval)+"\n")
with open(args.save_dir_finetune_image + 'training_result.txt', 'a') as f:
    f.write("eval dice: "+str(dice_eval)+"\t"+"eval jaccard: "+str(jaccard_eval)+"\n")


     
df_results.to_csv(args.save_dir_finetune_image + 'evaluation_results_ensemble.csv', index=False)




