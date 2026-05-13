#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import argparse
import os

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

from TransUNet.vit_seg_modeling import *
from TransUNet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg 
from TransUNet.vit_seg_modeling import VisionTransformer as ViT_seg

parser = argparse.ArgumentParser(description='TransUNet')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run') 
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--dataset_path', default = '/folder_path/', type=str) 
parser.add_argument('--train_csv', default='train_source.csv', type=str)
parser.add_argument('--val_csv', default='val_source.csv', type=str)
parser.add_argument('--save_dir', default = './results/', type=str, help = 'folder for pretrained model') 
parser.add_argument('--save_dir_finetune', default = './results_finetune/', type=str) 
parser.add_argument('--vit_name', default = 'R50-ViT-B_16', type=str , metavar='MODEL',
                    help='Name of model to train: R50-ViT-B_16/None; if None, model will be UNet') 
parser.add_argument('--pretrained_checkpoint_name', default = 'latest_checkpoint.pth', type=str, help = 'Pretrained model name') 
parser.add_argument('--batch_size', type=int, default=128, help='batch size (default: 16)') 
parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)') 
parser.add_argument('--lr', type=float, default=0.002, 
                        help='learning rate') 
parser.add_argument('--output_channel', default = 1, type=int, help = 'output prediction: 1/4') 
args = parser.parse_args()

    
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
os.makedirs(args.save_dir_finetune, exist_ok=True)


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
    
# evaluation metrics, we only evaluate dice and jaccard during training/validation phase
def evaluation_metrics(y_true, y_pred, smooth = 1):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    cm1 = confusion_matrix(y_true_f, y_pred_f)
    intersection = np.sum(y_true_f * y_pred_f)

    dice = (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    jaccard = (intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth - intersection)
    return(dice, jaccard)       


class SegmentationDataSet(data.Dataset):
    def __init__(self, dataset_path: str, df_input:str, image_transform=None, output_channel = 4):
        self.dataset_path = dataset_path
        self.input_path = os.path.join(self.dataset_path, 'Images/')
        self.output_path = os.path.join(self.dataset_path, 'Masks/')
        self.df = pd.read_csv(df_input)
        self.images_list = list(self.df['filename'])
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.float32
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
            image = self.image_transform(np.uint8(image))
              
        # Typecasting
        mask = torch.from_numpy(mask).type(self.targets_dtype)

        return image, mask 


# -------------------- Initialize DDP --------------------
dist.init_process_group(backend='nccl')
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(local_rank)





# -------------------- Model Definition --------------------
if args.vit_name:  # TransUNet
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.output_channel
    model = ViT_seg(config_vit, img_size=224, num_classes=args.output_channel)
else:
    print("check vit_name")

# -------------------- Move model to current GPU --------------------
model = model.to(local_rank)

# -------------------- Paths --------------------
PATH_ckpt = os.path.join(args.save_dir_finetune, 'latest_checkpoint.pth')
PATH_pretrained = os.path.join(args.save_dir, args.pretrained_checkpoint_name)

# -------------------- Initialize optimizer --------------------
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.99, weight_decay = args.weight_decay)

# -------------------- Load Weights / Checkpoint --------------------
start_epoch = 0
best_val_loss = float('inf')



if os.path.exists(PATH_ckpt):  #  Continue training (checkpoint)
    checkpoint = torch.load(PATH_ckpt, map_location=f'cuda:{local_rank}')
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        # remove "module."
        if k.startswith("module."):
            new_key = k[len("module."):]
        else:
            new_key = k
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict, strict=False)
    #model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  #  restore optimizer
    start_epoch = checkpoint['epoch'] + 1                      #  restore epoch
    results = checkpoint['results']
    if len(results['eval_loss']) == 0:
        best_val_loss = np.inf
    else:
        best_val_loss = min(results['eval_loss'])
    #best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    print(f"[Rank {local_rank}] Resume training from checkpoint {PATH_ckpt} at epoch {start_epoch}")

elif os.path.exists(PATH_pretrained):  
    checkpoint = torch.load(PATH_pretrained, map_location=f'cuda:{local_rank}')
    state_dict = checkpoint['model_state_dict']
    print('len(state_dict)', len(state_dict))
    print(list(state_dict.keys())[:10])
    
    # Remove 'module.' if exists
    new_state_dict = {
        k.replace("module.", ""): v
        for k, v in state_dict.items()}
    #if "projection_head" not in k

    print(PATH_pretrained)
    print('len(new_state_dict)', len(new_state_dict))
    print(list(new_state_dict.keys())[:20])

    model_dict = model.state_dict()
    print('len(model_dict)', len(model_dict))
    print(list(model_dict.keys())[:20])
    
    prefix = "transformer."
    
    matched = {}
    mismatched = []

    print("\n=== Matching Pretrained ? Model Keys ===")
    for k, v in new_state_dict.items():
        full_k = prefix + k

        if full_k in model_dict:
            matched[full_k] = v
        else:
            mismatched.append((k, full_k))

    #matched = {k: v for k, v in new_state_dict.items() if k in model_dict}
    #mismatched = [k for k in new_state_dict.keys() if k not in model_dict]
    print('len(matched)', len(matched))
    # 3. load matched
    model_dict.update(matched)
    model.load_state_dict(model_dict)

    print(f"\n[Rank {local_rank}] Loaded pretrained weights: {len(matched)} matched, {len(mismatched)} unmatched.")
    #print('model_dict.keys:', model_dict.keys())
    start_epoch = 0
    results = {
        'train_loss': [], 'eval_loss': [],
        'train_dice': [], 'eval_dice': [],
        'train_jaccard': [], 'eval_jaccard': []
    }
    best_val_loss = np.inf

else:
    print(f"[Rank {local_rank}] Model training from scratch (ImageNet initialization)")
    start_epoch = 0
    results = {
        'train_loss': [], 'eval_loss': [],
        'train_dice': [], 'eval_dice': [],
        'train_jaccard': [], 'eval_jaccard': []
    }
    best_val_loss = np.inf

# -------------------- Wrap model in DDP --------------------
model = DDP(model, device_ids=[local_rank], output_device=local_rank)


def train_model(dataloader, optimizer, model, local_rank, world_size):
    model.train()
    criterion = DiceBCELoss()


    loss_sum = torch.zeros(1, device=local_rank)
    dice_sum = torch.zeros(1, device=local_rank)
    jaccard_sum = torch.zeros(1, device=local_rank)
    count = torch.zeros(1, device=local_rank)

    for images, masks in dataloader:
        images, masks = images.cuda(local_rank), masks.cuda(local_rank)

        optimizer.zero_grad(set_to_none=True)
        preds = model(images)
        loss = criterion(preds, masks)
        loss.backward()
        optimizer.step()

        #  GPU batch loss
        loss_sum += loss.detach()

        # metrics
        with torch.no_grad():
            preds_bin = (preds >= 0.5).float().cpu()
            masks_cpu = masks.cpu()
            dice, jaccard = evaluation_metrics(masks_cpu, preds_bin)

            dice_sum += dice
            jaccard_sum += jaccard

        count += 1

    # -----------------------
    #  Step 1: all-reduce
    # -----------------------
    torch.distributed.all_reduce(loss_sum, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(dice_sum, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(jaccard_sum, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.SUM)

    # -----------------------
    # Step 2
    # -----------------------
    mean_loss = (loss_sum / count).item()
    mean_dice = (dice_sum / count).item()
    mean_jaccard = (jaccard_sum / count).item()

    return mean_loss, model, mean_dice, mean_jaccard


def eval_model(dataloader, model, local_rank, world_size):
    model.eval()
    criterion = DiceBCELoss()

    loss_sum = torch.zeros(1, device=local_rank)
    dice_sum = torch.zeros(1, device=local_rank)
    jaccard_sum = torch.zeros(1, device=local_rank)
    count = torch.zeros(1, device=local_rank)

    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.cuda(local_rank), masks.cuda(local_rank)

            preds = model(images)
            loss = criterion(preds, masks)

            # loss
            loss_sum += loss.detach()

            # metrics
            preds_bin = (preds >= 0.5).float()
            dice, jaccard = evaluation_metrics(masks, preds_bin)

            dice_sum += dice
            jaccard_sum += jaccard

            count += 1

    # -----------------------
    #  Step 1: all-reduce 
    # -----------------------
    torch.distributed.all_reduce(loss_sum, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(dice_sum, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(jaccard_sum, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.SUM)

    # -----------------------
    #  Step 2:
    # -----------------------
    mean_loss = (loss_sum / count).item()
    mean_dice = (dice_sum / count).item()
    mean_jaccard = (jaccard_sum / count).item()

    return mean_loss, mean_dice, mean_jaccard


train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([224, 224]),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])

# Create training dataset
training_dataset = SegmentationDataSet(dataset_path = args.dataset_path, df_input = args.train_csv, 
                                       image_transform=train_transform, output_channel = args.output_channel)
# Create validation dataset
validation_dataset = SegmentationDataSet(dataset_path = args.dataset_path, df_input = args.val_csv,
                                        image_transform=train_transform, output_channel = args.output_channel)


train_sampler = torch.utils.data.distributed.DistributedSampler(training_dataset)
validation_sampler = torch.utils.data.distributed.DistributedSampler(validation_dataset)



# Initialization

training_dataloader = data.DataLoader(dataset=training_dataset,
                                      batch_size=args.batch_size,
                                      shuffle = False, sampler=train_sampler)

validation_dataloader = data.DataLoader(dataset=validation_dataset,
                                      batch_size=args.batch_size,
                                      shuffle = False, sampler=validation_sampler)     

# Run training and evaluation cycles
print('------------------------------------------------------------------------')
print('Epochs:', args.epochs)
print('Batch size:', args.batch_size)
print('Learning rate:', args.lr)
print('')

with open(args.save_dir_finetune + 'training_result.txt', 'a') as f:
    f.write('Batch size:'+str(args.batch_size)+'Learning rate:'+str(args.lr))

torch.set_grad_enabled(True)

for epoch in range(start_epoch, args.epochs):
    train_sampler.set_epoch(epoch)
    train_loss, model, dice_train, jaccard_train = train_model(training_dataloader, optimizer, model, local_rank, world_size)
    validation_sampler.set_epoch(epoch)
    eval_loss, dice_eval, jaccard_eval = eval_model(validation_dataloader, model, local_rank, world_size)
    if torch.distributed.get_rank() == 0:
        # save model
        if epoch < 100:
            if eval_loss<best_val_loss:
                best_val_loss = eval_loss # save on val loss
                torch.save(model.state_dict(),args.save_dir_finetune + 'best_model_loss100.pth') #change
        else:
            if eval_loss<best_val_loss:
                best_val_loss = eval_loss # save on val loss
                torch.save(model.state_dict(),args.save_dir_finetune + 'best_model_loss200.pth') #change
                 
        print("(epoch "+str(epoch)+")", 
              "\t"+"train loss: "+str(train_loss)+
              "\t"+"eval loss: "+str(eval_loss)+
              "\t"+"training dice: "+str(dice_train)+
              "\t"+"eval dice: "+str(dice_eval)+
              "\t"+"training jaccard: "+str(jaccard_train)+
              "\t"+"eval jaccard: "+str(jaccard_eval)+
              "\t"+"best eval loss: "+str(best_val_loss))
        torch.save(model.state_dict(),args.save_dir_finetune + 'latest_model.pth') #change
        with open(args.save_dir_finetune + 'training_result.txt', 'a') as f:
            f.write("(epoch "+str(epoch)+")"+ 
                    "\t"+"train loss: "+str(train_loss)+
                    "\t"+"eval loss: "+str(eval_loss)+
                    "\t"+"training dice: "+str(dice_train)+
                    "\t"+"eval dice: "+str(dice_eval)+
                    "\t"+"training jaccard: "+str(jaccard_train)+
                    "\t"+"eval jaccard: "+str(jaccard_eval)+
                    "\t"+"best eval loss: "+str(best_val_loss)+"\n")
        results['train_loss'].append(train_loss)
        results['eval_loss'].append(eval_loss)
        results['train_dice'].append(dice_train)
        results['eval_dice'].append(dice_eval)
        results['train_jaccard'].append(jaccard_train)
        results['eval_jaccard'].append(jaccard_eval)
    
    
    
        # save checkpoint
        checkpoint = {'epoch': epoch,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'results': results}    
        torch.save(checkpoint, args.save_dir_finetune + 'latest_checkpoint.pth')

dist.destroy_process_group()     





