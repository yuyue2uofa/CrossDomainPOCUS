#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import cv2 as cv
import pandas as pd
import numpy as np
from TransUNet.vit_seg_modeling import VisionTransformer as ViT_seg
from TransUNet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
import os
import argparse
import pickle
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

parser = argparse.ArgumentParser(description='SimCLR_TransUNet')
parser.add_argument('--epochs', default=800, type=int, metavar='N',
                    help='number of total epochs to run, 200-800') 
parser.add_argument('--dataset_path', default = '/folder_path/', type=str) 
parser.add_argument('--train_csv', default='train.csv', type=str)
parser.add_argument('--save_dir', default = './results/', type=str, help = 'folder for pretrained model') 
parser.add_argument('--checkpoint_name', default = 'latest_checkpoint.pth', type=str, help = 'Pretrained model name') 
parser.add_argument('--batch_size', type=int, default=128, help='batch size') 
parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)') 
parser.add_argument('--lr', type=float, default=0.001, 
                        help='learning rate, test 0.005, 0.001, 0.0005, 0.0001, etc') 
parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
parser.add_argument('--transunet', default='R50-ViT-B_16', type=str,
                        help='transunet selection: R50-ViT-B_16/None; None: model will be UNet') 
parser.add_argument('--use_pretrained_transunet', action='store_true')                         
parser.add_argument('--save_frequency', default=100, type=int) 


args = parser.parse_args() 
os.makedirs(args.save_dir, exist_ok=True)


# ----------------------------
# 1. Data Augmentation for SimCLR
# ----------------------------
simclr_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(3/4, 4/3)), #random crop image, and rescale to 224
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.0)], p=0.8), # p: probabiliyu
    transforms.ToTensor(),
])

class SimCLRDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path: str, df_input:str, image_transform=None):
        self.dataset_path = dataset_path
        self.input_path = os.path.join(self.dataset_path, 'Images/')
        self.df = pd.read_csv(df_input)
        self.images_list = list(self.df['filename'])
        self.inputs_dtype = torch.float32
        self.transform = image_transform

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index: int):
        # Select the sample
        image_filename = self.images_list[index]
        # Load input
        image = cv.imread(os.path.join(self.input_path, image_filename),0)
        # padding
        width = max(image.shape) - image.shape[1]  # pad 0 on width
        height = max(image.shape) - image.shape[0]  # pad 0 on height
        image = np.pad(image, ((0, height), (0, width)), 'constant', constant_values=(0, 0))
        # add: 3 channel
        image = np.repeat(image[None,...], 3, axis=0).transpose(1, 2, 0)
        # add transform
        if self.transform:
            image1 = self.transform(np.uint8(image))    
            image2 = self.transform(np.uint8(image))         
        return image1, image2
    
# ----------------------------
# 2. Model: Backbone + Projection Head
# ----------------------------
class SimCLR(nn.Module):
    def __init__(self, base_model='resnet18', out_dim=128):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # remove original fc

        # projection head
        self.projection = nn.Sequential(
            nn.Linear(num_ftrs, num_ftrs),
            nn.ReLU(),
            nn.Linear(num_ftrs, out_dim)
        )

    def forward(self, x):
        h = self.backbone(x)
        z = self.projection(h)
        z = F.normalize(z, dim=1)
        return z

class SimCLRTransUNet(nn.Module):
    def __init__(self, transunet_model, projection_dim=128):
        super().__init__()
        self.embeddings = transunet_model.transformer.embeddings  
        self.encoder_out_dim = 768 
        self.encoder = transunet_model.transformer.encoder
        # projection head for SimCLR
        self.projection_head = nn.Sequential(
            nn.Linear(self.encoder_out_dim, self.encoder_out_dim),
            nn.ReLU(),
            nn.Linear(self.encoder_out_dim, projection_dim)
        )

    def forward(self, x):
        
        embedding_output, features = self.embeddings(x)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
  
        features = encoded
        if features.dim() == 3:  # [B, N, C]
            features = features.mean(dim=1)
        z = self.projection_head(features)
        z = F.normalize(z, dim=1)  
        return z

# ----------------------------
# 3. NT-Xent Loss
# ----------------------------
def nt_xent_loss(z_i, z_j, temperature=0.5,device='cpu'):
    batch_size = z_i.size(0)
    
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)
    
    z = torch.cat([z_i, z_j], dim=0)  # [2B, dim]
    
    sim = torch.matmul(z, z.T) / temperature  # [2B, 2B]
    
    mask = torch.eye(2*batch_size, device=z.device).bool()
    sim = sim.masked_fill(mask, -9e15)
    
    labels = torch.arange(batch_size, device=z.device)
    labels = torch.cat([labels + batch_size, labels], dim=0)  
    
    loss = F.cross_entropy(sim, labels)
    return loss


# -------------------- Initialize DDP --------------------
import torch.distributed as dist
import os

dist.init_process_group(backend='nccl')

local_rank = int(os.environ["LOCAL_RANK"])

torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")



# ViT + TransUNet
config_vit = CONFIGS_ViT_seg[args.transunet]
config_vit.n_classes = 1
model_transunet = ViT_seg(config_vit, img_size=224, num_classes=1)
model_transunet.transformer.encoder.gradient_checkpointing = True
model = SimCLRTransUNet(model_transunet).to(device)

# Wrap model in DDP
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)


optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# Load checkpoint (only by rank 0 or all ranks‚)
PATH = args.save_dir + args.checkpoint_name
if os.path.exists(PATH):
    checkpoint = torch.load(PATH, map_location=device)
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k.replace("module.", "")] = v
    model.load_state_dict(new_state_dict)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    results = checkpoint['results']
else:
    start_epoch = 0
    results = {'train_loss': []}

training_dataset = SimCLRDataset(args.dataset_path, args.train_csv, simclr_transform)
train_sampler = torch.utils.data.distributed.DistributedSampler(training_dataset)
train_loader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=False, sampler=train_sampler)  # batch_size 

   
import socket, torch, os, sys

local_rank = int(os.environ.get("LOCAL_RANK", -1))
global_rank = int(os.environ.get("RANK", -1))
pid = os.getpid()
host = socket.gethostname()

# make prints per-process clear and flush immediately
def dbg_print(*args, **kwargs):
    print(f"[{host}] rank={global_rank} local_rank={local_rank} pid={pid}:", *args, **kwargs)
    sys.stdout.flush()

dbg_print("torch.cuda.device_count()", torch.cuda.device_count())
dbg_print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

if torch.cuda.is_available():
    try:
        # ensure CUDA work is finished for accurate stats
        torch.cuda.synchronize(local_rank)
    except Exception:
        # synchronize may fail on some setups; ignore
        pass

    dbg_print("current_device:", torch.cuda.current_device())
    try:
        dbg_print("device_name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    except Exception:
        pass

    dbg_print("mem_alloc (MiB):", torch.cuda.memory_allocated() / 1024**2)
    dbg_print("mem_reserved (MiB):", torch.cuda.memory_reserved() / 1024**2)

    # Optional: show more stats in a safe way if available
    try:
        stats = torch.cuda.memory_stats()
        dbg_print("memory_stats keys:", list(stats.keys())[:10])
    except Exception:
        pass


for epoch in range(start_epoch, args.epochs):
    train_sampler.set_epoch(epoch)
    for x1, x2 in train_loader:
        x1, x2 = x1.to(device), x2.to(device)
        z1 = model(x1)
        torch.cuda.empty_cache()
        z2 = model(x2)

        #z1, z2 = model(x1), model(x2)
        loss = nt_xent_loss(z1, z2, device=device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ------------------  loss ------------------
        loss_value = loss.detach()
        dist.all_reduce(loss_value, op=dist.ReduceOp.SUM)
        loss_value /= dist.get_world_size()
    # ------------------------------------------------------

    if dist.get_rank() == 0:
        results['train_loss'].append(loss_value.item())
        print(f"Epoch {epoch}: loss={loss_value.item():.4f}\n")
        with open(args.save_dir + 'training_result.txt', 'a') as f:
            f.write(f"(epoch {epoch})\ttrain loss: {loss_value.item():.4f}\n")

        # save checkpoint
        checkpoint = {'epoch': epoch,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'loss': loss_value.item(),
                      'results': results}    
        torch.save(checkpoint, args.save_dir + 'latest_checkpoint.pth')

        if (epoch+1) % args.save_frequency == 0:
            torch.save(model.state_dict(), args.save_dir + f'latest_model_{epoch}.pth')

dist.destroy_process_group()  




