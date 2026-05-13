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

parser = argparse.ArgumentParser(description='TransUNet/UNet')
parser.add_argument('--epochs', default=400, type=int, metavar='N',
                    help='number of total epochs to run, 200-800') 
parser.add_argument('--dataset_path', default = '/folder_path/', type=str) 
parser.add_argument('--train_csv', default='/train.csv', type=str)
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
parser.add_argument('--min_frame', default=20, type=int) 
parser.add_argument('--scale_min', default=0.5, type=float) 
parser.add_argument('--scale_max', default=1, type=float) 

args = parser.parse_args()
os.makedirs(args.save_dir, exist_ok=True)


# ----------------------------
# 1. Data Augmentation for SimCLR
# ----------------------------
simclr_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(224, scale=(args.scale_min, args.scale_max), ratio=(3/4, 4/3)), #random crop image, and rescale to 224
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.0)], p=0.8), # p: probability
    transforms.ToTensor(),
])

class SimCLRDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path: str, df_input: str, batch_size: int, image_transform=None):
        self.dataset_path = dataset_path
        self.input_path = os.path.join(self.dataset_path, 'Images/')
        self.df = pd.read_csv(df_input)
        self.images_list = list(self.df['filename'])
        self.inputs_dtype = torch.float32
        self.transform = image_transform
        
        self.batch_size = batch_size
        unique_videos = self.df['videoname'].unique()
        self.video_to_id = {name: i for i, name in enumerate(unique_videos)}

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index: int):
        # Load the image
        image_filename = self.images_list[index]
        image = cv.imread(os.path.join(self.input_path, image_filename), 0)
        
        # Padding to make square
        width = max(image.shape) - image.shape[1]
        height = max(image.shape) - image.shape[0]
        image = np.pad(image, ((0, height), (0, width)), 'constant', constant_values=(0, 0))
        
        # Convert to 3 channels
        image = np.repeat(image[None, ...], 3, axis=0).transpose(1, 2, 0)
        
        # Apply SimCLR augmentations if any
        if self.transform:
            image1 = self.transform(np.uint8(image))
            image2 = self.transform(np.uint8(image))
        else:
            image1, image2 = torch.tensor(image, dtype=self.inputs_dtype), torch.tensor(image, dtype=self.inputs_dtype)
            
        videoname = self.df.loc[index, 'videoname'] 
        framenum = self.df.loc[index, 'framenum']
            
        return image1, image2, videoname, framenum

from torch.utils.data._utils.collate import default_collate

def custom_collate_fn(batch, video_to_id_map):
    images1 = [item[0] for item in batch]
    images2 = [item[1] for item in batch]
    videoname_strs = [item[2] for item in batch] 
    framenums = [item[3] for item in batch]
    
    images1_batched = default_collate(images1)
    images2_batched = default_collate(images2)
    
    video_ids = [video_to_id_map[name] for name in videoname_strs]
    video_ids_batched = torch.tensor(video_ids, dtype=torch.long)
    
    framenums_batched = torch.tensor(framenums, dtype=torch.float)

    return images1_batched, images2_batched, video_ids_batched, framenums_batched

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
        # encoder 
        features = encoded
        # global pooling�
        if features.dim() == 3:  # [B, N, C]
            features = features.mean(dim=1)
        z = self.projection_head(features)
        z = F.normalize(z, dim=1)  
        return z

# ----------------------------
# 3. MT-NXent Loss
# ----------------------------
def nt_xent_loss_masked(z_i, z_j, batch_videonames, batch_framenums, min_frame_gap: int = 20, temperature=0.5):
    batch_size = z_i.size(0)
    device = z_i.device
    
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)
    
    z = torch.cat([z_i, z_j], dim=0)    # [2B, dim]
    
    sim = torch.matmul(z, z.T) / temperature    # [2B, 2B]
    
    standard_mask = torch.eye(2 * batch_size, device=device).bool()
    

    vn = torch.cat([batch_videonames, batch_videonames], dim=0).unsqueeze(1) # [2B, 1]
    fn = torch.cat([batch_framenums, batch_framenums], dim=0).unsqueeze(1)   # [2B, 1]

    # is_same_video[i, j] = True iff z[i] and z[j] are from the same video
    is_same_video = (vn == vn.T) # [2B, 2B]

    # is_close_frame[i, j] = True iff |framenum_i - framenum_j| < min_frame_gap
    frame_diff = torch.abs(fn - fn.T) # [2B, 2B]
    is_close_frame = (frame_diff < min_frame_gap)

    # temporal_mask: True iff z[i] and z[j] are from the same video AND too close
    temporal_mask = is_same_video & is_close_frame

    positive_mask = torch.zeros_like(standard_mask)
    labels = torch.arange(batch_size, device=device)

    positive_labels = torch.cat([labels + batch_size, labels], dim=0) 
    positive_mask[torch.arange(2*batch_size), positive_labels] = True
    
    final_mask = (standard_mask | temporal_mask) & (~positive_mask) 
    
    sim = sim.masked_fill(final_mask, -9e15)

    loss = F.cross_entropy(sim, positive_labels)
    return loss


# -------------------- Initialize DDP --------------------
import torch.distributed as dist
import os

dist.init_process_group(backend='nccl')

local_rank = int(os.environ["LOCAL_RANK"])

torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")



config_vit = CONFIGS_ViT_seg[args.transunet]
config_vit.n_classes = 1
model_transunet = ViT_seg(config_vit, img_size=224, num_classes=1)
model_transunet.transformer.encoder.gradient_checkpointing = True
model = SimCLRTransUNet(model_transunet).to(device)

# Wrap model in DDP
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)


optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# Load checkpoint (only by rank 0 or all ranks, depends on你的需求)
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

training_dataset = SimCLRDataset(args.dataset_path, args.train_csv, args.batch_size, simclr_transform)
print('training_dataset loaded')

import functools

train_sampler = torch.utils.data.distributed.DistributedSampler(
    training_dataset,
    num_replicas=dist.get_world_size(),
    rank=local_rank,
    shuffle=True
)
print('train_sampler loaded')

collate_func_with_map = functools.partial(
    custom_collate_fn, 
    video_to_id_map=training_dataset.video_to_id
)

train_loader = torch.utils.data.DataLoader(
    training_dataset, 
    batch_size=args.batch_size,
    sampler=train_sampler, 
    collate_fn=collate_func_with_map, 
    num_workers=4, 
    pin_memory=True
)
   
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
    for x1, x2, videoname1, framenum1 in train_loader:
        x1, x2 = x1.to(device), x2.to(device)
        z1 = model(x1)
        torch.cuda.empty_cache()
        z2 = model(x2)
        videonames_gpu = videoname1.to(device)
        framenums_gpu = framenum1.to(device)
        loss = nt_xent_loss_masked(z1, z2, videonames_gpu, framenums_gpu, args.min_frame)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ------------------ � loss ------------------
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





