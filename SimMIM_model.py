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

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
import argparse
import os
from timm.models.layers import trunc_normal_


import segmentation_models_pytorch as smp
from segmentation_models_pytorch import Unet

from typing import Optional, Union, List

# TransUNet
from TransUNet.vit_seg_modeling import *
from TransUNet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg #add
from TransUNet.vit_seg_modeling import VisionTransformer as ViT_seg

#U-Net
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import (
    SegmentationModel,
    SegmentationHead,
    ClassificationHead,
)

from segmentation_models_pytorch.encoders.resnet import ResNetEncoder   
from segmentation_models_pytorch.encoders.resnet import resnet_encoders 
from segmentation_models_pytorch.encoders.resnet import * 
from segmentation_models_pytorch.encoders import * 
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder



# TransUNet
class SimMIM_TransUNet(nn.Module):
    def __init__(
        self, config, img_size=224, zero_head=False, vis=False, in_channels = 3, loss_function: str = "default"):
        super().__init__()
        
        #self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.activation = nn.Sigmoid() # remember to change later, for 4 channel output
        #self.patch_size = self.encoder.patch_size
        self.in_chans = in_channels
        self.loss_function = loss_function

    def forward(self, x, mask, x_original):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
            
            
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)  
        x = self.decoder(x, features)
        x_rec = self.segmentation_head(x)
        
        if self.loss_function == 'default': #default MAE loss
            loss_recon = F.l1_loss(x_original, x_rec, reduction='none')
            loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans
            return loss, x_rec #Add reconstruction
        if self.loss_function == 'rmse+mae':
            loss_fn1 = nn.MSELoss()
            RMSE_loss = torch.sqrt(loss_fn1(x_rec, x_original))
            mae_loss = F.l1_loss(x_original, x_rec, reduction='none')
            loss = ((RMSE_loss+mae_loss) * mask).sum() / (mask.sum() + 1e-5) / self.in_chans
            return loss, x_rec
               
    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1]-1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # Encoder whole
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)

                        
# U-Net                        

class SimMIM_UNet(nn.Module):
    def __init__(
        self, 
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 3,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
        loss_function: str = 'default',
        encoder_weights: Optional[str] = "imagenet"
    ):
        super().__init__()
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights #for UNet without ImageNet weights initialization: please delete this line
        )
        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        self.in_chans = in_channels
        #self.patch_size = self.encoder.patch_size
        self.loss_function = loss_function

    def forward(self, x, mask, x_original):
        #x: image_masked, mask: mask_fullsize, x_original: original image
        features = self.encoder(x)
        x_rec = self.decoder(*features)
        #x_rec = self.decoder(z)
        x_rec = self.segmentation_head(x_rec)

        if self.loss_function == 'default':
            loss_recon = F.l1_loss(x_original, x_rec, reduction='none')
            loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans
            return loss, x_rec #Add reconstruction
        if self.loss_function == 'rmse+mae':
            loss_fn1 = nn.MSELoss()
            RMSE_loss = torch.sqrt(loss_fn1(x_rec, x_original))
            mae_loss = F.l1_loss(x_original, x_rec, reduction='none')
            loss = ((RMSE_loss+mae_loss) * mask).sum() / (mask.sum() + 1e-5) / self.in_chans
            return loss, x_rec
