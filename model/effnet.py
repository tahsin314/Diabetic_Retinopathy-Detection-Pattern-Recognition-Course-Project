## Code copied from Lukemelas github repository have a look
## https://github.com/lukemelas/EfficientNet-PyTorch/tree/master/efficientnet_pytorch
"""
This file contains helper functions for building the model and for loading model parameters.
These helper functions are built to mirror those in the official TensorFlow implementation.
"""
import warnings

from torch.optim import optimizer
warnings.filterwarnings("ignore", category=DeprecationWarning)
import re
import math
import collections
from functools import partial
import timm
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import model_zoo
import pytorch_lightning as pl
from .utils import *
from losses.arcface import ArcMarginProduct
from losses.triplet_loss import *
from efficientnet_pytorch import EfficientNet

class EffNet(nn.Module):
    def __init__(self, pretrained_model='tf_efficientnet_b4', num_class=1, freeze_upto=1):
        super(EffNet, self).__init__()
        # Load imagenet pre-trained model 
        self.backbone = timm.create_model(pretrained_model, pretrained=True)
        self.num_named_param = 0
        # Dirty way of finding out number of named params
        for l, (name, param) in enumerate(self.backbone.named_parameters()):
            self.num_named_param = l
        # self.freeze_upto_blocks(freeze_upto)
        self.in_features = self.backbone.bn2.num_features
        # self.backbone._fc = nn.Linear(in_features=in_features, out_features=1, bias=True)
        # self.backbone._avg_pooling = GeM()
        # self.backbone.global_pool = Identity()
        # self.backbone.global_pool = nn.Conv2d(self.in_features, 2048, 3)
        # self.out_feature = self.backbone.global_pool.out_channels
        # self.backbone._avg_pooling = nn.Sequential(AdaptiveConcatPool2d(), Swish(), Flatten())
        # self.backbone._fc = nn.Sequential(*[bn_drop_lin(self.in_features*2, 512, True, 0.5, Swish()), bn_drop_lin(512, 2, True, 0.5)])
        # self.head = Head(in_features, 2, activation='mish', use_meta=self.use_meta)
        # self.output = nn.Linear(self.out_neurons, 2)
        # self.backbone._fc = nn.Linear(in_features=in_features, out_features=256, bias=True)
        # self.backbone.classifier = Identity()
        self.head = Head(self.in_features, num_class, activation='mish')
        # self.output = nn.Linear(256, 1)
        
    def forward(self, x):
        x = self.backbone.conv_stem(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        x = self.backbone.blocks(x)
        x = self.backbone.conv_head(x)
        # x = self.backbone.global_pool(x)
        # x = self.backbone.classifier(x)
        output = self.head(x)
        return output

    def freeze_upto_blocks(self, n_blocks):
        '''
        Freezes upto bottom n_blocks
        '''
        if n_blocks == -1:
            return

        num_freeze_params = 6 + 12*n_blocks
        for l, (name, param) in enumerate(self.backbone.named_parameters()):
            if not 'bn' in name and l<=self.num_named_param-num_freeze_params:
                param.requires_grad = False
