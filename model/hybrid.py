from copy import deepcopy
import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
getattr(ssl, '_create_unverified_context', None)):ssl._create_default_https_context = ssl._create_unverified_context
import torch
from torch import nn
from torch.nn import *
from torch.nn import functional as F
from torchvision import models
from pytorchcv.model_provider import get_model as ptcv_get_model
# from .utils import get_cadene_model
from typing import Optional
from .utils import *
        
import timm
from pprint import pprint

class Hybrid(nn.Module):
    def __init__(self, normalize_attn=False):
        super().__init__()
        # self.backbone = timm.create_model(model_name, pretrained=True)
        self.resnest = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50_fast_1s1x64d', pretrained=True)
        self.effnet = timm.create_model('tf_efficientnet_b4', pretrained=True)
        self.in_features = 2048
        self.eff_conv = nn.Conv2d(1792, 2048, (1, 1))
        self.head_res = Head(self.in_features,2, activation='mish', use_meta=False)
        self.head_eff = Head(self.in_features,2, activation='mish', use_meta=False)
        self.relu = Mish()
        self.maxpool = GeM()
        self.res_attn1 = AttentionBlock(256, 1024, 512, 4, normalize_attn=normalize_attn)
        self.res_attn2 = AttentionBlock(512, 1024, 512, 2, normalize_attn=normalize_attn)
        self.eff_attn = AttentionBlock(2048, 2048, 512, 1, normalize_attn=normalize_attn)
        self.output1 = nn.Linear(2820, 128)
        self.output = nn.Linear(128, 2)

    def forward(self, x, meta_data=None):
        res1 = self.resnest.conv1(x)
        res2 = self.resnest.bn1(res1)
        res3 = self.resnest.relu(res2)
        res4 = self.resnest.maxpool(res3)

        layer1 = self.resnest.layer1(res4)
        layer2 = self.resnest.layer2(layer1)
        layer3 = self.resnest.layer3(layer2)
        layer4 = self.resnest.layer4(layer3)
        a1, g1 = self.res_attn1(layer1, layer3)
        a2, g2 = self.res_attn2(layer2, layer3)
        g_res = self.head_res(layer4, meta_data)
        
        eff1 = self.effnet.conv_stem(x)
        eff2 = self.effnet.bn1(eff1)
        eff3 = self.effnet.act1(eff2)
        eff4 = self.effnet.blocks(eff3)
        eff5 = self.effnet.conv_head(eff4)
        eff5 = self.eff_conv(eff5)
        
        a3, g3 = self.eff_attn(layer4, eff5)
        g_eff = self.head_eff(eff5, meta_data)

        g_hat = torch.cat((g_res,g1,g2,g3,g_eff), dim=1) # batch_size x C
    
        out = self.output1(g_hat)
        out = self.output(out)
        return out
