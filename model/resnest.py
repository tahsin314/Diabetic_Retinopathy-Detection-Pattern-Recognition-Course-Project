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
from losses.arcface import ArcMarginProduct
        
import timm
from pprint import pprint

model_names = timm.list_models('*resnext*')
print(model_names)
# print(torch.hub.list('zhanghang1989/ResNeSt', force_reload=True))
class Resnest(nn.Module):

    def __init__(self, model_name='resnest50_fast_1s1x64d', num_class=1):
        super().__init__()
        # self.backbone = timm.create_model(model_name, pretrained=True)
        # self.backbone = torch.hub.load('zhanghang1989/ResNeSt', model_name, pretrained=True)
        # print(self.backbone)
        self.backbone = timm.create_model(model_name, pretrained=True)
        # print(self.backbone)
        # self.in_features = 2048
        self.in_features = self.backbone.fc.in_features
        self.head = Head(self.in_features,num_class, activation='mish')
        self.out = nn.Linear(self.in_features, num_class)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        # x = self.backbone.avgpool(x)
        # x = self.out(x)
        # print(x.size())
        x = self.head(x)
        return x

class Attn_Resnest(nn.Module):

    def __init__(self, model_name='resnest50_fast_1s1x64d', normalize_attn=False, num_class=1):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True)
        # self.backbone = torch.hub.load('zhanghang1989/ResNeSt', model_name, pretrained=True)
        self.in_features = self.backbone.fc.in_features
        # print(self.in_features)
        self.head = Head(self.in_features,num_class, activation='mish')
        self.relu = Mish()
        self.maxpool = GeM()
        self.attn1 = AttentionBlock(256, 1024, 512, 4, normalize_attn=normalize_attn)
        self.attn2 = AttentionBlock(512, 1024, 512, 2, normalize_attn=normalize_attn)
        self.output = nn.Sequential(nn.Linear(768+num_class, 128), nn.Linear(128, num_class))
        
    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        x = self.backbone.maxpool(x)

        layer1 = self.backbone.layer1(x)
        layer2 = self.backbone.layer2(layer1)
        layer3 = self.backbone.layer3(layer2)
        layer4 = self.backbone.layer4(layer3)
        a1, g1 = self.attn1(layer1, layer3)
        a2, g2 = self.attn2(layer2, layer3)
        g = self.head(layer4)
        g_hat = torch.cat((g,g1,g2), dim=1) # batch_size x C
        out = self.output(g_hat)
        return out

class Resnest_ArcFace(nn.Module):

    def __init__(self, model_name='resnest50_fast_1s1x64d', use_meta=True, out_neurons=600, meta_neurons=150):
        super().__init__()
        # self.backbone = timm.create_model(model_name, pretrained=False)
        self.backbone = torch.hub.load('zhanghang1989/ResNeSt', model_name, pretrained=True)
        self.use_meta = use_meta
        self.in_features = 2048
        self.head = Head(self.in_features,128, activation='mish', use_meta=self.use_meta)
        self.metric_classify = ArcMarginProduct(128, 2)
        self.relu = Mish()
        self.maxpool = GeM()

    def forward(self, x, meta_data=None):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        
        x = self.backbone.layer3(x)
        
        x = self.backbone.layer4(x)
        
        x = self.head(x, meta_data)
        x = self.metric_classify(x)
        return x

class Attn_Resnest_ArcFace(nn.Module):

    def __init__(self, model_name='resnest50_fast_1s1x64d', normalize_attn=False, use_meta=True, out_neurons=600, meta_neurons=150):
        super().__init__()
        # self.backbone = timm.create_model(model_name, pretrained=False)
        self.backbone = torch.hub.load('zhanghang1989/ResNeSt', model_name, pretrained=True)
        self.use_meta = use_meta
        self.in_features = self.backbone.fc.in_features
        self.head = Head(self.in_features,128, activation='mish', use_meta=self.use_meta)
        self.metric_classify = ArcMarginProduct(128+256+512, 2)
        self.relu = Mish()
        self.maxpool = GeM()
        self.attn1 = AttentionBlock(256, 1024, 512, 4, normalize_attn=normalize_attn)
        self.attn2 = AttentionBlock(512, 1024, 512, 2, normalize_attn=normalize_attn)

    def forward(self, x, meta_data=None):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        layer1 = self.backbone.layer1(x)
        layer2 = self.backbone.layer2(layer1)
        layer3 = self.backbone.layer3(layer2)
        layer4 = self.backbone.layer4(layer3)
        a1, g1 = self.attn1(layer1, layer3)
        a2, g2 = self.attn2(layer2, layer3)
        g = self.head(layer4, meta_data)
        g_hat = torch.cat((g,g1,g2), dim=1) # batch_size x C
        out = self.metric_classify(g_hat)
        return out

class Mixnet(nn.Module):

    def __init__(self, model_name='mixnet_xxl', use_meta=True, out_neurons=600, meta_neurons=150):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True)
        self.use_meta = use_meta
        self.in_features = self.backbone.classifier.in_features
        # self.backbone.act2 = GeM()
        # self.backbone.act1 = GeM()
        # to_GeM(self.backbone.blocks)
        self.backbone.classifier = nn.Linear(self.in_features, 128)
        self.output = nn.Linear(128, 2)
        self.head = Head(self.in_features,2, activation='mish', use_meta=self.use_meta)

    def forward(self, x, meta_data=None):
        x = self.backbone.conv_stem(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        x = self.backbone.blocks(x)
        x = self.backbone.conv_head(x)
        x = self.backbone.bn2(x)
        x = self.backbone.act2(x)
        x = self.backbone.global_pool(x)
        x = x.flatten(1)
        x = self.backbone.classifier(x)
        x = self.output(x)
        return x
