import torch
from torch import nn
import timm
from .utils import * 

from bottleneck_transformer_pytorch import BottleStack

class BotNet(nn.Module):
    def __init__(self, model_name='gluon_resnet50_v1b', dim=384, num_class=1):
        super().__init__()
        assert dim%64 == 0, "\033[91mImage Dimension must be multiple of 64"
        self.backbone = timm.create_model(model_name, pretrained=True)
        in_features = self.backbone.fc.in_features
        self.layer = BottleStack(
        dim = in_features,
        fmap_size = dim//32,        # set specifically for imagenet's 224 x 224
        dim_out = 512,
        proj_factor = 4,
        downsample = True,
        heads = 4,
        dim_head = 128,
        rel_pos_emb = True,
        activation = nn.ReLU())
        self.head1 = Head(256, 1)
        self.head = nn.Sequential(self.layer, self.head1)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.head(x)
        return x
