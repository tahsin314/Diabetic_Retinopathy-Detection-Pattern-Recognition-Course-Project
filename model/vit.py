from torch import nn
import timm

class ViT(nn.Module):
    def __init__(self, pretrained_model='vit_base_patch16_384'):
        super().__init__()
        self.backbone = timm.create_model(pretrained_model, pretrained=True)