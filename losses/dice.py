import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from losses.focal import criterion_margin_focal_binary_cross_entropy

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + self.smooth)/(inputs.sum() + targets.sum() + self.smooth)  
        
        return 1 - dice
    
class HybridLoss(nn.Module):
    def __init__(self, alpha=1, beta=1, weight=None, size_average=True, smooth=1):
        super(HybridLoss, self).__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta
        self.focal = criterion_margin_focal_binary_cross_entropy
        self.dice = DiceLoss(weight=weight, size_average=size_average)

    def forward(self, inputs, targets):
        
        focal_loss = self.focal(inputs, targets)
        dice_loss = self.dice(inputs, targets)

        
        return self.alpha*focal_loss + self.beta*dice_loss