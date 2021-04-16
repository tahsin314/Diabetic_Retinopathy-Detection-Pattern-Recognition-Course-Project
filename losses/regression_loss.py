import torch
from torch import nn
'''
https://github.com/tuantle/regression-losses-pytorch
'''
class XSigmoidLoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_t, y_prime_t):
        ey_t = (y_t - y_prime_t)**2
        loss = 2 * (ey_t**0.5) / (1 + 0.5*torch.exp(-ey_t)) - (ey_t**0.5)
        if self.reduction =='mean':
            return torch.mean(loss)
        elif self.reduction =='sum':
            return torch.sum(loss)
        elif self.reduction =='none':
            return loss

"""
From https://en.wikipedia.org/wiki/Coefficient_of_determination
"""
def r2_loss(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2

def hybrid_regression_loss(output, target, a=0.9, b=0.1):
    xsig = XSigmoidLoss(reduction='mean')
    mse = nn.MSELoss(reduction='mean')
    mse_loss_value = mse(output, target)
    xsig_loss_value = xsig(output, target)
    loss = a*mse_loss_value + b*xsig_loss_value
    return loss