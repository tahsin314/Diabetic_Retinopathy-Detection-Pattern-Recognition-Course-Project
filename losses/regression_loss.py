import torch

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