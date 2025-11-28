# regression.py

from torch import nn

__all__ = ['Regression']


class Regression(nn.Module):

    def __init__(self, reduction):
        super(Regression, self).__init__()
        self.loss = nn.MSELoss(reduction=reduction)
        self.weighted_total_loss = 0
        self.num_samples = 0

    def __call__(self, inputs, targets):
        loss = self.loss(inputs, targets)
        self.weighted_total_loss += loss * len(inputs)
        self.num_samples += len(inputs)
        self.weighted_total_loss += loss * len(inputs)
        self.num_samples += len(inputs)
        return loss
    
    def reset(self):
        self.weighted_total_loss = 0
        self.num_samples = 0
    
    def compute(self):
        # import pdb; pdb.set_trace()
        return self.weighted_total_loss / self.num_samples

