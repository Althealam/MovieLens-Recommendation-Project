import torch.nn as nn
import torch

class Dice(nn.Module):
    def __init__(self, num_features, epsilon=1e-8):
        super(Dice, self).__init__()
        self.bn = nn.BatchNorm1d(num_features, affine=False)
        self.alpha = nn.Parameter(torch.zeros(num_features))
        self.epsilon = epsilon

    def forward(self, x):
        x_normed = self.bn(x)
        x_p = torch.sigmoid(self.alpha * (x_normed - x_normed.detach()))
        return x * x_p + (1 - x_p) * x_normed
