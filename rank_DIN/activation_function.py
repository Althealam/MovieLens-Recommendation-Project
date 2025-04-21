# 自定义的激活函数
import torch
import torch.nn as nn

class CustomActivation(nn.Module):
    def forward(self, x):
        return torch.clamp(x, min=1, max=5)