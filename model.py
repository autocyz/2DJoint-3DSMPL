import numpy as np
import torch
import torch.nn as nn
import os


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(24*2, 24*3),
            nn.ReLU(inplace=True),
            nn.Linear(24*3, 24*3)
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.net(x)
        return x