import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()

        self.convoluted_reLu_stack = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.LeakyReLU(),          
            nn.MaxPool2d(2,2),

            nn.Conv2d(16,32, 3, 1, 1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(32, 64, 3, 1, 1),
            nn.LeakyReLU(),

            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        logits = self.convoluted_reLu_stack(x)
        return logits   