import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class CNN(nn.Module):
    def __init__(self, size:int = 1):
        super().__init__()
        self.size = size
        self.flatten = nn.Flatten()

        self.convoluted_reLu_stack = nn.Sequential(
            nn.Conv2d(3, self.size*16, 3, 1, 1),
            nn.LeakyReLU(),          
            nn.MaxPool2d(2,2),

            nn.Conv2d(self.size*16,self.size*32, 3, 1, 1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(self.size*32, self.size*64, 3, 1, 1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(self.size*64, self.size*128, 3, 1, 1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(self.size*128, self.size*256, 3, 1, 1),
            nn.LeakyReLU(),

            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(self.size*256, self.size*3)
        )

    def forward(self, x):
        logits = self.convoluted_reLu_stack(x)
        return logits   