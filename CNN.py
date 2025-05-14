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
            #first cnn layer with relu
            nn.Conv1d(1, 16, 3, 1,1),
            nn.ReLU(),          
            nn.Conv1d(16,32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 3, 1, 1),
            nn.ReLU(),
            #nn.Linear(64, 5)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.convoluted_reLu_stack(x)
        return logits   