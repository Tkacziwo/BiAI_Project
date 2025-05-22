import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class ImageLoader(DataLoader):
    def __init__(self, batch_size=64, num_workers=0):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.train_data = datasets.MNIST(root='data', train=True, download=True, transform=self.transform)
        self.test_data = datasets.MNIST(root='data', train=False, download=True, transform=self.transform)
    
    def get_train_loader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def get_test_loader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)