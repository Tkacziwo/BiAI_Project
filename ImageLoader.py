import os
import torch
import imageio.v2 as iio
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class ImageLoader():
    def __init__(self):
        super().__init__()
        # self.batch_size = batch_size
        # self.num_workers = num_workers
        self.input_photos_tensor = []
        self.input_photos_file_names = []
        # self.transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5,), (0.5,))
        # ])
        # self.train_data = datasets.MNIST(root='input-photos', train=True, download=True, transform=self.transform)
        # self.test_data = datasets.MNIST(root='input-photos', train=False, download=True, transform=self.transform)
    
    def get_train_loader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def get_test_loader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def load_input_photos_to_tensor(self, dir_path: str):
        for name in os.listdir(dir_path):
            photo_path = dir_path + "/" + name
            img = iio.imread(photo_path)
            transform = transforms.Compose([transforms.ToTensor()])
            tensor = transform(img)
            tensor = torch.unsqueeze(tensor, 0)
            self.input_photos_tensor.append(tensor)
            self.input_photos_file_names.append(name)

    def get_input_photo_tensors(self):
        return self.input_photos_tensor
    
    def get_input_photo_file_names(self):
        return self.input_photos_file_names
        