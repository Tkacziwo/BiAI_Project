from genericpath import isdir
from torch.utils.data import Dataset
from torchvision import transforms
import random
from ColorAnnotation import ColorAnnotations, ColorsArray
import DataFilter
import os
from PIL import Image
import torch
from skimage import color
import imageio.v2 as iio
import shutil

class ImageColorDataSet(Dataset):
    def __init__(self, root='data', annotations = ColorAnnotations(), train=True, transform=None):
        self.imageFolder = root
        self.transform = transform
        self.annotations = annotations

        # Shuffle indices for all images
        if isdir(root):
            self.imageFileNames = os.listdir(self.imageFolder)
            random.shuffle(self.imageFileNames) 

            # Calculate split sizes
            numImages = len(self.imageFileNames)
            trainEnd = int(0.7 * numImages)  # 70% for training
            valEnd = trainEnd + int(0.15 * numImages)  # 15% for validation, rest for testing

            # Split indices
            train_indices = list(range(0, trainEnd))
            val_indices = list(range(trainEnd, valEnd))
            test_indices = list(range(valEnd, numImages))

            # Assign images to datasets
            self.trainData = [self.imageFileNames[i] for i in train_indices]
            self.valData = [self.imageFileNames[i] for i in val_indices]
            self.testData = [self.imageFileNames[i] for i in test_indices]

            # By default, use training set
            self.data = self.trainData if train else self.testData

    def load_predetermined_dataset(self, dataset_folder: str, filtered_data: ColorAnnotations):
        # Assign filtered data, dataset folder and transform to fields
        self.imageFolder = dataset_folder
        self.annotations = filtered_data
        self.transform = None

        train_folder_path = dataset_folder + "/train/"
        test_folder_path = dataset_folder + "/test/"
        validate_folder_path = dataset_folder + "/validate/"
        
        self.trainData = []
        self.testData = []
        self.valData = []
        for file in os.listdir(train_folder_path):
            self.trainData.append(file)
        for file in os.listdir(test_folder_path):
            self.testData.append(file)
        for file in os.listdir(validate_folder_path):
            self.valData.append(file)

        self.data = self.trainData

    def __len__(self):
        return len(self.data)

    def get_item(self, idx: int, path: str):
        imageName = self.data[idx]
        # imagePath = os.path.join(os.path.curdir, self.imageFolder)
        # imagePath = os.path.join(imagePath, imageName)
        image = Image.open(path+"/"+imageName).convert('RGB')
        image = color.rgb2lab(image)
        label = self.annotations.getAnnotation(imageName)
        if label is None:
            label = [0, 0, 0]  # Default label if not found

        if self.transform:
            image_tensor = self.transform(image)
            # label = torch.tensor([c for c in label], dtype=torch.float32)
            label = torch.tensor(label[1], dtype=torch.float32)
        else:
            image_tensor = transforms.ToTensor()(image).to(dtype = torch.float32)
            # label = torch.tensor([c for c in label], dtype=torch.float32)
            label = torch.tensor(label[1], dtype=torch.float32)

        image_tensor = torch.unsqueeze(image_tensor, 0)
        return image_tensor, label

    def __getitem__(self, idx):
        imageName = self.data[idx]
        imagePath = os.path.join(os.path.curdir, self.imageFolder)
        imagePath = os.path.join(imagePath, imageName)
        image = Image.open(imagePath).convert('RGB')
        image = color.rgb2lab(image)
        label = self.annotations.getAnnotation(imageName)
        if label is None:
            label = [0, 0, 0]  # Default label if not found

        if self.transform:
            image_tensor = self.transform(image)
            # label = torch.tensor([c for c in label], dtype=torch.float32)
            label = torch.tensor(label[1], dtype=torch.float32)
        else:
            image_tensor = transforms.ToTensor()(image).to(dtype = torch.float32)
            # label = torch.tensor([c for c in label], dtype=torch.float32)
            label = torch.tensor(label[1], dtype=torch.float32)

        image_tensor = torch.unsqueeze(image_tensor, 0)
        return image_tensor, label
    
    def get_image_filename(self, idx):
        return self.data[idx]

    def get_image_tensor(self, idx):
        imageName = self.data[idx]
        imagePath = os.path.join(os.path.curdir, self.imageFolder)
        imagePath = os.path.join(imagePath, imageName)
        image = Image.open(imagePath).convert('RGB')
        image = color.rgb2lab(image)

        transform = transforms.Compose([transforms.ToTensor()])
        image_tensor = transform(image)
        image_tensor = torch.unsqueeze(image_tensor, 0)

        return image_tensor

    def get_prepared_image_tensor(self, idx:int):
        image_name = self.data[idx]
        img = iio.imread(self.imageFolder + "/" + image_name)
        transform = transforms.Compose([transforms.ToTensor()])
        image_tensor = transform(img)
        image_tensor = torch.unsqueeze(image_tensor, 0)
        return image_tensor
    
    def get_training_data(self):
        return self.trainData

    @property
    def classes(self):
        return self.data.classes
    
    def switchOnTraining(self):
        self.data = self.trainData
    
    def switchOnTesting(self):
        self.data = self.testData
    
    def switchOnValidating(self):
        self.data = self.valData

    def save_dataset(self):
        destination = "CreatedDataSet"
        source = "input-photos"
        for name in self.trainData:
            for file in os.listdir("input-photos"):
                if file == name:
                    shutil.copy("input-photos/"+file, destination + "/train/")
                    break

        for name in self.testData:
            for file in os.listdir("input-photos"):
                if file == name:
                    shutil.copy("input-photos/"+file, destination + "/test/")
                    break
        
        for name in self.valData:
            for file in os.listdir("input-photos"):
                if file == name:
                    shutil.copy("input-photos/"+file, destination + "/validate/")
                    break
        print("Save dataset to CreatedDataSetFolder")
        
    
#Example usage
#Custom ImageLoader not needed, for nn training use basic DataLoader from torch.utils.data
"""
colorsPath = os.path.join(os.path.curdir, 'expected-results')
dataFilter = DataFilter.DataFilter()
for filename in os.listdir(colorsPath):
    if "_Time" not in filename:
        dataFilter.loadColorAnnotations(os.path.join(colorsPath, filename))

dataFilter.filterData()
data = ImageColorDataSet(root='input-photos', annotations=ColorAnnotations(), train=True, transform=None)

colorAnnotations = dataFilter.getData("000000010432.jpg")
print(colorAnnotations[1])
print(colorAnnotations[2])
print(colorAnnotations[3])
print(colorAnnotations[4])
print(colorAnnotations[5])
colorTensor = torch.tensor([c for c in colorAnnotations.getOneColor()], dtype=torch.float32)
print(colorTensor)
print(data.__getitem__(0))
"""