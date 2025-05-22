from torch.utils.data import Dataset, ImageFolder
import random

class DataSet(Dataset):
    def __init__(self, root='data', train=True, transform=None):
        self.basicData = ImageFolder(root=root, transform=transform)
        self.transform = transform
        # Shuffle indices for all images
        indices = list(range(len(self.basicData)))
        random.shuffle(indices)

        # Calculate split sizes
        numImages = len(self.basicData)
        trainEnd = int(0.7 * numImages)  # 70% for training
        valEnd = trainEnd + int(0.15 * numImages)  # 15% for validation, rest for testing

        # Split indices
        train_indices = indices[:trainEnd]
        val_indices = indices[trainEnd:valEnd]
        test_indices = indices[valEnd:]

        # Assign images to datasets
        self.trainData = [self.basicData[i] for i in train_indices]
        self.valData = [self.basicData[i] for i in val_indices]
        self.testData = [self.basicData[i] for i in test_indices]

        # By default, use training set
        self.data = self.trainData if train else self.testData

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
    
    @property
    def classes(self):
        return self.data.classes
    
    def getTrainingItem(self, idx):
        return self.trainData[idx]
    
    def getValidationItem(self, idx):
        return self.valData[idx]
    
    def getTestingItem(self, idx):
        return self.testData[idx]