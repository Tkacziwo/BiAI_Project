import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import neuralNetwork
import CNN
import imageio.v2 as iio
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import brainTrainer as trainer
import DataFilter

def rgb_to_tensor(rgb: str):

    red = int(rgb[0:2], 16)
    green = int(rgb[2:4], 16)
    blue = int(rgb[4:6], 16)

    t = torch.tensor([red, green, blue], dtype=torch.float32) / 255.0
    return t

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

#loading of previously saved "good" model if none, train again

#image loading -> to change to data loader
img = iio.imread("input-photos/000000010432.jpg")
transform = transforms.Compose([transforms.ToTensor()])
tensor = transform(img)
tensor = torch.unsqueeze(tensor, 0)

#Using data filter to load expected results
filter = DataFilter.DataFilter()

directoryName = "expected-results"
colorList = []

for name in os.listdir(directoryName):
    if not name.endswith('Time.txt'):
        filePath = os.path.join(directoryName, name)
        filter.loadColorAnnotations(filePath)

        oneColor = filter.getOneColorAnnotation("000000010432.jpg")
        string = oneColor[0].lstrip('#')
        oneColorTensor = rgb_to_tensor(string)
        colorList.append(oneColorTensor)

#colorList has colors for one single photo

#loading of expected results
expectedResult = open("expected-results/AdrianR_2025_03_20-13_05_22.txt")
expectedResultTextLine = expectedResult.readline()
position = expectedResultTextLine.find('#') + 1
rgb = expectedResultTextLine[position:(len(expectedResultTextLine)-1):1]
tensor_rgb = rgb_to_tensor(rgb)


#declare loss function
loss_fn = torch.nn.MSELoss()

#train brain for 10 epochs
single_image_brain_trainer = trainer.SingleImageBrainTrainer(tensor,tensor_rgb,device,loss_fn)
single_image_brain_trainer.train_brain(10)

trained_single_image_brain = single_image_brain_trainer.get_model()

trained_single_image_brain.eval()
trained_output = trained_single_image_brain(tensor)
print("trained output: {}".format(trained_output))
print("correct result: {}".format(tensor_rgb))


single_image_multiple_results_trainer = trainer.SingleImageMultipleResultsBrainTrainer(tensor, colorList, device, loss_fn)
single_image_multiple_results_trainer.train_brain(10)

trained_single_image_multiple_result_brain = single_image_multiple_results_trainer.get_model()

trained_single_image_multiple_result_brain.eval()
trained_multiple_result_output = trained_single_image_multiple_result_brain(tensor)
print("trained multiple result output: {}".format(trained_multiple_result_output))
print("correct result: {}".format(tensor_rgb))