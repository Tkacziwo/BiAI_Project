import os
import torch
import math
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

#z score counting algorithm
red_sum = 0
green_sum = 0 
blue_sum = 0
for col in colorList:
    red_sum += col[0]
    green_sum += col[1]
    blue_sum += col[2]

red_avg = red_sum / len(colorList)
green_avg = green_sum / len(colorList)
blue_avg = blue_sum / len(colorList)

red_sum = 0
green_sum = 0
blue_sum = 0

for col in colorList:
    red_sum += math.pow(col[0] - red_avg, 2)
    green_sum += math.pow(col[1] - green_avg, 2)
    blue_sum += math.pow(col[2] - blue_avg, 2)

red_deviation = math.sqrt(red_sum / len(colorList))
green_deviation = math.sqrt(green_sum / len(colorList))
blue_deviation = math.sqrt(blue_sum / len(colorList))


z_score_red = []
z_score_green = []
z_score_blue = []

for col in colorList:
    z_score_red.append((col[0] - red_avg)/red_deviation)
    z_score_green.append((col[1] - green_avg)/green_deviation)
    z_score_blue.append((col[2] - blue_avg)/blue_deviation)

print("Averages red, green, blue: {}, {}, {}".format(red_avg, green_avg, blue_avg))
print("Deviation red green blue: {}, {}, {}".format(red_deviation, green_deviation, blue_deviation))

print("Z score for red: {}".format(z_score_red))
print("Z score for green: {}".format(z_score_green))
print("Z score for blue: {}".format(z_score_blue))


filtered_color_list = []
filtered_z_score_list = []
for i in range(len(colorList)):
    if z_score_red[i] < 1.5 and z_score_red[i] > -1.5 and z_score_green[i] < 1.5 and z_score_green[i] > -1.5 and z_score_blue[i] < 1.5 and z_score_blue[i] > -1.5:
        filtered_color_list.append(colorList[i])
        filtered_z_score_list.append((z_score_red[i], z_score_green[i], z_score_blue[i]))


print("Filtered colors: {}".format(filtered_color_list))
print("Filtered z_score: {}".format(filtered_z_score_list))

#average expected result
averaged_expected_result_tensor = sum(filtered_color_list) / len(filtered_color_list)

#loading of expected results
# expectedResult = open("expected-results/AdrianR_2025_03_20-13_05_22.txt")
# expectedResultTextLine = expectedResult.readline()
# position = expectedResultTextLine.find('#') + 1
# rgb = expectedResultTextLine[position:(len(expectedResultTextLine)-1):1]
# tensor_rgb = rgb_to_tensor(rgb)

#declare loss function
loss_fn = torch.nn.MSELoss()

#train brain for 10 epochs
single_image_brain_trainer = trainer.SingleImageBrainTrainer(tensor,averaged_expected_result_tensor,device,loss_fn)
single_image_brain_trainer.train_brain(10)

trained_single_image_brain = single_image_brain_trainer.get_model()

trained_single_image_brain.eval()
trained_output = trained_single_image_brain(tensor)
print("trained output: {}".format(trained_output))
print("correct result: {}".format(averaged_expected_result_tensor))


# ! works the same as single image brain trainer but worse ! #

# single_image_multiple_results_trainer = trainer.SingleImageMultipleResultsBrainTrainer(tensor, colorList, device, loss_fn)
# single_image_multiple_results_trainer.train_brain(10)

# trained_single_image_multiple_result_brain = single_image_multiple_results_trainer.get_model()

# trained_single_image_multiple_result_brain.eval()
# trained_multiple_result_output = trained_single_image_multiple_result_brain(tensor)
# print("trained multiple result output: {}".format(trained_multiple_result_output))
# print("correct result: {}".format(averaged_expected_result_tensor))