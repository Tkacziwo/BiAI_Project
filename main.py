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
brainModels = []
for modelName in os.listdir("models"):
    brain = CNN.CNN()
    path = "models/"+modelName
    brain.load_state_dict(torch.load(path))
    brainModels.append(brain)


if brainModels.__len__() == 0:
    print("No models found")
    image_tensor_loader = im()
    image_tensor_loader.load_input_photos_to_tensor("input-photos")

    loaded_images_tensors = image_tensor_loader.get_input_photo_tensors()
    if loaded_images_tensors.count == 0:
        print("Error while loading images")
        exit()
    else:
        print("Loaded all images!")



    print("Training brain... this may take a while...")

    #Using data filter to load expected results
    filter = DataFilter.DataFilter()
    color_dataset = icd("input-photos", filter.getDictionary())

    directoryName = "expected-results"
    photos_with_assigned_colors = []

    for photo_name in image_tensor_loader.get_input_photo_file_names():
        colors_for_image = cfi()
        colorList = []
        for name in os.listdir(directoryName):
            if not name.endswith('Time.txt'):
                filePath = os.path.join(directoryName, name)
                filter.loadColorAnnotations(filePath)

                oneColor = filter.getOneColorAnnotation(photo_name)
                string = oneColor[0].lstrip('#')
                oneColorTensor = rgb_to_tensor(string)
                colorList.append(oneColorTensor)

        img = iio.imread("input-photos/"+photo_name)
        transform = transforms.Compose([transforms.ToTensor()])
        image_tensor = transform(img)
        image_tensor = torch.unsqueeze(image_tensor, 0)

        colors_for_image.set_data(photo_name, colorList, image_tensor)
        photos_with_assigned_colors.append(colors_for_image)


    #declare loss function
    loss_fn = torch.nn.MSELoss()

    #train brain for 5 epochs
    brain = trainer.MultipleImageBrainTrainer(photos_with_assigned_colors, device, loss_fn)
    brain.train_brain(5)
    print("Brain trained. Run application again to verify model")
else:
    image_name = "000000034221.jpg"
    img = iio.imread("input-photos/000000034221.jpg")
    transform = transforms.Compose([transforms.ToTensor()])
    tensor = transform(img)
    tensor = torch.unsqueeze(tensor, 0)
    color_list = []
    filter = DataFilter.DataFilter()
    for name in os.listdir("expected-results"):
            if not name.endswith('Time.txt'):
                filePath = os.path.join("expected-results", name)
                filter.loadColorAnnotations(filePath)

                oneColor = filter.getOneColorAnnotation(image_name)
                string = oneColor[0].lstrip('#')
                oneColorTensor = rgb_to_tensor(string)
                color_list.append(oneColorTensor)

    test_colors_for_image = cfi()
    test_colors_for_image.set_data(image_name, color_list, tensor)
    z_handler = z_score_handler(test_colors_for_image)
    expected_result = z_handler.get_filtered_averaged_result()

    #test
    for brain in brainModels:
        trained_brain = brain
        trained_brain.eval()
        trained_output = trained_brain(tensor)
        print("Trained output: {}. Correct result: {}".format(trained_output, expected_result))