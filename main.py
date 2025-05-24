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

#loading of expected results
expectedResult = open("expected-results/AdrianR_2025_03_20-13_05_22.txt")
expectedResultTextLine = expectedResult.readline()
position = expectedResultTextLine.find('#') + 1
rgb = expectedResultTextLine[position:(len(expectedResultTextLine)-1):1]
tensor_rgb = rgb_to_tensor(rgb)

#declare loss function
loss_fn = torch.nn.MSELoss()

#training loop

# brain = CNN.CNN().to(device)
# optimizer = torch.optim.SGD(brain.parameters(), lr=0.001, momentum=0.9)


# def epoch_train(epoch_index: int):
#     expected_result_rgb = tensor_rgb
#     expected_result_rgb = torch.unsqueeze(expected_result_rgb, 0)
    
#     #for i, data in enumerate(training_loader):
        
    
#     for i in range(100):
#         optimizer.zero_grad()
#         cnn_result = brain(tensor)
#         loss = loss_fn(cnn_result, expected_result_rgb)
#         loss.backward()
#         optimizer.step()
#     return loss

# # image_loader = ImageLoader.ImageLoader()
# # training_set = image_loader.get_train_loader()
# # validation_set = image_loader.get_test_loader()
# # training_loader = torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=True)
# # validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=4, shuffle=False)


# best_vloss = 1_000_000.

# for i in range(2):
#     print("Epoch: " + str((i+1)))
#     brain.train(True)
#     loss_for_epoch = epoch_train(i)
    
#     brain.eval()
#     torch.no_grad()
#     avg_loss = loss_for_epoch / (i+1)
#     print("Loss: {}".format(avg_loss))
    
#     if avg_loss < best_vloss:
#         best_vloss = avg_loss
#         brain_filename = 'models/brain_{}_{}'.format("first", (i+1))
#         if  os.path.exists(brain_filename):
#             os.remove(brain_filename)
            
#         torch.save(brain.state_dict(), brain_filename)

#train brain for 10 epochs
single_image_brain_trainer = trainer.SingleImageBrainTrainer(tensor,tensor_rgb,device,loss_fn)
single_image_brain_trainer.train_brain(10)

trained_single_image_brain = single_image_brain_trainer.get_model()

trained_single_image_brain.eval()
trained_output = trained_single_image_brain(tensor)
print("trained output: {}".format(trained_output))
print("correct result: {}".format(tensor_rgb))