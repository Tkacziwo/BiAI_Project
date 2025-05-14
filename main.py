import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import neuralNetwork

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

brain = neuralNetwork.neuralNetwork().to(device)
print(brain)

X = torch.rand(1, 28, 28, device=device)
logits = brain(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")
print(f"Value: {y_pred.tolist()}")


logits2 = brain.forward(X)
probability = nn.Softmax(dim=1)(logits2)
mostProbable = probability.argmax(1)
print(f"Predicted class 2: {mostProbable}")
print(f"Value: {mostProbable.tolist()}")