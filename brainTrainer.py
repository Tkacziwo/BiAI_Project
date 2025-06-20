import math
import math
import os
import torch
import CNN as CNN_Brain
from torch import nn
from ImageColorDataSet import ImageColorDataSet

class FilteredBrainTrainer():
    def __init__(self, 
                 dataset: ImageColorDataSet,
                 device: str, 
                 loss_function: nn.MSELoss):
        self.dataset = dataset
        self.brain = CNN_Brain.CNN().to(device)
        self.optimizer = torch.optim.SGD(self.brain.parameters(), lr=0.001, momentum=0.9)
        self.loss_function = loss_function
        
    def epoch_train(self, index: int):
        total_loss = 0.0
        divider = 0

        for i in range(self.dataset.__len__()):
            # Get image tensor and expected result tensor from dataset
            img_tensor, expected_result = self.dataset.__getitem__(i)
            if img_tensor != None and expected_result != [0,0,0]:
                for i in range(10):
                    self.optimizer.zero_grad()
                    cnn_result = self.brain(img_tensor)
                    loss = self.loss_function(cnn_result, expected_result)
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()
                    divider += 1
        
        # Returns total loss divided by number of operations in training
        return total_loss / divider
        
    def train_brain(self, epochs_num: int):
        best_vloss = 1_000_000.
            
        for i in range(epochs_num):
            self.brain.train(True)
            loss_for_epoch = self.epoch_train(i)
            self.brain.eval()
            torch.no_grad()
            print("Epoch #{}. Loss: {}".format(i+1,loss_for_epoch))
            if loss_for_epoch < best_vloss:
                best_vloss = loss_for_epoch
                brain_filename = 'models/filtered_brain_{}_{}.pt'.format("first", (i+1))
                if  os.path.exists(brain_filename):
                    os.remove(brain_filename)
                    
                torch.save(self.brain.state_dict(), brain_filename)
        return
    
    def get_model(self):
        return self.brain