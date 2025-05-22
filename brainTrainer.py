import os
import torch
import CNN as CNN_Brain

from torch import nn

class SingleImageBrainTrainer():
    def __init__(self, 
                 image_tensor: torch.tensor, 
                 expected_result_tensor: torch.tensor, 
                 device: str, 
                 loss_function: nn.MSELoss):
        
        self.image_tensor = image_tensor
        self.expected_result_tensor = expected_result_tensor
        self.brain = CNN_Brain.CNN().to(device)
        self.optimizer = torch.optim.SGD(self.brain.parameters(), lr=0.001, momentum=0.9)
        self.loss_function = loss_function
        
    def epoch_train(self, index: int):
        unsqueezed_expected_result_rgb = torch.unsqueeze(self.expected_result_tensor, 0)
        for i in range(100):
            self.optimizer.zero_grad()
            cnn_result = self.brain(self.image_tensor)
            loss = self.loss_function(cnn_result, unsqueezed_expected_result_rgb)
            loss.backward()
            self.optimizer.step()
        return loss
        
    def train_brain(self, epochs_num: int):
        best_vloss = 1_000_000.
            
        for i in range(epochs_num):
                
            print("Epoch: " + str((i+1)))
            self.brain.train(True)
            loss_for_epoch = self.epoch_train(i)
                
            self.brain.eval()
            torch.no_grad()
            avg_loss = loss_for_epoch / (i+1)
            print("Loss: {}".format(avg_loss))
                
            if avg_loss < best_vloss:
                best_vloss = avg_loss
                brain_filename = 'models/single_image_brain_{}_{}'.format("first", (i+1))
                if  os.path.exists(brain_filename):
                    os.remove(brain_filename)
                    
                torch.save(self.brain.state_dict(), brain_filename)
        return
    
    def get_model(self):
        return self.brain