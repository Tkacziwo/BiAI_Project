import math
import math
import os
import torch
import CNN as CNN_Brain

from torch import nn
from Z_Score_Handler import Z_Score_Handler as z_score_handler
from Z_Score_Handler import Z_Score_Handler as z_score_handler

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
                brain_filename = 'models/single_image_brain_{}_{}.pt'.format("first", (i+1))
                if  os.path.exists(brain_filename):
                    os.remove(brain_filename)
                    
                torch.save(self.brain.state_dict(), brain_filename)
        return
    
    def get_model(self):
        return self.brain
    
class MultipleImageBrainTrainer():
    def __init__(self, 
                 images_with_assigned_colors,
                 device: str, 
                 loss_function: nn.MSELoss):
        
        self.images_with_assigned_colors = images_with_assigned_colors
        self.brain = CNN_Brain.CNN().to(device)
        self.optimizer = torch.optim.SGD(self.brain.parameters(), lr=0.001, momentum=0.9)
        self.loss_function = loss_function
        
    def epoch_train(self, index: int):
        total_loss = 0.0
        for image in self.images_with_assigned_colors:
            z_handler = z_score_handler(image)
            unsqueezed_filtered_result = torch.unsqueeze(z_handler.get_filtered_averaged_result(), 0)
            image_tensor = image.get_image_tensor()
            for i in range(10):
                self.optimizer.zero_grad()
                cnn_result = self.brain(image_tensor)
                loss = self.loss_function(cnn_result, unsqueezed_filtered_result)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
        return total_loss / (len(self.images_with_assigned_colors)*10)
        
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
                brain_filename = 'models/multiple_image_brain_{}_{}.pt'.format("first", (i+1))
                if  os.path.exists(brain_filename):
                    os.remove(brain_filename)
                    
                torch.save(self.brain.state_dict(), brain_filename)
        return
    
    def get_model(self):
        return self.brain

class SingleImageMultipleResultsBrainTrainer():
    def __init__(self, 
                image_tensor: torch.tensor, 
                expected_result_list: torch.tensor, 
                device: str, 
                loss_function: nn.MSELoss):
       
       self.image_tensor = image_tensor
       self.expected_result_list = expected_result_list 
       self.brain = CNN_Brain.CNN().to(device)
       self.optimizer = torch.optim.SGD(self.brain.parameters(), lr=0.001, momentum=0.9)
       self.loss_function = loss_function
    
    def epoch_train(self, index: int):

        for result in self.expected_result_list:
            result = torch.unsqueeze(result, 0)
            
            for i in range(100):
                self.optimizer.zero_grad()


                cnn_result = self.brain(self.image_tensor)
                loss = self.loss_function(cnn_result, result)
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
                brain_filename = 'models/single_image_multiple_results_brain_{}'.format((i+1))
                if  os.path.exists(brain_filename):
                    os.remove(brain_filename)
                    
                torch.save(self.brain.state_dict(), brain_filename)
        return
    
    def get_model(self):
        return self.brain

class SingleImageMultipleResultsBrainTrainer():
    def __init__(self, 
                image_tensor: torch.tensor, 
                expected_result_list: torch.tensor, 
                device: str, 
                loss_function: nn.MSELoss):
       
       self.image_tensor = image_tensor
       self.expected_result_list = expected_result_list 
       self.brain = CNN_Brain.CNN().to(device)
       self.optimizer = torch.optim.SGD(self.brain.parameters(), lr=0.001, momentum=0.9)
       self.loss_function = loss_function
    
    def epoch_train(self, index: int):

        for result in self.expected_result_list:
            result = torch.unsqueeze(result, 0)
            
            for i in range(100):
                self.optimizer.zero_grad()


                cnn_result = self.brain(self.image_tensor)
                loss = self.loss_function(cnn_result, result)
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
                brain_filename = 'models/single_image_multiple_results_brain_{}'.format((i+1))
                if  os.path.exists(brain_filename):
                    os.remove(brain_filename)
                    
                torch.save(self.brain.state_dict(), brain_filename)
        return
    
    def get_model(self):
        return self.brain