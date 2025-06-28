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
        self.dataset = dataset
        self.device = device
        self.brain = CNN_Brain.CNN(1).to(device)
        self.two_brain = CNN_Brain.CNN(2).to(device)
        self.three_brain = CNN_Brain.CNN(3).to(device)
        self.four_brain = CNN_Brain.CNN(4).to(device)
        self.five_brain = CNN_Brain.CNN(5).to(device)
        self.optimizer = torch.optim.SGD(self.brain.parameters(), lr=0.001, momentum=0.9)
        self.loss_function = loss_function
        
    def epoch_train(self, number_of_colors: int = 1):
        total_loss = 0.0
        divider = 0

        for i in range(self.dataset.__len__()):
            # Get image tensor and expected result tensor from dataset
            img_tensor, expected_result_list = self.dataset.__getitem__(i)

            ready_expected_result = expected_result_list[number_of_colors-1]

            if img_tensor != None and ready_expected_result != [0,0,0]:
                
                # for i in range(10):
                self.optimizer.zero_grad()
                cnn_result = self.brain(img_tensor)
                # if cnn_result.shape == ready_expected_result.view(1, -1).shape:
                    # reshaped = ready_expected_result.view(1, -1)
                loss = self.loss_function(cnn_result, ready_expected_result)
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
            loss_for_epoch = self.epoch_train(number_of_colors=1)
            self.brain.eval()
            torch.no_grad()
            print("Epoch #{}. Loss: {}".format(i+1,loss_for_epoch))
            if loss_for_epoch < best_vloss:
                best_vloss = loss_for_epoch
                brain_filename = 'models/filtered_brain_color_{}_{}.pt'.format(1, (i+1))
                if  os.path.exists(brain_filename):
                    os.remove(brain_filename)
                        
                torch.save(self.brain.state_dict(), brain_filename)
        best_vloss =  1_000_000.
        for i in range(epochs_num):
            self.two_brain.train(True)
            loss_for_epoch = self.epoch_train(2)
            self.two_brain.eval()
            torch.no_grad()
            print("Epoch #{}. Loss: {}".format(i+1,loss_for_epoch))
            if loss_for_epoch < best_vloss:
                best_vloss = loss_for_epoch
                brain_filename = 'models/filtered_brain_color_{}_{}.pt'.format(2, (i+1))
                if  os.path.exists(brain_filename):
                    os.remove(brain_filename)
                        
                torch.save(self.two_brain.state_dict(), brain_filename)
        best_vloss =  1_000_000.
        for i in range(epochs_num):
            self.three_brain.train(True)
            loss_for_epoch = self.epoch_train(3)
            self.three_brain.eval()
            torch.no_grad()
            print("Epoch #{}. Loss: {}".format(i+1,loss_for_epoch))
            if loss_for_epoch < best_vloss:
                best_vloss = loss_for_epoch
                brain_filename = 'models/filtered_brain_color_{}_{}.pt'.format(3, (i+1))
                if  os.path.exists(brain_filename):
                    os.remove(brain_filename)
                        
                torch.save(self.three_brain.state_dict(), brain_filename)
        best_vloss =  1_000_000.
        for i in range(epochs_num):
            self.four_brain.train(True)
            loss_for_epoch = self.epoch_train(4)
            self.four_brain.eval()
            torch.no_grad()
            print("Epoch #{}. Loss: {}".format(i+1,loss_for_epoch))
            if loss_for_epoch < best_vloss:
                best_vloss = loss_for_epoch
                brain_filename = 'models/filtered_brain_color_{}_{}.pt'.format(4, (i+1))
                if  os.path.exists(brain_filename):
                    os.remove(brain_filename)
                        
                torch.save(self.four_brain.state_dict(), brain_filename)
        best_vloss =  1_000_000.
        for i in range(epochs_num):
            self.five_brain.train(True)
            loss_for_epoch = self.epoch_train(5)
            self.five_brain.eval()
            torch.no_grad()
            print("Epoch #{}. Loss: {}".format(i+1,loss_for_epoch))
            if loss_for_epoch < best_vloss:
                best_vloss = loss_for_epoch
                brain_filename = 'models/filtered_brain_color_{}_{}.pt'.format(5, (i+1))
                if  os.path.exists(brain_filename):
                    os.remove(brain_filename)
                        
                torch.save(self.five_brain.state_dict(), brain_filename)
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