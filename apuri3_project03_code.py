# -*- coding: utf-8 -*-
"""apuri3_project03_code.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12FvQZCrn9Uokz4XsOMnv_UJmGPkE7-kH
"""

from google.colab import drive
drive.mount('/content/drive')

from __future__ import print_function

import os
import torch
from PIL import Image

import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models
import torch.utils.data
import torchvision.datasets

from torch.autograd import Variable

import copy    
import time
import numpy as np
import os

"""LOAD IMAGEs"""

TRAIN_PATH = '/content/drive/My Drive/Data'
loader1=transforms.ToTensor()# if not normalize then in range[0,1]
#normalization for simple feedforward neural network
loader2=transforms.Compose(
    [transforms.ToTensor(),#convert an image to tensor
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#without normalization for LeNet5
loader3=transforms.Compose(
        [transforms.Resize((32,32)),
         transforms.ToTensor()])

#normalization for LeNet5
loader4=transforms.Compose(
        [transforms.Resize((32,32)),
         transforms.ToTensor(),
         transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

def load_images_flow(batch_size,which_trans):
    if(which_trans == 1):
         transform = loader1
    if(which_trans == 2):
         transform = loader2
    if(which_trans == 3):
         transform = loader3
    if(which_trans == 4):
         transform = loader4

    train_set = torchvision.datasets.ImageFolder(root= TRAIN_PATH, transform=transform)
    VALID_RATIO = 0.9 # 90%- Train, 10%- Validation
    n_train_examples = int(len(train_set) * VALID_RATIO)
    print("Training Examples - ",n_train_examples)
    n_valid_examples = len(train_set) - n_train_examples
    print("Validation Examples - ",n_valid_examples)
    train_data, valid_data = data.random_split(train_set, 
                                            [n_train_examples, n_valid_examples])
    valid_data = copy.deepcopy(valid_data)
    valid_data.dataset.transform = transform

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True, num_workers=2)    
    return train_loader,valid_loader

"""OPTIMIZER"""

def createLossAndOptimizer(net, learning_rate = 0.001, weight_decay = 0, loss_method = "SGD"):#weight_decay: tuning parameter of L2 term
     #Loss function
    loss = torch.nn.CrossEntropyLoss()
    
    #Optimizer
    if loss_method == "SGD":
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=weight_decay) 
    if loss_method == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay) 
    return(loss, optimizer)

"""Feedforward Neural Network and LeNet5"""

input_size = 3*60*60; hidden_size = 50; output_size = 2   
class Two_Layers_NN(torch.nn.Module):
    
    def __init__(self):
        super(Two_Layers_NN,self).__init__()
       
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        return out
    

     

class LeNet(torch.nn.Module):
    
    def __init__(self):
        super(LeNet,self).__init__()
        
        self.conv1 = torch.nn.Conv2d(3, 6, kernel_size=(5,5), stride=1)#input 3 channels, output 6 channels
        
        self.maxpool = nn.MaxPool2d(kernel_size=(2,2),stride=2)
        # torch.nn.AdaptiveAvgPool2d() can avoid overfitting
        self.conv2 = torch.nn.Conv2d(6, 16, kernel_size=(5,5), stride=1)#output may not be the times of input
        
        
        self.fc1 = torch.nn.Linear(5*5*16,120)
        
        self.fc2 = torch.nn.Linear(120,84)
        
        self.fc3 = torch.nn.Linear(84,2)

  
    def forward(self, x):
        out = self.conv1(x) #input 32*32*3 output 28*28*6
       
        #x = self.batchnorm1(x)
        
        out = self.maxpool(out) #output 14*14*6
    
        out = self.conv2(out) #output 10*10*16
 
        out = self.maxpool(out) #output 5*5*16
  
        out = out.view(-1, 5 * 5 * 16)#flatten
   
        out = self.fc1(out)
  
        out = self.fc2(out)
      
        out = self.fc3(out)
        return(out)

"""TRAINING FUNCTION"""

def train_net(net, loss_method, which_model, whether_norm, batch_size, n_epochs, learning_rate, weight_decay, print_train_process = True, print_test_process= True, validation = True):
    assert(which_model == "NN" or which_model == "LeNet")# the input should be reasonable
    
    #choose right transformation
    if (whether_norm == False and which_model == "NN"):
        which_trans = 1
    elif (whether_norm == True and which_model == "NN"):
        which_trans = 2
    elif (whether_norm == False and which_model == "LeNet"):
        which_trans = 3
    elif (whether_norm == True and which_model == "LeNet"):
        which_trans = 4
        
    print(which_trans)
        
    #Get training data and test data
    train_loader,test_loader = load_images_flow(batch_size,which_trans)
     
    
    n_batches = len(train_loader)
    loss, optimizer =  createLossAndOptimizer(net, learning_rate = learning_rate, weight_decay = weight_decay,loss_method=loss_method)
    
    #Time for printing
    start_time = time.time()
    print_every = n_batches // 10
        
    for epoch in range(n_epochs):
        #epoch = 0
        running_loss = 0
        running_correct_num = 0
        
        total_train_loss = 0
        total_train_num = 0
        total_correct_train_num = 0
                
        for i, data in enumerate(train_loader): # handle every batch_size pictures
             
            (inputs,labels) = data
            
            if(which_model == "NN"):
                 inputs = inputs.view(inputs.size()[0],3*60*60)#flatten
                 #inputs = inputs.view(batch_size,3*60*60) not batch size since 
            
            inputs, labels = Variable(inputs), Variable(labels)
            
            optimizer.zero_grad() # whether zero setting is okay ?
            #print(inputs.size())
            #Forward pass, backward pass, optimize
            outputs = net(inputs) #why ? same as forward
            
            m, predicted = torch.max(outputs.data,1)
            total_train_num += labels.size(0)
            running_correct_num += (predicted == labels).sum().item()
            total_correct_train_num += (predicted == labels).sum().item()
            
            loss_size = loss(outputs, labels)
            if np.isnan(loss_size.data):
                raise ValueError("loss explode due to large regularization or learning rate")

            loss_size.backward()
            optimizer.step()       
            
            #print statistics
            running_loss += loss_size.data
            total_train_loss += loss_size.data
              
            #print every 10th batch
            if(print_train_process == True):
                  if (i+1) % (print_every) == 0:
                        print("Epoch {}, {:d}% \t train_loss: {:.4f} train_accuracy:{:d}% took: {:.2f}s".format(
                             epoch+1, int(100*(i+1)/n_batches), running_loss/print_every/batch_size, int(100 * running_correct_num /print_every/batch_size),
                             time.time()-start_time)) # loss for currect running_loss and running_correct_num not accumulated ones
                        #reset running loss and time
                        running_loss = 0.0
                        running_correct_num = 0.0
                        start_time = time.time()
                        
                                                
        #For validation
        if(validation == True):       
            total_test_loss = 0
            total_test_num = 0
            correct_test_num = 0
            for inputs, labels in test_loader:  
                if (which_model == "NN"):
                    inputs = inputs.view(inputs.size()[0], 3*60*60)
                    inputs, labels = Variable(inputs), Variable(labels)
        
                #Forward pass
                test_outputs = net(inputs)
        
                #test accuracy rate
                m, predicted = torch.max(test_outputs.data, 1)
                #print(predicted)
                total_test_num += labels.size(0)
                
                correct_test_num += (predicted == labels).sum().item()
                test_loss_size=loss(test_outputs, labels)
                total_test_loss += test_loss_size.data
                                
            if(print_test_process == True):        
                print("Test loss = {:.4f} Test Accuracy = {:d}%".format(total_test_loss / len(test_loader), 
                       int(100 * correct_test_num / total_test_num)))
        #print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct_test_num / total_test_num))
            elif(epoch == n_epochs-1):
                 print("Test loss = {:.4f} Test Accuracy = {:d}%".format(total_test_loss / len(test_loader), 
                       int(100 * correct_test_num / total_test_num)))
            
            #print("Training finished, took {:.2f}s".format(time.time() - start_time))
           
    if(not print_train_process == True):
          print("train_loss: {:.8f} train_accuracy:{:d}% learning rate:{:.8f} regularization:{:.8f} running time:{:.4f}" .format(running_loss/total_train_num, int(100 * total_correct_train_num / total_train_num),
           learning_rate, weight_decay, time.time()-start_time))

"""First We do for FEED Forward Neural Network and then LeNet5

Compare the Results with and without normalization
"""

weight_decay = 0; learning_rate = 0.001
print("weight decay:{:.4f} learning rate:{:.4f}".format(weight_decay, learning_rate))
NN = Two_Layers_NN()
train_net(NN, which_model = "NN", whether_norm = False, batch_size=5, n_epochs=1, learning_rate = learning_rate, 
              weight_decay = weight_decay, print_train_process = True, print_test_process = False, validation = False,loss_method = "SGD")
del NN

"""Without normalization, optimization converge slow.

Sanity Check : 

Varying Regularisation and keeping learning rate same
"""

import random

weight_decay = 0; learning_rate = 0.001
print("weight decay:{:.4f} learning rate:{:.4f}".format(weight_decay, learning_rate))
NN = Two_Layers_NN()
train_net(NN, which_model = "NN", whether_norm = True, batch_size=5, n_epochs=1, learning_rate = learning_rate, 
              weight_decay = weight_decay, print_train_process = True, print_test_process = False, validation = False,loss_method = "SGD")
    
del NN


weight_decay = 1000; learning_rate = 0.001
print("weight decay:{:.4f} learning rate:{:.4f}".format(weight_decay, learning_rate))
NN = Two_Layers_NN()
train_net(NN, which_model = "NN", whether_norm = True, batch_size=5, n_epochs=1, learning_rate = learning_rate, 
              weight_decay = weight_decay, print_train_process = True, print_test_process = False, validation = False, loss_method = "SGD")
    
del NN

weight_decay = 10000; learning_rate = 0.001
print("weight decay:{:.4f} learning rate:{:.4f}".format(weight_decay, learning_rate))
NN = Two_Layers_NN()
train_net(NN, which_model = "NN", whether_norm = True, batch_size=5, n_epochs=1, learning_rate = learning_rate, 
              weight_decay = weight_decay, print_train_process = True, print_test_process = False, validation = False, loss_method = "SGD")
    
del NN
#comments:  when regularization increase, loss goes up and when regularization is too large, loss explode

"""Case 1: As seen, when we increase the weight decay / regularization from 0 to 1000, Loss also increases.

Case 2: when Regularization is 10000, Loss EXPLODES.

Now, Varying the Learning Rate and keeping the Regularisation constant at 0.001
"""

weight_decay = 0.001; learning_rate = 0.000001
print("weight decay:{:.4f} learning rate:{:.8f}".format(weight_decay, learning_rate))
NN = Two_Layers_NN()
train_net(NN, which_model = "NN", whether_norm = True, batch_size=5, n_epochs=1, learning_rate = learning_rate, 
              weight_decay = weight_decay, print_train_process = True, print_test_process = False, validation = False, loss_method = "SGD")
del NN

weight_decay = 0.001; learning_rate = 0.001
print("weight decay:{:.4f} learning rate:{:.8f}".format(weight_decay, learning_rate))
NN = Two_Layers_NN()
train_net(NN, which_model = "NN", whether_norm = True, batch_size=5, n_epochs=1, learning_rate = learning_rate, 
              weight_decay = weight_decay, print_train_process = True, print_test_process = False, validation = False, loss_method = "SGD")
    
del NN

weight_decay = 0.001; learning_rate = 0.1
print("weight decay:{:.4f} learning rate:{:.8f}".format(weight_decay, learning_rate))
NN = Two_Layers_NN()
train_net(NN, which_model = "NN", whether_norm = True, batch_size=5, n_epochs=1, learning_rate = learning_rate, 
              weight_decay = weight_decay, print_train_process = True, print_test_process = False, validation = False, loss_method = "SGD")
    
del NN

"""Case 1: When learning rate is 0.000001(too small), Loss barely changes.

Case 2: When learning rate is 0.001, Loss changes reasonably.

Case 3: When learning rate is 0.1, Loss EXPLODES.

After this, we will do 

Hyperparameter optimization (random search)
"""

for count in range(10):
    learning_rate = 10 ** random.uniform(-6,-1)
    weight_decay = 10 ** random.uniform(-6,0)
    
    NN = Two_Layers_NN()
    
    try:
        train_net(NN, which_model = "NN", whether_norm = True, batch_size=5, n_epochs=5, learning_rate = learning_rate, 
              weight_decay = weight_decay, print_train_process = False, print_test_process = False, validation = True,loss_method = "SGD")
    except:
        print("loss explodes. learning rate:{:.8f} regularization:{:.8f}".format(learning_rate, weight_decay))
        
    del NN

"""When Learning Rate: 0.03080337 and Regularization:0.00443945,

 then train loss is the smallest and accuracy is largest
"""

learning_rate = 0.03080337; weight_decay =0.00443945
NN = Two_Layers_NN()
train_net(NN, which_model = "NN", whether_norm = True, batch_size=5, n_epochs=10, learning_rate = learning_rate, 
              weight_decay = weight_decay, print_train_process = True, print_test_process = True, validation = True,loss_method = "SGD")
del NN

"""Sanity Check to LENET5"""

#sanity check
weight_decay = 0; learning_rate = 0.001
print("weight decay:{:.4f} learning rate:{:.8f}".format(weight_decay, learning_rate))
LN = LeNet()
train_net(LN, which_model = "LeNet", whether_norm = True, batch_size=5, n_epochs=1, learning_rate = learning_rate, 
              weight_decay = weight_decay, print_train_process = True, print_test_process = False, validation = False,loss_method = "SGD")
    
del LN

weight_decay = 0.001; learning_rate = 0.001
print("weight decay:{:.4f} learning rate:{:.8f}".format(weight_decay, learning_rate))
LN = LeNet()
train_net(LN, which_model = "LeNet", whether_norm = True, batch_size=5, n_epochs=1, learning_rate = learning_rate, 
              weight_decay = weight_decay, print_train_process = True, print_test_process = False, validation = False,loss_method = "SGD")
    
del LN

weight_decay = 10000; learning_rate = 0.001
print("weight decay:{:.4f} learning rate:{:.8f}".format(weight_decay, learning_rate))
LN = LeNet()
train_net(LN, which_model = "LeNet", whether_norm = True, batch_size=5, n_epochs=1, learning_rate = learning_rate, 
              weight_decay = weight_decay, print_train_process = True, print_test_process = False, validation = False,loss_method = "SGD")
    
del LN

"""Case 1: As Regularization increases, loss also increases
Case 2: When regularization is too large i.e. 10000, LOSS EXPLODES.
"""

weight_decay = 0.001; learning_rate = 0.00001
print("weight decay:{:.4f} learning rate:{:.8f}".format(weight_decay, learning_rate))
LN = LeNet()
train_net(LN, which_model = "LeNet", whether_norm = True, batch_size=5, n_epochs=1, learning_rate = learning_rate, 
              weight_decay = weight_decay, print_train_process = True, print_test_process = False, validation = False,loss_method = "SGD")
    
del LN

weight_decay = 0.001; learning_rate = 0.001
print("weight decay:{:.4f} learning rate:{:.8f}".format(weight_decay, learning_rate))
LN = LeNet()
train_net(LN, which_model = "LeNet", whether_norm = True, batch_size=5, n_epochs=1, learning_rate = learning_rate, 
              weight_decay = weight_decay, print_train_process = True, print_test_process = False, validation = False,loss_method = "SGD")
    
del LN

weight_decay = 0.001; learning_rate = 0.125
print("weight decay:{:.4f} learning rate:{:.8f}".format(weight_decay, learning_rate))
LN = LeNet()
train_net(LN, which_model = "LeNet", whether_norm = True, batch_size=5, n_epochs=1, learning_rate = learning_rate, 
              weight_decay = weight_decay, print_train_process = True, print_test_process = False, validation = False,loss_method = "SGD")
    
del LN

"""Case 1: When learning rate is too small (0.00001), Loss barely changes.

Case 2: When learning rate is 0.001, Loss changes reasonably.

Case 3: When learning rate is too large 0.125 here, loss explodes.

Hyperparameter Optimization(random search) LENET5
"""

for count in range(10):
    learning_rate = 10 ** random.uniform(-5,-2)
    weight_decay = 10 ** random.uniform(-6,-1)
    
    LN = LeNet()
    
    try:
        train_net(LN, which_model = "LeNet", whether_norm = True, batch_size=5, n_epochs=5, learning_rate = learning_rate, 
              weight_decay = weight_decay, print_train_process = False, print_test_process = False, validation = True,loss_method = "SGD")
    except:
        print("loss explodes. learning rate:{:.8f} regularization:{:.8f}".format(learning_rate, weight_decay))
        
    del LN

"""Choose the pair with highest accuracy rate and smallest loss on LENET5

Learning Rate : 0.00515260  and Regularization : 0.00002367
"""

learning_rate = 0.00515260; weight_decay = 0.00002367
LN = LeNet()
train_net(LN, which_model = "LeNet", whether_norm = True, batch_size=5, n_epochs=10, learning_rate = learning_rate, 
              weight_decay = weight_decay, print_train_process = True, print_test_process = True, validation = True,loss_method = "SGD")
del LN