'''
this script is for the training code of Project 2. It should be similar as the one in Project 1.

-------------------------------------------
INTRO:
You should write your codes or modify codes between the 
two '#####' lines. The codes between two '=======' lines 
are used for logging or printing, please keep them unchanged 
in your submission. 

You need to debug and run your codes in somewhere else (e.g. Jupyter 
Notebook). This file is only used for the evaluation stage and for
your submission and marks. Please make sure the codes are running 
smoothly using your trained model.

-------------------------------------------
USAGE:
In your final update, please keep the file name as 'python2_test.py'.

>> python project2_test.py
This will run the program on CPU to test on your trained nets for the Fruit test dataset in Task 1.

>> python project2_test.py --cuda
This will run the program on GPU to test on your trained nets for the Fruit test dataset in Task 1. 
You can ignore this if you do not have GPU or CUDA installed.

>> python project2_test.py --da
This will run the program on CPU to test your domain adaptive model for the Office target dataset in Task 2.

>> python project2_test.py --da --cuda
This will run the program on GPU to test your domain adaptive model for the Office target dataset in Task 2.

-------------------------------------------
NOTE:
this file might be incomplete, feel free to contact us
if you found any bugs or any stuff should be improved.
Thanks :)

Email: wzha8158@uni.sydney.edu.au, dzho8854@uni.sydney.edu.au
'''

# import the packages
import argparse
import logging
import sys
import time
import os

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import numpy as np
import matplotlib.pyplot as plt

from network import Network # the network you used


# training process. 
def train_net(net, trainloader, valloader,criterion,optimizer,scheduler,epochs=1):
########## ToDo: Your codes goes below #######
    net = net.train()
    train_set_num = []
    vali_set_num = []
    vali_accuracy = []
    train_loss_set = []
    vali_loss_set = []
    for epoch in range(epochs):
        if type(scheduler).__name__ != 'NoneType':
            scheduler.step()

        train_running_loss = 0.0
        vali_running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss = loss.cpu()

        # print statistics
            train_running_loss += loss.item()
            if i % 100 == 99:    # print every 2000 mini-batches
                train_loss_set.append(train_running_loss/100)
                train_set_num.append(len(train_loss_set))
                print('[%d, %5d] train_loss: %.3f' %
                  (epoch + 1, i + 1, train_running_loss / 100))
                train_running_loss = 0.0
        for i, data in enumerate(valloader, 0):
        # get the inputs
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
        # zero the parameter gradients
            optimizer.zero_grad()

        # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        # print statistics
            loss = loss.cpu()
            vali_running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if i % 20 == 19:    # print every 2000 mini-batches
                vali_loss_set.append(vali_running_loss/20)
                vali_set_num.append(len(vali_loss_set))
                vali_accuracy.append(correct / total)
                print('[%d, %5d] validation_loss: %.3f' %
                  (epoch + 1, i + 1, vali_running_loss / 20))
                print('Accuracy of validation: %d %%' % (
                     100 * correct / total))
                vail_running_loss = 0.0

    print('Finished Training')
    val_accuracy = 0
    # val_accuracy is the validation accuracy of each epoch. You can save your model base on the best validation accuracy.

    return val_accuracy
##############################################

############################################
# Transformation definition
# NOTE:
# Write the train_transform here. We recommend you use
# Normalization, RandomCrop, Resize and any other transform you think is useful. 
# Remember to make the normalize value same as in the training transformation.

train_transform = transforms.Compose([
    transforms.Resize((224,224),interpolation = 1),
    transforms.RandomCrop(224),
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])




####################################

####################################
# Define the training dataset and dataloader.
# You can make some modifications, e.g. batch_size, adding other hyperparameters, etc.

train_image_path = 'D:/project2/5307Project2/train' 
#validation_image_path = '../validation/' 

trainset = ImageFolder(train_image_path, train_transform)

train_set, val_set = torch.utils.data.random_split(trainset, [1000-64, 151])

trainloader = torch.utils.data.DataLoader(train_set, batch_size=2,
                                         shuffle=True, num_workers=2)


valloader = torch.utils.data.DataLoader(val_set, batch_size=2,
                                         shuffle=True, num_workers=2)
####################################



# ==================================
# use cuda if called with '--cuda'. 
# DO NOT CHANGE THIS PART.

network = Network()
network = network.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(network.parameters(), lr=0.005, momentum=0.80) # adjust optimizer settings
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.90)

# train and eval your trained network
# you have to define your own 
if __name__ == '__main__':
    data = next(iter(trainloader))
    print(data[0].mean())
    print(data[0].std())
    print("print 1")
    val_acc = train_net(network, trainloader, valloader,criterion,optimizer,scheduler)

    print("final validation accuracy:", val_acc)

# ==================================