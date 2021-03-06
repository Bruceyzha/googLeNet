# import the packages
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

def parse_args():
    parser = argparse.ArgumentParser(description= \
        'scipt for part 3 of project 1')
    parser.add_argument('--cuda', action='store_true', default=False,
        help='Used when there are cuda installed.')
    

    pargs = parser.parse_args()
    return pargs

# training process. 
def train_net(net, trainloader, valloader,criterion,optimizer,scheduler,epochs=2):
########## ToDo: Your codes goes below #######
    net = net.train()
    for epoch in range(epochs):
        scheduler.step()

        train_running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            if args.cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if args.cuda:
                loss = loss.cpu()

        # print statistics
            train_running_loss += loss.item()
            if i % 50 == 49:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, train_running_loss / 2000))
                train_running_loss = 0.0
        

        for data in valloader:
            inputs, labels = data
            if args.cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        
        print('Accuracy of validation: %d %%' % (
                     100 * correct / total))
        val_accuracy = 100 * correct / total
    print('Finished Training')
    
    # val_accuracy is the validation accuracy of each epoch. You can save your model base on the best validation accuracy.

    return val_accuracy
##############################################

############################################
# Transformation definition
# NOTE:
# Write the train_transform here. We recommend you use
# Normalization, RandomCrop, Resize and any other transform you think is useful. 
# Remember to make the normalize value same as in the training transformation.

args = parse_args()

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

train_image_path = 'D:/project2/5307Project2/train/' 

validation_image_path = 'D:/project2/5307Project2/test/' 

train_set = ImageFolder(train_image_path, train_transform)
val_set = ImageFolder(train_image_path, train_transform)




trainloader = torch.utils.data.DataLoader(train_set, batch_size=4,
                                         shuffle=True, num_workers=2)


valloader = torch.utils.data.DataLoader(val_set, batch_size=4,
                                         shuffle=True, num_workers=2)
####################################



# ==================================
# use cuda if called with '--cuda'. 
# DO NOT CHANGE THIS PART.

network = Network()
if args.cuda:
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
    val_acc = train_net(network, trainloader, valloader,criterion,optimizer,scheduler)

    print("final validation accuracy:", val_acc)

# ==================================