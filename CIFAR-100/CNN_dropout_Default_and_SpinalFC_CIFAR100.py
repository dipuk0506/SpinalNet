# -*- coding: utf-8 -*-
"""
This Script contains the default and Spinal CNN dropout code for CIFAR-100.

This code trains both NNs as two different models.

The code is collected and changed from:
    https://zhenye-na.github.io/2018/09/28/pytorch-cnn-cifar10.html

This code gradually decreases the learning rate to get a good result.

@author: Dipu
"""


import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import random

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 160
learning_rate = 0.001

torch.manual_seed(1)
random.seed(1)

Half_width =2048
layer_width = 256

# Image preprocessing modules
transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()])

# CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform)

testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                           batch_size=200, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=testset,
                                          batch_size=200, 
                                          shuffle=False)

class CNN(nn.Module):
    """CNN."""

    def __init__(self):
        """CNN Builder."""
        super(CNN, self).__init__()

        self.conv_layer = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 100)
        )


    def forward(self, x):
        """Perform forward."""
        
        # conv layers
        x = self.conv_layer(x)
        
        # flatten
        x = x.view(x.size(0), -1)
        
        # fc layer
        x = self.fc_layer(x)

        return x

# 3x3 convolution
class SpinalCNN(nn.Module):
    """CNN."""

    def __init__(self):
        """CNN Builder."""
        super(SpinalCNN, self).__init__()

        self.conv_layer = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.fc_spinal_layer1 = nn.Sequential(
            nn.Dropout(p=0.1), nn.Linear(Half_width, layer_width),
            nn.ReLU(inplace=True),
            )
        self.fc_spinal_layer2 = nn.Sequential(
            nn.Dropout(p=0.1), nn.Linear(Half_width + layer_width, layer_width),
            nn.ReLU(inplace=True),
            )
        self.fc_spinal_layer3 = nn.Sequential(
            nn.Dropout(p=0.1), nn.Linear(Half_width + layer_width, layer_width),
            nn.ReLU(inplace=True),
            )
        self.fc_spinal_layer4 = nn.Sequential(
            nn.Dropout(p=0.1), nn.Linear(Half_width + layer_width, layer_width),
            nn.ReLU(inplace=True),
            )
        self.fc_out = nn.Sequential(
            nn.Dropout(p=0.1), nn.Linear(layer_width*4, 100)            
            )


    def forward(self, x):
        """Perform forward."""
        
        # conv layers
        x = self.conv_layer(x)
        
        # flatten
        x = x.view(x.size(0), -1)
        
        x1 = self.fc_spinal_layer1(x[:, 0:Half_width])
        x2 = self.fc_spinal_layer2(torch.cat([ x[:,Half_width:2*Half_width], x1], dim=1))
        x3 = self.fc_spinal_layer3(torch.cat([ x[:,0:Half_width], x2], dim=1))
        x4 = self.fc_spinal_layer4(torch.cat([ x[:,Half_width:2*Half_width], x3], dim=1))
        
        x = torch.cat([x1, x2], dim=1)
        x = torch.cat([x, x3], dim=1)
        x = torch.cat([x, x4], dim=1)
        
        # fc layer
        x = self.fc_out(x)

        return x



# For updating learning rate
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Train the model
total_step = len(train_loader)
curr_lr1 = learning_rate

curr_lr2 = learning_rate



model1 = CNN().to(device)

model2 = SpinalCNN().to(device)



# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer1 = torch.optim.Adam(model1.parameters(), lr=learning_rate)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=learning_rate) 
  
# Train the model
total_step = len(train_loader)

best_accuracy1 = 0
best_accuracy2 =0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model1(images)
        loss1 = criterion(outputs, labels)

        # Backward and optimize
        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()
        
        outputs = model2(images)
        loss2 = criterion(outputs, labels)

        # Backward and optimize
        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()

        if i == 249:
            print ("Ordinary Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                   .format(epoch+1, num_epochs, i+1, total_step, loss1.item()))
            print ("Spinal Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                   .format(epoch+1, num_epochs, i+1, total_step, loss2.item()))


        
    # Test the model
    model1.eval()
    model2.eval()
    with torch.no_grad():
        correct1 = 0
        total1 = 0
        correct2 = 0
        total2 = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            
            outputs = model1(images)
            _, predicted = torch.max(outputs.data, 1)
            total1 += labels.size(0)
            correct1 += (predicted == labels).sum().item()
            
            outputs = model2(images)
            _, predicted = torch.max(outputs.data, 1)
            total2 += labels.size(0)
            correct2 += (predicted == labels).sum().item()
    
        
        if best_accuracy1> correct1 / total1:
            curr_lr1 = curr_lr1/3
            update_lr(optimizer1, curr_lr1)
            print('Test Accuracy of NN: {} % '.format(100 * correct1 / total1))
        else:
            best_accuracy1 = correct1 / total1
            print('Test Accuracy of NN: {} % (improvement)'.format(100 * correct1 / total1))
            
        if best_accuracy2> correct2 / total2:
            curr_lr2 = curr_lr2/3
            update_lr(optimizer2, curr_lr2)
            print('Test Accuracy of SpinalNet: {} % '.format(100 * correct2 / total2))
        else:
            best_accuracy2 = correct2 / total2
            print('Test Accuracy of SpinalNet: {} % (improvement)'.format(100 * correct2 / total2))

        
            
        model1.train()
        model2.train()
        
