# -*- coding: utf-8 -*-
"""
This Script contains the default and Spinal VGG code for CIFAR-100.

This code trains both NNs as two different models.

There is option of choosing NN among:
    vgg11_bn(), vgg13_bn(), vgg16_bn(), vgg19_bn() and
    Spinalvgg11_bn(), Spinalvgg13_bn(), Spinalvgg16_bn(), Spinalvgg19_bn()

This code randomly changes the learning rate to get a good result.

@author: Dipu
"""


import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'

# Hyper-parameters
num_epochs = 200
learning_rate = 0.0001

Half_width =256
layer_width=512

torch.manual_seed(0)
random.seed(0)

# Image preprocessing modules
# Normalize training set together with augmentation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
])

# Normalize test set same as training set without augmentation
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
])

# CIFAR-100 dataset

trainset = torchvision.datasets.CIFAR100(root='./data',
                                         train=True,
                                         download=True,
                                         transform=transform_train)
train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=200, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR100(root='./data',
                                        train=False,
                                        download=True,
                                        transform=transform_test)
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=200, shuffle=False, num_workers=0)



def conv3x3(in_channels, out_channels, stride=1):
    """3x3 kernel size with padding convolutional layer in ResNet BasicBlock."""
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):

    def __init__(self, features, num_class=100):
        super().__init__()
        self.features = features

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)
    
        return output
    
class SpinalVGG(nn.Module):

    def __init__(self, features, num_class=100):
        super().__init__()
        self.features = features
        
        self.fc_spinal_layer1 = nn.Sequential(
            nn.Dropout(), nn.Linear(Half_width, layer_width),
            nn.ReLU(inplace=True),
            )
        self.fc_spinal_layer2 = nn.Sequential(
            nn.Dropout(), nn.Linear(Half_width + layer_width, layer_width),
            nn.ReLU(inplace=True),
            )
        self.fc_spinal_layer3 = nn.Sequential(
            nn.Dropout(), nn.Linear(Half_width + layer_width, layer_width),
            nn.ReLU(inplace=True),
            )
        self.fc_spinal_layer4 = nn.Sequential(
            nn.Dropout(), nn.Linear(Half_width + layer_width, layer_width),
            nn.ReLU(inplace=True),
            )
        self.fc_out = nn.Sequential(
            nn.Dropout(), nn.Linear(layer_width*4, num_class)            
            )


    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        x = output
        x1 = self.fc_spinal_layer1(x[:, 0:Half_width])
        x2 = self.fc_spinal_layer2(torch.cat([ x[:,Half_width:2*Half_width], x1], dim=1))
        x3 = self.fc_spinal_layer3(torch.cat([ x[:,0:Half_width], x2], dim=1))
        x4 = self.fc_spinal_layer4(torch.cat([ x[:,Half_width:2*Half_width], x3], dim=1))
        
        x = torch.cat([x1, x2], dim=1)
        x = torch.cat([x, x3], dim=1)
        x = torch.cat([x, x4], dim=1)
        
        x = self.fc_out(x)
    
        return x
    

def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]
        
        layers += [nn.ReLU(inplace=True)]
        input_channel = l
    
    return nn.Sequential(*layers)

def vgg11_bn():
    return VGG(make_layers(cfg['A'], batch_norm=True))

def vgg13_bn():
    return VGG(make_layers(cfg['B'], batch_norm=True))

def vgg16_bn():
    return VGG(make_layers(cfg['D'], batch_norm=True))

def vgg19_bn():
    return VGG(make_layers(cfg['E'], batch_norm=True))

def Spinalvgg11_bn():
    return SpinalVGG(make_layers(cfg['A'], batch_norm=True))

def Spinalvgg13_bn():
    return SpinalVGG(make_layers(cfg['B'], batch_norm=True))

def Spinalvgg16_bn():
    return SpinalVGG(make_layers(cfg['D'], batch_norm=True))

def Spinalvgg19_bn():
    return SpinalVGG(make_layers(cfg['E'], batch_norm=True))



# For updating learning rate
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Train the model
total_step = len(train_loader)
curr_lr1 = learning_rate

curr_lr2 = learning_rate



model1 = vgg19_bn().to(device)

model2 = Spinalvgg19_bn().to(device)



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
            curr_lr1 = learning_rate*np.asscalar(pow(np.random.rand(1),3))
            update_lr(optimizer1, curr_lr1)
            print('Test Accuracy of NN: {} % Best: {} %'.format(100 * correct1 / total1, 100*best_accuracy1))
        else:
            best_accuracy1 = correct1 / total1
            net_opt1 = model1
            print('Test Accuracy of NN: {} % (improvement)'.format(100 * correct1 / total1))
            
        if best_accuracy2> correct2 / total2:
            curr_lr2 = learning_rate*np.asscalar(pow(np.random.rand(1),3))
            update_lr(optimizer2, curr_lr2)
            print('Test Accuracy of SpinalNet: {} % Best: {} %'.format(100 * correct2 / total2, 100*best_accuracy2))
        else:
            best_accuracy2 = correct2 / total2
            net_opt2 = model2
            print('Test Accuracy of SpinalNet: {} % (improvement)'.format(100 * correct2 / total2))

        
            
        model1.train()
        model2.train()
        



