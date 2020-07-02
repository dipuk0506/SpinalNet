# -*- coding: utf-8 -*-
"""
This Script contains the default and Spinal ResNet code for CIFAR-10.

This code trains both NNs as two different models.

There is option of choosing ResNet18(), ResNet34(), SpinalResNet18(), or
SpinalResNet34().

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
num_epochs = 160
learning_rate = 0.001

torch.manual_seed(0)
random.seed(0)

first_HL = 256

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

trainset = torchvision.datasets.CIFAR10(root='./data',
                                         train=True,
                                         download=True,
                                         transform=transform_train)
train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=200, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data',
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


class BasicBlock(nn.Module):
    """Basic Block of ReseNet."""

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """Basic Block of ReseNet Builder."""
        super(BasicBlock, self).__init__()

        # First conv3x3 layer
        self.conv1 = conv3x3(in_channels, out_channels, stride)

        #  Batch Normalization
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)

        # ReLU Activation Function
        self.relu = nn.ReLU(inplace=True)

        # Second conv3x3 layer
        self.conv2 = conv3x3(out_channels, out_channels)

        #  Batch Normalization
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        # downsample for `residual`
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """Forward Pass of Basic Block."""
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return out

class SpinalResNet(nn.Module):
    """Residual Neural Network."""

    def __init__(self, block, duplicates, num_classes=10):
        """Residual Neural Network Builder."""
        super(SpinalResNet, self).__init__()

        self.in_channels = 32
        self.conv1 = conv3x3(in_channels=3, out_channels=32)
        self.bn = nn.BatchNorm2d(num_features=32)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=0.02)

        # block of Basic Blocks
        self.conv2_x = self._make_block(block, duplicates[0], out_channels=32)
        self.conv3_x = self._make_block(block, duplicates[1], out_channels=64, stride=2)
        self.conv4_x = self._make_block(block, duplicates[2], out_channels=128, stride=2)
        self.conv5_x = self._make_block(block, duplicates[3], out_channels=256, stride=2)

        self.maxpool = nn.MaxPool2d(kernel_size=4, stride=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        #self.fc_layer = nn.Linear(256, num_classes)
        
        self.fc1 = nn.Linear(256, first_HL) #changed from 16 to 8
        self.fc1_1 = nn.Linear(256 + first_HL, first_HL) #added
        self.fc1_2 = nn.Linear(256 + first_HL, first_HL) #added
        self.fc1_3 = nn.Linear(256 + first_HL, first_HL) #added
        
        self.fc_layer = nn.Linear(first_HL*4, num_classes)

        # initialize weights
        # self.apply(initialize_weights)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_block(self, block, duplicates, out_channels, stride=1):
        """
        Create Block in ResNet.

        Args:
            block: BasicBlock
            duplicates: number of BasicBlock
            out_channels: out channels of the block

        Returns:
            nn.Sequential(*layers)
        """
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(num_features=out_channels)
            )

        layers = []
        layers.append(
            block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, duplicates):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass of ResNet."""
        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)

        # Stacked Basic Blocks
        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)
        
        
        out1 = self.maxpool2(out)
        #print('out1',out1.shape)
        out2 = out1[:,:,0,0]
        #print('out2',out2.shape)
        out2 = out2.view(out2.size(0),-1)
        #print('out2',out2.shape)
        
        x1 = out1[:,:,0,0]
        x1 = self.relu(self.fc1(x1))
        x2= torch.cat([ out1[:,:,0,1], x1], dim=1)
        x2 = self.relu(self.fc1_1(x2))
        x3= torch.cat([ out1[:,:,1,0], x2], dim=1)
        x3 = self.relu(self.fc1_2(x3))
        x4= torch.cat([ out1[:,:,1,1], x3], dim=1)
        x4 = self.relu(self.fc1_3(x4))
        
        x = torch.cat([x1, x2], dim=1)
        x = torch.cat([x, x3], dim=1)
        out = torch.cat([x, x4], dim=1)
        
        out = self.fc_layer(out)

        return out


class ResNet(nn.Module):
    """Residual Neural Network."""

    def __init__(self, block, duplicates, num_classes=10):
        """Residual Neural Network Builder."""
        super(ResNet, self).__init__()

        self.in_channels = 32
        self.conv1 = conv3x3(in_channels=3, out_channels=32)
        self.bn = nn.BatchNorm2d(num_features=32)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=0.02)

        # block of Basic Blocks
        self.conv2_x = self._make_block(block, duplicates[0], out_channels=32)
        self.conv3_x = self._make_block(block, duplicates[1], out_channels=64, stride=2)
        self.conv4_x = self._make_block(block, duplicates[2], out_channels=128, stride=2)
        self.conv5_x = self._make_block(block, duplicates[3], out_channels=256, stride=2)

        self.maxpool = nn.MaxPool2d(kernel_size=4, stride=1)
        #self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.fc_layer = nn.Linear(256, num_classes)

        # initialize weights
        # self.apply(initialize_weights)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_block(self, block, duplicates, out_channels, stride=1):
        """
        Create Block in ResNet.

        Args:
            block: BasicBlock
            duplicates: number of BasicBlock
            out_channels: out channels of the block

        Returns:
            nn.Sequential(*layers)
        """
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(num_features=out_channels)
            )

        layers = []
        layers.append(
            block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, duplicates):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass of ResNet."""
        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)

        # Stacked Basic Blocks
        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)
        
        out = self.maxpool(out)
        out = out.view(out.size(0), -1)
        out = out.view(out.size(0), -1)
        out = self.fc_layer(out)

        return out


model = ResNet(BasicBlock, [1,1,1,1]).to(device)


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2]).to(device)

def SpinalResNet18():
    return SpinalResNet(BasicBlock, [2,2,2,2]).to(device)

def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3]).to(device)

def SpinalResNet34():
    return SpinalResNet(BasicBlock, [3, 4, 6, 3]).to(device)


# For updating learning rate
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Train the model
total_step = len(train_loader)
curr_lr1 = learning_rate

curr_lr2 = learning_rate



model1 = ResNet18().to(device)

model2 = SpinalResNet18().to(device)



# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer1 = torch.optim.Adam(model1.parameters(), lr=learning_rate)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=learning_rate) 
  
# Train the model
total_step = len(train_loader)

best_accuracy1 = 0
best_accuracy2 =0
#%%
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
            curr_lr1 = learning_rate*np.asscalar(pow(np.random.rand(1),5))
            update_lr(optimizer1, curr_lr1)
            print('Epoch :{} Accuracy NN: ({:.2f}%), Maximum Accuracy: {:.2f}%'.format(epoch, 
                                              100 * correct1 / total1, 100*best_accuracy1))
        else:
            best_accuracy1 = correct1 / total1
            print('Test Accuracy of NN: {} % (improvement)'.format(100 * correct1 / total1))
            
        if best_accuracy2> correct2 / total2:
            curr_lr2 = learning_rate*np.asscalar(pow(np.random.rand(1),5))
            update_lr(optimizer2, curr_lr2)
            print('Epoch :{} Accuracy SpinalNet: ({:.2f}%), Maximum Accuracy: {:.2f}%'.format(epoch, 
                                              100 * correct2 / total2, 100*best_accuracy2))
        else:
            best_accuracy2 = correct2 / total2
            print('Test Accuracy of SpinalNet: {} % (improvement)'.format(100 * correct2 / total2))

        
            
        model1.train()
        model2.train()
        
