# -*- coding: utf-8 -*-
"""
Created on Thu May  7 11:31:06 2020

@author: Dipu
"""

import torch
import torchvision
import torchvision.transforms as transforms

first_HL =20


print('Layer sizes:',first_HL)
NUM_EPOCH = 2
print('Epoch', NUM_EPOCH)

torch.manual_seed(0)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(8 * 5 * 5, first_HL) #changed from 16 to 8
        self.fc1_1 = nn.Linear(8 * 5 * 5 + first_HL, first_HL) #added
        self.fc1_2 = nn.Linear(8 * 5 * 5 + first_HL, first_HL) #added
        self.fc1_3 = nn.Linear(8 * 5 * 5 + first_HL, first_HL) #added
        self.fc1_4 = nn.Linear(8 * 5 * 5 + first_HL, first_HL) #added
        self.fc1_5 = nn.Linear(8 * 5 * 5 + first_HL, first_HL) #added
        self.fc3 = nn.Linear(first_HL*6, 10) # changed first_HL from second_HL

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #print("size after Pool",x.shape)
        
        x = x.view(-1, 16 * 5 * 5) # changed from 16 to 8
        x1 = x[:, 0:8 * 5 * 5]
        
        x1 = F.relu(self.fc1(x1))
        x2= torch.cat([ x[:,8 * 5 * 5:16 * 5 * 5], x1], dim=1)
        x2 = F.relu(self.fc1_1(x2))
        x3= torch.cat([ x[:,0:8 * 5 * 5], x2], dim=1)
        x3 = F.relu(self.fc1_2(x3))
        x4= torch.cat([ x[:,8 * 5 * 5:16 * 5 * 5], x3], dim=1)
        x4 = F.relu(self.fc1_3(x4))
        x5= torch.cat([ x[:,0:8 * 5 * 5], x4], dim=1)
        x5 = F.relu(self.fc1_4(x5))
        x6= torch.cat([ x[:,8 * 5 * 5:16 * 5 * 5], x5], dim=1)
        x6 = F.relu(self.fc1_5(x6))

        
        x = torch.cat([x1, x2], dim=1)
        x = torch.cat([x, x3], dim=1)
        x = torch.cat([x, x4], dim=1)
        x = torch.cat([x, x5], dim=1)
        x = torch.cat([x, x6], dim=1)

        x = self.fc3(x)
        #print("size after Layer3",x.shape)
        return x


net = Net()


import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


for epoch in range(NUM_EPOCH):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 6000 == 5999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 6000))
            running_loss = 0.0

print('Finished Training')



#%%

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))


net = Net()
net.load_state_dict(torch.load(PATH))

outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

#%%

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))







