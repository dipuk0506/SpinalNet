# -*- coding: utf-8 -*-
"""
This Script contains the SpinalNet MNIST code.

It ususlly provides better performance for the same number of epoch.

The same code can also be used for KMNIST, QMNIST and FashionMNIST.

torchvision.datasets.MNIST needs to be changed to  
torchvision.datasets.FashionMNIST for FashionMNIST simulations

@author: Dipu
"""

import torch
import torchvision

n_epochs = 8
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 100
first_HL =8


torch.backends.cudnn.enabled = False


train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

print(example_data.shape)

import matplotlib.pyplot as plt

fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Ground Truth: {}".format(example_targets[i]))
  plt.xticks([])
  plt.yticks([])
fig

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(160, first_HL) #changed from 16 to 8
        self.fc1_1 = nn.Linear(160 + first_HL, first_HL) #added
        self.fc1_2 = nn.Linear(160 + first_HL, first_HL) #added
        self.fc1_3 = nn.Linear(160 + first_HL, first_HL) #added
        self.fc1_4 = nn.Linear(160 + first_HL, first_HL) #added
        self.fc1_5 = nn.Linear(160 + first_HL, first_HL) #added
        self.fc2 = nn.Linear(first_HL*6, 10) # changed first_HL from second_HL
        
        #self.fc1 = nn.Linear(320, 50)
        #self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x1 = x[:, 0:160]
        
        x1 = F.relu(self.fc1(x1))
        x2= torch.cat([ x[:,160:320], x1], dim=1)
        x2 = F.relu(self.fc1_1(x2))
        x3= torch.cat([ x[:,0:160], x2], dim=1)
        x3 = F.relu(self.fc1_2(x3))
        x4= torch.cat([ x[:,160:320], x3], dim=1)
        x4 = F.relu(self.fc1_3(x4))
        x5= torch.cat([ x[:,0:160], x4], dim=1)
        x5 = F.relu(self.fc1_4(x5))
        x6= torch.cat([ x[:,160:320], x5], dim=1)
        x6 = F.relu(self.fc1_5(x6))

        
        x = torch.cat([x1, x2], dim=1)
        x = torch.cat([x, x3], dim=1)
        x = torch.cat([x, x4], dim=1)
        x = torch.cat([x, x5], dim=1)
        x = torch.cat([x, x6], dim=1)

        x = self.fc2(x)
        
        #x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        #x = self.fc2(x)
        return F.log_softmax(x)
    
    
    
network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)


train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]


def train(epoch):
  network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = network(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))

      
def test():
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      output = network(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
  
  
test()
for epoch in range(1, n_epochs + 1):
  train(epoch)
  test()
#%%

fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
fig

  


