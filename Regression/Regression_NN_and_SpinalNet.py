# -*- coding: utf-8 -*-
"""
This script performs regression on toy datasets. 
There exist several relations between inputs and output.
We investigate both of the traditional feed-forward and SpinalNet  
for all of these input-output relations.

----------
Multiplication:
y = x1*x2*x3*x4*x5*x6*x7*x8 +  0.2*torch.rand(x1.size())
Spinal
Epoch [100/200], Loss: 0.0573, Minimum Loss 0.003966
Epoch [200/200], Loss: 0.0170, Minimum Loss 0.002217
Normal
Epoch [100/200], Loss: 0.0212, Minimum Loss 0.003875
Epoch [200/200], Loss: 0.0373, Minimum Loss 0.003875

Sine multiplication:
y = torch.sin(x1*x2*x3*x4*x5*x6*x7*x8) +  0.2*torch.rand(x1.size())
Spinal
Epoch [100/200], Loss: 0.0013, Minimum Loss 0.000910
Epoch [200/200], Loss: 0.0023, Minimum Loss 0.000910

Normal
Epoch [100/200], Loss: 0.0090, Minimum Loss 0.003403
Epoch [200/200], Loss: 0.0041, Minimum Loss 0.001554

Addition:
    y = (x1+x2+x3+x4+x5+x6+x7+x8) +  0.2*torch.rand(x1.size())

Spinal
Epoch [100/200], Loss: 0.0038, Minimum Loss 0.001007
Epoch [200/200], Loss: 0.0022, Minimum Loss 0.000855
Normal
Epoch [100/200], Loss: 0.0024, Minimum Loss 0.001178
Epoch [200/200], Loss: 0.0021, Minimum Loss 0.000887

Sine Addition:
y = torch.sin(x1+x2+x3+x4+x5+x6+x7+x8) +  0.2*torch.rand(x1.size())   
Spinal
Epoch [100/200], Loss: 0.0254, Minimum Loss 0.001912
Epoch [200/200], Loss: 0.0029, Minimum Loss 0.001219
Normal
Epoch [100/200], Loss: 0.0019, Minimum Loss 0.001918
Epoch [200/200], Loss: 0.0038, Minimum Loss 0.001086



@author: Dipu
"""

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data

import matplotlib.pyplot as plt

import numpy as np
#import imageio

torch.manual_seed(0)    


size_x=1000

x1 = torch.unsqueeze(torch.randn(size_x), dim=1)
x2 = torch.unsqueeze(torch.randn(size_x), dim=1)
x3 = torch.unsqueeze(torch.randn(size_x), dim=1)
x4 = torch.unsqueeze(torch.randn(size_x), dim=1)
x5 = torch.unsqueeze(torch.randn(size_x), dim=1)
x6 = torch.unsqueeze(torch.randn(size_x), dim=1)
x7 = torch.unsqueeze(torch.randn(size_x), dim=1)
x8 = torch.unsqueeze(torch.randn(size_x), dim=1)

half_in_size=4

y = (x1*x2*x3*x4*x5*x6*x7*x8) +  0.2*torch.rand(size_x)
              # noisy y data (tensor), shape=(100, 1)

x=torch.cat([x1,x2,x3,x4,x5,x6,x7,x8], dim=1)
x, y = Variable(x), Variable(y)

# another way to define a network
net = torch.nn.Sequential(
        torch.nn.Linear(half_in_size*2, 200),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(200, 100),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(100, 1),
    )

import torch.nn as nn


first_HL = 50

class SpinalNet(nn.Module):
    def __init__(self):
        super(SpinalNet, self).__init__()
        self.lru = nn.LeakyReLU()
        self.fc1 = nn.Linear(half_in_size, first_HL)
        self.fc2 = nn.Linear(half_in_size+first_HL, first_HL)
        self.fc3 = nn.Linear(half_in_size+first_HL, first_HL)
        self.fc4 = nn.Linear(half_in_size+first_HL, first_HL)
        self.fc5 = nn.Linear(half_in_size+first_HL, first_HL)
        self.fc6 = nn.Linear(half_in_size+first_HL, first_HL)
        
        self.fcx = nn.Linear(first_HL*6, 1)

    def forward(self, x):
        x1 = x[:,0:half_in_size]
        x1 = self.lru(self.fc1(x1))
        x2= torch.cat([ x[:,half_in_size:half_in_size*2], x1], dim=1)
        x2 = self.lru(self.fc2(x2))
        x3= torch.cat([x[:,0:half_in_size], x2], dim=1)
        x3 = self.lru(self.fc3(x3))
        x4= torch.cat([x[:,half_in_size:half_in_size*2], x3], dim=1)
        x4 = self.lru(self.fc4(x4))
        x5= torch.cat([x[:,0:half_in_size], x4], dim=1)
        x5 = self.lru(self.fc3(x5))
        x6= torch.cat([x[:,half_in_size:half_in_size*2], x5], dim=1)
        x6 = self.lru(self.fc4(x6))
        
        
        
        x = torch.cat([x1, x2], dim=1)
        x = torch.cat([x, x3], dim=1)
        x = torch.cat([x, x4], dim=1)
        x = torch.cat([x, x5], dim=1)
        x = torch.cat([x, x6], dim=1)
        

        x = self.fcx(x)
        return x

#------------------------------------------------------------------------------
"""
Comment these two lines for traditional NN training.
"""
net = SpinalNet()      
print('SpinalNet')   

#------------------------------------------------------------------------------


optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

BATCH_SIZE = 64
EPOCH = 200

torch_dataset = Data.TensorDataset(x, y)

loader = Data.DataLoader(
    dataset=torch_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, num_workers=0,)

min_loss =100
# start training
for epoch in range(EPOCH):
    for step, (batch_x, batch_y) in enumerate(loader): # for each training step
        
        b_x = Variable(batch_x)
        b_y = Variable(batch_y)

        prediction = net(b_x)     # input x and predict based on x

        loss = loss_func(prediction, b_y)     # must be (1. nn output, 2. target)
        
               
        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients
    loss = loss.item()
    if loss<min_loss:
            min_loss = loss  
            net_opt = net
    if epoch%100 == 99:
      print ("Epoch [{}/{}], Loss: {:.4f}, Minimum Loss {:.6f}"  .format(epoch+1, EPOCH, loss, min_loss))

    

