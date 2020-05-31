

import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(0)

learning_rate = 0.001
momentum = 0.95
num_classes = 10
num_epochs = 50
loss_check = []

first_HL = 64
Half_Data = 64*4*2


#Dataset.py
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



class dense_block(nn.Module):
    def __init__(self, in_channels):
        super(dense_block, self).__init__()
        
        self.relu = nn.ReLU()
        self.bn=nn.BatchNorm2d(num_features= in_channels)
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size = 3, padding =1, stride =1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size = 3, padding =1, stride =1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size = 3, padding =1, stride =1)
        self.conv4 = nn.Conv2d(in_channels=96, out_channels=32, kernel_size = 3, padding =1, stride =1)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size = 3, padding =1, stride =1)
        
    def forward(self, x):
        
        bn=self.bn(x)
        conv1=self.relu(self.conv1(bn))
        conv2=self.relu(self.conv2(conv1))
        cat1_dense=self.relu(torch.cat([conv1, conv2], 1))
        conv3=self.relu(self.conv3(cat1_dense))
        cat2_dense=self.relu(torch.cat([conv1, conv2, conv3], 1))
        conv4=self.relu(self.conv4(cat2_dense))
        cat3_dense=self.relu(torch.cat([conv1, conv2, conv3, conv4], 1))
        conv5=self.relu(self.conv5(cat3_dense))
        cat4_dense=self.relu(torch.cat([conv1, conv2, conv3, conv4, conv5], 1))
                             
        return cat4_dense

#Transition blocks.py
class transition_block(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(transition_block, self).__init__()
        
        self.relu=nn.ReLU(inplace=True)
        self.bn=nn.BatchNorm2d(num_features=out_channels)
        
        self.conv=nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size =1, bias=False)
        self.avg_pl=nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        
    def forward(self, x):
        bn=self.bn(self.relu(self.conv(x)))
        output= self.avg_pl(bn)
        
        return output   



class DenseNet(nn.Module):
    def __init__(self, num_classes):
        super(DenseNet, self).__init__()
        
        self.in_conv=nn.Conv2d(in_channels=3, out_channels=64, kernel_size = 7, padding =3, bias= False)
        self.relu=nn.ReLU()
        
        self.denseblock1 = self.add_dense_block(dense_block, 64)
        self.transitionLayer1 = self.add_transition_block(transition_block, in_channels=160, out_channels=128)
         
        self.denseblock2 = self.add_dense_block(dense_block, 128)
        self.transitionLayer2 = self.add_transition_block(transition_block, in_channels=160, out_channels=128)
        
        self.denseblock3 = self.add_dense_block(dense_block, 128)
        self.transitionLayer3 = self.add_transition_block(transition_block, in_channels=160, out_channels=64)
        
        self.denseblock4 = self.add_dense_block(dense_block, 64)
        
        self.bn=nn.BatchNorm2d(num_features=64)
        self.lastlayer=nn.Linear(64*4*4, 512)
        
        self.fc1 = nn.Linear(Half_Data, first_HL) #changed from 16 to 8
        self.fc1_1 = nn.Linear(Half_Data + first_HL, first_HL) #added
        self.fc1_2 = nn.Linear(Half_Data + first_HL, first_HL) #added
        self.fc1_3 = nn.Linear(Half_Data + first_HL, first_HL) #added
        self.fc1_4 = nn.Linear(Half_Data + first_HL, first_HL) #added
        self.fc1_5 = nn.Linear(Half_Data + first_HL, first_HL) #added
        self.fc1_6 = nn.Linear(Half_Data + first_HL, first_HL) #added
        self.fc1_7 = nn.Linear(Half_Data + first_HL, first_HL) #added
        self.fc3 = nn.Linear(first_HL*8, num_classes) # changed first_HL from second_HL
        
        #self.final = nn.Linear (512, num_classes)
       
    
    def add_dense_block(self, block, in_channels):
        layer=[]
        layer.append(block(in_channels))
        D_seq=nn.Sequential(*layer)
        return D_seq
    
    def add_transition_block(self, layers, in_channels, out_channels):
        trans=[]
        trans.append(layers(in_channels, out_channels))
        T_seq=nn.Sequential(*trans)
        return T_seq
    
    
    
    def forward(self, x):
        out= self.relu(self.in_conv(x))
        out= self.denseblock1(out)
        out=self.transitionLayer1(out)
        
        out= self.denseblock2(out)
        out=self.transitionLayer2(out)
        
        out= self.denseblock3(out)
        out=self.transitionLayer3(out)
        
        out=self.bn(out)
        out=out.view(-1, 64*4*4)
        
        #out=self.lastlayer(out)
        x = out 
        x1 = x[:, 0:Half_Data]
        x1 = self.relu(self.fc1(x1))
        x2= torch.cat([ x[:,Half_Data:Half_Data*2], x1], dim=1)
        x2 = self.relu(self.fc1_1(x2))
        x3= torch.cat([ x[:, 0:Half_Data], x2], dim=1)
        x3 = self.relu(self.fc1_2(x3))
        x4= torch.cat([ x[:,Half_Data:Half_Data*2], x3], dim=1)
        x4 = self.relu(self.fc1_3(x4))
        x5= torch.cat([ x[:, 0:Half_Data], x4], dim=1)
        x5 = self.relu(self.fc1_4(x5))
        x6= torch.cat([ x[:,Half_Data:Half_Data*2], x5], dim=1)
        x6 = self.relu(self.fc1_5(x6))
        x7= torch.cat([ x[:, 0:Half_Data], x6], dim=1)
        x7 = self.relu(self.fc1_6(x7))
        x8= torch.cat([ x[:,Half_Data:Half_Data*2], x7], dim=1)
        x8 = self.relu(self.fc1_7(x8))


        
        x = torch.cat([x1, x2], dim=1)
        x = torch.cat([x, x3], dim=1)
        x = torch.cat([x, x4], dim=1)
        x = torch.cat([x, x5], dim=1)
        x = torch.cat([x, x6], dim=1)
        x = torch.cat([x, x7], dim=1)
        x = torch.cat([x, x8], dim=1)
        out=x
        
        out=self.fc3(out)
        
        return out

Net = DenseNet(num_classes)
#if gpu:
#    Net.cuda()

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(Net.parameters(), lr=0.001, momentum=0.9)


#criterion = nn.CrossEntropyLoss().cuda() if gpu else nn.CrossEntropyLoss()
optimizer = optim.SGD(Net.parameters(), lr=learning_rate, momentum=momentum, nesterov = False)

print("Start Training..")
for epoch in range(num_epochs):
    Net.train()

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = Net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 2000 mini-batches
            loss_check.append(running_loss / 200)
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0
    if epoch % 5 == 4 or epoch ==0:
        Net.eval()
        correct = 0.0
        total = 0
        
        for data in testloader:
                images, labels = data
                outputs = Net(images)
                _, predicted = torch.max(outputs.cpu().data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
                
        print('Test Accuracy: %.2f'%(100 * correct / total),'%')
    
print("^^^^^^^^^^^^^^^^^")
print('Finished Training.')
#print("The total time to train the model on Google Colab is : {:.1f} minutes.".format((end - start)/60))


