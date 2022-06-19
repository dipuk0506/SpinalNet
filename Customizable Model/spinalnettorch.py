# Customizable SpinalNet. Supports up to 30 layers.

import torch
import torch.nn as nn
import numpy as np

class SpinalNet(nn.Module):
    def __init__(self, Input_Size, Number_of_Split, HL_width, number_HL, Output_Size, Activation_Function):
        
        super(SpinalNet, self).__init__()
        Splitted_Input_Size = int(np.round(Input_Size/Number_of_Split))
        self.lru = Activation_Function
        self.fc1 = nn.Linear(Splitted_Input_Size, HL_width)
        if number_HL>1:
            self.fc2 = nn.Linear(Splitted_Input_Size+HL_width, HL_width)
        if number_HL>2:
            self.fc3 = nn.Linear(Splitted_Input_Size+HL_width, HL_width)
        if number_HL>3:
            self.fc4 = nn.Linear(Splitted_Input_Size+HL_width, HL_width)
        if number_HL>4:
            self.fc5 = nn.Linear(Splitted_Input_Size+HL_width, HL_width)
        if number_HL>5:
            self.fc6 = nn.Linear(Splitted_Input_Size+HL_width, HL_width)
        if number_HL>6:
            self.fc7 = nn.Linear(Splitted_Input_Size+HL_width, HL_width)
        if number_HL>7:
            self.fc8 = nn.Linear(Splitted_Input_Size+HL_width, HL_width)
        if number_HL>8:
            self.fc9 = nn.Linear(Splitted_Input_Size+HL_width, HL_width)
        if number_HL>9:
            self.fc10 = nn.Linear(Splitted_Input_Size+HL_width, HL_width)
        if number_HL>10:
            self.fc11 = nn.Linear(Splitted_Input_Size+HL_width, HL_width)
        if number_HL>11:
            self.fc12 = nn.Linear(Splitted_Input_Size+HL_width, HL_width)
        if number_HL>12:
            self.fc13 = nn.Linear(Splitted_Input_Size+HL_width, HL_width)
        if number_HL>13:
            self.fc14 = nn.Linear(Splitted_Input_Size+HL_width, HL_width)
        if number_HL>14:
            self.fc15 = nn.Linear(Splitted_Input_Size+HL_width, HL_width)
        if number_HL>15:
            self.fc16 = nn.Linear(Splitted_Input_Size+HL_width, HL_width)
        if number_HL>16:
            self.fc17 = nn.Linear(Splitted_Input_Size+HL_width, HL_width)
        if number_HL>17:
            self.fc18 = nn.Linear(Splitted_Input_Size+HL_width, HL_width)
        if number_HL>18:
            self.fc19 = nn.Linear(Splitted_Input_Size+HL_width, HL_width)
        if number_HL>19:
            self.fc20 = nn.Linear(Splitted_Input_Size+HL_width, HL_width)
        if number_HL>20:
            self.fc21 = nn.Linear(Splitted_Input_Size+HL_width, HL_width)
        if number_HL>21:
            self.fc22 = nn.Linear(Splitted_Input_Size+HL_width, HL_width)
        if number_HL>22:
            self.fc23 = nn.Linear(Splitted_Input_Size+HL_width, HL_width)
        if number_HL>23:
            self.fc24 = nn.Linear(Splitted_Input_Size+HL_width, HL_width)
        if number_HL>24:
            self.fc25 = nn.Linear(Splitted_Input_Size+HL_width, HL_width)
        if number_HL>25:
            self.fc26 = nn.Linear(Splitted_Input_Size+HL_width, HL_width)
        if number_HL>26:
            self.fc27 = nn.Linear(Splitted_Input_Size+HL_width, HL_width)
        if number_HL>27:
            self.fc28 = nn.Linear(Splitted_Input_Size+HL_width, HL_width)
        if number_HL>28:
            self.fc29 = nn.Linear(Splitted_Input_Size+HL_width, HL_width)
        if number_HL>29:
            self.fc30 = nn.Linear(Splitted_Input_Size+HL_width, HL_width)
        
        self.fcx = nn.Linear(HL_width*number_HL, Output_Size)

    def forward(self, x):
        x_all =x        
        
        Splitted_Input_Size = self.fc1.in_features
        HL_width = self.fc2.in_features - self.fc1.in_features
        number_HL = int(np.round(self.fcx.in_features/HL_width))
        length_x_all = number_HL*Splitted_Input_Size      
        
        while x_all.size(dim=1) < length_x_all:
            x_all = torch.cat([x_all, x],dim=1)
            
        x = self.lru(self.fc1(x_all[:,0:Splitted_Input_Size]))
        x_out = x
        
        counter1 = 1
        if number_HL>counter1:
            x_from_all = x_all[:,Splitted_Input_Size* counter1:Splitted_Input_Size*(counter1+1)]
            x = self.lru(self.fc2(torch.cat([x_from_all, x], dim=1)))
            x_out = torch.cat([x_out, x], dim=1)
            
        counter1 = counter1 + 1
        if number_HL>counter1:
            x_from_all = x_all[:,Splitted_Input_Size* counter1:Splitted_Input_Size*(counter1+1)]
            x = self.lru(self.fc3(torch.cat([x_from_all, x], dim=1)))
            x_out = torch.cat([x_out, x], dim=1)
            
        counter1 = counter1 + 1
        if number_HL>counter1:
            x_from_all = x_all[:,Splitted_Input_Size* counter1:Splitted_Input_Size*(counter1+1)]
            x = self.lru(self.fc4(torch.cat([x_from_all, x], dim=1)))
            x_out = torch.cat([x_out, x], dim=1)
            
        counter1 = counter1 + 1
        if number_HL>counter1:
            x_from_all = x_all[:,Splitted_Input_Size* counter1:Splitted_Input_Size*(counter1+1)]
            x = self.lru(self.fc5(torch.cat([x_from_all, x], dim=1)))
            x_out = torch.cat([x_out, x], dim=1)
            
        counter1 = counter1 + 1
        if number_HL>counter1:
            x_from_all = x_all[:,Splitted_Input_Size* counter1:Splitted_Input_Size*(counter1+1)]
            x = self.lru(self.fc6(torch.cat([x_from_all, x], dim=1)))
            x_out = torch.cat([x_out, x], dim=1)
            
        counter1 = counter1 + 1
        if number_HL>counter1:
            x_from_all = x_all[:,Splitted_Input_Size* counter1:Splitted_Input_Size*(counter1+1)]
            x = self.lru(self.fc7(torch.cat([x_from_all, x], dim=1)))
            x_out = torch.cat([x_out, x], dim=1)
            
        counter1 = counter1 + 1
        if number_HL>counter1:
            x_from_all = x_all[:,Splitted_Input_Size* counter1:Splitted_Input_Size*(counter1+1)]
            x = self.lru(self.fc8(torch.cat([x_from_all, x], dim=1)))
            x_out = torch.cat([x_out, x], dim=1)
            
        counter1 = counter1 + 1
        if number_HL>counter1:
            x_from_all = x_all[:,Splitted_Input_Size* counter1:Splitted_Input_Size*(counter1+1)]
            x = self.lru(self.fc9(torch.cat([x_from_all, x], dim=1)))
            x_out = torch.cat([x_out, x], dim=1)
            
        counter1 = counter1 + 1
        if number_HL>counter1:
            x_from_all = x_all[:,Splitted_Input_Size* counter1:Splitted_Input_Size*(counter1+1)]
            x = self.lru(self.fc10(torch.cat([x_from_all, x], dim=1)))
            x_out = torch.cat([x_out, x], dim=1)  
            
        counter1 = counter1 + 1
        if number_HL>counter1:
            x_from_all = x_all[:,Splitted_Input_Size* counter1:Splitted_Input_Size*(counter1+1)]
            x = self.lru(self.fc11(torch.cat([x_from_all, x], dim=1)))
            x_out = torch.cat([x_out, x], dim=1)
            
        counter1 = counter1 + 1
        if number_HL>counter1:
            x_from_all = x_all[:,Splitted_Input_Size* counter1:Splitted_Input_Size*(counter1+1)]
            x = self.lru(self.fc12(torch.cat([x_from_all, x], dim=1)))
            x_out = torch.cat([x_out, x], dim=1)
            
        counter1 = counter1 + 1
        if number_HL>counter1:
            x_from_all = x_all[:,Splitted_Input_Size* counter1:Splitted_Input_Size*(counter1+1)]
            x = self.lru(self.fc13(torch.cat([x_from_all, x], dim=1)))
            x_out = torch.cat([x_out, x], dim=1)
            
        counter1 = counter1 + 1
        if number_HL>counter1:
            x_from_all = x_all[:,Splitted_Input_Size* counter1:Splitted_Input_Size*(counter1+1)]
            x = self.lru(self.fc14(torch.cat([x_from_all, x], dim=1)))
            x_out = torch.cat([x_out, x], dim=1)
            
        counter1 = counter1 + 1
        if number_HL>counter1:
            x_from_all = x_all[:,Splitted_Input_Size* counter1:Splitted_Input_Size*(counter1+1)]
            x = self.lru(self.fc15(torch.cat([x_from_all, x], dim=1)))
            x_out = torch.cat([x_out, x], dim=1)
            
        counter1 = counter1 + 1
        if number_HL>counter1:
            x_from_all = x_all[:,Splitted_Input_Size* counter1:Splitted_Input_Size*(counter1+1)]
            x = self.lru(self.fc16(torch.cat([x_from_all, x], dim=1)))
            x_out = torch.cat([x_out, x], dim=1)
            
        counter1 = counter1 + 1
        if number_HL>counter1:
            x_from_all = x_all[:,Splitted_Input_Size* counter1:Splitted_Input_Size*(counter1+1)]
            x = self.lru(self.fc17(torch.cat([x_from_all, x], dim=1)))
            x_out = torch.cat([x_out, x], dim=1) 
            
        counter1 = counter1 + 1
        if number_HL>counter1:
            x_from_all = x_all[:,Splitted_Input_Size* counter1:Splitted_Input_Size*(counter1+1)]
            x = self.lru(self.fc18(torch.cat([x_from_all, x], dim=1)))
            x_out = torch.cat([x_out, x], dim=1)
            
        counter1 = counter1 + 1
        if number_HL>counter1:
            x_from_all = x_all[:,Splitted_Input_Size* counter1:Splitted_Input_Size*(counter1+1)]
            x = self.lru(self.fc19(torch.cat([x_from_all, x], dim=1)))
            x_out = torch.cat([x_out, x], dim=1)
            
        counter1 = counter1 + 1
        if number_HL>counter1:
            x_from_all = x_all[:,Splitted_Input_Size* counter1:Splitted_Input_Size*(counter1+1)]
            x = self.lru(self.fc20(torch.cat([x_from_all, x], dim=1)))
            x_out = torch.cat([x_out, x], dim=1) 
        counter1 = counter1 + 1
        if number_HL>counter1:
            x_from_all = x_all[:,Splitted_Input_Size* counter1:Splitted_Input_Size*(counter1+1)]
            x = self.lru(self.fc21(torch.cat([x_from_all, x], dim=1)))
            x_out = torch.cat([x_out, x], dim=1)
            
        counter1 = counter1 + 1
        if number_HL>counter1:
            x_from_all = x_all[:,Splitted_Input_Size* counter1:Splitted_Input_Size*(counter1+1)]
            x = self.lru(self.fc22(torch.cat([x_from_all, x], dim=1)))
            x_out = torch.cat([x_out, x], dim=1)
            
        counter1 = counter1 + 1
        if number_HL>counter1:
            x_from_all = x_all[:,Splitted_Input_Size* counter1:Splitted_Input_Size*(counter1+1)]
            x = self.lru(self.fc23(torch.cat([x_from_all, x], dim=1)))
            x_out = torch.cat([x_out, x], dim=1)
            
        counter1 = counter1 + 1
        if number_HL>counter1:
            x_from_all = x_all[:,Splitted_Input_Size* counter1:Splitted_Input_Size*(counter1+1)]
            x = self.lru(self.fc24(torch.cat([x_from_all, x], dim=1)))
            x_out = torch.cat([x_out, x], dim=1)
            
        counter1 = counter1 + 1
        if number_HL>counter1:
            x_from_all = x_all[:,Splitted_Input_Size* counter1:Splitted_Input_Size*(counter1+1)]
            x = self.lru(self.fc25(torch.cat([x_from_all, x], dim=1)))
            x_out = torch.cat([x_out, x], dim=1)
            
        counter1 = counter1 + 1
        if number_HL>counter1:
            x_from_all = x_all[:,Splitted_Input_Size* counter1:Splitted_Input_Size*(counter1+1)]
            x = self.lru(self.fc26(torch.cat([x_from_all, x], dim=1)))
            x_out = torch.cat([x_out, x], dim=1)
            
        counter1 = counter1 + 1
        if number_HL>counter1:
            x_from_all = x_all[:,Splitted_Input_Size* counter1:Splitted_Input_Size*(counter1+1)]
            x = self.lru(self.fc27(torch.cat([x_from_all, x], dim=1)))
            x_out = torch.cat([x_out, x], dim=1) 
            
        counter1 = counter1 + 1
        if number_HL>counter1:
            x_from_all = x_all[:,Splitted_Input_Size* counter1:Splitted_Input_Size*(counter1+1)]
            x = self.lru(self.fc28(torch.cat([x_from_all, x], dim=1)))
            x_out = torch.cat([x_out, x], dim=1)
            
        counter1 = counter1 + 1
        if number_HL>counter1:
            x_from_all = x_all[:,Splitted_Input_Size* counter1:Splitted_Input_Size*(counter1+1)]
            x = self.lru(self.fc29(torch.cat([x_from_all, x], dim=1)))
            x_out = torch.cat([x_out, x], dim=1)
            
        counter1 = counter1 + 1
        if number_HL>counter1:
            x_from_all = x_all[:,Splitted_Input_Size* counter1:Splitted_Input_Size*(counter1+1)]
            x = self.lru(self.fc30(torch.cat([x_from_all, x], dim=1)))
            x_out = torch.cat([x_out, x], dim=1)        
        #print("Size before output layer:",x_out.size(dim=1))
        x = self.fcx(x_out)
        return x