# -*- coding: UTF-8 -*-
import torch.nn as nn
import torch
# CNN Model (2 conv layer)
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outchannel)
        )
    
        self.shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride),
            nn.BatchNorm2d(outchannel)
        )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class RES(nn.Module):
    def __init__(self):
        super(RES, self).__init__()
        self.inchannel = 32
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            self.make_layer(ResidualBlock, 64,  1, stride=1),
            nn.MaxPool2d(2),
            nn.Dropout(0.3))

        self.layer3 = nn.Sequential(
            self.make_layer(ResidualBlock, 64,  1, stride=1),
            nn.MaxPool2d(2),
            nn.Dropout(0.3))

        self.layer4 = nn.Conv2d(64, 64, kernel_size=3, dilation = 2)
            
        self.layer5 = nn.Sequential(
            self.make_layer(ResidualBlock, 128,  1, stride=1),
            nn.MaxPool2d(2),
            nn.Dropout(0.3))

        self.layer6= nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Dropout(0.3))
        

        self.out_layer1 = nn.Sequential(
            nn.Linear(2048,28),
            nn.Softmax(1)
           )
        self.out_layer2 = nn.Sequential(
            nn.Linear(2048,28),
            nn.Softmax(1)
           )
        self.out_layer3 = nn.Sequential(
            nn.Linear(2048,28),
            nn.Softmax(1)
           )
        self.out_layer4 = nn.Sequential(
            nn.Linear(2048,28),
            nn.Softmax(1)
           )
        self.out_layer5 = nn.Sequential(
            nn.Linear(2048,28),
            nn.Softmax(1)
           )

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)



    def forward(self, x):
        out = self.layer1(x)
        #print(out.size(2),"test2")
        out = self.layer2(out)
        #print(out.size(2),"test2")
        out = self.layer3(out)
        #print(out.size(2),"test3")
        out = self.layer4(out)
        #print(out.size(2),"test4")
        out = self.layer5(out)
        #print(out.size(2),"test5")
        out = self.layer6(out)
        #print(out.size(0),"test6")
        out1 = self.out_layer1(out)
        out2 = self.out_layer2(out)
        out3 = self.out_layer3(out)
        out4 = self.out_layer4(out)
        out5 = self.out_layer5(out)
  
        return out1,out2,out3,out4,out5
        ##return out

