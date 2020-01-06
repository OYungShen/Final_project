# -*- coding: UTF-8 -*-
import torch.nn as nn
import torch
# CNN Model (2 conv layer)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Dropout(0.3))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout(0.3))
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Dropout(0.3))
        self.layer7= nn.Sequential(
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

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        #print(out.size(1),"test")
        out = self.layer3(out)
        #print(out.size(0),"test")
        out = self.layer4(out)
        #print(out.size(0),"test")
        out = self.layer5(out)
        #print(out.size(0),"test")
        out = self.layer6(out)
        #print(out.size(0),"test")
        out = self.layer7(out)
        out1 = self.out_layer1(out)
        out2 = self.out_layer2(out)
        out3 = self.out_layer3(out)
        out4 = self.out_layer4(out)
        out5 = self.out_layer5(out)
        '''
        out = {
            'out1':out1,
            'out2':out2,
            'out3':out3,
            'out4':out4,
            'out5':out5
        }
        '''
        return out1,out2,out3,out4,out5
        ##return out

