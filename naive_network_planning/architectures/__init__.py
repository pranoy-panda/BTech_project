import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torch.autograd import Variable
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy import io
import os

class ConvNet(torch.nn.Module):

    def __init__(self,vis):
        super(ConvNet, self).__init__()

        # activation
        self.LReLU = torch.nn.LeakyReLU()

        # display network layer output sizes
        self.vis = vis

        # for torch.nn.Conv2d :
        # Input: (N,Cin,Hin,Win)
        # Output: (N,Cout,Hout,Wout)       

        self.conv1=torch.nn.Conv2d(in_channels=1,out_channels=8, kernel_size=7,stride=2)
        torch.nn.init.xavier_uniform(self.conv1.weight)

        self.conv2=torch.nn.Conv2d(in_channels=8,out_channels=16, kernel_size=5,stride=2)
        torch.nn.init.xavier_uniform(self.conv2.weight)

        self.conv3=torch.nn.Conv2d(in_channels=16,out_channels=16, kernel_size=3,stride=2)
        torch.nn.init.xavier_uniform(self.conv3.weight)

        self.conv4=torch.nn.Conv2d(in_channels=16,out_channels=16, kernel_size=3,stride=2)
        torch.nn.init.xavier_uniform(self.conv4.weight)

        self.linear1 = torch.nn.Linear(2304,256)
        self.linear2 = torch.nn.Linear(256,128)
        self.linear3 = torch.nn.Linear(128,64)
        self.linear4 = torch.nn.Linear(64,32)
        self.linear5 = torch.nn.Linear(32,2)

    def forward(self,x):
        if self.vis:
            print(x.shape)
            x = self.LReLU(self.conv1(x))
            print(x.shape)
            x = self.LReLU(self.conv2(x))
            print(x.shape)
            x = self.LReLU(self.conv3(x))
            print(x.shape)
            x = self.LReLU(self.conv4(x))
            print(x.shape)
            x = x.view(x.size(0),-1)
            print(x.shape)
            x = self.LReLU(self.linear1(x))
            print(x.shape)
            x = self.LReLU(self.linear2(x))
            print(x.shape)
            x = self.LReLU(self.linear3(x))
            print(x.shape)  
            x = self.LReLU(self.linear4(x))
            print(x.shape) 
            x = self.LReLU(self.linear5(x))
            print(x.shape)                                               
            self.vis = False
        else:
            x = self.LReLU(self.conv1(x))
            x = self.LReLU(self.conv2(x))
            x = self.LReLU(self.conv3(x))
            x = self.LReLU(self.conv4(x))
            x = x.view(x.size(0),-1)
            x = self.LReLU(self.linear1(x))
            x = self.LReLU(self.linear2(x))
            x = self.LReLU(self.linear3(x))
            x = self.LReLU(self.linear4(x))
            x = self.LReLU(self.linear5(x))
            
        return x              





