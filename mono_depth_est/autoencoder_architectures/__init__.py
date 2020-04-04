import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torch.autograd import Variable
from torch.optim import lr_scheduler
#import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy import io
import os
import custom_loss_functions

# transfer learning based depth estimation naive network
class ConvAutoencoder(torch.nn.Module):

    def __init__(self,vis):
        super(ConvAutoencoder, self).__init__()

        model = models.resnet18(pretrained = False)
        new_model = torch.nn.Sequential(*list(model.children())[0:7])

        for child in new_model.children():
            for param in child.parameters():
                param.requires_grad = True	

        self.model = new_model

        # activation
        self.LReLU = torch.nn.LeakyReLU()

        # display network layer output sizes
        self.vis = vis

        # for torch.nn.Conv2d :
        # Input: (N,Cin,Hin,Win)
        # Output: (N,Cout,Hout,Wout)       

        self.deconv1=torch.nn.ConvTranspose2d(in_channels=256,out_channels=128, kernel_size=3,stride=2)
        torch.nn.init.xavier_uniform(self.deconv1.weight)

        self.maxunpool1=torch.nn.MaxUnpool2d(2, stride=2)

        self.deconv2=torch.nn.ConvTranspose2d(in_channels=128,out_channels=64, kernel_size=3,stride=2)
        torch.nn.init.xavier_uniform(self.deconv2.weight)

        self.maxunpool2=torch.nn.MaxUnpool2d(2, stride=2)

        self.deconv3=torch.nn.ConvTranspose2d(in_channels=64,out_channels=64, kernel_size=3,stride=2)
        torch.nn.init.xavier_uniform(self.deconv3.weight)

        self.deconv4=torch.nn.ConvTranspose2d(in_channels=64,out_channels=64, kernel_size=3,stride=2)
        torch.nn.init.xavier_uniform(self.deconv4.weight)        

        self.conv1=torch.nn.Conv2d(in_channels=64,out_channels=16, kernel_size=7,stride=1, padding=0)
        torch.nn.init.xavier_uniform(self.conv1.weight) #Xaviers Initialisation                
        
        self.conv2=torch.nn.Conv2d(in_channels=16,out_channels=16, kernel_size=7,stride=1, padding=0)
        torch.nn.init.xavier_uniform(self.conv2.weight) #Xaviers Initialisation   

        self.conv3=torch.nn.Conv2d(in_channels=16,out_channels=16, kernel_size=7,stride=1, padding=0)
        torch.nn.init.xavier_uniform(self.conv3.weight) #Xaviers Initialisation                

        self.conv4=torch.nn.Conv2d(in_channels=16,out_channels=32, kernel_size=5,stride=1, padding=0)
        torch.nn.init.xavier_uniform(self.conv4.weight) #Xaviers Initialisation  

        self.conv5=torch.nn.Conv2d(in_channels=32,out_channels=32, kernel_size=5,stride=1, padding=0)
        torch.nn.init.xavier_uniform(self.conv5.weight) #Xaviers Initialisation                

        self.conv6=torch.nn.Conv2d(in_channels=32,out_channels=1, kernel_size=3,stride=1, padding=0)
        torch.nn.init.xavier_uniform(self.conv6.weight) #Xaviers Initialisation  

        self.dropout0 = torch.nn.Dropout2d(0.1)   
        self.dropout1 = torch.nn.Dropout2d(0.1)                                     	


    def forward(self,x):
        x = self.model(x)
        if self.vis:
            print(x.shape)
            x = self.LReLU(self.deconv1(x))
            print(x.shape)
            x = self.LReLU(self.deconv2(x))
            print(x.shape)
            x = self.LReLU(self.deconv3(x))
            print(x.shape)
            x = self.LReLU(self.deconv4(x))
            print(x.shape)
            x = self.LReLU(self.conv1(x))
            print(x.shape)
            x = self.LReLU(self.conv2(x))
            print(x.shape)
            #x = self.dropout0(x)
            x = self.LReLU(self.conv3(x))
            print(x.shape)
            x = self.LReLU(self.conv4(x))
            print(x.shape)
            #x = self.dropout0(x)
            x = self.LReLU(self.conv5(x))
            print(x.shape)
            y = self.LReLU(self.conv6(x))
            print(y.shape)
            self.vis = False
        else:
            x = self.LReLU(self.deconv1(x))
            x = self.LReLU(self.deconv2(x))
            x = self.LReLU(self.deconv3(x))
            x = self.LReLU(self.deconv4(x))
            x = self.LReLU(self.conv1(x))
            x = self.LReLU(self.conv2(x))
            #x = self.dropout0(x)
            x = self.LReLU(self.conv3(x))
            x = self.LReLU(self.conv4(x))
            #x = self.dropout1(x)
            x = self.LReLU(self.conv5(x))
            y = self.LReLU(self.conv6(x))        
		
        return y              






