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

class DepthMapDS(Dataset):

    def __init__(self,IMAGE_PATH,DEPTH_PATH):
        image_list = os.listdir(IMAGE_PATH)
        image_list.sort()
        depth_list = os.listdir(DEPTH_PATH)
        depth_list.sort()
        inp = []
        out = []
        for i,j in zip(image_list,depth_list):
            img = cv2.imread(IMAGE_PATH+i) # H,W,C

            # pre-processing
            img.astype(float)                      # convert to float
            img=img/255                            # scale the range to [0,1]
            #mean = np.array([0.485, 0.456, 0.406]) # mean (for the pretrained model)
            #std = np.array([0.229, 0.224, 0.225])  # standard dev.
            #img = std * img + mean                 # normalizing img with the above mean and std. dev. 
            #img = np.clip(img, 0, 1)               # again rescale to [0,1] 
            
            img = cv2.resize(img,(227,227))
            img = img.reshape((img.shape[2],img.shape[0],img.shape[1]))
            depth = io.loadmat(DEPTH_PATH+j)['a']
            #depth.astype(float)
            #depth = (depth*10)/(2**16 - 1)
            depth = cv2.resize(depth,(227,227))
            depth = depth.reshape((1,depth.shape[0],depth.shape[1]))
            inp.append(img)
            out.append(depth)

        #convert to tensors
        #self.inp = torch.from_numpy(np.array(inp))
        #self.inp = self.inp.type('torch.FloatTensor')
        #self.out = torch.from_numpy(np.array(out))
        #self.out = self.out.type('torch.FloatTensor')
        self.inp = torch.from_numpy(np.array(inp))
        self.inp = self.inp.type('torch.cuda.FloatTensor')
        self.out = torch.from_numpy(np.array(out))
        self.out = self.out.type('torch.cuda.FloatTensor')
        self.len = len(image_list)	

    def __getitem__(self, index):
        return self.inp[index],self.out[index]

    def __len__(self):
        return self.len
