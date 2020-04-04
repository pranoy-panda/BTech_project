import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy import io
import os
from torchvision import models
import dataset_Class
from architectures import ConvNet
import time

def main_test():
    model = ConvNet(vis = False)
    #model.load_state_dict(torch.load('249.pkl'), strict=True)
    #model.load_state_dict(torch.load('new_model_trial_2/252.pkl'), strict=False)
    #model.load_state_dict(torch.load('new_model/362.pkl'), strict=False)
    #model.load_state_dict(torch.load('model_trial_84_sampl/510.pkl'), strict=True)
    model.load_state_dict(torch.load('model_weights/490.pkl'), strict=False)

    p = "/media/pranoy/New Volume1/Pytor/Untitled Folder/Dataset_generation_fin_yr_prj/image_sets/validation/set1"
    l1 = os.listdir(p+'/images')
    #l1.sort()

    for i in l1:
        print p+'/images/'+i
        img = cv2.imread(p+'/images/'+i)
        img_copy = img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # pre-processing
        img.astype(float)                      # convert to float
        img=img/255                            # scale the range to [0,1]
   
        img = cv2.resize(img,(227,227))
        img = img.reshape((1,1,img.shape[0],img.shape[1]))

        img = torch.from_numpy(img)
        img = img.type('torch.FloatTensor')

        y_pred = model.forward(img)
        y_pred = y_pred.detach().numpy()

        # cv2.imshow('img',img_copy)
        # print y_pred[0]*255
        # cv2.waitKey(500) 

        
        plt.subplot(121)
        plt.bar([0,1],[y_pred[0][0]*255,y_pred[0][1]*255],width = 0.4)

        #print val
        #cv2.waitKey(500)
        plt.subplot(122)
        plt.imshow(img_copy)
        
        plt.pause(1)
        plt.clf()

        print "***************************"       


if __name__ == '__main__':
	main_test()