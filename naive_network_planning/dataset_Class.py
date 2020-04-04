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

# class DataSet(Dataset):

#     def __init__(self,IMAGE_PATH,ANGLE_PATH):
#         image_list = os.listdir(IMAGE_PATH)
#         image_list.sort()
#         angle_list = os.listdir(ANGLE_PATH)
#         angle_list.sort()
#         inp = []
#         out = []
#         for i,j in zip(image_list,angle_list):
#             img = cv2.imread(IMAGE_PATH+i) # H,W,C
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
#             print img.shape

#             # pre-processing
#             img.astype(float)                      # convert to float
#             img=img/255                            # scale the range to [0,1]
            
#             img = cv2.resize(img,(227,227))
#             img = img.reshape((img.shape[0],img.shape[1],1))
#             img = img.reshape((img.shape[2],img.shape[0],img.shape[1]))

#             inp.append(img)
#             out.append(np.array([1]))

#         #convert to tensors
#         self.inp = torch.from_numpy(np.array(inp))
#         self.inp = self.inp.type('torch.FloatTensor')
#         self.out = torch.from_numpy(np.array(out))
#         self.out = self.out.type('torch.FloatTensor')

#         self.len = len(image_list)	

#     def __getitem__(self, index):
#         return self.inp[index],self.out[index]

#     def __len__(self):
#         return self.len

def minof(a,b):
    if a>b:
        return b
    else:
        return a

class DataSet(Dataset):

    def __init__(self,PATH):
        list_of_files = os.listdir(PATH)
        list_of_files.sort()
        inp = []
        out = []

        for k in list_of_files:
            IMAGE_PATH = PATH+k+"/images/"
            image_list = os.listdir(IMAGE_PATH)
            image_list.sort()
            txt_file = open(PATH+k+"/pwm.txt")
            for i in xrange(len(image_list)):
                img = cv2.imread(IMAGE_PATH+str(i)+'.jpg') # H,W,C
                val = txt_file.readline()

                # cv2.imshow('img',img)
                # print val
                # cv2.waitKey(500)

                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                #print img.shape

                # pre-processing
                img.astype(float)                      # convert to float
                img=img/255                            # scale the range to [0,1]
                
                img = cv2.resize(img,(227,227))
                img = img.reshape((img.shape[0],img.shape[1],1))
                img = img.reshape((img.shape[2],img.shape[0],img.shape[1]))

                try:
                    pwmL = int(val.split('\r')[0].split(',')[0])/255.0
                    pwmR = int(val.split('\r')[0].split(',')[1])/255.0

                    if (i%4==0 or abs(pwmL-pwmR)>20) and (not (pwmL==0 and pwmR==0)):
                        inp.append(img)
                        out.append(np.array([minof(1,pwmL),minof(1,pwmR)]))
                except:
                    pass
            txt_file.close()

            #convert to tensors
            self.inp = torch.from_numpy(np.array(inp))
            self.inp = self.inp.type('torch.FloatTensor')
            self.out = torch.from_numpy(np.array(out))
            self.out = self.out.type('torch.FloatTensor')

            self.len = len(inp)

    def __getitem__(self, index):
        return self.inp[index],self.out[index]

    def __len__(self):
        return self.len        