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
import dataset_Class
from architectures import ConvNet

#global vars
PATH = "/media/pranoy/New Volume1/Pytor/Untitled Folder/Dataset_generation_fin_yr_prj/image_sets/final_compilation/"
PATH_VAL = "/media/pranoy/New Volume1/Pytor/Untitled Folder/Dataset_generation_fin_yr_prj/image_sets/validation/"
# IMAGE_PATH = "data/images/"
# DEPTH_PATH = "data/depths/"
num_of_epochs = 500
vis = True # print each conv layer output size
batch_size = 2

def training_procedure():
    global num_of_epochs
    #dataset = dataset_Class.DataSet(IMAGE_PATH,DEPTH_PATH)
    dataset = dataset_Class.DataSet(PATH)
    train_loader = DataLoader(dataset = dataset,batch_size = batch_size,shuffle = True,num_workers = 1)

    # for validation
    dataset_val = dataset_Class.DataSet(PATH_VAL)
    train_loader_val = DataLoader(dataset = dataset_val,batch_size = batch_size,shuffle = True,num_workers = 1)

    #instantiate model
    model = ConvNet(vis = vis)

    #loss function
    mse_LossFunc = torch.nn.MSELoss(size_average = True)
    optimizer = torch.optim.Adam(model.parameters(),lr = 3e-05)  
    #optimizer = torch.optim.SGD(model.parameters(), lr=3e-02, momentum = 0.9, weight_decay = 0.0005)
    #exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)  
    

    list_for_loss = []
    list_for_epoch_loss = []
    list_for_valid_loss = []

    #Training loop
    for epoch in xrange(num_of_epochs):
        #exp_lr_scheduler.step()
        print("***********************************************************")       
        batch_loss_avg = 0
        for i,data in enumerate(train_loader,0):
            inputs,labels = data
            #print inputs.shape
            inputs,labels = Variable(inputs),Variable(labels)
            
            #forward pass
            y_pred = model.forward(inputs)

            mse_loss = mse_LossFunc(y_pred,labels)

            # # #L2 regularization
            # l2_reg = None
            # reg_lambda = 0.001
            # for W in model.parameters():
            #     if l2_reg is None:
            #         l2_reg = W.norm(2)
            #     else:
            #         l2_reg = l2_reg + W.norm(2)    

            batch_loss_avg+=mse_loss.item()
            list_for_loss.append(mse_loss.item())
            
            print epoch,"\t",i,"\t",'total loss: ',mse_loss.item()

            #zero gradients
            optimizer.zero_grad()
            # backprop
            mse_loss.backward()
            optimizer.step()
        if epoch%10==0:    
            torch.save(model.state_dict(),'model_weights/'+str(epoch)+'.pkl')  
        batch_loss_avg/=(i+1)
        print "average loss: ",batch_loss_avg
        list_for_epoch_loss.append(batch_loss_avg)

        # for validation
        batch_loss_avg_val = 0
        for i,data in enumerate(train_loader_val,0):
            inputs,labels = data
            #print inputs.shape
            inputs,labels = Variable(inputs),Variable(labels)
            
            #forward pass
            y_pred = model.forward(inputs)
            mse_loss = mse_LossFunc(y_pred,labels)
            batch_loss_avg_val+=mse_loss.item()

        batch_loss_avg_val/=(i+1)
        print "valid average loss: ",batch_loss_avg_val
        list_for_valid_loss.append(batch_loss_avg_val)
            
    #torch.save(model.state_dict(),'9.pkl')
    plt.plot(list_for_epoch_loss,'b')
    plt.plot(list_for_valid_loss,'r')
    plt.show()               
                    

def main():
    training_procedure()

if __name__ == '__main__':
	main()