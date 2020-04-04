import torch
# for overcoming CUDA initialization error
try:
    torch.multiprocessing.set_start_method("spawn")
except RuntimeError:
    pass    
#    
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
import DepthMap_dataset_Class
from autoencoder_architectures import ConvAutoencoder

#global vars
IMAGE_PATH = "data/images/"
DEPTH_PATH = "data/depths/"

num_of_epochs = 1000
vis = True # print each conv layer output size
batch_size = 16


def main():
    global num_of_epochs
    dataset = DepthMap_dataset_Class.DepthMapDS(IMAGE_PATH,DEPTH_PATH)
    #dataset_valid = DepthMap_dataset_Class.DepthMapDS(IMAGE_PATH_VALID,DEPTH_PATH_VALID)
    train_loader = DataLoader(dataset = dataset,batch_size = batch_size,shuffle = True,num_workers = 1)
    #train_loader_valid = DataLoader(dataset = dataset_valid, batch_size = batch_size, shuffle = True,num_workers = 1)
    #instantiate model
    model = ConvAutoencoder(vis = vis)
    model.cuda()

    #loss function
    sc_LossFunc = custom_loss_functions.SC_loss()
    mse_LossFunc = torch.nn.MSELoss(size_average = True)
    #optimizer = torch.optim.Adam(model.parameters(),lr = 3e-04)  
    optimizer = torch.optim.SGD(model.parameters(), lr=3e-02, momentum = 0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)  
    

    list_for_loss = []
    list_for_epoch_loss = []
    f = open("loss_model.txt",'w')
    f.write("epoch,avg_loss\n")
    #Training loop
    for epoch in range(num_of_epochs):
        exp_lr_scheduler.step()
        print("***********************************************************")       
        batch_loss_avg = 0
        for i,data in enumerate(train_loader,0):
            inputs,labels = data
            inputs,labels = Variable(inputs).cuda(),Variable(labels).cuda()
            
            #forward pass
            y_pred = model.forward(inputs)

            #comuting the mixed loss (l1 and ssim)
            alpha = 0.85
            sc_loss = (1-sc_LossFunc(y_pred,labels))
            #mse_loss = (y_pred - labels).pow(2).mean()
            mse_loss = mse_LossFunc(y_pred,labels)

            #L2 regularization
            l2_reg = None
            reg_lambda = 0.02
            for W in model.parameters():
                if l2_reg is None:
                    l2_reg = W.norm(2)
                else:
                    l2_reg = l2_reg + W.norm(2)    

            #complete loss equation
            loss = alpha*sc_loss + (1-alpha)*mse_loss + l2_reg*reg_lambda

            batch_loss_avg+=loss.item()
            list_for_loss.append(loss.item())
            
            print (epoch,"\t",i,"\t",'total loss: ',loss.item(),' mse:',mse_loss.item(),' sc_loss:',sc_loss.item(),' l2_reg:',l2_reg.item()*reg_lambda)

            #zero gradients
            optimizer.zero_grad()
            # backprop
            loss.backward()
            optimizer.step()

        if epoch%10==0:
            torch.save(model.state_dict(),'saved_models/'+str(epoch)+'.pkl')    

        batch_loss_avg/=(i+1)
        print ("average loss: ",batch_loss_avg)
        s = str(epoch)+","+str(batch_loss_avg)+'\n'#+","+str(valid_loss_avg)+"\n"
        f.write(s)
        #list_for_epoch_loss.append(batch_loss_avg)
    
    f.close()    

if __name__ == '__main__':
	main()