import cv2
import numpy as np
from scipy import io
import os

#global vars
IMAGE_PATH = "data/images/"
DEPTH_PATH = "data/depths/"

image_list = os.listdir(IMAGE_PATH)
depth_list = os.listdir(DEPTH_PATH)

for i,j in zip(image_list,depth_list):
	img = cv2.imread(IMAGE_PATH+i) # H,W,C
	#img = img.reshape((img.shape[2],img.shape[0],img.shape[1]))
	depth = io.loadmat(DEPTH_PATH+j)['a']
	depth*=(255/np.max(depth)) # this mult. is done only for visualization
	cv2.imshow('image',img)
	cv2.imshow('depth',depth)
	cv2.imwrite('depth.png',depth)
	cv2.waitKey(0)