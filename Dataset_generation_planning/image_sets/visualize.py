import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

path_to_folder = "/media/pranoy/New Volume1/Pytor/Untitled Folder/Dataset_generation_fin_yr_prj/image_sets/final_compilation/set2/"
txt_file = open(path_to_folder+"pwm.txt")
IMAGE_PATH = path_to_folder+"images/"

list_of_files = os.listdir(IMAGE_PATH)

for i in xrange(len(list_of_files)):
	img = plt.imread(IMAGE_PATH+str(i)+'.jpg') # H,W,C
	# img = img_BGR
	# img[:,:,0] = img_BGR[:,:,2]
	# img[:,:,2] = img_BGR[:,:,0]
	val = txt_file.readline()

	pwmL = int(val.split('\r')[0].split(',')[0])
	pwmR = int(val.split('\r')[0].split(',')[1])
	#cv2.imshow('img',img)

	plt.subplot(121)
	plt.bar([0,1],[pwmL,pwmR],width = 0.4)

	#print val
	#cv2.waitKey(500)
	plt.subplot(122)
	plt.imshow(img)
	
	plt.pause(1)
	plt.clf()
	#plt.close()
