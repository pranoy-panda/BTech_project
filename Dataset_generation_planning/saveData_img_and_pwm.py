import serial
import cv2
import numpy as np

port = '/dev/ttyUSB0'
baud = 38400
 
ser = serial.Serial(port, baud, timeout=1)
ser.flushInput()

# for saving video
# Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi',fourcc, 30.0, (640,480))
cam = cv2.VideoCapture(1)
f = open("output.txt","w")
counter = 0
c = 0
while True:
	ret, frame = cam.read()

	#cv2.imshow('img',frame)

	if ser.isOpen():
		ser_bytes = ser.readline()
		if len(ser_bytes)>=7:
			cv2.waitKey(1000) 
			ret,frame = cam.read()
			#out.write(frame)
			cv2.imwrite('images/'+str(c)+'.jpg',frame)
			f.write(ser_bytes)
			print str(c)+' '+ser_bytes
			print ret
			ser_bytes = ''

			#f.write("\n")
			c+=1
			##
			#ser.write()


	else:
		print("Port unaccessible")
		break 
	cv2.waitKey(250)	
	counter+=1
cam.release()
out.release()        
f.close()