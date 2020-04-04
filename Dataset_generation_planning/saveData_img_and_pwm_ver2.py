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
f = open("output.txt","w")
counter = 0
c = 0
cam = cv2.VideoCapture(1)
ret,frame = cam.read()
cv2.imwrite('images/'+str(c)+'.jpg',frame)

while True:
	if ser.isOpen():
		ser.write('a') # send ACK SIG
		
		try:
			print "ACK sent"
			ser_bytes = ser.readline()
			while ord(ser_bytes[0])!=98: 
				# wait until MEGA ACK SIG
				ser_bytes = ser.readline()

			print "MEGA has acknowledged"
			ser_bytes = ser.readline()
			print "read Data from MEGA"
			f.write(ser_bytes)
			# wait some time for the vehicle to move

			ser_bytes = ser.readline()
			while ord(ser_bytes[0])!=99:
				ser_bytes = ser.readline()
			cv2.waitKey(250)	

			print "robot movement complete"	
			
			# read image
			ret,frame = cam.read()
			cv2.imwrite('images/'+str(c)+'.jpg',frame)
			
			print "read new frame"
			c+=1	
		except:
			pass		

	else:
		print("Port unaccessible")
		break 

	counter+=1

cam.release()
#out.release()        
f.close()