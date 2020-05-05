#import library
# import RPi.GPIO as GPIO
import time
import cv2
from yolo import *
from data_processing import *
import numpy as np
import sys
from threading import Thread

#______________________________Define_______________________________________
# angle_put = 3.5                #DutyCycle need to put the Object out off conveyer
# default_angle = 1.8            #Default DutyCycle when it don't have Object

# cam_c_check = 0                #it make the process more exactly
# chanh_c_check = 0

# servo_CAM = 4                  #Servo Orange on GPIO 4
# servo_CHANH = 27               #Servo Lemon on GPIO 27
# #Setup Servo
# GPIO.setmode(GPIO.BCM)
# GPIO.setup(servo_CAM, GPIO.OUT)
# GPIO.setup(servo_CHANH, GPIO.OUT)

# p_CAM = GPIO.PWM(servo_CAM, 50) # GPIO 4 for PWM with 50Hz
# p_CAM.start(default_angle)

# p_CHANH = GPIO.PWM(servo_CHANH, 50) # GPIO 27 for PWM with 50Hz
# p_CHANH.start(default_angle)

#load model deep learning
model = load_model('weight/objdect3.h5', compile=False)

#______________________________Main loop______________________________________
while(1):

    type_value = 0  #to check this object is Orange or Lemon
    cap = cv2.VideoCapture(0)   #connect to Camera
    if (cap.isOpened()== False):    #if can't connect to Camera, the main loop will break out
        print("Error opening video stream or file")
        break

    ret, frame = cap.read() #read camera
    cap.release()           #disconect to Camera (it make the process run faster)

    if ret == True:
        img = frame/(255)
        img = cv2.resize(img, (416, 416))
        img = np.expand_dims(img, axis = 0)
        img = np.array(img)
        start = time.time()
        #predict object
        y = model.predict(img)
        print("Execute time = ", time.time() - start)

        true_boxs = interpret_netout(img[0]*255, y[0])

        img, type_value = draw_box_predict(img[0]*255, true_boxs)   #draw box predict around object and return type of Object (Orange or Lemon in type_value)
        img = np.array(img, dtype = np.uint8)
        cv2.imshow("result", img)       #Show img after predict

        #Control servo
        #type_value have 3 case
        #type_value = 1: Object is Orange
        #type_value = 2: Object is Lemon
        #type_value = 3: Object isn't Orange or Lemon

        # if type_value == 1: #is Orange
        #     cam_c_check += 1
        #     chanh_c_check = 0
        #     if cam_c_check == 2:    #when it's taken continuously, at least 2 times
        #         cam_c_check = 0
        #         p_CAM.ChangeDutyCycle(angle_put)
        #         time.sleep(10)      #wait 10s (take the Object out of the conveyer
        # else:
        #     if type_value == 2:     #like Orange
        #         chanh_c_check += 1
        #         cam_c_check = 0
        #         if chanh_c_check == 2:
        #             chanh_c_check = 0
        #             p_CHANH.ChangeDutyCycle(angle_put)
        #             time.sleep(5)
        #     else:
        #         if type_value == 3:     #set servo angle to default
        #             p_CAM.ChangeDutyCycle(default_angle)
        #             p_CHANH.ChangeDutyCycle(default_angle)
        #             time.sleep(0.1)
        #             p_CAM.ChangeDutyCycle(0)
        #             p_CHANH.ChangeDutyCycle(0)
        #             print("vo dinh")

        if cv2.waitKey(25) & 0xFF == ord('q'):  #press 'q' to quit main loop
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()
