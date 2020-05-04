from flask import Flask, render_template, Response, request
from imutils.video import VideoStream
from yolo import *
from data_processing import *
from threading import Thread
import RPi.GPIO as GPIO
import threading
import cv2
import imutils
import time
import ctypes
import sys
import numpy as np

outputFrame = None
threadID = None
stopProcess = False
CamNum = 0
ChanhNum = 0
lock = threading.Lock()
lock_frame =  threading.Lock()
vs = VideoStream(src=0).start()
time.sleep(2.0)

app = Flask(__name__)

#______________________________Define_______________________________________
angle_put = 3.5                #DutyCycle need to put the Object out off conveyer
default_angle = 1.8            #Default DutyCycle when it don't have Object

cam_c_check = 0                #it make the process more exactly
chanh_c_check = 0

servo_CAM = 4                  #Servo Orange on GPIO 4
servo_CHANH = 27               #Servo Lemon on GPIO 27
#Setup Servo
GPIO.setmode(GPIO.BCM)
GPIO.setup(servo_CAM, GPIO.OUT)
GPIO.setup(servo_CHANH, GPIO.OUT)

p_CAM = GPIO.PWM(servo_CAM, 50) # GPIO 4 for PWM with 50Hz
p_CAM.start(default_angle)

p_CHANH = GPIO.PWM(servo_CHANH, 50) # GPIO 27 for PWM with 50Hz
p_CHANH.start(default_angle)

#load model deep learning
model = load_model('/home/pi/yolo_tiny_v2/weight/objdect3.h5', compile=False)

#______________________________Main Funct______________________________________

@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html", CamNum=CamNum, ChanhNum=ChanhNum)

@app.route('/controlModel', methods=['POST'])
def controlModel():
	print("in POST--------------------")
	global threadID, stopProcess
	if request.method == 'POST':
		msg = request.json['msg']
		print(msg)
		if msg == '0':
			print("stop")
			with lock:
				stopProcess = True
				threadID.join()
				
		elif (msg == '1') & (stopProcess == True):
			print("run")
			with lock:
				stopProcess = False
				threadID = None
				threadID = threading.Thread(target=ImageProcessing, daemon=True)
				threadID.start()

	print(threading.active_count())
	print("End POST-------------------")
	return render_template('index.html', CamNum=CamNum, ChanhNum=ChanhNum)

def ImageProcessing():
	global vs, outputFrame, lock_frame, stopProcess, CamNum, ChanhNum
	while (not stopProcess):
		frame = vs.read() #read camera

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
        # cv2.imshow("result", img)       #Show img after predict

        x, y, _ = img.shape

		newFrame = np.zeros((x + 200, y, 3), dtype=int)	
		newFrame[0:x, :] = img[:,:]

		CamNum = np.sum(newFrame)
		ChanhNum = CamNum - 10

		cv2.putText(newFrame, f"CamNum: {CamNum}", (50, x + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
		cv2.putText(newFrame, f"ChanhNum: {ChanhNum}", (50, x + 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        with lock_frame:
			outputFrame = newFrame.copy()
        #Control servo
        #type_value have 3 case
        #type_value = 1: Object is Orange
        #type_value = 2: Object is Lemon
        #type_value = 3: Object isn't Orange or Lemon

        if type_value == 1: #is Orange
            cam_c_check += 1
            chanh_c_check = 0
            if cam_c_check == 2:    #when it's taken continuously, at least 2 times
                cam_c_check = 0
                p_CAM.ChangeDutyCycle(angle_put)
                time.sleep(10)      #wait 10s (take the Object out of the conveyer
        else:
            if type_value == 2:     #like Orange
                chanh_c_check += 1
                cam_c_check = 0
                if chanh_c_check == 2:
                    chanh_c_check = 0
                    p_CHANH.ChangeDutyCycle(angle_put)
                    time.sleep(5)
            else:
                if type_value == 3:     #set servo angle to default
                    p_CAM.ChangeDutyCycle(default_angle)
                    p_CHANH.ChangeDutyCycle(default_angle)
                    time.sleep(0.1)
                    p_CAM.ChangeDutyCycle(0)
                    p_CHANH.ChangeDutyCycle(0)
                    print("vo dinh")

        # if cv2.waitKey(25) & 0xFF == ord('q'):  #press 'q' to quit main loop
        #     break

def generate():
	# grab global references to the output frame and lock variables
	global outputFrame, lock

	# loop over frames from the output stream
	while True:
		# wait until the lock is acquired
		with lock_frame:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if outputFrame is None:
				continue

			# encode the frame in JPEG format
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

			# ensure the frame was successfully encoded
			if not flag:
				continue

		# yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
	threadID = threading.Thread(target=ImageProcessing, daemon=True)
	threadID.start()
	app.run()

vs.stop()