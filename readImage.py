from flask import Flask, render_template, Response, request
from imutils.video import VideoStream
from yolo import *
from data_processing import *
from threading import Thread
import threading
import sys
import cv2
import imutils
import time
import ctypes
import numpy as np

outputFrame = None
threadID = None
stopProcess = False
CamNum = 0
ChanhNum = 0
lock = threading.Lock()
lock_frame =  threading.Lock()
# vs = VideoStream(src=0).start()
cap = cv2.VideoCapture(0)
time.sleep(2.0)

model = load_model('weight/objdect3.h5', compile=False)

app = Flask(__name__)


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
				print("helu")
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
	global cap, outputFrame, lock_frame, stopProcess, CamNum, ChanhNum
	while (not stopProcess):
		ret, frame = cap.read()
		# frame = imutils.resize(frame, width=700)

		if ret == True:
			img = frame/(255)
			img = cv2.resize(frame, (416, 416))
			img = np.expand_dims(img, axis = 0)
			img = np.array(img)
		# 	start = time.time()
		# 	#predict object
			y = model.predict(img)
		# 	# print("Execute time = ", time.time() - start)

			true_boxs = interpret_netout(img[0]*255, y[0])

			img, type_value = draw_box_predict(img[0]*255, true_boxs)   #draw box predict around object and return type of Object (Orange or Lemon in type_value)
			img = np.array(img, dtype = np.uint8)

		with lock_frame:
			outputFrame = img.copy()

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
cap.release()