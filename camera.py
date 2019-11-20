import cv2
from yolo import *
from data_processing import *
import numpy as np
import sys

# cap = cv2.VideoCapture(sys.argv[1])
cap = cv2.VideoCapture(0)
model = load_model('./weight/objdect3.h5', compile=False)

if (cap.isOpened()== False): 
    print("Error opening video stream or file")

check = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    print(frame.shape)
    if ret == True:
        img = frame/(255)
        img = cv2.resize(img, (416, 416))
        img = np.expand_dims(img, axis = 0)
        img = np.array(img)
        # print(img.shape)
        start = time.time()
        y = model.predict(img)
        print("Execute time = ", time.time() - start)
        
        true_boxs = interpret_netout(img[0]*255, y[0])

        img, check = draw_box_predict(img[0]*255, true_boxs)

        if check == 1:
            print("Cam ne")
        else:
            if check == 2:
                print("Chanh ne")

        img = np.array(img, dtype = np.uint8)
        # print(img.shape)
        cv2.imshow("result", img)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
 
    else: 
        break

cap.release()
cv2.destroyAllWindows()