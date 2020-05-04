# IPS-Orange_Lemon_Classification
----
# Introduction
----
Using Yolo tiny v2 to classify Orange and Lemon on conveyor with Raspberry pi 3

# Source
----
Yolo tiny v2 at [this script](https://bitbucket.org/minhtan97/yolo_v2_tiny/src/master/)

# Usage
----
## 1.Download weight:
Download [this weight](https://drive.google.com/open?id=1FAXgugjOhFOtA0BzlK9HstzqPs_YbJSv) and extract file in "IPS-Orange_Lemon_Classification" folder. You will have weight folder

## 2.Test code:
Run camera.py code in your PC (Raspberry, window)
```
Raspberry:      python3 camera.py
Windows:        python camera.py
```
## 3.Setup:
I you Raspberry to classify Orange and Lemon, using servo to put Object out of the conveyor
```
Servo 1 --> GPIO 4
Servo 2 --> GPIO 27
```
## 4.Run model:
```
python3 servo_control.py
```