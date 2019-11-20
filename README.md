# YOLO_v2_tiny
----
# Introduction
----
This is the YOLO V2 tiny model implemented on keras.
# Requirement
----
Using [this script](https://bitbucket.org/minhtan97/yolo_v2_tiny/src/master/requirement.txt) to install indispensable libs:
```
sudo chmod +x requirement.txt
./requirement.txt
```
# Usage
----
## 1. Training:
Add training data (images and announcements) to the data folder. Then using [this script](https://bitbucket.org/minhtan97/yolo_v2_tiny/src/master/train.py) to train the model. The weight after training is stored in the weight folder.
```
python3 train.py
```
## 2. Test image:
```
python3 test.py <IMAGE_PATH>
```
## 3. Test on camera:
```
python3 camera.py
```
# References
----
## Documentations
* [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/pdf/1506.02640)
* [YOLO 9000](https://arxiv.org/pdf/1612.08242)
* [Understanding YOLO](https://hackernoon.com/understanding-yolo-f5a74bbc7967)
* [YOLO tutorial](https://trungthanhnguyen0502.github.io/computer%20vision/2018/12/10/yolo_tutorial-2-yolo2-algorithms/)
* [A Comprehensive Guide To Object Detection Using YOLO Framework](https://medium.com/@pratheesh.27998/object-detection-part1-4dbe5147ad0a)
