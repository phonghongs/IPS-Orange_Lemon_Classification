import cv2
import copy
import os
import numpy as np
import xml.etree.ElementTree as ET 
from matplotlib import pyplot as plt
from random import randint
from bound_box import BoundBox
import time

def get_data(img_dir):
    directories = [d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))]
    data = []
    for d in directories:
        data_dir = os.path.join(img_dir, d)
        images = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".jpg")]
        for img in images:
            new_image = {}
            if os.path.exists(img.replace(".jpg", ".xml")):
                new_image['file_name'] = img
                print(new_image['file_name'])
                tree = ET.parse(img.replace(".jpg", ".xml"))
                print(img.replace(".jpg", ".xml"))
                root = tree.getroot()
                num_obj = len(root) - 6
                new_image['fruit_num'] = int(num_obj)
                new_image['fruit_type'] = []
                new_image['fruit_coordinate'] = []
                for d in range(num_obj):
                    print(root[d + 6][0].text)
                    new_image['fruit_type'].append(root[d + 6][0].text)
                    coor = []
                    for j in range(4):
                        coor.append(int(root[d + 6][4][j].text))
                    new_image['fruit_coordinate'].append(coor)

                data.append(new_image)
            else:
                continue
    return data


def plot_img(img):
    plt.figure()
    plt.axis('off')
    img = np.array(img, dtype = np.uint8)
    plt.imshow(img[:, :, :: -1]) 
    plt.show()


def draw_box(img, boxes, labels):
    copy_image = copy.deepcopy(img)
    for i, box in enumerate(boxes):
        cv2.rectangle(copy_image, (box['xmin'],box['ymin']), (box['xmax'], box['ymax']), (0,255,0), 10)
        startX = box['xmin']
        startY = box['ymin'] - 15 if box['ymin'] - 15 > 15 else box['ymin'] + 15
        cv2.putText(copy_image, labels[i], (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    return copy_image


def draw_box_predict(img, boxes):
    copy_image = copy.deepcopy(img)
    check = 3
    for box in boxes:
        cv2.rectangle(copy_image, (box[0],box[1]), (box[2],box[3]), (0,255,0), 10)
        startX = box[0]
        startY = box[1] - 15 if box[1] - 15 > 15 else box[1] + 15
        print(softmax(box[5]))
        if np.argmax(box[5]) == 0:
            cv2.putText(copy_image, "Cam", (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            check = 1
        elif np.argmax(box[5]) == 1:
            cv2.putText(copy_image, "Chanh", (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            check = 2
    return copy_image, check


def aug_img(img_data, NORM_H= 416, NORM_W = 416):
    img = cv2.imread(img_data['file_name'])
    if(img is None):
        return None, None
    h, w, c = img.shape

    # scale the image
    scale = np.random.uniform() / 10. + 1.
    img = cv2.resize(img, (0,0), fx = scale, fy = scale)

    # translate the image
    max_offx = (scale-1.) * w
    max_offy = (scale-1.) * h
    offx = int(np.random.uniform() * max_offx)
    offy = int(np.random.uniform() * max_offy)
    img = img[offy : (offy + h), offx : (offx + w)]

    # flip the image
    flip = np.random.binomial(1, .5)
    if flip > 0.5:
        img = cv2.flip(img, 1)
    # cv2.imshow("pre_img", img)
    # cv2.waitKey(0)
    img = img/(255)
    # cv2.imshow("pre_img", img)
    # cv2.waitKey(0)

    # resize the image to standard size
    img = cv2.resize(img, (NORM_H, NORM_W))

    # fix object's position and size
    bound_box = []
    labels = []

    for lb in img_data['fruit_type']:
        labels.append(lb)

    for rect in img_data['fruit_coordinate']:
        box = {}
        box['xmin'] = rect[0]
        box['xmin'] = int(box['xmin'] * scale - offx)
        box['xmin'] = int(box['xmin'] * float(NORM_W) / w)
        box['xmin'] = max(min(box['xmin'], NORM_W), 0)

        box['xmax'] = rect[2]
        box['xmax'] = int(box['xmax'] * scale - offx)
        box['xmax'] = int(box['xmax'] * float(NORM_W) / w)
        box['xmax'] = max(min(box['xmax'], NORM_W), 0)

        box['ymin'] = rect[1]
        box['ymin'] = int(box['ymin'] * scale - offy)
        box['ymin'] = int(box['ymin'] * float(NORM_H) / h)
        box['ymin'] = max(min(box['ymin'], NORM_H), 0)

        box['ymax'] = rect[3]
        box['ymax'] = int(box['ymax'] * scale - offy)
        box['ymax'] = int(box['ymax'] * float(NORM_H) / h)
        box['ymax'] = max(min(box['ymax'], NORM_H), 0)
        if flip > 0.5:
            xmin = box['xmin']
            box['xmin'] = NORM_W - box['xmax']
            box['xmax'] = NORM_W - xmin
        bound_box.append(box)
    return img, bound_box, labels


def aug_test_image(img_data, NORM_H= 416, NORM_W = 416):
    imgs = []
    img = cv2.imread(img_data)
    if(img is None):
        return None, None
    h, w, c = img.shape

    # scale the image
    scale = np.random.uniform() / 10. + 1.
    img = cv2.resize(img, (0,0), fx = scale, fy = scale)

    # translate the image
    max_offx = (scale-1.) * w
    max_offy = (scale-1.) * h
    offx = int(np.random.uniform() * max_offx)
    offy = int(np.random.uniform() * max_offy)
    img = img[offy : (offy + h), offx : (offx + w)]

    # flip the image
    flip = np.random.binomial(1, .5)
    if flip > 0.5:
        img = cv2.flip(img, 1)
    img = img/(255)

    # resize the image to standard size
    img = cv2.resize(img, (NORM_H, NORM_W))
    imgs.append(img)
    imgs = np.array(imgs)
    return imgs



def data_gen(imgs_data, batch_size, NORM_W = 416, NORM_H = 416, GRID_W = 13, GRID_H = 13, BOX=5):
    imgs_len = len(imgs_data)
    shuffled_indices = np.random.permutation(np.arange(imgs_len))
    left = 0
    right = batch_size if batch_size < imgs_len else imgs_len
    CLASS = 2
    x_batch = []
    y_batch = []
  
    for index in shuffled_indices[left:right]:
        img_data = imgs_data[index]
        img, bboxes, labels = aug_img(img_data)
        # re = draw_box(img, bboxes, labels)
        # cv2.imshow("re", re)
        # cv2.waitKey(0)
        if(img is not None):
            x_img = img
            y_img = np.zeros((GRID_W, GRID_H, BOX, 5+CLASS))
            for i, bbox in enumerate(bboxes):
                center_x = (bbox['xmin'] + bbox['xmax'])/2
                center_x = center_x / (float(NORM_W) / GRID_W)
                center_y = (bbox['ymin'] + bbox['ymax'])/2
                center_y = center_y / (float(NORM_H) / GRID_H)

                grid_x = int(np.floor(center_x))
                grid_y = int(np.floor(center_y))
                box = [bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']]
               
                if grid_x < GRID_W and grid_y < GRID_H:
                    y_img[grid_y, grid_x, :, 0:4] = BOX * [box]
                    y_img[grid_y, grid_x, :, 4]  = BOX * [1.]
                    if labels[i] == "Cam":
                        y_img[grid_y, grid_x, :, 5]  = 1.0
                        y_img[grid_y, grid_x, :, 6]  = 0.
                    elif labels[i] == "Chanh" or labels[i] == "lemon":
                        y_img[grid_y, grid_x, :, 5]  = 0.
                        y_img[grid_y, grid_x, :, 6]  = 1.0
            x_batch.append(x_img)
            y_batch.append(y_img)

    x_batch = np.array(x_batch)
    y_batch = np.array(y_batch)
    return x_batch, y_batch


def interpret_netout(image, netout, GRID_H = 13, GRID_W = 13):
    BOX = 5
    boxes = []
    THRESHOLD = 0.3
    ANCHORS = '1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52'
    ANCHORS = [float(ANCHORS.strip()) for ANCHORS in ANCHORS.split(',')]
    
  # interpret the output by the network
    for row in range(GRID_H):
        for col in range(GRID_W):
            for b in range(BOX):
                box = BoundBox()
                # first 5 weights for x, y, w, h and confidence
                box.x, box.y, box.w, box.h, box.c = netout[row,col,b,:5]
                box.lb = netout[row,col,b,5:]
                box.col, box.row = col, row
                box.x = (col + sigmoid(box.x)) / GRID_W
                box.y = (row + sigmoid(box.y)) / GRID_H
                box.w = ANCHORS[2 * b + 0] * np.exp(box.w) / GRID_W
                box.h = ANCHORS[2 * b + 1] * np.exp(box.h) / GRID_H
                box.c = sigmoid(box.c)

                classes = netout[row,col,b,5]
                box.probs = softmax(classes) * box.c
                box.probs *= box.probs > THRESHOLD
                boxes.append(box)
    sorted_indices = list(reversed(np.argsort([box.probs for box in boxes])))
    for i in range(len(sorted_indices)):
        index_i = sorted_indices[i]
        if boxes[index_i].probs <= 0:
            continue
        else:
            for j in range(i+1, len(sorted_indices)):
                index_j = sorted_indices[j]
                iou = boxes[index_i].iou(boxes[index_j])
                if iou >= 0.4:
                    boxes[index_j].probs = 0
                elif iou == -1:
                    boxes[index_i].probs = 0
    true_boxs = []
    for box in boxes:
        if box.probs > THRESHOLD:
            try:
                xmin  = int((box.x - box.w/2) * image.shape[1])
                xmax  = int((box.x + box.w/2) * image.shape[1])
                ymin  = int((box.y - box.h/2) * image.shape[0])
                ymax  = int((box.y + box.h/2) * image.shape[0])
                true_boxs.append([xmin, ymin, xmax, ymax, box.probs, box.lb])
            except Exception as e:
                print("some error")
    return true_boxs


def sigmoid(x):
    return 1. / (1.  + np.exp(-x))


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def test(model, x_test, enable_time = False):
    # j = randint(0,len(x_test) - 2)
    if enable_time:
        start = time.time()
        y = model.predict(x_test)
        print("Execute time = ", time.time() - start)
    else:
        y = model.predict(x_test)
    true_boxs = interpret_netout(x_test[0]*255, y[0])
    # print(len(true_boxs))
    img = draw_box_predict(x_test[0]*255, true_boxs)
    plot_img(img)