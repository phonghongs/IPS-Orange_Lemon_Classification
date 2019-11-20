from keras.models import Sequential, Model, load_model
from keras.layers import Reshape, Activation, Conv2D, MaxPooling2D, Input, BatchNormalization, Flatten, Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import SGD, Adam
import numpy as np 
import scipy.io
import os
import tensorflow as tf 
import keras

_grid_h = 13
_grid_w = 13
_box = 5
_class = 2
_anchors = '1.08, 1.19,    3.42, 4.41,    6.63, 11.38,    9.42, 5.11,    16.62, 10.52'
_norm_h = 416
_norm_w = 416
_scale_nood = 0.5
_scale_conf = 5.0
_scale_coor = 5.0
_scale_prob = 1.0

def yolo_model():
    model = Sequential()

    # Input 416x416x3 -> Output 208x208x16
    model.add(Conv2D(16, (3,3), strides = (1,1), padding = 'same', use_bias = False, input_shape = (416, 416, 3)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha = 0.1))
    model.add(MaxPooling2D(pool_size = (2,2)))

    # Input 208x208x16 -> Output 104x104x32
    model.add(Conv2D(32, (3,3), strides = (1,1), padding = 'same', use_bias = False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha = 0.1))
    model.add(MaxPooling2D(pool_size = (2,2)))

    #Input 104x104x32 -> Output 52x52x64
    model.add(Conv2D(64, (3,3), strides = (1,1), padding = 'same', use_bias = False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha = 0.1))
    model.add(MaxPooling2D(pool_size = (2,2)))

    #Input 52x52x64 -> Output 26x26x128
    model.add(Conv2D(128, (3,3), strides = (1,1), padding = 'same', use_bias = False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha = 0.1))
    model.add(MaxPooling2D(pool_size = (2,2)))

    #Input 26x26x128 -> Output 13x13x256
    model.add(Conv2D(256, (3,3), strides = (1,1), padding = 'same', use_bias = False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha = 0.1))
    model.add(MaxPooling2D(pool_size = (2,2)))

    #Input 13x13x256 -> Output 13x13x512
    model.add(Conv2D(512, (3,3), strides = (1,1), padding = 'same', use_bias = False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha = 0.1))
    model.add(MaxPooling2D(pool_size = (2,2), strides = (1,1), padding = 'same'))

    #Input 13x13x512 -> Output 13x13x1024
    model.add(Conv2D(1024, (3,3), strides = (1,1), padding = 'same', use_bias = False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha = 0.1))

    #Input 13x13x1024 -> Output 13x13x512
    model.add(Conv2D(512, (3,3), strides = (1,1), padding = 'same', use_bias = False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha = 0.1))

    # Input 13x13x512 -> Output 13x13x5x(4+1+1)
    model.add(Conv2D(_box * (4 + 1 + _class), (1,1), strides = (1,1), kernel_initializer='he_normal'))
    model.add(Activation('linear'))
    model.add(Reshape((_grid_h, _grid_w, _box, 4 + 1 + _class)))
    return model

def yolo_loss(_y_true, _y_pred):
    _anchor = [float(_an.strip()) for _an in _anchors.split(',')]
    _anchor = np.reshape(_anchor, [1,1,1,5,2])

    _pred_box_xy = tf.sigmoid(_y_pred[:,:,:,:,:2])

    _pred_box_wh = tf.exp(_y_pred[:,:,:,:,2:4]) * _anchor
    _pred_box_wh = tf.sqrt(_pred_box_wh / np.reshape([float(_grid_w), float(_grid_h)], [1,1,1,1,2]))
    
    _pred_box_conf = tf.expand_dims(tf.sigmoid(_y_pred[:,:,:,:,4]), -1)

    _pred_box_prob = tf.nn.softmax(_y_pred[:, :, :, :, 5:])

    # _pred_box_prob = tf.expand_dims(tf.sigmoid(_y_pred[:,:,:,:,5]), -1)

    _y_pred = tf.concat([_pred_box_xy, _pred_box_wh, _pred_box_conf, _pred_box_prob], 4)

    print("Y_pred shape: {}".format(_y_pred.shape))

    _center_xy = .5*(_y_true[:,:,:,:,0:2] + _y_true[:,:,:,:,2:4])
    _center_xy = _center_xy / np.reshape([(float(_norm_w)/_grid_w), (float(_norm_h)/_grid_h)], [1,1,1,1,2])
    _true_box_xy = _center_xy - tf.floor(_center_xy)

    _true_box_wh = (_y_true[:,:,:,:,2:4] - _y_true[:,:,:,:,0:2])
    _true_box_wh = tf.sqrt(_true_box_wh / np.reshape([float(_norm_w), float(_norm_h)], [1,1,1,1,2]))

    _pred_tem_wh = tf.pow(_pred_box_wh, 2) * np.reshape([_grid_w, _grid_h], [1,1,1,1,2])        
    _pred_box_area = _pred_tem_wh[:,:,:,:,0] * _pred_tem_wh[:,:,:,:,1]
    _pred_box_ul = _pred_box_xy - 0.5 * _pred_tem_wh
    _pred_box_br = _pred_box_xy + 0.5 * _pred_tem_wh

    _true_tem_wh = tf.pow(_true_box_wh, 2) * np.reshape([_grid_w, _grid_h], [1,1,1,1,2])
    _true_box_area = _true_tem_wh[:,:,:,:,0] * _true_tem_wh[:,:,:,:,1]
    _true_box_ul = _true_box_xy - 0.5 * _true_tem_wh
    _true_box_br = _true_box_xy + 0.5 * _true_tem_wh

    _intersect_ul = tf.maximum(_pred_box_ul, _true_box_ul)
    _intersect_br = tf.minimum(_pred_box_br, _true_box_br)
    _intersect_wh = _intersect_br - _intersect_ul
    _intersect_wh = tf.maximum(_intersect_wh, 0.0)
    _intersect_area = _intersect_wh[:,:,:,:,0] * _intersect_wh[:,:,:,:,1]

    _iou = tf.truediv(_intersect_area, _true_box_area + _pred_box_area - _intersect_area)

    print("iou shape: {}".format(_iou.shape))

    # https://blog.csdn.net/dmy88888/article/details/81144835
    _reduce_max = tf.reduce_max(_iou, [3], True)

    print("reduce_max shape: {}".format(_reduce_max.shape))

    _best_box = tf.equal(_iou, _reduce_max)
    _best_box = tf.to_float(_best_box)

    print("best_box shape{}".format(_best_box.shape))

    _true_box_conf = tf.expand_dims(_best_box * _y_true[:,:,:,:,4], -1)
    _true_box_prob = _y_true[:,:,:,:,5:]

    _y_true = tf.concat([_true_box_xy, _true_box_wh, _true_box_conf, _true_box_prob], 4)

    print("Y_true shape: {}".format(_y_true.shape))

    # https://github.com/magee256/yolo_v2/blob/master/training/yolo_loss.py
    # https://medium.com/@jonathan_hui/real-time-object-detection-with-yolo-yolov2-28b1b93e2088
    # https://trungthanhnguyen0502.github.io/computer%20vision/2018/12/10/yolo_tutorial-2-yolo2-algorithms/

    _weight_coor = tf.concat(4*[_true_box_conf], 4)
    _weight_coor = _scale_coor * _weight_coor
    _weight_conf = _scale_nood * (1. - _true_box_conf) + _scale_conf * _true_box_conf
    _weight_prob = tf.concat(_class * [_true_box_conf], 4)
    _weight_prob = _scale_prob * _weight_prob
    _weight = tf.concat([_weight_coor, _weight_conf, _weight_prob], 4)

    _loss = tf.pow(_y_pred - _y_true, 2)
    _loss = _loss * _weight
    _loss = tf.reshape(_loss, [-1, _grid_h * _grid_w * _box * (4 + 1 + _class)])
    _loss = tf.reduce_sum(_loss, 1)
    _loss = .5 * tf.reduce_mean(_loss)
    return _loss




        