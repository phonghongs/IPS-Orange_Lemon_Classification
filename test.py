import os
from yolo import *
import pickle
import copy
from matplotlib import pyplot as plt
from data_processing import *
from random import randint
import sys

model = load_model('./weight/objdect3.h5', compile=False)


x_test = aug_test_image(sys.argv[1])
print(x_test.shape)

test(model, x_test, True)
# test(model, x_test, True)
# test(model, x_test, True)
# test(model, x_test, True)

