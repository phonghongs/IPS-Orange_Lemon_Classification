import os
from yolo import *
import pickle
import copy
from matplotlib import pyplot as plt
from data_processing import *
from random import randint

# Fist time
IMAGE_DIR = "data"
img_data = get_data(IMAGE_DIR)
with open("preprocessed_data.txt", "wb") as fp:
    pickle.dump(img_data, fp)


# img_data = []
# with open("preprocessed_data.txt", "rb") as fp:
#     img_data = pickle.load(fp)



print(len(img_data))

model = yolo_model()
# model = load_model('./weight/objdect2.h5', compile=False)
adam = Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss = yolo_loss, optimizer= adam, metrics=['mse'])
# model.compile(loss = yolo_loss, optimizer= adam, metrics=['accuracy'])

# i = randint(0, len(img_data))
# print(img_data[i]['file_name'], " ", img_data[i]['fruit_coordinate'], " ", img_data[i]['fruit_type'])
# img = cv2.imread(img_data[i]['file_name'])
# img = draw_box(img, img_data[i]['fruit_coordinate'])
# plot_img(img)

train_data = img_data[:int(0.98*len(img_data))]
test_data = img_data[int(0.98*len(img_data)):]
x_test, y_test = data_gen(test_data, 100)

# print(len(train_data))

i = 0
x_batch = None
y_batch = None
model_index = 1
batch_size = 256
max_iter = 1000

print("start train")
while True:
    x_batch, y_batch = data_gen(train_data, batch_size)
    print("Stored weight: " + str(i))
    model.fit(x_batch, y_batch, epochs = 50, verbose = 1, validation_data=(x_test, y_test))
    i += 1
    # score = model.evaluate(x_test, y_test, verbose=0)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])
    # if( i % 5 == 0):
    #     test(model, x_test)
    # if(i % 50 == 0):
    model_link = "./weight/objdect" + str(model_index) + ".h5"
    model.save(model_link)
    model_index += 1
    if(i > max_iter):
        break

