from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from matplotlib import pyplot as plt
import cv2

model=keras.models.load_model('save.h5')
model.summary()
layer_dict = dict([(layer.name, layer) for layer in model.layers])

layer_name = 'conv2d'
layer_name_1= 'conv2d_1'
model = tf.keras.models.Model(inputs=model.input,outputs=layer_dict[layer_name_1].output)

img=cv2.imread("pl1.jpg")
img1=img
img=cv2.resize(img,(32,32))
img = (np.expand_dims(img,0))
feature_maps=model.predict(img)
square = 8
index = 1
print(feature_maps.shape)
print(img.shape)

for _ in range(square):
    for _ in range(square):
        ax = plt.subplot(square, square, index)
        ax.set_xticks([])
        ax.set_yticks([])

        plt.imshow(feature_maps[0, :, :, index - 1], cmap='viridis')
        index += 1

plt.show()
plt.imshow(feature_maps[0, :, :, 43], cmap='viridis')
plt.show()
img_width = img.shape[1]
img_height = img.shape[2]
width = feature_maps.shape[1]
height = feature_maps.shape[2]
w_stride = img_width / width
h_stride = img_height / height
print(w_stride,h_stride)
shift_x = np.arange(0, width) * w_stride
shift_y = np.arange(0, height) * h_stride
shift_x, shift_y = np.meshgrid(shift_x, shift_y)
shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
plt.imshow(shifts)
plt.show()