from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
import glob
from keras.preprocessing.image import ImageDataGenerator
import natsort
from skimage.io import imread_collection
import numpy as np
import pathlib
import pandas as pd
raw_dataset = pd.read_csv('train_labels.csv')
data_dir = pathlib.Path('pot')
tr_labels = raw_dataset['xmin'].copy()
tr_labels_stats=tr_labels.describe()
tr_labels_stats = tr_labels_stats.transpose()

def norm(x):
  return (x - tr_labels_stats['mean']) / tr_labels_stats['std']
tr_labels_norm = norm(tr_labels)
image_count = len(list(data_dir.glob('*.jpg')))

model=keras.models.load_model('xcod.h5')

image_count = len(list(data_dir.glob('*.png')))
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
BATCH_SIZE = 32
IMG_HEIGHT = 32
IMG_WIDTH = 32
train_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=False,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH))
image_batch, label_batch = next(train_data_gen)
raw_dataset = pd.read_csv('train_labels.csv')
print(raw_dataset)
lables =[0,0,0,0]
tr_labels = raw_dataset['xmin'].copy()
tr_labels = tr_labels.to_numpy()
lables[0] = [raw_dataset.pop('xmin')]
lables[1] = [raw_dataset.pop('xmax')]
lables[2] = [raw_dataset.pop('ymin')]
lables[3] = [raw_dataset.pop('ymax')]
tr_labels_b = tr_labels[0:500]
print(tr_labels_b.shape)


#print(col)
img=cv2.imread('img-1.jpg')
img1=img
img=cv2.resize(img,(64,64))
img = (np.expand_dims(img,0))
model.summary()
ps=model.predict(img)
print('predicted=',ps)
print('lable=',tr_labels_norm[0])
