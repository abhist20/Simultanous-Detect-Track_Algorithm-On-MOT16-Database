import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import pathlib
from keras.layers import Dropout
import pandas as pd
from keras.callbacks import ModelCheckpoint
data_dir = pathlib.Path('pot')
image_count = len(list(data_dir.glob('*.jpg')))
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
BATCH_SIZE = 500
IMG_HEIGHT = 720
IMG_WIDTH = 720
train_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=False,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH))
image_batch, label_batch = next(train_data_gen)
raw_dataset = pd.read_csv('train_labels.csv')

print(raw_dataset)
lables =[0,0,0,0]
tr_labels = raw_dataset['xmin'].copy()
tr_labels_stats=tr_labels.describe()
tr_labels_stats = tr_labels_stats.transpose()

def norm(x):
  return (x - tr_labels_stats['mean']) / tr_labels_stats['std']
tr_labels = norm(tr_labels)
img = cv2.imread("img-1.jpg")
img = (np.expand_dims(img,axis=0))

tr_labels = tr_labels.to_numpy()
lables[0] = [raw_dataset.pop('xmin')]
lables[1] = [raw_dataset.pop('xmax')]
lables[2] = [raw_dataset.pop('ymin')]
lables[3] = [raw_dataset.pop('ymax')]
tr_labels_b = tr_labels[0:500]
print(tr_labels_b[1])

model = models.Sequential()
model.add(layers.Flatten(input_shape=(720,720,3)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1))

A = tf.keras.optimizers.Adam()
model.compile(loss = 'mean_squared_error', optimizer=A)
# ideally batch_size must be 2^n
EPOCHS=10
model.fit(img, tr_labels_b[0], epochs=EPOCHS)
model.save('xcod.h5')
