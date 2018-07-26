# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 15:38:51 2018

@author: Xi Yu
"""

import keras
from keras.datasets import mnist
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import os
from keras.layers import Input
import h5py

num_classes = 10
img_rows =28
img_cols =28
input_shape = (img_rows, img_cols, 1)
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',name='conv1',input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu',name='conv2'))
model.add(MaxPooling2D(pool_size=(2, 2),name='maxpooling1'))
model.add(Dropout(0.25,name='dropout1'))
model.add(Flatten(name='flatten1'))
model.add(Dense(128, activation='relu',name='dense1'))
model.add(Dropout(0.5,name='dropout2'))
model.add(Dense(num_classes, activation='softmax',name='dense2'))
#%%
# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers])
#%%
weights_path = './model_weight/weight_7.22.h5'
model.load_weights(weights_path)

#%%
for layer in model.layers:
    weights = layer.get_weights() 
#%%
inputs = Input((28, 28, 1))
conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu',name='conv1') (inputs)
conv2 = Conv2D(64, kernel_size=(3, 3), activation='relu',name='conv2')(conv1)
maxpooling1 = MaxPooling2D(pool_size=(2, 2),name='maxpooling1')(conv2)
dropout1 = Dropout(0.25,name='dropout1')(maxpooling1)
flatten = Flatten(name='flatten')(dropout1)
dense1 = Dense(128, activation='relu',name='dense1')(flatten)
dropout2 = Dropout(0.5,name='dropout2')(dense1)
dense2 = Dense(num_classes, activation='softmax',name='dense2')(dropout2)
new_model = Model(inputs=inputs,outputs=dense2)

#%%
import numpy as np
#import keras

x = keras.layers.Input(shape=(3,))
mid = keras.layers.Dense(8,name='dense2')
mid.trainable = False
y = mid(x)
output = keras.layers.Dense(5)(y)

model = keras.models.Model(x, output)
model.summary()
#%%
#model.trainable = False
model.load_weights('./model_weight/weight_test.h5')
#%%
model.compile(optimizer='rmsprop', loss='mse')

x = np.random.random((10, 3))
y = np.random.random((10, 5))
model.fit(x, y, epochs=10)
#%%
k = len(model.layers)
for i in range(k):
    model_weight = model.layers[i].get_weights()
    new_model.layers[i+1].set_weights(model_weight)

#%%
x = Input(shape=(32,))
layer = Dense(32)
layer.trainable = False
y = layer(x)

frozen_model = Model(x, y)
frozen_model.summary()
#%%
save_dir = os.path.join(os.getcwd(), 'model_weight')
model_name = 'weight_test.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)












