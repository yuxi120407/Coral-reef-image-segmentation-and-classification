# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 14:30:05 2018

@author: Xi Yu
"""

from crf import CRF
from keras.layers import Dense, Embedding, Conv1D, Input,Dropout
from keras.models import Model # 这里我们学习使用Model型的模型
import keras.backend as K # 引入Keras后端来自定义loss，注意Keras模型内的一切运算

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from function import newtxt,newimagedata,create_plots,plot_confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np 
import os
#%%
batch_size = 128
num_classes = 6
epochs = 10
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = '2012images_trained_augmentation——6.1.h5'
data_augmentation = False

# input crop image dimensions
img_rows, img_cols = 30,30

# the data, split between train and test sets
#crop_image,label = newimagedata(img_rows, img_cols)
crop_image = final_data
label = final_label
x_train = crop_image[0:60000]
x_test = crop_image[60000:]

y_train_vector = label[0:60000]
y_test_vector = label[60000:]
#%%
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train_vector, num_classes)
y_test = keras.utils.to_categorical(y_test_vector, num_classes)

#%%
def cnn_crf_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(30,30,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25)) 

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))      
    
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    model.summary()

    return model

model = cnn_crf_model()
#%%
input_image = Input(shape = (30,30,3))
conv1 = Conv2D(32,(3,3),padding = 'same',activation='relu')(input_image)
conv2 = Conv2D(32,(3,3),activation='relu')(conv1)
maxpooling1 = MaxPooling2D(pool_size=(2, 2))(conv2)
dropout1 = Dropout(0.25)(maxpooling1)

conv3 = Conv2D(64,(3,3),padding = 'same',activation='relu')(dropout1)
conv4 = Conv2D(64,(3,3),activation='relu')(conv3)
maxpooling2 = MaxPooling2D(pool_size=(2, 2))(conv4)
dropout2 = Dropout(0.25)(maxpooling2)

conv5 = Conv2D(128,(3,3),padding = 'same',activation='relu')(dropout2)
conv6 = Conv2D(128,(3,3),activation='relu')(conv5)
maxpooling3 = MaxPooling2D(pool_size=(2, 2))(conv6)
dropout2 = Dropout(0.25)(maxpooling3)

#flatten1 = Flatten()(dropout2)
#dense1 = Dense(512,activation='relu')(flatten1)
#dropout3 = Dropout(0.5)(dense1)
#dense2 = Dense(6,activation='softmax')(dropout3)

crf = CRF(True)
tag_score = Dense(5)(dropout2)
tag_score = crf(tag_score)
model = Model(inputs = input_image , outputs = tag_score)
model.summary()
#%%
model.compile(loss=crf.loss,optimal ='adma',metrics=[crf.accuracy])


















