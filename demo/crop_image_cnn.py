# -*- coding: utf-8 -*-
"""
Created on Tue May 15 10:18:21 2018

@author: yuxi
"""

'''Trains a simple convnet on the crop image
'''

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from function import newtxt,newimagedata,create_plots,plot_confusion_matrix,cnn_model,cnn_model1
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np 
import os
from keras.utils.vis_utils import plot_model
#%%
batch_size = 128
num_classes = 6
epochs = 10
save_dir = os.path.join(os.getcwd(), 'model_weight')
model_name = '2012images_hsv_trained_augmentation-6.26.h5'
data_augmentation = False

# input crop image dimensions
img_rows, img_cols = 30,30

# the data, split between train and test sets
#crop_image,label = newimagedata(img_rows, img_cols)
crop_image = hsv_data
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
#x_train /= 255
#x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train_vector, num_classes)
y_test = keras.utils.to_categorical(y_test_vector, num_classes)

#%%
model = cnn_model()
model.summary()
#%%
# initiate Adam optimizer
opt = keras.optimizers.Adam(lr=0.0001, decay=1e-6)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


#%%
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)
#%%
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
#%%
class_names = ['Coral','DCP','Rock','RA','GA','others']
Y_pred = model.predict(x_train,verbose=2)
y_pred = np.argmax(Y_pred,axis=1)
for ix in range(6):
        print (ix, confusion_matrix(np.argmax(y_train,axis=1), y_pred)[ix].sum())
print (confusion_matrix(np.argmax(y_train,axis=1), y_pred))    
plot_confusion_matrix(confusion_matrix(np.argmax(y_train,axis=1), y_pred), classes=class_names)
