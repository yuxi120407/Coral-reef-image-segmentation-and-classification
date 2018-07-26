# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 20:48:07 2018

@author: Xi Yu
"""

'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs

'''
import keras
from keras.datasets import mnist
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import os
from keras.layers import Input
import h5py
#%%
save_dir = os.path.join(os.getcwd(), 'model_weight')
model_name = 'weight_7.22.h5'


batch_size = 128
num_classes = 10
epochs = 10



# input image dimensions
img_rows, img_cols = 28, 28
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)



x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')



# convert class vectors to binary class matrices

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


#%%
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',name='conv1',input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu',name='conv2'))
model.add(MaxPooling2D(pool_size=(2, 2),name='maxpooling1'))
model.add(Dropout(0.25,name='dropout1'))
model.add(Flatten(name='flatten1'))
model.add(Dense(128, activation='relu',name='dense1'))
model.add(Dropout(0.5,name='dropout2'))
model.add(Dense(num_classes, activation='softmax',name='dense2'))

model.summary()

#%%
inputs = Input((28, 28, 1))
conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu',name='conv1') 
conv1.trainable = False
conv1_output = conv1(inputs)
conv2 = Conv2D(64, kernel_size=(3, 3), activation='relu',name='conv2')
conv2.trainable = False
conv2_output = conv2(conv1_output)


maxpooling1 = MaxPooling2D(pool_size=(2, 2),name='maxpooling1')(conv2_output)
dropout1 = Dropout(0.25,name='dropout1')(maxpooling1)
flatten = Flatten(name='flatten')(dropout1)
dense1 = Dense(128, activation='relu',name='dense1')(flatten)
dropout2 = Dropout(0.5,name='dropout2')(dense1)
dense2 = Dense(num_classes, activation='softmax',name='dense2')(dropout2)
new_model = Model(inputs=inputs,outputs=dense2)

new_model.summary()
#%%
new_model.load_weights('./model_weight/weight_7.22.h5')
weights_path = './model_weight/weight_7.22.h5'

#%%
new_model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])

new_model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1, validation_data=(x_test, y_test))
score = new_model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#%%
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

