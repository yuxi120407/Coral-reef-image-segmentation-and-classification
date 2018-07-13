# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 15:52:00 2018

@author: Xi Yu
"""

import os
import sys
import random
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import keras
import keras.backend as K

#from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.layers.merge import multiply
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.vis_utils import plot_model

from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D,MaxPooling2D,Merge
from keras.models import Sequential 

from keras.layers import Conv2D,MaxPooling2D,UpSampling2D,BatchNormalization,Reshape,Permute,Activation  

import tensorflow as tf
import pydot_ng as pydot
pydot.find_graphviz()
#%%
# Set some parameters
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
TRAIN_PATH = './stage1_train/'
TEST_PATH = './stage1_test/'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed
train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]
#%%
# Get and resize train images and masks
X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
print('Getting and resizing train images and masks ... ')
sys.stdout.flush()
for n, id_ in enumerate(train_ids):
    path = TRAIN_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_train[n] = img
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', 
                                      preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)
    Y_train[n] = mask

# Get and resize test images
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Getting and resizing test images ... ')
sys.stdout.flush()
for n, id_ in enumerate(test_ids):
    path = TEST_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img

print('Done!')
#%%
p = np.zeros([670,128,128],dtype = np.uint8)
for i in range(670):
    pixel = Y_train[i]
    for m in range(128):  
        for n in range(128):
            if (pixel[m,n]):
                p[i,m,n]=1
            else:
                p[i,m,n]=0
#%%
Y_train = keras.utils.to_categorical(p,2)
#%%
weight_mask = np.ones([670,128,128],dtype = np.float32)
#%%
ix = random.randint(0, len(train_ids))
imshow(X_train[ix])
plt.show()
imshow(np.squeeze(Y_train[ix]))
plt.show()
#%%
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)
#%%
def pixel_loss(y_true,y_pred):   
    #class_weights = np.array([[0,1]])
    #weights = K.sum(class_weights*y_true,axis = 1)
    #unweighted_losses =K.categorical_crossentropy(y_true,y_pred)
    #weight_map = np.((1,128,128))
    #arg = tf.convert_to_tensor(weight_map, dtype=tf.float32)
    #weighted_losses = tf.matmul(arg,unweighted_losses)
    #weight_map[y_true[:, :, 1]==[0,1]] = 0e
    #weight_map = weight_map.reshape(128,128,1)
    #weighted_losses = np.multiply(unweighted_losses,weight_map)
    #loss = K.mean(weighted_losses, axis=-1)
    
    mask_labels_shape = tf.shape(mask_labels)
    mask_label = tf.reshape(mask_labels,[mask_labels_shape[2]*mask_labels_shape[1]*mask_labels_shape[0]])
    y_pred_shape = tf.shape(y_pred)
    y_pred = tf.reshape(y_pred,[y_pred_shape[2]*y_pred_shape[1]*y_pred_shape[0],y_pred_shape[3]])
    y_true_shape = tf.shape(y_true)
    y_true = tf.reshape(y_true,[y_true_shape[2]*y_true_shape[1]*y_true_shape[0],y_true_shape[3]])
    loss = tf.losses.softmax_cross_entropy(y_true,y_pred,weights=mask_label)
    return loss
#%%
def weighted_categorical_crossentropy(X):
    
    y_pred, weights, y_true = X
    loss = K.categorical_crossentropy(y_pred, y_true)
    loss = multiply([loss, weights])
    #loss = K.mean(loss,axis=-1)
    return loss
#%%
def weighted_crossentropy(X):
    y_pred,mask_labels,y_true = X
    mask_labels_shape = tf.shape(mask_labels)
    mask_label = tf.reshape(mask_labels,[mask_labels_shape[2]*mask_labels_shape[1]*mask_labels_shape[0]])
    y_pred_shape = tf.shape(y_pred)
    y_pred = tf.reshape(y_pred,[y_pred_shape[2]*y_pred_shape[1]*y_pred_shape[0],y_pred_shape[3]])
    y_true_shape = tf.shape(y_true)
    y_true = tf.reshape(y_true,[y_true_shape[2]*y_true_shape[1]*y_true_shape[0],y_true_shape[3]])
    loss = tf.losses.softmax_cross_entropy(y_true,y_pred,weights=mask_label)
    #loss = K.mean(loss,axis=-1)
    return loss
#%%
def categorical_crossentropy(y_true, y_pred):
        '''Expects a binary class matrix instead of a vector of scalar classes.
        '''
        return K.mean(K.categorical_crossentropy(y_pred, y_true), axis=-1)    
    
def identity_loss(y_true, y_pred):
    return y_pred

    
    
#%%
#weight_masks = Input((128,128))
#true_masks = Input((128,128,2))
inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = Lambda(lambda x: x / 255) (inputs)

c1 = Conv2D(8, (3, 3), activation='relu', padding='same',name='conv1') (s)
c1 = Conv2D(8, (3, 3), activation='relu', padding='same',name='conv2') (c1)
p1 = MaxPooling2D((2, 2),name='MaxPooling1') (c1)

c2 = Conv2D(16, (3, 3), activation='relu', padding='same',name='conv3') (p1)
c2 = Conv2D(16, (3, 3), activation='relu', padding='same',name='conv4') (c2)
p2 = MaxPooling2D((2, 2),name='MaxPooling2') (c2)

c3 = Conv2D(32, (3, 3), activation='relu', padding='same',name='conv5') (p2)
c3 = Conv2D(32, (3, 3), activation='relu', padding='same',name='conv6') (c3)
p3 = MaxPooling2D((2, 2),name='MaxPooling3') (c3)

c4 = Conv2D(64, (3, 3), activation='relu', padding='same',name='conv7') (p3)
c4 = Conv2D(64, (3, 3), activation='relu', padding='same',name='conv8') (c4)
p4 = MaxPooling2D(pool_size=(2, 2),name='MaxPooling4') (c4)

c5 = Conv2D(128, (3, 3), activation='relu', padding='same',name='conv9') (p4)
c5 = Conv2D(128, (3, 3), activation='relu', padding='same',name='conv10') (c5)

u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same',name='deconv1') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(64, (3, 3), activation='relu', padding='same',name='deconv2') (u6)
c6 = Conv2D(64, (3, 3), activation='relu', padding='same',name='deconv3') (c6)

u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same',name='deconv4') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(32, (3, 3), activation='relu', padding='same',name='deconv5') (u7)
c7 = Conv2D(32, (3, 3), activation='relu', padding='same',name='deconv6') (c7)

u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same',name='deconv7') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(16, (3, 3), activation='relu', padding='same',name='deconv8') (u8)
c8 = Conv2D(16, (3, 3), activation='relu', padding='same',name='deconv9') (c8)

u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same',name='deconv10') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(8, (3, 3), activation='relu', padding='same',name = 'deconv11') (u9)
c9 = Conv2D(8, (3, 3), activation='relu', padding='same',name = 'deconv12') (c9)

outputs = Conv2D(2, (1, 1), activation='softmax',name = 'output') (c9)
#loss = Lambda(weighted_categorical_crossentropy)([outputs, weight_masks, true_masks])


#model = Model(inputs=[inputs,weight_masks,true_masks], outputs=loss)
model = Model(inputs=inputs, outputs=outputs)
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[mean_iou])
#model.compile(optimizer='adam', loss=lambda y_true, y_pred:y_pred)
#model.compile(optimizer='adam', loss=identity_loss)
#model.compile(optimizer='adam', loss=pixel_loss, metrics=[mean_iou])
model.summary()
#%%
plot_model(model,show_shapes=True,to_file = 'U-net1.png')
#%%
#y = np.zeros([600,128,128,2],dtype = np.float32)
#y_test = np.zeros([70,128,128,2],dtype = np.float32)
#%%
#x_train = X_train[0:600]
#y_train = Y_train[0:600:]
#x_test = X_train[600:]
#y_test = Y_train[600:]
##%%
#x_train = np.array(x_train)
#y_train = np.array(y_train)
#weight_mask = np.array(weight_mask)
#mask = weight_mask[0:600]
#mask_test = weight_mask[600:]
##%%
#earlystopper = EarlyStopping(patience=5, verbose=1) 
#checkpointer = ModelCheckpoint('model-dsbowl2018-1.h5', verbose=1, save_best_only=True)
#results = model.fit([X_train, weight_masks,Y_train],None, batch_size=8, epochs=3)#, 
                    #callbacks=[earlystopper, checkpointer])
#model.fit([x_train,mask,y_train],y,epochs=3,batch_size=32,validation_data=([x_test,mask_test,y_test], y_test),verbose=2)
#model.fit(x_train,y_train,epoch = 3, batch_size = 32, validation=(x_test,y_test),verbose=1)
earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('model-dsbowl2018-1.h5', verbose=1, save_best_only=True)
results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=8, epochs=30, 
                    callbacks=[earlystopper, checkpointer])
#%%make prediction
model = load_model('model-dsbowl2018-1.h5', custom_objects={'mean_iou': mean_iou})
preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

# Threshold predictions
preds_train_t = (np.argmax(preds_train,axis=-1)).astype(np.uint8)
preds_val_t = (np.argmax(preds_val,axis=-1)).astype(np.uint8)
preds_test_t = (np.argmax(preds_test,axis=-1)).astype(np.uint8)
#%%
# Create list of upsampled test masks
preds_test_upsampled = []
for i in range(len(preds_test)):
    preds_test_upsampled.append(resize(np.squeeze(preds_test[i]), 
                                       (sizes_test[i][0], sizes_test[i][1]), 
                                       mode='constant', preserve_range=True))
#%%
ix = random.randint(0, len(preds_train_t))
imshow(X_train[ix])
plt.show()
y = (np.argmax(Y_train,axis=-1)).astype(np.uint8)
imshow(np.squeeze(y[ix]))
plt.show()
imshow(np.squeeze(preds_train_t[ix]))
plt.show()