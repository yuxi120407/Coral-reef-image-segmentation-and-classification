# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 11:46:16 2018

@author: yuxi
"""

import os
import sys
import random
import warnings

import numpy as np
import pandas as pd
import glob

import matplotlib.pyplot as plt
import keras
import keras.backend as K

#from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
#from skimage.morphology import label

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
#%%load label 
label= np.zeros([126,1536,2048],dtype = np.uint8)
labelName = glob.glob("./2012new/*.txt")
for n,name in enumerate(labelName):
    lab_name = name.split("\\")[1]
    lab_name = lab_name.split(".")[0]
    label_path = str('./label/')+lab_name+str('.txt')
    label[n] = np.loadtxt(label_path,dtype=np.int)
#%%load iamge data
image = np.zeros([126,1536,2048,3],dtype = np.uint8)
imageName = glob.glob("./test_image/*.jpg")
for m,name in enumerate(imageName):
    img_name = name.split("\\")[1]
    img_name = img_name.split(".")[0]
    img_path = str('./test_image/')+img_name+str('.jpg')
    image[m] = imread(img_path)
#%%
weight_masks = np.zeros([126,1536,2048],dtype = np.uint8)
weight_masksName = glob.glob("./weight_mask/*.txt")
for p,name in enumerate(weight_masksName):
    wei_name = name.split("\\")[1]
    wei_name = wei_name.split(".")[0]
    wei_path = str('./weight_mask/')+wei_name+str('.txt')
    weight_masks[p] = np.loadtxt(wei_path,dtype=np.int)
#%%
x_train = np.float16(image[0:110]/256)
#%%
x_test = (image[110:]/256).astype(np.float16)
#%%
x_test = np.float16(image[110:]/256)
#%%
y_train = label[0:110].astype(np.uint8)
y_test = label[110:].astype(np.uint8)   
#%%
y_train = keras.utils.to_categorical(y_train,7).astype(np.float16)
y_test = keras.utils.to_categorical(y_test,7).astype(np.float16)
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

def identity_loss(y_true, y_pred):
    return y_pred
#%%building U-net
weight_masks = Input((1536,2048),name='weight_mask')
true_masks = Input((1536,2048,7),name='true_mask')
inputs = Input((1536, 2048, 3),name='input_image')

c1 = Conv2D(8, (3, 3), activation='relu', padding='same',name='conv1') (inputs)
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

outputs = Conv2D(7, (1, 1), activation='softmax',name = 'output') (c9)
loss = Lambda(weighted_crossentropy)([outputs, weight_masks, true_masks])


model = Model(inputs=[inputs,weight_masks,true_masks], outputs=loss)
#model = Model(inputs=inputs, outputs=loss)
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[mean_iou])
#model.compile(optimizer='adam', loss=lambda y_true, y_pred:y_pred)
model.compile(optimizer='adam', loss=identity_loss)
#model.compile(optimizer='adam', loss=pixel_loss, metrics=[mean_iou])
model.summary()
#%%
y_null_train = np.zeros([110,1536,2048],dtype=np.float16)
y_null_test = np.zeros([16,1536,2048],dtype=np.float16)
weight_mask_train = weight_masks[0:110]
weight_mask_test = weight_masks[110:]
#%%
#earlystopper = EarlyStopping(patience=5, verbose=1)
#checkpointer = ModelCheckpoint('model-dsbowl2018-1.h5', verbose=1, save_best_only=True)
#results = model.fit([x_train,weight_mask_train,y_train],y_null_train,validation_data=([x_test,weight_mask_test,y_test],y_null_test), batch_size=1, epochs=2, 
                    #callbacks=[earlystopper, checkpointer])

model.fit([x_train,weight_mask_train,y_train],y_null_train,epochs = 2, batch_size=1,validation_data = ([x_test,weight_mask_test,y_test],y_null_test))
#%%
label0 = label[0]
image0 = image[0]
#%%
lab_0 = label0[0:384,0:512]
image_0 = image0[0:384,0:512]
#%%
ground_truth = np.zeros([1536,2048,3])
for i in range(1536):
    for j in range(2048):
        la = int(y_pred[i,j])
        if(la == 1):#coral---blue
            ground_truth[i,j]=np.array([0,0,255])
        elif(la == 2): #DCP---slight gray
            ground_truth[i,j]=np.array([105,105,105])
        elif(la == 3): #ROC---deep gray
            ground_truth[i,j]=np.array([169,169,169])
        elif(la == 4): #CCA---red
            ground_truth[i,j]=np.array([255,0,0])
        elif(la == 5): #Ana---green
            ground_truth[i,j]=np.array([0,255,0])
        elif(la == 6):
            ground_truth[i,j]=np.array([255,255,0])
        else:
            ground_truth[i,j]=np.array([0,0,0])
ground_truth = ground_truth.astype(np.uint8)
#%%
x = x_test[0]
x = x.reshape(1,1536,2048,3)
model.load_weights('./model-dsbowl2018-1.h5')
#%%
preds_test = model.predict(x, verbose=1)
#%%
y_pred = np.argmax(preds_test,axis=-1)














