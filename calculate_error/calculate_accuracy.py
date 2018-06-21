# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 10:21:11 2018

@author: yuxi
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from PIL import Image
from function import cnn_model,cnn_model1
import glob
import time
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
#%%
def percent_coral(testimage):
    count = 0
    #blue = np.array([0,0,255], dtype=np.uint8)
    test = np.uint8(testimage)
    pixel = np.zeros([1,1,3],dtype = np.uint8)
    for r in range(0,test.shape[0]):
        for c in range(0,test.shape[1]):
            pixel = test[r,c,:]
            if(np.all(pixel == blue)):
            #if((pixel[0]==0)and(pixel[1]==0)):
                count = count +1
    all_pixel = test.shape[0]*test.shape[1]
    coral_percent = (count/all_pixel)*100 
    coral_percent = round(coral_percent,2)
    return coral_percent
#%%
model = cnn_model()
model.summary()
model.load_weights('./saved_models/2012images_trained_augmentation-6.6.h5')
#%%
model1 = cnn_model1()
model1.summary()
model1.load_weights('./saved_models/2012images_trained_model.h5')
#%%
path_image = './image_ground truth/image/201208172_T-12-52-20_Dive_01_029_json/img.png'

testimage = Image.new("RGB",(2048,1536),(0,0,0))
test_image = imread(path_image)
# Define the window size
windowsize_r = 30
windowsize_c = 30

# Crop out the window and calculate the histogram
for r in range(0,test_image.shape[0] - windowsize_r+1, 6):
    for c in range(0,test_image.shape[1] - windowsize_c+1, 6):
        window = test_image[r:r+windowsize_r,c:c+windowsize_c]
        window = window.reshape(-1,30,30,3)
        pred = model.predict(window,verbose=2)
        y_pred = np.argmax(pred,axis=1)
        start_time = time.time()
        if(y_pred==0):#coral---blue
            testimage.paste((0,0,255),[c,r,c+windowsize_c,r+windowsize_r])
        elif(y_pred==1):#DCP---slight gray
            testimage.paste((105,105,105),[c,r,c+windowsize_c,r+windowsize_r])
        elif(y_pred==2):#ROC---deep gray
            testimage.paste((169,169,169),[c,r,c+windowsize_c,r+windowsize_r])
        elif(y_pred==3):#CCA---red
            testimage.paste((255,0,0),[c,r,c+windowsize_c,r+windowsize_r])
        elif(y_pred==4):#Ana---green
            testimage.paste((0,255,0),[c,r,c+windowsize_c,r+windowsize_r])
        else:#others---yellow
            testimage.paste((255,255,0),[c,r,c+windowsize_c,r+windowsize_r])           
percent = percent_coral(testimage)
time_perimage = time.time()-start_time
time_perimage = round(time_perimage,2)
print("image finished in {0}s".format(time_perimage))
print("there are {0}% coral in this image".format(percent))
plt.imshow(testimage)
plt.axis('off')
#%%
labels = imread("./image_ground truth/image/201208172_T-12-52-20_Dive_01_029_json/label.png")
#%%
blue = np.array([0,0,255])
pixel = np.zeros([1,1,3],dtype = np.uint8)
test = np.uint8(testimage)
prediction = np.zeros([test.shape[0],test.shape[1]])
for r in range(0,test.shape[0]):
    for c in range(0,test.shape[1]):
        pixel = test[r,c,:]
        if(np.all(pixel == blue)):
            prediction[r,c] = 1
        else:
            prediction[r,c] = 0
        
#%%
labels = labels.reshape(labels.shape[0]*labels.shape[1])
prediction = prediction.reshape(prediction.shape[0]*prediction.shape[1])
cf = confusion_matrix(labels, prediction)
print (cf) 
#%%
false_alarm = cf[1,0]/(cf[1,0]+cf[1,1])
misdetection = cf[0,1]/(cf[0,1]+cf[0,0])
#%%
with tf.Session() as sess:
    ypredT = tf.constant(prediction)
    ytrueT = tf.constant(labels)
    iou,conf_mat = tf.metrics.mean_iou(ytrueT, ypredT, num_classes=2)
    sess.run(tf.local_variables_initializer())
    conf_mat = sess.run([conf_mat])
    miou = sess.run([iou])
    print(miou)

























