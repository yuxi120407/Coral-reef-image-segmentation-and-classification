# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 22:43:34 2018

@author: yuxi
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from PIL import Image,ImageDraw
from function import cnn_model,cnn_model1,cnn_model2
from skimage.measure import block_reduce
from sklearn.cluster import KMeans
import glob
import time
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from skimage import color

def fun(x,y):
    f = test_image[x:x+30,y:y+30]
    f = f.reshape(2700)
    return f

#%%
start_time = time.time()
color_vector = np.zeros([7,3])
color_vector[0] = np.array((0,0,0))
color_vector[1] = np.array((0,0,255))
color_vector[2] = np.array((105,105,105))
color_vector[3] = np.array((169,169,169))
color_vector[4] = np.array((255,0,0))
color_vector[5] = np.array((0,255,0))
color_vector[6] = np.array((255,255,0))
#step1 dowmsample the image
path_image = "./2012image/201208172_T-12-54-11_Dive_01_032.jpg"
img = imread(path_image)
test_image = img[0:1536:3,0:2048:3,:]
plt.imshow(test_image)
plt.axis('off')
##step2 get the results of downsample images with lower threshold CNN
stride = 30
check_palindrome = np.frompyfunc(fun, 2, 1)
x = np.arange(0,test_image.shape[0] - 30+1, stride)
y = np.arange(0,test_image.shape[1] - 30+1, stride)
X,Y = np.meshgrid(x, y)
zs = check_palindrome(np.ravel(X.T), np.ravel(Y.T))
fs = np.concatenate(zs,axis=0).astype(np.uint8)
image_all_patches = fs.reshape(len(x)*len(y),30,30,3)
pred = model.predict(image_all_patches,verbose=1)
y_pred = np.argmax(pred,axis=1)
y_pred = y_pred.reshape(len(x),len(y))
y_pred = np.uint8(y_pred)
##step3 mapping back at original image
factor = 3
dis = 20
g = np.where(y_pred==0)
local_x = g[0]*30 + 15
local_y = g[1]*30 + 15
are_x1 = factor*(local_x-dis-15)
are_x1[are_x1<0]=0
are_x2 = factor*(local_x+dis+15)
are_x2[are_x2>img.shape[0]]=img.shape[0]
are_y1 = factor*(local_y-dis-15)
are_y1[are_y1<0]=0
are_y2 = factor*(local_y+dis+15)
are_y2[are_y2>img.shape[1]]=img.shape[1]
length = len(are_y2)
image_label = np.zeros([img.shape[0],img.shape[1]]) 
for i in range(length):
    image_label[are_x1[i]:are_x2[i],are_y1[i]:are_y2[i]] = 1
plt.imshow(image_label,cmap='Greys_r')
##only search areas in the potential areas
nu_pa = np.ones([1,30,30,3])*300
nu_pa = nu_pa.reshape(2700)
def new_fun(x,y):
    if image_label[x,y]==1:
        m = img[x:x+30,y:y+30]
        m = m.reshape(2700)
        return m
    else:
        return nu_pa               
check_palindrome = np.frompyfunc(new_fun, 2, 1)
stride = 6
x = np.arange(0,img.shape[0] - 30+1, stride)
y = np.arange(0,img.shape[1] - 30+1, stride)
X,Y = np.meshgrid(x, y)
zs= check_palindrome(np.ravel(X.T), np.ravel(Y.T))
fs = np.concatenate(zs,axis=0).astype(np.uint16)
ff = fs[fs!=300]
image_some_patches = ff.reshape(-1,30,30,3)
pred_some = detail_model.predict(image_some_patches,verbose=1)
y_pred_some = np.argmax(pred_some,axis=1)
j = np.arange(0,229294800,2700)
ffs = fs[j]
gg = np.where(ffs!=300)
gg = gg[0]
index_x = gg//337
index_y = gg%337
size_some_patch = len(index_x)
re_pre = np.zeros([252,337])
for i in range (size_some_patch):
    re_pre[index_x[i],index_y[i]] = y_pred_some[i]+1
re_pre = re_pre.repeat(stride, axis = 0).repeat(stride, axis = 1)
result=color.label2rgb(re_pre,colors =color_vector, kind = 'overlay')
result = np.uint8(result)
plt.imshow(result)
print('the time of coarse_to_fine is: {:.4f}'.format(time.time()-start_time))



