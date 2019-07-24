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
#%%
def max_downsample(img,rate):
    im_r = img[:,:,0]
    im_g = img[:,:,1]
    im_b = img[:,:,2]
    im_r_max = block_reduce(im_r, (rate,rate), np.max)
    im_g_max = block_reduce(im_g, (rate,rate), np.max)
    im_b_max = block_reduce(im_b, (rate,rate), np.max)
    im_max = np.zeros([im_r_max.shape[0],im_r_max.shape[1],3],dtype=np.uint8)
    im_max[:,:,0]=im_r_max
    im_max[:,:,1]=im_g_max
    im_max[:,:,2]=im_b_max
    return im_max

def mean_downsample(img,rate):
    im_r = img[:,:,0]
    im_g = img[:,:,1]
    im_b = img[:,:,2]
    im_r_mean = block_reduce(im_r, (rate,rate), np.mean)
    im_g_mean = block_reduce(im_g, (rate,rate), np.mean)
    im_b_mean = block_reduce(im_b, (rate,rate), np.mean)
    im_mean = np.zeros([im_r_mean.shape[0],im_r_mean.shape[1],3],dtype=np.uint8)
    im_mean[:,:,0]=im_r_mean
    im_mean[:,:,1]=im_g_mean
    im_mean[:,:,2]=im_b_mean
    return im_mean

def plot_rectangle(draw,area_x1,area_x2,area_y1,area_y2):
    #draw = ImageDraw.Draw(im)
    draw.line((area_y1, area_x1, area_y1, area_x2), fill="blue", width=10)
    draw.line((area_y1, area_x2, area_y2, area_x2), fill="blue", width=10)
    draw.line((area_y2, area_x2, area_y2, area_x1), fill="blue", width=10)
    draw.line((area_y2, area_x1, area_y1, area_x1), fill="blue", width=10)

#%%
def fun(x,y):
    f = test_image[x:x+30,y:y+30]
    f = f.reshape(2700)
    return f
#%%
def percent_coral(testimage):
    count = 0
    #blue = np.array([0,0,255], dtype=np.uint8)
    test = np.uint8(testimage)
    f = np.where(test[:,:,2]==255)
    count = len(f[0])
    all_pixel = test.shape[0]*test.shape[1]
    coral_percent = (count/all_pixel)*100 
    coral_percent = round(coral_percent,2)
    return coral_percent
#%%

color_vector = np.zeros([7,3])
color_vector[0] = np.array((0,0,0))
color_vector[1] = np.array((0,0,255))
color_vector[2] = np.array((105,105,105))
color_vector[3] = np.array((169,169,169))
color_vector[4] = np.array((255,0,0))
color_vector[5] = np.array((0,255,0))
color_vector[6] = np.array((255,255,0))
## detail model used for fine search
detail_model = cnn_model()
detail_model.summary()
detail_model.load_weights('./model_weight/2012images-areas-7.5-50epoch.h5')
## model used for coarse search
model = cnn_model()
model.summary()
model.load_weights('./model_weight/2018_10_14_weight0.2.h5')
#step1 dowmsample the image
path_image = "./2012image/201208172_T-12-54-11_Dive_01_032.jpg"
name_image = path_image.split('/')[2]
name_image = name_image.split('.')[0]
img = imread(path_image)
test_image = mean_downsample(img,5)
plt.imshow(test_image)
plt.axis('off')
#%%
##step2 get the results of downsample images with lower threshold CNN
start_time = time.time()
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
factor = 5
dis = 15
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
im = Image.open(path_image)
draw = ImageDraw.Draw(im)

for i in range(length):
    image_label[are_x1[i]:are_x2[i],are_y1[i]:are_y2[i]] = 1
    plot_rectangle(draw,are_x1[i],are_x2[i],are_y1[i],are_y2[i])

im.save('area_coarse_image/'+str(name_image)+'.pdf')  
#plt.imshow(image_label,cmap='Greys_r')
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
stride = 8
x = np.arange(0,img.shape[0] - 30+1, stride)
y = np.arange(0,img.shape[1] - 30+1, stride)
X,Y = np.meshgrid(x, y)
zs= check_palindrome(np.ravel(X.T), np.ravel(Y.T))
fs = np.concatenate(zs,axis=0).astype(np.uint16)
ff = fs[fs!=300]
image_some_patches = ff.reshape(-1,30,30,3)
pred_some = detail_model.predict(image_some_patches,verbose=1)
y_pred_some = np.argmax(pred_some,axis=1)
j = np.arange(0,x.shape[0]*y.shape[0]*2700,2700)#84924*2700=229294800
ffs = fs[j]
gg = np.where(ffs!=300)
gg = gg[0]
index_x = gg//y.shape[0]
index_y = gg%y.shape[0]
size_some_patch = len(index_x)
re_pre = np.zeros([x.shape[0],y.shape[0]])
for i in range (size_some_patch):
    re_pre[index_x[i],index_y[i]] = y_pred_some[i]+1
re_pre = re_pre.repeat(stride, axis = 0).repeat(stride, axis = 1)
result=color.label2rgb(re_pre,colors =color_vector, kind = 'overlay')
result = np.uint8(result)
percent = percent_coral(result)
print('the time of coarse_to_fine is: {:.2f}s'.format(time.time()-start_time))
plt.imshow(result)

fig = plt.gcf()
fig.set_size_inches(re_pre.shape[1]/100.0/3.0, re_pre.shape[0]/100.0/3.0)  
plt.gca().xaxis.set_major_locator(plt.NullLocator()) 
plt.gca().yaxis.set_major_locator(plt.NullLocator()) 
plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0) 
plt.margins(0,0)
#the address to save the result
plt.savefig('./result_coarse_image/'+'/'+str(name_image)+' result'+str(percent)+'.pdf', bbox_inches='tight')

print("there are {0}% coral in this image".format(percent))
#%%

#plt.close()















