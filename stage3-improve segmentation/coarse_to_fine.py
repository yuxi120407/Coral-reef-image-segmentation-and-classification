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
#%%find the precent areas of coral
def percent_coral(testimage):
    count = 0
    #blue = np.array([0,0,255], dtype=np.uint8)
    test = np.uint8(testimage)
    pixel = np.zeros([1,1,3],dtype = np.uint8)
    for r in range(0,test.shape[0]):
        for c in range(0,test.shape[1]):
            pixel = test[r,c,:]
            #if(np.all(pixel == blue)):
            if((pixel[0]==0)and(pixel[1]==0)and(pixel[2]==255)):
                count = count +1
    all_pixel = test.shape[0]*test.shape[1]
    coral_percent = (count/all_pixel)*100 
    coral_percent = round(coral_percent,2)
    return coral_percent

#%%test the image
def test_img_loc(img,model):
    windowsize_r = 30
    windowsize_r = 30
    windowsize_c = 30
    count = 0
    #test_image = img_new
    testimage = Image.new('RGB', (683,512))
    result = np.zeros([510,690,3],dtype=np.uint8)
    result[:,0:683,:] = img[0:510,:]
    start_time = time.time()
    loc_x = []
    loc_y = []
    for r in range(0,result.shape[0] - windowsize_r+1, 30):
            for c in range(0,result.shape[1] - windowsize_c+1, 30):
                window = result[r:r+windowsize_r,c:c+windowsize_c]
                window = window.reshape(-1,30,30,3)
                pred = model.predict(window,verbose=2)
                y_pred = np.argmax(pred,axis=1)
                if(y_pred==0):#coral---blue
                    testimage.paste((0,0,255),[c,r,c+windowsize_c,r+windowsize_r])
                    loc_x = np.append(loc_x,r+15)
                    loc_y = np.append(loc_y,c+15) 
                    
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
    count = count+1
    print(("---image%d finished in %s seconds ---" % (count,(time.time()-start_time))))
    plt.imshow(testimage)
    plt.axis('off')
    return loc_x,loc_y
#%%
def test_img_label(img,model,img_label):
    start_time = time.time()
    windowsize_r = 30
    windowsize_c = 30
    count = 0
    testimage = Image.new('RGB', (img.shape[1],img.shape[0]))
    for r in range(0,img.shape[0] - windowsize_r+1, 6):
        for c in range(0,img.shape[1] - windowsize_c+1, 6):
            if(image_label[r,c]==1):
                window = img[r:r+windowsize_r,c:c+windowsize_c]
                window = window.reshape(-1,30,30,3)
                pred = detail_model.predict(window,verbose=2)
                y_pred = np.argmax(pred,axis=1)
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
    count = count+1    
    percent = percent_coral(testimage)
    print(("---image%d finished in %s seconds ---" % (count,(time.time()-start_time))))
    print("there are {0}% coral in this image".format(percent))
    plt.imshow(testimage)    
    plt.axis('off')
#%%
#get the area of the original image
def get_back_img(img,center,dis,factor):
    area_x1 = factor*np.int(center[0]-dis-15)
    area_x2 = factor*np.int(center[0]+dis+15)
    area_y1 = factor*np.int(center[1]-dis-15)
    area_y2 = factor*np.int(center[1]+dis+15)
    
    if(area_x1<0):
        area_x1 = 0
    if(area_x2>img.shape[0]):
        area_x2 = img.shape[0]
    if(area_y1<0):
        area_y1 = 0
    if(area_y2>img.shape[1]):
        area_y2 = img.shape[1]
    back_img = img[area_x1:area_x2,area_y1:area_y2]
    plt.imshow(np.uint8(back_img))
    return back_img,area_x1,area_x2,area_y1,area_y2
#get the length of the area
def get_dis(data,center):
    temp_dis = np.zeros(len(data))
    for i in range(len(data)):
        temp = center-data[i,:]
        temp_value = temp@temp.T
        temp_dis[i] =  np.sqrt(temp_value)
    dis = np.max(temp_dis)
    return dis
#plot the rectangle in the original image
def plot_rectangle(draw,area_x1,area_x2,area_y1,area_y2):
    #draw = ImageDraw.Draw(im)
    draw.line((area_y1, area_x1, area_y1, area_x2), fill="blue", width=10)
    draw.line((area_y1, area_x2, area_y2, area_x2), fill="blue", width=10)
    draw.line((area_y2, area_x2, area_y2, area_x1), fill="blue", width=10)
    draw.line((area_y2, area_x1, area_y1, area_x1), fill="blue", width=10)
#%%#%%load coarse model and weight
model = cnn_model()
model.summary()
model.load_weights('./model_weight/2018_10_14_weight1.5.h5')
#%%
#step1 downsample image
path_image = "./2012image/201208172_T-12-56-18_Dive_01_036.jpg"
img = imread(path_image)
k = img[0:1536:3,0:2048:3,:]
plt.imshow(k)
plt.axis('off')
#%%
#step2 coarse search the downsampe image
loc_x,loc_y = test_img_loc(k,model)
#%%store the location of the coral
coral_loc = np.vstack((loc_x,loc_y)).T
image_label = np.zeros([img.shape[0],img.shape[1]]) 
size = len(coral_loc)
im = Image.open(path_image)
img = imread(path_image)
draw = ImageDraw.Draw(im)
for i in range(size):
    _,area_x1,area_x2,area_y1,area_y2 = get_back_img(img,coral_loc[i,:],20,3)
    image_label[area_x1:area_x2,area_y1:area_y2] = 1
    plot_rectangle(draw,area_x1,area_x2,area_y1,area_y2)
im.save('test7.jpg')   
plt.imshow(image_label,cmap='Greys_r')
plt.axis('off') 
#%%
detail_model = cnn_model()
detail_model.summary()
detail_model.load_weights('./model_weight/2012images-areas-7.5-50epoch.h5')
#%%
test_img_label(img,detail_model,image_label)















