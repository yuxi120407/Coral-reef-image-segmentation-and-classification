# -*- coding: utf-8 -*-
"""
Created on Wed May 16 10:01:03 2018

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
def test_img(img):
    windowsize_r = 30
    windowsize_c = 30
    count = 0
    testimage = Image.new('RGB', (img.shape[1],img.shape[0]))
    #result = np.zeros([510,690,3],dtype=np.uint8)
    #result[:,:683,:] = k[:510,:,:]
    start_time = time.time()
    for r in range(0,img.shape[0] - windowsize_r+1, 6):
            for c in range(0,img.shape[1] - windowsize_c+1, 6):
                window = back_img[r:r+windowsize_r,c:c+windowsize_c]
                window = window.reshape(-1,30,30,3)
                pred = model.predict(window,verbose=2)
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
    #im.show()
#%%load model and weight
model = cnn_model()
model.summary()
model.load_weights('./model_weight/2012images-areas-7.5-50epoch.h5')
#%%
windowsize_r = 30
windowsize_c = 30
raw_image_files = glob.glob("./New folder/*.jpg")
count = 0
for name in raw_image_files:
    name_piece = name.split("\\")[1]
    name_image = name_piece.split(".")[0]
    path_image = './New folder/'+str(name_image)+'.jpg'
    testimage = Image.open(path_image)
    test_image = imread(path_image)
    start_time = time.time()
    for r in range(0,test_image.shape[0] - windowsize_r+1, 6):
        for c in range(0,test_image.shape[1] - windowsize_c+1, 6):
            window = test_image[r:r+windowsize_r,c:c+windowsize_c]
            window = window.reshape(-1,30,30,3)
            pred = model.predict(window,verbose=2)
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
    testimage.save('./result_image/'+str(name_image)+' result'+str(percent)+'.jpg')
    print(("---image%d finished in %s seconds ---" % (count,(time.time()-start_time))))
    print("there are {0}% coral in this image".format(percent))
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#downsample
path_image = "./2012image/201208172_T-12-58-58_Dive_01_041.jpg"
img = imread(path_image)
img_downsample = block_reduce(img, block_size=(3, 3, 1), func=np.max)
img_new = np.uint8(img_downsample)
plt.imshow(img_new)

#%%
#step1 downsample image
path_image = "./2012image/201208172_T-12-54-11_Dive_01_032.jpg"
img = imread(path_image)
k = img[0:1536:3,0:2048:3,:]
plt.imshow(k)
plt.axis('off')
#%%
#step2 coarse search
windowsize_r = 30
windowsize_c = 30
count = 0
test_image = img_new
testimage = Image.new('RGB', (683,512))
result = np.zeros([510,690,3],dtype=np.uint8)
result[:,0:683,:] = k[0:510,:]


#testimage = Image.new('RGB', (410,308))
#result = np.zeros([300,420,3],dtype=np.uint8)
#result[:,0:410,:] = k[0:300,:]

#testimage = Image.new('RGB', (293,220))
#result = np.zeros([210,300,3],dtype=np.uint8)
#result[:,0:293:] = k[0:210,:]
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
percent = percent_coral(testimage)
print(("---image%d finished in %s seconds ---" % (count,(time.time()-start_time))))
print("there are {0}% coral in this image".format(percent))
plt.imshow(testimage)
plt.axis('off')
#%%
##step3 find center of each area with k-mean
coral_loc = np.vstack((loc_x,loc_y)).T
#%%
kmeans = KMeans(n_clusters=4,random_state=0,max_iter=30).fit(coral_loc)
center = kmeans.cluster_centers_
#%%store the location of the coral
coral_loc = np.vstack((loc_x,loc_y)).T
#%%plot the clusting
pre_lab = kmeans.predict(coral_loc)
size_pre_lab = len(pre_lab)
pre_lab = pre_lab.reshape(size_pre_lab,1)
all_data = np.hstack((coral_loc,pre_lab))
data0 = np.array([[0,0]])
data1 = np.array([[0,0]])
data2 = np.array([[0,0]])
data3 = np.array([[0,0]])

for i in range(size_pre_lab):
    if(all_data[i,2]==0): 
        data0 = np.append(data0,[[all_data[i,0],all_data[i,1]]],axis=0)
        plt.plot(all_data[i,0],all_data[i,1],'b^')
    if(all_data[i,2]==1):
        data1 = np.append(data1,[[all_data[i,0],all_data[i,1]]],axis=0)
        plt.plot(all_data[i,0],all_data[i,1],'bs')
    if(all_data[i,2]==2):
        data2 = np.append(data2,[[all_data[i,0],all_data[i,1]]],axis=0)
        plt.plot(all_data[i,0],all_data[i,1],'b*')
    if(all_data[i,2]==3):
        data3 = np.append(data3,[[all_data[i,0],all_data[i,1]]],axis=0)
        plt.plot(all_data[i,0],all_data[i,1],'b+')
data0 = data0[1:,:]
data1 = data1[1:,:]
data2 = data2[1:,:]
data3 = data3[1:,:]

plt.scatter(center[:,0],center[:,1],color = 'red')
#%%step 5 find the length of the area
dis = get_dis(data3,center[3,:])
#%%step 6 back to the image
back_img,area_x1,area_x2,area_y1,area_y2 = get_back_img(img,center[3,:],dis)
#%%step7 test the areas in the back image
test_img(back_img)
#%%
im = Image.open("./2012image/201208172_T-12-58-58_Dive_01_041.jpg")
draw = ImageDraw.Draw(im)
#%%
plot_rectangle(draw,area_x1,area_x2,area_y1,area_y2)
#%%
im.save('test1.jpg')
#%%
image_label = np.zeros([img.shape[0],img.shape[1]]) 
size = len(coral_loc)
im = Image.open("./2012image/201208172_T-12-54-11_Dive_01_032.jpg")
img = imread("./2012image/201208172_T-12-54-11_Dive_01_032.jpg")
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
labels = imread("./image_ground truth/image/201208172_T-12-54-11_Dive_01_032_json/label.png")
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
print(false_alarm)
print(misdetection)
#%%
with tf.Session() as sess:
    ypredT = tf.constant(prediction)
    ytrueT = tf.constant(labels)
    iou,conf_mat = tf.metrics.mean_iou(ytrueT, ypredT, num_classes=2)
    sess.run(tf.local_variables_initializer())
    conf_mat = sess.run([conf_mat])
    miou = sess.run([iou])
    print(miou)   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
 
 
 
 