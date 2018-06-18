# -*- coding: utf-8 -*-
"""
Created on Wed May 16 10:01:03 2018

@author: yuxi
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from PIL import Image
from function import cnn_model
import glob
import time

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
            if((pixel[0]==0)and(pixel[1]==0)):
                count = count +1
    all_pixel = test.shape[0]*test.shape[1]
    coral_percent = (count/all_pixel)*100 
    coral_percent = round(coral_percent,2)
    return coral_percent
    

#%%load model and weight
model = cnn_model()
model.summary()
#model.load_weights('CNN_weights-08.h5')
model.load_weights('./saved_models/2012images_trained_augmentation-6.6.h5')
#model.load_weights('./saved_models/2012images_trained_model.h5')
#model.load_weights('./saved_models/cifar10_ResNet14v1_model.007.h5')
#model.load_weights('./saved_models/2012images_data_augmentation_trained_model-50epoch.h5')
#%%
windowsize_r = 30
windowsize_c = 30
raw_image_files = glob.glob("./2013-image/*.jpg")
count = 0
for name in raw_image_files:
    name_piece = name.split("\\")[1]
    name_image = name_piece.split(".")[0]
    path_image = './2013-image/'+str(name_image)+'.jpg'
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
    testimage.save('./2013-image/'+str(name_image)+' result'+str(percent)+'.jpg')
    print(("---image%d finished in %s seconds ---" % (count,(time.time()-start_time))))
    print("there are {0}% coral in this image".format(percent))



#%%    
path_image = './New folder (2)/201308203-T-16-16-18_005 .jpg'

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
count = 0
result_image_files = glob.glob("./2012-result/*.jpg")
for name in result_image_files:
    testimage = Image.open(name)
    name_piece = name.split("\\")[1]
    name_image = name_piece.split(".")[0]
    percent = percent_coral(testimage)
    testimage.save('./2012-result/'+str(name_image)+str(percent)+'.jpg')
    count = count +1
    print("image {0} finished".format(count))
     
#%%
count = 0
blue = np.array([0,0,255], dtype=np.uint8)
test = np.uint8(testimage)
pixel = np.zeros([1,1,3],dtype = np.uint8)
for r in range(0,test.shape[0]):
    for c in range(0,test.shape[1]):
        pixel = test[r,c,:]
        #if(np.all(pixel == blue)):
        if((pixel[0]==0)and(pixel[1]==0)):
            count = count +1
all_pixel = test.shape[0]*test.shape[1]
coral_percent = (count/all_pixel)*100 
coral_percent = round(coral_percent,2)        
        
        
        


























