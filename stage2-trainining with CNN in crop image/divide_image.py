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

#grey_levels = 256
# Generate a test image
#test_image = np.random.randint(0,grey_levels, size=(11,11))
#2012images_data_augmentation_trained_model-50epoch
#%%load model and weight
model = cnn_model()
model.summary()
model.load_weights('./saved_models/2012images_trained_model.h5')
#model.load_weights('./saved_models/2012images_trained_model.h5')
#model.load_weights('./saved_models/cifar10_ResNet14v1_model.007.h5')
#model.load_weights('./saved_models/2012images_data_augmentation_trained_model-50epoch.h5')
#%%
windowsize_r = 30
windowsize_c = 30
raw_image_files = glob.glob("./2012-image/*.jpg")
count = 0
for name in raw_image_files:
    name_piece = name.split("\\")[1]
    name_image = name_piece.split(".")[0]
    path_image = './2012-image/'+str(name_image)+'.jpg'
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
    testimage.save('./2012-image/'+str(name_image)+'result.jpg')
    print(("---image%d finished in %s seconds ---" % (count,(time.time()-start_time))))
    



#%%    
path_image = '201208172_T-12-58-58_Dive_01_041.jpg'
testimage = Image.open(path_image)
test_image = imread(path_image)
# Define the window size
windowsize_r = 30
windowsize_c = 30
start_time = time.time()
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
print("--- %s seconds ---" % (time.time()-start_time))
#%%
for a in range(0,2018,6):
    print(a)
#%%
def testimage():
    testimage.paste((255,0,0),[1748,1236,2049,1537])
    testimage.paste((255,0,0),[0,0,300,300])
    plt.imshow(testimage)
    plt.axis('off')
#%%
plt.imshow(testimage)
plt.axis('off')
























