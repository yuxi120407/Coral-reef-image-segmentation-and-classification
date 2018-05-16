# -*- coding: utf-8 -*-
"""
Created on Wed May 16 10:01:03 2018

@author: yuxi
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from PIL import Image
from crop_image_cnn import cnn_model

#grey_levels = 256
# Generate a test image
#test_image = np.random.randint(0,grey_levels, size=(11,11))

#%%load model and weight
model = cnn_model()
model.summary()
model.load_weights('./saved_models/2012images_trained_model.h5')











#%%
path_image = '201208172_T-12-58-58_Dive_01_041.jpg'
testimage = Image.open('201208172_T-12-58-58_Dive_01_041.jpg')
test_image = imread(path_image)
# Define the window size
windowsize_r = 30
windowsize_c = 30

# Crop out the window and calculate the histogram
for r in range(0,test_image.shape[0] - windowsize_r+1, 1):
    for c in range(0,test_image.shape[1] - windowsize_c+1, 1):
        window = test_image[r:r+windowsize_r,c:c+windowsize_c]
        window = window.reshape(-1,30,30,3)
        pred = model.predict(window,verbose=2)
        y_pred = np.argmax(pred,axis=1)
        if(y_pred==4):
            testimage.paste((0,255,0),[c,r,c+windowsize_c,r+windowsize_r])
        elif(y_pred==6):
            testimage.paste((255,255,0),[c,r,c+windowsize_c,r+windowsize_r])
        elif(y_pred==7):
            testimage.paste((0,255,255),[c,r,c+windowsize_c,r+windowsize_r])
        else:
            testimage.paste((0,0,255),[c,r,c+windowsize_c,r+windowsize_r])
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

#%%
for a in range(0,7,1):
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























