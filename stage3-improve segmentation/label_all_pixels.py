# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 14:16:48 2018

@author: yuxi
"""

# import the necessary packages
import keras
from skimage.segmentation import slic
from skimage.data import astronaut
import matplotlib.pyplot as plt
from skimage.io import imread
import pylab
from data_augmentation import shuffle
import matplotlib.pyplot as plt
import numpy as np
import random
import glob
import time

from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries,find_boundaries
from skimage.util import img_as_float

#%%read image
path_image = "./201208172_T-12-58-58_Dive_01_041.jpg"
img = imread(path_image)
img = img_as_float(img)
#%%

segments_slic = slic(img, n_segments=300, compactness=15, sigma=1)
fig, ax = plt.subplots(1, 1, figsize=(10,10), sharex=True, sharey=True)
print('SLIC number of segments: {}'.format(len(np.unique(segments_slic))))
ax.imshow(mark_boundaries(img, segments_slic,color=(0,0,0)))
ax.set_title('SLIC')
plt.tight_layout()
plt.show()
#%%read original pixels labels
count_points = 50
label_image = np.zeros([1536,2048])
path_txt = "./2012new/201208172_T-12-58-58_Dive_01_041.txt"
txtfile = open(path_txt) 
lines = txtfile.readlines()[2:52]
for i in range(count_points):
    line_piece = lines[i]
    list_element = line_piece.split(',')
    l_x = int(list_element[0])
    l_y = int(list_element[1])
    original_label = int(list_element[2])
        
    label = segments_slic[l_y,l_x]
    #generate_sample(segments_slic,original_label,label,10,path_txt)
    a = np.where(segments_slic==label)
    a_y = a[0]
    a_x = a[1]
    zipped = zip(a_y,a_x)
    for i,j in zipped:
        label_image[i,j]=original_label+1
plt.imshow(label_image)    
txtfile.close()
#%%
label_image = label_image.reshape(1,1536,2048)
label_image_onehot = keras.utils.to_categorical(label_image, 7)





























