# -*- coding: utf-8 -*-
"""
Created on Fri May 11 10:18:55 2018

@author: Xi Yu
"""

import matplotlib.pyplot as plt
import numpy as np 
from skimage.io import imread
from PIL import Image


original_x = 30720
original_y = 23040

true_x = 2048
true_y = 1536
#%% load the original dimension and the account of random points
f = open('201208172_T-12-58-58_Dive_01_041.txt')
line = f.readlines()
line1 = line[0]
new = line1.split(",") 
demision_length = int(new[2])
demision_width = int(new[3])
count_points = int(line[5])
f.close()
#%% load the true coordinate
data_new = []
f = open('201208172_T-12-58-58_Dive_01_041.txt')
data = f.readlines()[6:56]
corrdinate = np.zeros([50,2])
for n in range(50):   
    data1 = data[n].split(",")
    corrdinate[n,0] =int(int(data1[0])*2048/30720)
    corrdinate[n,1] =int(int(data1[1])*1536/23040)
f.close()
#%%load the label of each pixels
label_encode = np.zeros(50)
f = open('201208172_T-12-58-58_Dive_01_041.txt')
label = f.readlines()[56:106]
for m in range(50):
    label1 = label[m].split(",")
    label2 = label1[1]
    new_l = label2.replace('\"', '')
    if (new_l=="Agalg"):
        label_encode[m] = 0
    elif (new_l=="DCP"):
        label_encode[m] = 1
    elif (new_l=="ROC"):
        label_encode[m] = 2
    elif (new_l=="TWS"):
        label_encode[m] = 3
    elif (new_l=="CCA"):
        label_encode[m] = 4
    elif (new_l=="Davi"):
        label_encode[m] = 5
    elif (new_l=="Ana"):
        label_encode[m] = 6 
    else:
        label_encode[m] = 7
f.close()
        
#%%open a file to write coordinate, label and the name of the image file
path_image = './image/201208172_T-13-51-38_Dive_01_094.jpg'
name_image = path_image.split("/")[2] 
name_txt = name_image.split(".")[0]
with open(str('new/')+name_txt+str('.txt'), 'w+') as f:
  f.write(name_image+str('\n'))
  f.write('x,y,label\n')
  for i in range(50):
      x = corrdinate[i,0]
      y = corrdinate[i,1]
      label = int(label_encode[i])
      f.write('{0},{1},{2}\n'.format(x,y,label))
    
#%%load the image and crop 
image = imread('201208172_T-12-58-58_Dive_01_041.jpg')
#plt.imshow(image)
all_image = np.zeros([50,30,30,3])
for i in range(50):
   all_image[i,:,:,:] = image[corrdinate[i,1]-15:corrdinate[i,1]+15,corrdinate[i,0]-15:corrdinate[i,0]+15] 





#%% write data into txt files
#f1 = open('data.txt','w')
#f1.write('hello boy!')
#f1.close()
